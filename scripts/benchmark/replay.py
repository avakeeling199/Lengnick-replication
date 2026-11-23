import argparse, asyncio, subprocess, json, os, sys, time, zlib

import pandas as pd
from openai import AsyncOpenAI

# make 'src' importable when run as 'python scripts/benchmark/replay,py'
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from src.llm.prompts import PRICE_RESPONSE_FORMAT
from src.llm.parsing import parse_llm_response

def load_prompts(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def sampling_kwargs(args):
    return dict(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        response_format=PRICE_RESPONSE_FORMAT,
        extra_body={"top_k": -1, "repetition_penalty": 1.0},
    )

async def call_one(client, sem, record, repeat, kwargs, model):
    seed = zlib.crc32(f"{record['prompt_id']}|{repeat}".encode())
    row = dict(
    prompt_id=record['prompt_id'],
    month=record['month'],
    firm_id=record['firm_id'],
    stratum=record['stratum'],
    expected=record['expected'],
    repeat=repeat,
    seed=seed,
    latency_s=None, raw_response=None, finish_reason=None,
    prompt_tokens=None, completion_tokens=None, error=None,
    thinking_chars=None,
    )
    async with sem:
        t0 = time.perf_counter()
        try:
            resp = await client.chat.completions.create(
                model=model, messages=record['messages'], seed=seed, **kwargs)
            # 3. success: fill in what came back
            row['latency_s'] = time.perf_counter() - t0
            choice = resp.choices[0]
            row['raw_response'] = choice.message.content
            row['finish_reason'] = choice.finish_reason
            usage = getattr(resp, "usage", None)
            row['prompt_tokens'] = getattr(usage, "prompt_tokens", None)
            row['completion_tokens'] = getattr(usage, "completion_tokens", None)
            # thinking models: vLLM's reasoning parser puts the thinking text in a separate field
            extra = getattr(choice.message, "model_extra", None) or {}
            thinking = extra.get("reasoning_content") or extra.get("reasoning")
            row['thinking_chars'] = len(thinking) if thinking else 0
        except Exception as e:
            #4. failure: record how long it took and what went wrong
            row['latency_s'] = time.perf_counter() - t0
            row['error'] = repr(e)

    # 5. outside the sem: interpret the text (no network, so no need to holf a slot)
    if row['raw_response'] is not None:
        row.update(interpret(row['raw_response'], record))
    return row

def interpret(raw_text, record):
    # start with 'nothing worked' values. anything we manage to read overwrites these.
    out = dict(json_ok=False, raw_action=None, raw_new_price=None, raw_reasoning=None)

    # 1. what did the model ask for?
    try:
        parsed = json.loads(raw_text)
        out['raw_action'] = parsed.get('action')
        out['raw_new_price'] = parsed.get('new_price')
        out['raw_reasoning'] = parsed.get('reasoning')
        out['json_ok'] = True
    except (json.JSONDecodeError, AttributeError):
        pass
    
    # 2. what would the simulation have done with that answer? 
    final_price, final_reasoning, final_ok = parse_llm_response(
        raw_text, 
        field = "new_price",
        current_value=record['p_f'],
        lowerbar=record['p_f_lowerbar'],
        upperbar=record['p_f_upperbar'],
    )
    out['final_price'] = final_price
    out['final_reasoning'] = final_reasoning
    out['final_ok'] = final_ok

    return out 

async def run_quality(client, model, records, k, concurrency, kwargs):
    sem = asyncio.Semaphore(concurrency)
    jobs = [call_one(client, sem, r, rep, kwargs, model)
            for r in records for rep in range(k)]
    return await asyncio.gather(*jobs)

async def run_throughput(client, model, records, concurrency, kwargs):
    # group the records by month: {30: [100 records], 31: [100 records], ...}
    by_month = {}
    for r in records:
        by_month.setdefault(r['month'], []).append(r)

    sem = asyncio.Semaphore(concurrency)
    rows = []
    for month in sorted(by_month):
        t0 = time.perf_counter()
        month_rows = await asyncio.gather(
            *[call_one(client, sem, r, 0, kwargs, model) for r in by_month[month]])
        month_wall_s = time.perf_counter() - t0

        for row in month_rows:
            row['month_wall_s'] = month_wall_s
        rows.extend(month_rows)
        print(f"month {month}: {month_wall_s:.1f}s", flush=True)
    return rows

async def check_model(client, model):
    page = await client.models.list()
    served = [m.id for m in page.data]
    if model not in served:
        raise SystemExit(f"model {model!r} not served here; server has {served}")

async def run(args, client, records, kwargs):
    await check_model(client, args.model)
    if args.mode == "quality":
        return await run_quality(client, args.model, records, args.k, args.concurrency, kwargs)
    else: 
        return await run_throughput(client, args.model, records, args.concurrency, kwargs)

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts", required=True, help="quality_set.jsonl or throughput_set.jsonl")
    p.add_argument("--mode", required=True, choices=["quality", "throughput"], help="quality or throughput")
    p.add_argument("--model", required=True, help="name vLLM serves the model under (e.g. qwen3.8-27b)")
    p.add_argument("--label", required=True, help="short name for the result files")
    p.add_argument("--api-base", default="http://localhost:8000/v1" )
    p.add_argument("--k",default=5, type=int, help="repeats per prompt (quality mode)")
    p.add_argument("--concurrency", type=int, default=100)
    p.add_argument("--out", required=True, help="output CSV path")
    p.add_argument("--temperature", type=float, default=0.3, help="temperature sent to the LLM")
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=float, default=120, help="seconds before a request counts as failed")
    args=p.parse_args()

    started = time.strftime("%Y-%m-%d %H:%M:%S")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    client = AsyncOpenAI(base_url=args.api_base, api_key='dummy', max_retries=0)

    records = load_prompts(args.prompts)
    kwargs = sampling_kwargs(args)

    rows = asyncio.run(run(args, client, records, kwargs))

    df = pd.DataFrame(rows)
    df['label'] = args.label
    df.to_csv(args.out, index=False)

    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                            text=True, cwd=REPO_ROOT).stdout.strip()
    meta = dict(
        args=vars(args), 
        git_commit=commit,
        started=started,
        finished=time.strftime("%Y-%m-%d %H:%M:%S"),
        n_rows=len(df),
        n_errors=int(df['error'].notna().sum()),
    )
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()

