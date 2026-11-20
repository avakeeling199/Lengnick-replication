"""
Select overstocked situations from simulation prompt logs as an extra benchmark set.

Example:
  python scripts/benchmark/select_overstock.py \
      --source gemma=diagnostics/sim60/gemma4_31b_q4/prompts.jsonl \
      --source qwen38=diagnostics/sim60/qwen38_27b_fp8_mb368/prompts.jsonl
"""
import argparse
import random

from render_prompts import read_jsonl, write_jsonl   # reuse your functions (same folder)

STARTUP_LAST_MONTH = 14


def price_pos(r):
    """where a firms current price sits relative to its allowed range"""
    if r["p_f"] >= r["p_f_upperbar"]:
        return "ceiling"
    elif r["p_f"] <= r["p_f_lowerbar"]:
        return "floor"
    else:
        return "interior"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", required=True, help="NAME=PATH to a prompt log; repeatable")
    p.add_argument("--per-group", type=int, default=20, help="max prompts per price position per source")
    p.add_argument("--out", default="diagnostics/benchmark/prompt_sets/overstock_set.jsonl")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)

    chosen = []
    for source in args.source:
        name, path = source.split("=", 1)
        over = [r for r in read_jsonl(path) if r["i_f"] > r["i_f_upperbar"]]        
        for pos in ["ceiling", "interior", "floor"]:
            group = [r for r in over if price_pos(r) == pos]
            picked = rng.sample(group, min(args.per_group, len(group)))
            for r in picked:
                new = dict(r)
                phase = "startup" if r["month"] <= STARTUP_LAST_MONTH else "steady"
                new["prompt_id"] = f"{name}_m{r['month']:03d}_f{r['firm_id']}"
                new["source"] = name
                new["stratum"] = f"{phase}|above|{pos}"
                new["expected"] = "hold" if pos == "floor" else "lower"                    
                chosen.append(new)
        print(f"{name}: {len(over)} overstocked, selected {sum(c['source'] == name for c in chosen)}")

    write_jsonl(args.out, chosen)
    print(f"wrote {len(chosen)} prompts to {args.out}")


if __name__ == "__main__":
    main()
