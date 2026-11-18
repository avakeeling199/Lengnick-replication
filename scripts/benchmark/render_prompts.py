"""
Rebuild the benchmark prompt sets with a different prompt variant.

Example:
  python scripts/benchmark/render_prompts.py --variant goal_state
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from src.llm.prompts import PROMPT_BUILDERS

SET_FILES = ["quality_set.jsonl", "throughput_set.jsonl"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, records):
    with open(path, 'w') as f:
        for record in records:
            f.write(json.dumps(record, default=float) + "\n")

def prompt_inputs(r):
    """Map a recorded entry's field names onto the prompt functions' argument names."""
    return dict(
        i_f=r["i_f"], p_f=r["p_f"], mc_f=r["mc_f"], demand=r["demand"],
        price_history=r["price_hist"], demand_history=r["demand_hist"], inventory_history=r["inv_hist"],
        i_f_lowerbar=r["i_f_lowerbar"], i_f_upperbar=r["i_f_upperbar"],
        p_f_lowerbar=r["p_f_lowerbar"], p_f_upperbar=r["p_f_upperbar"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True, choices=sorted(PROMPT_BUILDERS))
    p.add_argument("--recording", default="diagnostics/benchmark/recording/vllm_60m_prompts.jsonl")
    p.add_argument("--sets-dir", default="diagnostics/benchmark/prompt_sets")
    args = p.parse_args()

    build = PROMPT_BUILDERS[args.variant]
    original = {(r["month"], r["firm_id"]): r for r in read_jsonl(args.recording)}
    out_dir = os.path.join(args.sets_dir, args.variant)
    os.makedirs(out_dir, exist_ok=True)

    for name in SET_FILES:
        selected = read_jsonl(os.path.join(args.sets_dir, name))
        rendered, matches = [], 0
        for r in selected:
            orig = original[(r["month"], r["firm_id"])]                          
            prompt = build(**prompt_inputs(orig))                     
            new = dict(r)                       
            new["messages"] = [{"role": "system", "content": ""},
                               {"role": "user", "content": prompt}]
            new["prompt_variant"] = args.variant
            matches += prompt == orig["messages"][1]["content"]
            rendered.append(new)
        write_jsonl(os.path.join(out_dir, name), rendered)
        print(f"{name}: {len(rendered)} prompts -> {out_dir}  "
              f"(identical to recorded prompt: {matches}/{len(rendered)})")


if __name__ == "__main__":
    main()
