"""
Build the "change one input" set: real situations copied at several inventory levels.

For each chosen situation, every copy is identical (sales, price, marginal cost, bands and the
whole history) except the current inventory, which is set to a multiple of last month's sales.
Any change in a model's answers across a situation's copies is therefore due to inventory alone.

Situations are drawn from the original recording (exact inputs), restricted to steady state
(month > 14), a current price strictly inside its allowed range (so the band limits never block a
move under the rule prompts) and at least --min-sales units sold last month.

The default levels sit clear of the rule's band edges (a quarter of sales and one month of sales):
two below the band, two inside it and two above it.

Example:
  python scripts/benchmark/make_perturbations.py
then render it for each prompt variant:
  python scripts/benchmark/render_prompts.py --variant goal_state --sets perturb_set.jsonl --self-inputs
"""
import argparse
import random

from render_prompts import read_jsonl, write_jsonl, prompt_inputs
from src.llm.prompts import PROMPT_BUILDERS

STARTUP_LAST_MONTH = 14
INPUT_FIELDS = ["month", "firm_id", "i_f", "p_f", "mc_f", "demand", "price_hist", "demand_hist",
                "inv_hist", "i_f_lowerbar", "i_f_upperbar", "p_f_lowerbar", "p_f_upperbar"]


def inv_state(r):
    if r["i_f"] < r["i_f_lowerbar"]:
        return "below"
    if r["i_f"] > r["i_f_upperbar"]:
        return "above"
    return "within"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recording", default="diagnostics/benchmark/recording/vllm_60m_prompts.jsonl")
    p.add_argument("--n", type=int, default=50, help="number of base situations")
    p.add_argument("--levels", default="0.05,0.15,0.4,0.8,1.5,2.5",
                   help="inventory as multiples of last month's sales")
    p.add_argument("--min-sales", type=float, default=20)
    p.add_argument("--out", default="diagnostics/benchmark/prompt_sets/perturb_set.jsonl")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    levels = [float(x) for x in args.levels.split(",")]
    rng = random.Random(args.seed)

    pool = [r for r in read_jsonl(args.recording)
            if r["month"] > STARTUP_LAST_MONTH
            and r["p_f_lowerbar"] < r["p_f"] < r["p_f_upperbar"]
            and r["demand"] >= args.min_sales]
    # one situation per firm at most, so no single firm dominates
    by_firm = {}
    for r in rng.sample(pool, len(pool)):
        by_firm.setdefault(r["firm_id"], r)
    bases = rng.sample(sorted(by_firm.values(), key=lambda r: r["firm_id"]), min(args.n, len(by_firm)))

    out = []
    for base in bases:
        base_id = f"m{base['month']:03d}_f{base['firm_id']}"
        for level in levels:
            new = {k: base[k] for k in INPUT_FIELDS}             # inputs only: no recorded answer
            new["i_f"] = float(round(level * base["demand"]))   # whole units, like the simulation
            state = inv_state(new)
            new.update({
                "prompt_id": f"pert_{base_id}_x{level:g}",
                "base_id": base_id,
                "inv_level": level,
                "source": "perturb",
                "stratum": f"steady|{state}|interior",
                "expected": {"below": "raise", "above": "lower", "within": "hold"}[state],
                "messages": [{"role": "system", "content": ""},
                             {"role": "user", "content": PROMPT_BUILDERS["strict"](**prompt_inputs(new))}],
            })
            out.append(new)

    write_jsonl(args.out, out)
    states = {}
    for r in out:
        states[(r["inv_level"], r["stratum"].split("|")[1])] = states.get((r["inv_level"], r["stratum"].split("|")[1]), 0) + 1
    print(f"{len(bases)} situations x {len(levels)} levels = {len(out)} prompts -> {args.out}")
    for (level, state), n in sorted(states.items()):
        print(f"  inventory = {level:g} x sales: {n} prompts {state} the band")


if __name__ == "__main__":
    main()
