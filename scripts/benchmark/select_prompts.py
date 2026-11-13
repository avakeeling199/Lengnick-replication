"""
Select the fixed prompt sets for the LLM prompt-replay benchmark.

Reads a prompt log written by `main.py --prompt-log` and writes:
  quality_set.jsonl     ~200 prompts, stratified by economic situation, for scoring decisions
  throughput_set.jsonl  every prompt from a block of consecutive months, for timing
  strata_counts.csv     how many prompts each situation had and how many were selected
  selection_meta.json   inputs, arguments and a hash of the input file, for provenance

Every candidate model is replayed against these same files, so run this once and keep the outputs.

Example:
  python scripts/select_prompts.py \
      --input diagnostics/benchmark/recording/vllm_60m_prompts.jsonl \
      --out-dir diagnostics/benchmark/prompt_sets
"""
import argparse
import hashlib
import json
import os

import pandas as pd

STARTUP_LAST_MONTH = 14  # economy reaches full employment around month 14


def parse_month_range(text):
    """'30-39' -> (30, 39); '30' -> (30, 30)."""
    lo, _, hi = text.partition("-")
    return int(lo), int(hi or lo)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def label_situations(df):
    """Add the columns that describe each prompt's economic situation."""
    df = df.copy()
    df["prompt_id"] = df.apply(lambda r: f"m{int(r.month):03d}_f{int(r.firm_id)}", axis=1)

    df["phase"] = df["month"].le(STARTUP_LAST_MONTH).map({True: "startup", False: "steady"})

    # inventory relative to its target band (the same band the prompt shows the model)
    df["inv_state"] = "within"
    df.loc[df["i_f"] < df["i_f_lowerbar"], "inv_state"] = "below"
    df.loc[df["i_f"] > df["i_f_upperbar"], "inv_state"] = "above"

    # price relative to its allowed band. At or past an edge, the parser blocks further moves that way
    df["price_pos"] = "interior"
    df.loc[df["p_f"] >= df["p_f_upperbar"], "price_pos"] = "ceiling"
    df.loc[df["p_f"] <= df["p_f_lowerbar"], "price_pos"] = "floor"

    # direction the rule-based pricer would allow. Rule-based firms only move with
    # probability theta, so "hold" is never wrong; "expected" is the move that is allowed.
    df["expected"] = "hold"
    df.loc[(df["inv_state"] == "below") & (df["price_pos"] != "ceiling"), "expected"] = "raise"
    df.loc[(df["inv_state"] == "above") & (df["price_pos"] != "floor"), "expected"] = "lower"

    df["stratum"] = df["phase"] + "|" + df["inv_state"] + "|" + df["price_pos"]
    return df


def allocate(available, n_total):
    """
    Split n_total across strata as evenly as possible.
    Strata smaller than their fair share give everything they have, and the
    shortfall is shared out among the larger strata (repeated until stable).
    `available` is a Series: stratum -> number of prompts available.
    """
    quota = pd.Series(0, index=available.index)
    open_ = set(available.index[available > 0])
    remaining = min(n_total, int(available.sum()))
    while remaining > 0 and open_:
        share = max(remaining // len(open_), 1)
        for s in sorted(open_):  # sorted so the result does not depend on dict order
            take = min(share, available[s] - quota[s], remaining)
            quota[s] += take
            remaining -= take
            if quota[s] == available[s]:
                open_.discard(s)
            if remaining == 0:
                break
    return quota


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="prompt log .jsonl from main.py --prompt-log")
    p.add_argument("--out-dir", required=True, help="folder to write the selected sets into")
    p.add_argument("--n-quality", type=int, default=200, help="size of the quality set (default 200)")
    p.add_argument("--throughput-months", default="30-39",
                   help="inclusive month range for the throughput set (default 30-39)")
    p.add_argument("--seed", type=int, default=0, help="random seed for sampling (default 0)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = label_situations(pd.read_json(args.input, lines=True))
    n_firms = df["firm_id"].nunique()
    print(f"read {len(df)} prompts, months {df.month.min()}-{df.month.max()}, {n_firms} firms")

    # quality set: stratified sample
    available = df["stratum"].value_counts().sort_index()
    quota = allocate(available, args.n_quality)
    parts = []
    for stratum, k in quota.items():
        if k > 0:
            parts.append(df[df["stratum"] == stratum].sample(n=int(k), random_state=args.seed))
    quality = pd.concat(parts).sort_values(["month", "firm_id"])

    counts = pd.DataFrame({"available": available, "selected": quota})
    counts.index.name = "stratum"
    print("\nprompts per situation (phase|inventory|price position):")
    print(counts.to_string())
    print(f"\nquality set: {len(quality)} prompts; expected direction: "
          f"{quality['expected'].value_counts().to_dict()}")

    # throughput set: every prompt from a block of consecutive months, as the simulation sends them
    lo, hi = parse_month_range(args.throughput_months)
    throughput = df[df["month"].between(lo, hi)].sort_values(["month", "firm_id"])
    per_month = throughput.groupby("month").size()
    missing = [m for m in range(lo, hi + 1) if per_month.get(m, 0) != n_firms]
    if throughput.empty:
        print(f"\nWARNING: no prompts in months {lo}-{hi}; throughput set is empty")
    elif missing:
        print(f"\nWARNING: months {missing} do not have one prompt per firm ({n_firms})")
    print(f"throughput set: {len(throughput)} prompts from months {lo}-{hi}")

    # write outputs
    quality.to_json(os.path.join(args.out_dir, "quality_set.jsonl"), orient="records", lines=True)
    throughput.to_json(os.path.join(args.out_dir, "throughput_set.jsonl"), orient="records", lines=True)
    counts.to_csv(os.path.join(args.out_dir, "strata_counts.csv"))
    meta = {
        "input": os.path.abspath(args.input),
        "input_sha256": file_sha256(args.input),
        "args": vars(args),
        "startup_last_month": STARTUP_LAST_MONTH,
        "n_quality": len(quality),
        "n_throughput": len(throughput),
    }
    with open(os.path.join(args.out_dir, "selection_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
