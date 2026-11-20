"""
Summarise the prompt-replay benchmark results.

Reads every model folder in --results-dir (written by replay.py via cluster/benchmark_model.sh)
and writes to --out-dir:
  summary_table.csv / .md   one row per model: reliability, direction, consistency, agreement, speed
  by_stratum.csv            what each model did in each economic situation
  quality_vs_speed.pdf      wrong-direction rate against seconds per simulated month
  wrong_dir_by_stratum.pdf  wrong-direction rate where the rules call for raising

Run from the repository root:
  python scripts/benchmark/analyse.py
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")           # draw to files, no screen needed (works on the cluster)
import matplotlib.pyplot as plt
import pandas as pd

PRICE_TOL = 1e-6                # price changes smaller than this count as "no change"
MOVES = ["raise", "hold", "lower"]


# ---------------------------------------------------------------- loading

def load_model(results_dir, label):
    """Read one model's three result files."""
    folder = os.path.join(results_dir, label)
    q = pd.read_csv(os.path.join(folder, "quality.csv"))
    t = pd.read_csv(os.path.join(folder, "throughput.csv"))
    # older runs wrote startup.csv; runs since the mode option write startup_full.csv
    startup_s = float("nan")
    for name in ("startup_full.csv", "startup.csv"):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            with open(path) as f:
                startup_s = float(f.read().strip().split(",")[1])     # file is "startup_s,571"
            break
    return q, t, startup_s


def add_prompt_info(q, prompts):
    """Attach each prompt's current price and price band to every result row."""
    info = prompts[["prompt_id", "p_f", "p_f_lowerbar", "p_f_upperbar"]]
    return q.merge(info, on="prompt_id", how="left")


# ---------------------------------------------------------------- per-answer labels

def classify(q):
    """Add columns describing what each answer did. One row = one answer."""
    q = q.copy()
    change = q["raw_new_price"] - q["p_f"]

    # what the price actually did (this is what would change the economy)
    q["move"] = "hold"
    q.loc[change > PRICE_TOL, "move"] = "raise"
    q.loc[change < -PRICE_TOL, "move"] = "lower"
    q.loc[q["raw_new_price"].isna(), "move"] = None           # no usable answer

    # the stated action disagrees with the price it gave, e.g. "raise" with a lower price
    q["mismatch"] = q["move"].notna() & q["raw_action"].notna() & (q["move"] != q["raw_action"])

    # moved the opposite way to what the rules allow
    q["wrong_dir"] = (((q["expected"] == "raise") & (q["move"] == "lower")) |
                      ((q["expected"] == "lower") & (q["move"] == "raise")))

    # asked for the economically right move but was held at the band edge (not an error)
    understocked = q["stratum"].str.contains("|below|", regex=False)
    overstocked = q["stratum"].str.contains("|above|", regex=False)
    q["blocked_ok"] = (((q["p_f"] >= q["p_f_upperbar"]) & understocked & (q["move"] == "raise")) |
                       ((q["p_f"] <= q["p_f_lowerbar"]) & overstocked & (q["move"] == "lower")))
    return q


# ---------------------------------------------------------------- per-prompt summaries

def modal_move(q):
    """For each prompt, the move the model chose most often across its repeats."""
    return q.dropna(subset=["move"]).groupby("prompt_id")["move"].agg(lambda s: s.mode().iloc[0])


def repeat_consistency(q):
    """For each prompt, the share of repeats that gave its most common move; averaged over prompts."""
    per_prompt = q.dropna(subset=["move"]).groupby("prompt_id")["move"].agg(
        lambda s: s.value_counts(normalize=True).iloc[0])
    return per_prompt.mean()


def self_agreement(q):
    """
    How often one answer matches the model's own usual answer, leaving that answer out.
    This is the most agreement anyone could expect with this model, given its own randomness.
    """
    q = q.dropna(subset=["move"])
    hits = []
    for _, g in q.groupby("prompt_id"):
        moves = g["move"].tolist()
        for i, m in enumerate(moves):
            others = pd.Series(moves[:i] + moves[i + 1:])
            if len(others):
                hits.append(m == others.mode().iloc[0])
    return sum(hits) / len(hits)


# ---------------------------------------------------------------- per-model metrics

def quality_metrics(q, baseline_modal):
    ok = q[q["error"].isna()]                     # failed requests have no answer to judge
    directional = ok[ok["expected"].isin(["raise", "lower"])]
    return {
        "requests": len(q),
        "error_rate": q["error"].notna().mean(),
        "cut_off_rate": (ok["finish_reason"] == "length").mean(),
        "mismatch_rate": ok["mismatch"].mean(),
        "wrong_dir_rate": directional["wrong_dir"].mean(),
        "blocked_ok_rate": ok["blocked_ok"].mean(),
        "repeat_consistency": repeat_consistency(ok),
        "agree_with_baseline": (ok["move"] == ok["prompt_id"].map(baseline_modal)).mean(),
        "share_raise": (ok["move"] == "raise").mean(),
        "share_hold": (ok["move"] == "hold").mean(),
        "share_lower": (ok["move"] == "lower").mean(),
        "median_latency_s": ok["latency_s"].median(),
        "median_output_tokens": ok["completion_tokens"].median(),
    }


def speed_metrics(t, startup_s):
    per_month = t.groupby("month")["month_wall_s"].first()   # same value on all 100 rows of a month
    return {
        "s_per_month": per_month.mean(),
        "hours_for_7000_months": per_month.mean() * 7000 / 3600,
        "throughput_errors": int(t["error"].notna().sum()),
        "startup_min": startup_s / 60,
    }


def by_stratum(q, label):
    """What one model did in each economic situation."""
    ok = q[q["error"].isna()]
    rows = []
    for stratum, g in ok.groupby("stratum"):
        row = {"model": label, "stratum": stratum, "expected": g["expected"].iloc[0], "answers": len(g)}
        for m in MOVES:
            row[f"share_{m}"] = (g["move"] == m).mean()
        row["wrong_dir_rate"] = g["wrong_dir"].mean()
        rows.append(row)
    return rows


# ---------------------------------------------------------------- figures

# categorical palette, assigned in fixed model order (validated colour-blind-safe order)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# fixed colour per model, shared by every figure (also imported by compare_prompts.py)
MODEL_COLOURS = {"llama70b": PALETTE[0], "qwen38_27b": PALETTE[1], "qwen38_27b-fp8": PALETTE[2],
                 "qwen3_30b_a3b": PALETTE[3], "qwen3_32b_awq": PALETTE[4], "gemma4_31b_q4": PALETTE[6],
                 "qwen35_9b": PALETTE[7], "qwen35_9b_think": PALETTE[1]}
STYLE = {"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
         "axes.edgecolor": "#e4e3df", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
         "ytick.color": "#52514e", "text.color": "#0b0b0b", "font.size": 10, "axes.titlesize": 11,
         "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
         "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "legend.frameon": False}


def plot_quality_vs_speed(summary, colours, path):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        # label nearby points alternately above and below so their names do not overlap
        ranked = summary.sort_values("s_per_month")
        for i, (label, row) in enumerate(ranked.iterrows()):
            ax.scatter(row["s_per_month"], row["wrong_dir_rate"] * 100, s=90, color=colours[label],
                       edgecolor="#fcfcfb", linewidth=2, zorder=3, clip_on=False)   # points at 0% stay visible
            ax.annotate(label, (row["s_per_month"], row["wrong_dir_rate"] * 100),
                        xytext=(8, 8 if i % 2 == 0 else -14), textcoords="offset points",
                        fontsize=9, color="#52514e")
        ax.set_xlim(0, summary["s_per_month"].max() * 1.25)
        ax.set_ylim(0, max(summary["wrong_dir_rate"].max() * 100 * 1.25, 5))
        ax.set_xlabel("Seconds per simulated month (lower is faster)")
        ax.set_ylabel("Wrong-direction decisions (%)")
        ax.set_title("Decision quality vs speed (bottom-left is best)", loc="left")
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)


def plot_wrong_dir_by_stratum(strata, colours, order, path):
    raise_strata = strata[strata["expected"] == "raise"]
    pivot = raise_strata.pivot(index="stratum", columns="model", values="wrong_dir_rate")[order] * 100
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 0.9 * len(pivot) + 2.0))
        ax.set_axisbelow(True)                   # grid lines behind the bars
        n = len(order)
        height = 0.8 / n
        for i, label in enumerate(order):
            y = [k + (i - (n - 1) / 2) * height for k in range(len(pivot))]
            ax.barh(y, pivot[label], height=height * 0.9, color=colours[label], label=label)
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels([s.replace("|", " / ") for s in pivot.index])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Cut the price when the rules call for raising (%)")
        ax.set_title("Wrong-direction rate in understocked situations", loc="left")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=n)   # legend below the plot
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)


# ---------------------------------------------------------------- main

def to_markdown(df):
    """Simple markdown table without extra dependencies (for pasting into Notion)."""
    lines = ["| model | " + " | ".join(df.columns) + " |", "|---" * (len(df.columns) + 1) + "|"]
    for label, row in df.iterrows():
        lines.append(f"| {label} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="diagnostics/benchmark/results")
    p.add_argument("--prompt-set", default="diagnostics/benchmark/prompt_sets/quality_set.jsonl")
    p.add_argument("--baseline", default="llama70b", help="model the others are compared against")
    p.add_argument("--out-dir", default="diagnostics/benchmark/analysis")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    prompts = pd.read_json(args.prompt_set, lines=True)
    # only models whose quality and throughput runs have both finished
    labels = sorted(m for m in os.listdir(args.results_dir)
                    if os.path.exists(os.path.join(args.results_dir, m, "quality.csv"))
                    and os.path.exists(os.path.join(args.results_dir, m, "throughput.csv")))
    skipped = sorted(set(os.listdir(args.results_dir)) - set(labels))
    if skipped:
        print(f"skipping (results not complete): {', '.join(skipped)}")
    order = [args.baseline] + [m for m in labels if m != args.baseline]   # baseline first
    # each model keeps the same colour in every run; new models take the next unused colour
    spare = [c for c in PALETTE if c not in MODEL_COLOURS.values()]
    colours = {m: MODEL_COLOURS.get(m) or spare.pop(0) for m in order}

    # baseline's usual answer per prompt, and how often it agrees with itself
    q_base, _, _ = load_model(args.results_dir, args.baseline)
    q_base = classify(add_prompt_info(q_base, prompts))
    q_base_ok = q_base[q_base["error"].isna()]
    baseline_modal = modal_move(q_base_ok)
    baseline_self = self_agreement(q_base_ok)

    summary_rows, strata_rows = [], []
    for label in order:
        q, t, startup_s = load_model(args.results_dir, label)
        q = classify(add_prompt_info(q, prompts))
        row = {"model": label}
        row.update(quality_metrics(q, baseline_modal))
        row.update(speed_metrics(t, startup_s))
        if label == args.baseline:
            row["agree_with_baseline"] = baseline_self       # compare with itself, leaving each answer out
        summary_rows.append(row)
        strata_rows.extend(by_stratum(q, label))

    summary = pd.DataFrame(summary_rows).set_index("model")
    strata = pd.DataFrame(strata_rows)

    summary.round(4).to_csv(os.path.join(args.out_dir, "summary_table.csv"))
    strata.round(4).to_csv(os.path.join(args.out_dir, "by_stratum.csv"), index=False)

    # a readable version: percentages and rounded numbers
    pct = ["error_rate", "cut_off_rate", "mismatch_rate", "wrong_dir_rate", "blocked_ok_rate",
           "repeat_consistency", "agree_with_baseline", "share_raise", "share_hold", "share_lower"]
    readable = summary.copy()
    readable[pct] = (readable[pct] * 100).round(1)
    readable = readable.round(1)
    with open(os.path.join(args.out_dir, "summary_table.md"), "w") as f:
        f.write(to_markdown(readable) + "\n\n(rates in %; agree_with_baseline for the baseline "
                "itself is its leave-one-out self-agreement)\n")

    plot_quality_vs_speed(summary, colours, os.path.join(args.out_dir, "quality_vs_speed.pdf"))
    plot_wrong_dir_by_stratum(strata, colours, order, os.path.join(args.out_dir, "wrong_dir_by_stratum.pdf"))

    print(readable[["wrong_dir_rate", "repeat_consistency", "agree_with_baseline", "error_rate",
                    "s_per_month", "hours_for_7000_months"]].to_string())
    print(f"\nwrote tables and figures to {args.out_dir}")


if __name__ == "__main__":
    main()
