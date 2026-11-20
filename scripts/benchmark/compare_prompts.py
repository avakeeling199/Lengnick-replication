"""
Compare how each model's pricing behaviour changes across the three prompt variants.

Reads diagnostics/benchmark/results/<prompt>/<model>/ (quality.csv, throughput.csv, and
overstock.csv when present) and writes to --out-dir:
  prompt_comparison.csv / .md   one row per model x prompt
  prompt_comparison.pdf         key behaviours by model, one bar per prompt
  quality_vs_speed_by_prompt.pdf  one quality-vs-speed panel per prompt, shared axes

Behaviour columns (shares of answers, %):
  under_raise / under_lower   understocked with room to raise: raised / cut the price
  under_ceiling_raise         understocked but already at the price ceiling: still asked to raise
  within_hold                 inventory inside the target band: held the price
  over_lower                  overstocked (overstock set): cut the price
  consistency                 same move across repeats of a prompt
  median_move                 typical size of a price change when the model moved (%)
Note: "understocked" / "within" / "overstocked" use the rule's band, which the goal_state
prompt never shows the model.

Run from the repository root:
  python scripts/benchmark/compare_prompts.py
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analyse import add_prompt_info, classify, repeat_consistency, STYLE, MODEL_COLOURS

PROMPTS = ["strict", "rule_context", "goal_state"]
MODELS = ["llama70b", "qwen38_27b-fp8", "gemma4_31b_q4", "qwen3_32b_awq", "qwen3_30b_a3b", "qwen35_9b", "qwen35_9b_think"]
PROMPT_COLOURS = {"strict": "#2a78d6", "rule_context": "#eb6834", "goal_state": "#1baf7a"}
PCT = ["under_raise", "under_lower", "under_ceiling_raise", "within_hold", "over_lower",
       "opposite_rule", "consistency", "median_move", "error_rate"]


def answered(q):
    """Rows with a usable answer (no request error, a price to judge)."""
    return q[q["error"].isna() & q["move"].notna()]


def behaviour(q):
    """Shares of each move in the situations that matter. q = classified quality rows."""
    ok = answered(q)
    under = ok[ok["expected"] == "raise"]
    under_ceiling = ok[ok["stratum"].str.contains("|below|ceiling", regex=False)]
    within = ok[ok["stratum"].str.contains("|within|", regex=False)]
    moved = ok[ok["move"] != "hold"]
    directional = ok[ok["expected"].isin(["raise", "lower"])]
    return {
        # same as analyse.py's wrong_dir_rate: moved opposite to the rule where the rule calls for a move
        "opposite_rule": directional["wrong_dir"].mean(),
        "under_raise": (under["move"] == "raise").mean(),
        "under_lower": (under["move"] == "lower").mean(),
        "under_ceiling_raise": (under_ceiling["move"] == "raise").mean(),
        "within_hold": (within["move"] == "hold").mean(),
        "consistency": repeat_consistency(ok),
        "median_move": ((moved["raw_new_price"] - moved["p_f"]).abs() / moved["p_f"]).median(),
        "error_rate": q["error"].notna().mean(),
    }


def load_row(results_root, prompt, model, prompts):
    folder = os.path.join(results_root, prompt, model)
    q = classify(add_prompt_info(pd.read_csv(os.path.join(folder, "quality.csv")), prompts))
    t = pd.read_csv(os.path.join(folder, "throughput.csv"))
    row = {"model": model, "prompt": prompt}
    row.update(behaviour(q))

    over_path = os.path.join(folder, "overstock.csv")
    if os.path.exists(over_path):
        o = answered(classify(add_prompt_info(pd.read_csv(over_path), prompts)))
        row["over_lower"] = (o["move"] == "lower").mean()
        row["over_n"] = len(o)
    else:
        row["over_lower"] = float("nan")          # overstock job not finished yet
        row["over_n"] = 0

    row["s_per_month"] = t.groupby("month")["month_wall_s"].first().mean()
    row["prompt_tokens"] = t["prompt_tokens"].median()
    row["out_tokens"] = q["completion_tokens"].median()
    return row


def to_markdown(df):
    cols = list(df.columns)
    lines = ["| model | prompt | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for (model, prompt), row in df.iterrows():
        cells = ["" if pd.isna(v) else str(v) for v in row]
        lines.append(f"| {model} | {prompt} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot(table, path):
    """One panel per behaviour; for each model, one bar per prompt."""
    panels = [("under_raise", "Understocked: raised"),
              ("within_hold", "Within band: held"),
              ("over_lower", "Overstocked: cut")]
    models = [m for m in MODELS if m in table.index.get_level_values("model")]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(panels), figsize=(13, 4.2), sharey=True)
        width = 0.8 / len(PROMPTS)
        for ax, (col, title) in zip(axes, panels):
            ax.set_axisbelow(True)
            for i, p in enumerate(PROMPTS):
                vals = [table[col].get((m, p), float("nan")) for m in models]
                x = [k + (i - (len(PROMPTS) - 1) / 2) * width for k in range(len(models))]
                ax.bar(x, vals, width=width * 0.9, color=PROMPT_COLOURS[p], label=p)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=30, ha="right")
            ax.set_ylim(0, 100)
            ax.set_title(title, loc="left")
        axes[0].set_ylabel("Share of answers (%)")
        axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.42), ncol=len(PROMPTS))
        fig.suptitle("How pricing behaviour changes as the rule is removed from the prompt",
                     fontweight="bold", x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)


def plot_quality_vs_speed(table, path):
    """One 'decision quality vs speed' panel per prompt, on shared axes, one dot per model."""
    models = [m for m in MODELS if m in table.index.get_level_values("model")]
    x_max = table["s_per_month"].max() * 1.15
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(PROMPTS), figsize=(13, 4.6), sharex=True, sharey=True)
        for ax, prompt in zip(axes, PROMPTS):
            for m in models:
                if (m, prompt) not in table.index:
                    continue
                row = table.loc[(m, prompt)]
                ax.scatter(row["s_per_month"], row["opposite_rule"], s=90, color=MODEL_COLOURS[m],
                           edgecolor="#fcfcfb", linewidth=2, zorder=3, clip_on=False, label=m)
            ax.set_xlim(0, x_max)
            ax.set_ylim(0, 100)
            ax.set_title(prompt, loc="left")
            ax.set_xlabel("Seconds per simulated month")
        axes[0].set_ylabel("Moves opposite to the rule (%)")
        # one legend for the whole figure, below the panels (a legend inside one panel squashes the layout)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(models), frameon=False)
        fig.suptitle("Decision quality vs speed by prompt (bottom-left = rule-consistent and fast)",
                     fontweight="bold", x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.08, 1, 1))     # leave room at the bottom for the legend
        fig.savefig(path)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default="diagnostics/benchmark/results")
    p.add_argument("--sets-dir", default="diagnostics/benchmark/prompt_sets")
    p.add_argument("--out-dir", default="diagnostics/benchmark/analysis/prompts")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # look-up table of every prompt's current price and band (quality + overstock sets)
    sets = [os.path.join(args.sets_dir, "quality_set.jsonl"), os.path.join(args.sets_dir, "overstock_set.jsonl")]
    prompts = pd.concat([pd.read_json(s, lines=True) for s in sets if os.path.exists(s)])

    rows = []
    for prompt in PROMPTS:
        for model in MODELS:
            folder = os.path.join(args.results_root, prompt, model)
            if not (os.path.exists(os.path.join(folder, "quality.csv")) and
                    os.path.exists(os.path.join(folder, "throughput.csv"))):
                print(f"skipping {prompt}/{model}: results not complete")
                continue
            rows.append(load_row(args.results_root, prompt, model, prompts))

    table = pd.DataFrame(rows).set_index(["model", "prompt"]).sort_index(
        key=lambda idx: idx.map({**{m: i for i, m in enumerate(MODELS)}, **{p: i for i, p in enumerate(PROMPTS)}}))
    table[PCT] = table[PCT] * 100
    table = table.round(1)

    table.to_csv(os.path.join(args.out_dir, "prompt_comparison.csv"))
    with open(os.path.join(args.out_dir, "prompt_comparison.md"), "w") as f:
        f.write(to_markdown(table) + "\n")
    plot(table, os.path.join(args.out_dir, "prompt_comparison.pdf"))
    plot_quality_vs_speed(table, os.path.join(args.out_dir, "quality_vs_speed_by_prompt.pdf"))

    show = ["under_raise", "under_lower", "under_ceiling_raise", "within_hold", "over_lower",
            "consistency", "median_move", "s_per_month"]
    print(table[show].to_string())
    print(f"\nwrote prompt_comparison.csv / .md / .pdf to {args.out_dir}")


if __name__ == "__main__":
    main()
