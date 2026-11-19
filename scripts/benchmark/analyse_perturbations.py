"""
Analyse the "change one input" test: how answers change when only inventory changes.

Each of 50 situations was sent at 6 inventory levels (multiples of last month's sales) with
everything else identical, so differences between a situation's copies are due to inventory alone.
Intervals resample whole situations (all levels and repeats together), and contrasts compare
levels *within* each situation (paired).

Writes to --out-dir:
  perturb_levels.csv     per model x prompt x level: P(raise), P(hold), P(cut), mean price change (%)
  perturb_contrasts.csv  per model x prompt: low-stock vs high-stock differences with 95% CIs
  perturb_response.pdf   rows = prompt, columns = P(raise) / P(cut) / mean price change, one line per model

Run from the repository root:
  python scripts/benchmark/analyse_perturbations.py
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyse import add_prompt_info, classify, STYLE, MODEL_COLOURS

PROMPTS = {"goal_state": "Goal + state", "rule_context": "Rule as context"}
MODELS = {"gemma4_31b_q4": "Gemma 4 31B", "qwen38_27b-fp8": "Qwen3.8 27B", "llama70b": "Llama 3.3 70B"}
LOW, HIGH = [0.05, 0.15], [1.5, 2.5]          # clearly below and clearly above the rule's band


def load(results_root, prompt, model, prompts):
    q = pd.read_csv(os.path.join(results_root, prompt, model, "perturb.csv"))
    q = classify(add_prompt_info(q, prompts))
    q = q[q["error"].isna() & q["move"].notna()].copy()
    info = prompts[["prompt_id", "base_id", "inv_level"]]
    q = q.merge(info, on="prompt_id", how="left")
    q["raise"] = (q["move"] == "raise").astype(float)
    q["cut"] = (q["move"] == "lower").astype(float)
    q["hold"] = (q["move"] == "hold").astype(float)
    q["dprice"] = 100 * (q["raw_new_price"] - q["p_f"]) / q["p_f"]
    return q


def by_situation_level(q):
    """Per situation x level: share raising / cutting / holding and mean price change."""
    return q.groupby(["base_id", "inv_level"])[["raise", "cut", "hold", "dprice"]].mean().reset_index()


def bootstrap(values_by_situation, n_boot, rng):
    """values_by_situation: array (n_situations, ...) -> mean and 95% CI over resampled situations."""
    v = np.asarray(values_by_situation, dtype=float)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    draws = np.nanmean(v[idx], axis=1)
    return np.nanmean(v, axis=0), np.nanpercentile(draws, 2.5, axis=0), np.nanpercentile(draws, 97.5, axis=0)


def plot(levels, path):
    cols = [("raise", "Raised the price (%)", (0, 100)), ("cut", "Cut the price (%)", (0, 100)),
            ("dprice", "Average price change (%)", None)]
    lv = sorted(levels["inv_level"].unique())
    xs = np.arange(len(lv))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(PROMPTS), len(cols), figsize=(13, 3.6 * len(PROMPTS)), squeeze=False)
        for i, (prompt, plabel) in enumerate(PROMPTS.items()):
            for j, (col, title, ylim) in enumerate(cols):
                ax = axes[i][j]
                ax.set_axisbelow(True)
                # shade the rule's band (a quarter to one month of sales) between the categories
                inside = [k for k, l in enumerate(lv) if 0.25 < l < 1.0]
                if inside:
                    ax.axvspan(min(inside) - 0.5, max(inside) + 0.5, color="#e4e3df", alpha=0.5, lw=0)
                for model, mlabel in MODELS.items():
                    s = levels[(levels["prompt"] == prompt) & (levels["model"] == model)].sort_values("inv_level")
                    if s.empty:
                        continue
                    scale = 1 if col == "dprice" else 100
                    ax.fill_between(xs, s[f"{col}_lo"] * scale, s[f"{col}_hi"] * scale,
                                    color=MODEL_COLOURS[model], alpha=0.15, lw=0)
                    ax.plot(xs, s[col] * scale, color=MODEL_COLOURS[model], marker="o", ms=4, label=mlabel)
                if col == "dprice":
                    ax.axhline(0, color="#52514e", lw=0.8)
                if ylim:
                    ax.set_ylim(*ylim)
                ax.set_xticks(xs)
                ax.set_xticklabels([f"{l * 100:g}%" for l in lv])
                ax.set_title(f"{plabel}: {title}", loc="left")
        # one shared x-axis label (a label under each panel overlaps its neighbours)
        fig.supxlabel("Inventory as % of last month's sales (shaded = inside the rule's band)",
                      y=0.045, fontsize=10, color="#52514e")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(MODELS), frameon=False)
        fig.suptitle("Same situations, only inventory changed: how each model's price decision responds",
                     fontweight="bold", x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(path)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default="diagnostics/benchmark/results")
    p.add_argument("--perturb-set", default="diagnostics/benchmark/prompt_sets/perturb_set.jsonl")
    p.add_argument("--out-dir", default="diagnostics/benchmark/analysis/perturb")
    p.add_argument("--n-boot", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    prompts = pd.read_json(args.perturb_set, lines=True)

    level_rows, contrast_rows = [], []
    for prompt in PROMPTS:
        for model in MODELS:
            if not os.path.exists(os.path.join(args.results_root, prompt, model, "perturb.csv")):
                print(f"skipping {prompt}/{model}: no perturb results")
                continue
            sl = by_situation_level(load(args.results_root, prompt, model, prompts))
            wide = {c: sl.pivot(index="base_id", columns="inv_level", values=c) for c in ["raise", "cut", "hold", "dprice"]}
            lv = list(wide["raise"].columns)

            # level curves with CIs over situations
            for c in ["raise", "cut", "hold", "dprice"]:
                mean, lo, hi = bootstrap(wide[c].to_numpy(), args.n_boot, rng)
                for k, level in enumerate(lv):
                    level_rows.append({"model": model, "prompt": prompt, "inv_level": level,
                                       "measure": c, "mean": mean[k], "lo": lo[k], "hi": hi[k]})

            # paired contrasts within situations: low stock vs high stock
            low = lambda c: wide[c][LOW].mean(axis=1)          # noqa: E731
            high = lambda c: wide[c][HIGH].mean(axis=1)        # noqa: E731
            for name, diff in [("raise: low minus high stock (pp)", 100 * (low("raise") - high("raise"))),
                               ("cut: high minus low stock (pp)", 100 * (high("cut") - low("cut"))),
                               ("price change: low minus high stock (%)", low("dprice") - high("dprice"))]:
                est, lo, hi = bootstrap(diff.to_numpy(), args.n_boot, rng)
                contrast_rows.append({"model": model, "prompt": prompt, "contrast": name,
                                      "estimate": est, "ci_low": lo, "ci_high": hi,
                                      "share_of_situations_in_expected_direction": float((diff > 0).mean()),
                                      "n_situations": int(diff.notna().sum())})

    levels_long = pd.DataFrame(level_rows)
    levels = levels_long.pivot_table(index=["model", "prompt", "inv_level"], columns="measure",
                                     values=["mean", "lo", "hi"]).reset_index()
    levels.columns = [b if a == "mean" else (f"{b}_{a}" if b else a) for a, b in levels.columns]
    contrasts = pd.DataFrame(contrast_rows)
    levels.to_csv(os.path.join(args.out_dir, "perturb_levels.csv"), index=False)
    contrasts.to_csv(os.path.join(args.out_dir, "perturb_contrasts.csv"), index=False)
    plot(levels, os.path.join(args.out_dir, "perturb_response.pdf"))

    pd.set_option("display.width", 200)
    show = levels[["prompt", "model", "inv_level", "raise", "hold", "cut", "dprice"]].copy()
    show[["raise", "hold", "cut"]] *= 100
    print("Share of answers (%) and mean price change (%) by inventory level (x sales):")
    print(show.round(1).to_string(index=False))
    print("\nWithin-situation contrasts, 95% CI over situations:")
    print(contrasts.round(2).to_string(index=False))
    print(f"\nwrote CSVs and perturb_response.pdf to {args.out_dir}")


if __name__ == "__main__":
    main()
