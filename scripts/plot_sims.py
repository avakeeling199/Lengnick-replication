"""
Plot simulation runs over time on shared axes (one line per run), saved as a PDF.

Panels (each line is a rolling mean over --smooth months, to make trends readable):
  employment                       households with a job (out of 1000)
  price / marginal cost            median across firms, shaded band = 10th to 90th percentile
  firms priced below cost (%)      firms selling at a loss
  price spread (CV)                standard deviation / mean of firms' prices
  firms understocked (%)           inventory < 0.25 x last month's demand
  workers in the 10 biggest firms  share of all employed workers (10% = all firms equal size)

Works on unfinished runs too (run.csv and snap.csv are rewritten every 100 months).

Example:
  python scripts/plot_sims.py diagnostics/sim/rule/rule_seed33_m7000 \
      diagnostics/sim/goal_state/*_m7000_noband
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# reuse the benchmark figures' colours and style, so each model keeps its colour everywhere
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "benchmark"))
from analyse import MODEL_COLOURS, STYLE  # noqa: E402

RULE_COLOUR = "#52514e"


def run_label(folder):
    """'.../gemma4_31b_q4_seed33_m7000_noband' -> 'gemma4_31b_q4'; rule runs -> 'rule'"""
    name = os.path.basename(os.path.normpath(folder))
    return name.split("_seed")[0]


def monthly_series(folder, ld, inv_lower, active_only):
    """One row per month with every quantity we plot."""
    run = pd.read_csv(os.path.join(folder, "run.csv"), index_col=0)
    snap = pd.read_csv(os.path.join(folder, "snap.csv"))
    sizes = snap.groupby("month").num_workers   # firm sizes always use every firm
    if active_only:
        # firms with no workers sell nothing; their wage (so marginal cost) keeps falling and
        # their price follows it down, which can dominate the price panels
        snap = snap[snap.num_workers > 0].copy()

    snap["ratio"] = snap.price / (snap.wage / (21 * ld))
    snap["below_cost"] = snap.ratio < 1
    has_demand = snap.demand > 0
    snap["understock"] = np.where(has_demand, snap.inventory < inv_lower * snap.demand, np.nan)

    g = snap.groupby("month")
    out = pd.DataFrame({
        "ratio_med": g.ratio.median(),
        "ratio_p10": g.ratio.quantile(0.10),
        "ratio_p90": g.ratio.quantile(0.90),
        "below_cost": 100 * g.below_cost.mean(),
        "cv": g.price.std() / g.price.mean(),
        "understock": 100 * g.understock.mean(),
        "top10_share": 100 * sizes.apply(
            lambda w: w.nlargest(10).sum() / w.sum() if w.sum() > 0 else np.nan),
    })
    out["employment"] = run.Employment.reindex(out.index)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="+")
    p.add_argument("--smooth", type=int, default=12, help="rolling-mean window in months")
    p.add_argument("--max-month", type=int, default=None, help="cut the x axis here")
    p.add_argument("--ld", type=float, default=3, help="goods per worker per day")
    p.add_argument("--inv-lower", type=float, default=0.25)
    p.add_argument("--active-only", action="store_true",
                   help="price, below-cost, spread and stock panels use only firms with at least one worker")
    p.add_argument("--out", default="diagnostics/sim/figures/sim_compare.pdf")
    args = p.parse_args()

    panels = [("employment", "Employment (of 1,000 households)"),
              ("ratio_med", "Price ÷ marginal cost (median, 10th–90th pct)"),
              ("below_cost", "Firms priced below cost (%)"),
              ("cv", "Price spread across firms (CV)"),
              ("understock", "Firms understocked (%)"),
              ("top10_share", "Workers in the 10 biggest firms (%)")]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
        for folder in args.folders:
            label = run_label(folder)
            colour = RULE_COLOUR if label == "rule" else MODEL_COLOURS.get(label)
            df = monthly_series(folder, args.ld, args.inv_lower, args.active_only)
            if args.max_month:
                df = df.loc[:args.max_month]
            sm = df.rolling(args.smooth, min_periods=1).mean()
            print(f"{label}: {int(df.index.max())} months")

            for ax, (col, title) in zip(axes.flat, panels):
                ax.plot(sm.index, sm[col], color=colour, lw=1.2, label=label)
                if col == "ratio_med":
                    ax.fill_between(sm.index, sm.ratio_p10, sm.ratio_p90, color=colour, alpha=0.15, lw=0)
                ax.set_title(title, loc="left")

        axes.flat[1].axhline(1, color="#0b0b0b", lw=0.8, ls="--")   # price = cost
        axes.flat[5].axhline(10, color="#0b0b0b", lw=0.8, ls="--")  # all firms equal size
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels))
        fig.supxlabel("Month" + ("   (price and stock panels: firms with at least one worker)"
                                 if args.active_only else ""))
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        fig.savefig(args.out)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
