"""
Sanity checks for one or more simulation output folders (run.csv + snap.csv).

For each folder it prints a table with one column per window of months, so you can see
whether prices stay sensible relative to marginal cost and whether firms keep stock.

  mc         marginal cost = wage / (21 * ld)          (ld = 3 goods per worker per day)
  p/mc       price divided by marginal cost (median, and 10th / 90th percentile across firms)
  <mc        share of firm-months priced below marginal cost (selling at a loss)
  <band/>band share below 1.025*mc / above 1.15*mc: outside the rule's old price band
  understock share with inventory < 0.25 * last month's demand
  overstock  share with inventory > 1 * last month's demand
  changed    share of firm-months whose price changed since the previous month
  drift %/m  average monthly change in the average price, in percent
  CV price   price standard deviation / average price (how spread out firms' prices are)
  employment, unsatisfied demand %: straight from run.csv

Example:
  python scripts/check_sim.py diagnostics/sim/rule/rule_seed33_m60 \
      diagnostics/sim/goal_state/gemma4_31b_q4_seed33_m60_noband
"""
import argparse
import os

import numpy as np
import pandas as pd


def parse_windows(text):
    """'1-14,15-30,31-60' -> [(1, 14), (15, 30), (31, 60)]"""
    out = []
    for part in text.split(","):
        a, b = part.split("-")
        out.append((int(a), int(b)))
    return out


def window_stats(snap, run, lo, hi, args):
    s = snap[(snap.month >= lo) & (snap.month <= hi)]
    r = run[(run.month >= lo) & (run.month <= hi)]
    ratio = s.price / s.mc
    has_demand = s.demand > 0
    avg_price = run.set_index("month").AvgPrice
    drift = np.log(avg_price).diff().loc[lo:hi].mean() * 100

    stats = {
        "p/mc median": ratio.median(),
        "p/mc p10": ratio.quantile(0.10),
        "p/mc p90": ratio.quantile(0.90),
        "<mc %": 100 * (ratio < 1).mean(),
        "<band %": 100 * (ratio < args.band_lower).mean(),
        ">band %": 100 * (ratio > args.band_upper).mean(),
        "understock %": 100 * (s.inventory[has_demand] < args.inv_lower * s.demand[has_demand]).mean(),
        "overstock %": 100 * (s.inventory[has_demand] > args.inv_upper * s.demand[has_demand]).mean(),
        "changed %": 100 * s.changed.mean(),
        "drift %/m": drift,
        "CV price": (r.PriceStd / r.AvgPrice).mean(),
        "employment": r.Employment.mean(),
        "unsat. demand %": r.UnsatisfiedDemandPct.mean(),
    }
    if "llm_ok" in s.columns:
        stats["LLM ok %"] = 100 * s.llm_ok.astype(bool).mean()
    return stats


def check_folder(folder, args):
    snap = pd.read_csv(os.path.join(folder, "snap.csv"))
    run = pd.read_csv(os.path.join(folder, "run.csv"), index_col=0)
    run["month"] = run.index

    snap = snap.sort_values(["firm_id", "month"])
    snap["mc"] = snap.wage / (21 * args.ld)
    prev = snap.groupby("firm_id").price.shift()
    # 1.0 / 0.0, left empty for each firm's first month (nothing to compare with)
    snap["changed"] = ((snap.price - prev).abs() > 1e-6 * snap.price).astype(float).where(prev.notna())

    last = int(snap.month.max())
    windows = [(lo, min(hi, last)) for lo, hi in parse_windows(args.windows) if lo <= last]
    table = pd.DataFrame({f"m{lo}-{hi}": window_stats(snap, run, lo, hi, args)
                          for lo, hi in windows})

    print(f"\n=== {folder}  ({last} months, {snap.firm_id.nunique()} firms)")
    timings = os.path.join(folder, "timings.csv")
    if os.path.exists(timings):
        t = pd.read_csv(timings)
        print(f"speed: {t.month_wall_s.mean():.1f} s/month on average")
    print(table.round(2).to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="+")
    p.add_argument("--windows", default="1-14,15-30,31-60",
                   help="month windows, e.g. 1-14,15-1000,1001-7000")
    p.add_argument("--ld", type=float, default=3, help="goods per worker per day")
    p.add_argument("--band-lower", type=float, default=1.025)
    p.add_argument("--band-upper", type=float, default=1.15)
    p.add_argument("--inv-lower", type=float, default=0.25)
    p.add_argument("--inv-upper", type=float, default=1.0)
    args = p.parse_args()
    for folder in args.folders:
        check_folder(folder, args)


if __name__ == "__main__":
    main()
