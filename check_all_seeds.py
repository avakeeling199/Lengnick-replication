#!/usr/bin/env python3
"""Quick sanity check across all seed runs from the HPC array job.

Usage:
    python check_all_seeds.py [diagnostics_dir]

Checks, per seed:
  - row count (should be 7000*21 = 147000 for the full protocol)
  - post-burn-in (month 1000+) employment range / unemployment peak
  - post-burn-in UnsatisfiedDemandPct (should be ~0, almost all months < 0.03%)

Also produces a quick multi-panel plot overlaying a handful of seeds'
Employment and TotalInv trajectories, saved as check_all_seeds.png.
"""
import glob
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

diag_dir = sys.argv[1] if len(sys.argv) > 1 else 'diagnostics'

paths = sorted(glob.glob(os.path.join(diag_dir, 'run_seed*.csv')))
if not paths:
    print(f"No run_seed*.csv files found in {diag_dir}/")
    sys.exit(1)

print(f"Found {len(paths)} seed runs\n")

EXPECTED_ROWS = 7000 * 21
BURN_IN_MONTHS = 1000

summary_rows = []
all_data = {}

for path in paths:
    seed = os.path.basename(path).replace('run_seed', '').replace('.csv', '')
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[seed {seed}] FAILED TO READ: {e}")
        continue

    n_rows = len(df)
    complete = n_rows == EXPECTED_ROWS
    monthly = df.iloc[20::21].reset_index(drop=True)
    post = monthly.iloc[BURN_IN_MONTHS:]

    if len(post) == 0:
        print(f"[seed {seed}] INCOMPLETE RUN: only {n_rows} rows (expected {EXPECTED_ROWS})")
        continue

    unsat_ok_frac = (post['UnsatisfiedDemandPct'] < 0.03).mean() if 'UnsatisfiedDemandPct' in post else float('nan')

    summary_rows.append({
        'seed': seed,
        'n_rows': n_rows,
        'complete': complete,
        'emp_mean': post['Employment'].mean(),
        'emp_min': post['Employment'].min(),
        'emp_max': post['Employment'].max(),
        'unemp_peak_pct': (1000 - post['Employment'].min()) / 10,
        'unsat_mean_pct': post['UnsatisfiedDemandPct'].mean() if 'UnsatisfiedDemandPct' in post else float('nan'),
        'unsat_frac_under_0.03pct': unsat_ok_frac,
    })
    all_data[seed] = monthly

summary = pd.DataFrame(summary_rows)
pd.set_option('display.width', 160)
pd.set_option('display.max_columns', None)
print(summary.to_string(index=False))

n_incomplete = (~summary['complete']).sum()
print(f"\n{n_incomplete} of {len(summary)} runs did not reach the expected row count")
print(f"Employment mean across seeds: {summary['emp_mean'].mean():.1f} (std across seeds: {summary['emp_mean'].std():.2f})")
print(f"Unemployment peak mean across seeds: {summary['unemp_peak_pct'].mean():.2f}% (paper benchmark: up to 4.3%)")
print(f"Seeds with >=99% of months under 0.03% unsatisfied demand: {(summary['unsat_frac_under_0.03pct'] >= 0.99).sum()} / {len(summary)}")

# quick plot: overlay first 8 seeds (or fewer if less available)
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
plot_seeds = list(all_data.keys())[:8]
for seed in plot_seeds:
    m = all_data[seed]
    axes[0].plot(m.index, m['Employment'], alpha=0.6, label=f"seed {seed}")
    axes[1].plot(m.index, m['TotalInv'], alpha=0.6, label=f"seed {seed}")

axes[0].axvline(BURN_IN_MONTHS, color='k', linestyle='--', alpha=0.4, label='burn-in cutoff')
axes[0].set_title('Employment (first 8 seeds)')
axes[0].legend(fontsize=7, ncol=2)
axes[1].axvline(BURN_IN_MONTHS, color='k', linestyle='--', alpha=0.4)
axes[1].set_title('Total Inventory (first 8 seeds)')
axes[1].set_xlabel('month')

plt.tight_layout()
out_png = os.path.join(diag_dir, 'check_all_seeds.png')
plt.savefig(out_png, dpi=150)
print(f"\nWrote {out_png}")
