#!/usr/bin/env python3
"""Visualize business cycles from a single seed run: zoomed time series + ACF.

Usage:
    python plot_business_cycles.py diagnostics/run_seed<SEED>.csv [window_start_month]
"""
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

path = sys.argv[1]
window_start = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

df = pd.read_csv(path)
monthly = df.iloc[20::21].reset_index(drop=True)
BURN_IN = 1000
post = monthly.iloc[BURN_IN:].reset_index(drop=True)

window_len = 600  # 50 years, matching the paper's own subperiod length
rel_start = window_start - BURN_IN
window = post.iloc[rel_start:rel_start + window_len]

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(window.index + BURN_IN, window['Employment'])
axes[0].set_title(f'Employment, months {window_start}-{window_start + window_len} (50-year window)')
axes[0].set_ylabel('Employed households')

axes[1].plot(window.index + BURN_IN, window['TotalInv'], color='tab:orange')
axes[1].set_title('Total Inventory, same window')
axes[1].set_ylabel('Inventory units')
axes[1].set_xlabel('month')

emp = post['Employment'] - post['Employment'].mean()
max_lag = 300
acf = [1.0] + [emp.autocorr(lag=k) for k in range(1, max_lag + 1)]
axes[2].stem(range(max_lag + 1), acf, markerfmt=' ')
axes[2].axhline(0, color='k', linewidth=0.5)
axes[2].set_title('Autocorrelation of Employment (full post-burn-in run)')
axes[2].set_xlabel('lag (months)')
axes[2].set_ylabel('ACF')

plt.tight_layout()
out = path.replace('.csv', '_cycles.png')
plt.savefig(out, dpi=150)
print(f"Wrote {out}")

acf_arr = np.array(acf)
peaks = [k for k in range(2, max_lag) if acf_arr[k] > acf_arr[k-1] and acf_arr[k] > acf_arr[k+1]]
if peaks:
    print(f"First ACF peak after lag 0 at lag = {peaks[0]} months (~{peaks[0]/12:.1f} years) -- rough cycle-period estimate")
else:
    print("No clear ACF peak found within max_lag -- try increasing max_lag in the script")
