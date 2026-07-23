#!/usr/bin/env python3
"""
Reproduce Lengnick (2013) Tier 1 figures (Fig 4-7) from a SINGLE run,
matching the paper's own methodology exactly: one 6000-month simulation
(plus 1000-month burn-in), with all distributional figures pooling
observations across months WITHIN that one run, and the time-series
panels showing a 50-year subperiod of that same run.

Usage:
    python make_tier1_figures_singlerun.py diagnostics/run_seed<SEED>.csv \
        diagnostics/firm_snapshots_seed<SEED>.csv
"""
import sys
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

run_path = sys.argv[1]
firm_path = sys.argv[2]
out_dir = os.path.dirname(run_path) or '.'

BURN_IN_MONTHS = 1000
REP_WINDOW = 600  # 50-year illustrative window, matching the paper

df = pd.read_csv(run_path)
monthly = df.iloc[20::21].reset_index(drop=True)
post = monthly.iloc[BURN_IN_MONTHS:].reset_index(drop=True)
print(f"Using single run: {run_path} ({len(post)} post-burn-in months)")

fs = pd.read_csv(firm_path)
fs_post = fs[fs['month'] > BURN_IN_MONTHS]

# =========================================================================
# FIGURE 4: Excess demand (left) + Employment, 50-yr window (right)
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].hist(post['UnsatisfiedDemandPct'].values, bins=np.linspace(0, 0.2, 200), density=True,
             histtype='step', color='black', linewidth=1.0)
axes[0].set_xlim(0, 0.2)
axes[0].set_xlabel('Unsatisfied demand (in %)')
axes[0].set_ylabel('Probability Density Function')
axes[0].set_title('Excess demand')

window = post.iloc[:REP_WINDOW]
years = window.index / 12
axes[1].plot(years, window['Employment'], color='black', linewidth=0.7)
axes[1].set_xlabel('Years')
axes[1].set_ylabel('Employed households')
axes[1].set_title('Employment, 50-yr subperiod')

plt.tight_layout()
out4 = os.path.join(out_dir, 'fig4_excess_demand_employment.png')
plt.savefig(out4, dpi=150)
plt.close(fig)
print(f"Wrote {out4}")

# =========================================================================
# FIGURE 5: Phillips curve (left) + Beveridge curve (right)
# =========================================================================
unemployment = (1000 - post['Employment'])  # absolute count, matching the paper (not %)
delta_p = post['AvgPrice'].diff()  # paper's x-axis is raw price change (Delta P), not % inflation
vacancy = post['NumOpenPositions']
f5 = pd.DataFrame({'unemployment': unemployment, 'delta_p': delta_p, 'vacancy': vacancy}).dropna()

rng = np.random.default_rng(0)
n = len(f5)
# paper: "a very small pseudo random number ~U[-0.5, 0.5] is added to unemployment
# and vacancies before plotting them in Fig. 5" (footnote 31), since both are integers
jitter_u = f5['unemployment'].values + rng.uniform(-0.5, 0.5, size=n)
jitter_v = f5['vacancy'].values + rng.uniform(-0.5, 0.5, size=n)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
# paper puts unemployment (absolute count) on the y-axis for both panels
axes[0].scatter(f5['delta_p'].values, jitter_u, s=4, alpha=0.4, color='black')
axes[0].set_ylabel('Unemployment (absolute)')
axes[0].set_xlabel(r'$\Delta P$')
axes[0].set_title('Phillips curve')

axes[1].scatter(jitter_v, jitter_u, s=4, alpha=0.4, color='black')
axes[1].set_ylabel('Unemployment (absolute)')
axes[1].set_xlabel('Vacancies')
axes[1].set_title('Beveridge curve')

plt.tight_layout()
out5 = os.path.join(out_dir, 'fig5_phillips_beveridge.png')
plt.savefig(out5, dpi=150)
plt.close(fig)
print(f"Wrote {out5}")

# =========================================================================
# FIGURE 6: Firm size distribution (left) + price-change frequency (right)
# paper draws these as step histograms (PDF), not smooth KDE curves.
# "Firm size" is measured in demand (units sold/month), not worker count.
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

if 'demand' in fs_post.columns:
    sizes = fs_post['demand'].values.astype(float)
    axes[0].hist(sizes, bins=200, density=True, histtype='step', color='black', linewidth=1.0)
    axes[0].set_xlabel('Firm Size (in demand)')
else:
    # 'demand' wasn't recorded in this run's firm_snapshots yet -- fall back to
    # worker count as a placeholder until a fresh run with the updated model.py
    print("  NOTE: 'demand' column not found in firm_snapshots -- using num_workers "
          "as a placeholder. Rerun with the updated model.py for the real Fig 6 left panel.")
    sizes = fs_post['num_workers'].values.astype(float)
    axes[0].hist(sizes, bins=50, density=True, histtype='step', color='black', linewidth=1.0)
    axes[0].set_xlabel('Number of workers (PLACEHOLDER, rerun needed for "in demand")')
axes[0].set_ylabel('Probability Density Function')
axes[0].set_title('Firm size distribution')

price_change_freqs = []
for fid, g in fs_post.sort_values('month').groupby('firm_id'):
    prices = g['price'].values
    if len(prices) > 1:
        price_change_freqs.append(np.mean(np.diff(prices) != 0))
price_change_freqs = np.array(price_change_freqs)

freq_pct = price_change_freqs * 100
axes[1].hist(freq_pct, bins=60, density=True, histtype='step', color='black', linewidth=1.0)
axes[1].set_xlabel('Firms Changing Price (in %)')
axes[1].set_ylabel('Probability Density Function')
axes[1].set_title('Frequency of price changes')

plt.tight_layout()
out6 = os.path.join(out_dir, 'fig6_firmsize_pricefreq.png')
plt.savefig(out6, dpi=150)
plt.close(fig)
print(f"Wrote {out6}")
print(f"  median price-change frequency: {np.median(price_change_freqs)*100:.2f}% (paper: 9%)")
if 'demand' in fs_post.columns:
    print(f"  skewness of firm size (demand): {fs_post['demand'].skew():.2f} (paper: ~1.88)")
print(f"  skewness of price-change freq: {pd.Series(price_change_freqs).skew():.2f} (paper: ~0.47)")

# =========================================================================
# FIGURE 7: GDP-price cross-correlation (left) + aggregate liquidity (right)
# =========================================================================
def ccf_at_lag(x, y, lag):
    if lag >= 0:
        a = x[:len(x) - lag] if lag > 0 else x
        b = y[lag:]
    else:
        a = x[-lag:]
        b = y[:len(y) + lag]
    if len(a) < 10:
        return np.nan
    return np.corrcoef(a, b)[0, 1]

from statsmodels.tsa.filters.hp_filter import hpfilter

n_quarters = len(post) // 3
emp_q = np.array([post['Employment'].iloc[3*i:3*i+3].mean() for i in range(n_quarters)])
price_q = np.array([post['AvgPrice'].iloc[3*i:3*i+3].mean() for i in range(n_quarters)])

emp_cycle, emp_trend = hpfilter(emp_q, lamb=1600)
price_cycle, price_trend = hpfilter(price_q, lamb=1600)

lags = list(range(-8, 9))
ccf_vals = [ccf_at_lag(emp_cycle, price_cycle, lag) for lag in lags]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(lags, ccf_vals, 'o-')
axes[0].axhline(0, color='k', linewidth=0.5)
axes[0].set_xlabel('lag k (quarters)')
axes[0].set_ylabel('correlation')
axes[0].set_title('GDP (Employment) vs lagged prices')

daily_window = df.iloc[BURN_IN_MONTHS * 21: BURN_IN_MONTHS * 21 + 21 * 6].reset_index(drop=True)
axes[1].plot(daily_window.index, daily_window['HHLiquidity'], label='Household liquidity')
axes[1].plot(daily_window.index, daily_window['FirmLiquidity'], label='Firm liquidity')
axes[1].set_xlabel('day')
axes[1].set_ylabel('Aggregate liquidity')
axes[1].set_title('Liquidity circulation, 6-month window')
axes[1].legend()

plt.tight_layout()
out7 = os.path.join(out_dir, 'fig7_gdpcorr_liquidity.png')
plt.savefig(out7, dpi=150)
plt.close(fig)
print(f"Wrote {out7}")
