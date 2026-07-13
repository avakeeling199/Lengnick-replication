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

axes[0].hist(post['UnsatisfiedDemandPct'].values, bins=100, density=True, color='tab:blue')
xmax = np.percentile(post['UnsatisfiedDemandPct'].values, 99.5)
axes[0].set_xlim(0, max(xmax, 0.01))
axes[0].set_xlabel('Unsatisfied demand relative to planned demand (%)')
axes[0].set_ylabel('Density')
axes[0].set_title('Excess demand')

window = post.iloc[:REP_WINDOW]
axes[1].plot(window.index + BURN_IN_MONTHS, window['Employment'])
axes[1].set_xlabel('month')
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
unemployment = (1000 - post['Employment']) / 10
inflation = post['AvgPrice'].pct_change() * 100
vacancy = post['NumOpenPositions']
f5 = pd.DataFrame({'unemployment': unemployment, 'inflation': inflation, 'vacancy': vacancy}).dropna()

rng = np.random.default_rng(0)
n = len(f5)
jitter_u = f5['unemployment'].values + rng.uniform(-0.05, 0.05, size=n)
jitter_v = f5['vacancy'].values + rng.uniform(-0.5, 0.5, size=n)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].scatter(jitter_u, f5['inflation'].values, s=4, alpha=0.3)
axes[0].set_xlabel('Unemployment (%)')
axes[0].set_ylabel('Inflation (% month-on-month)')
axes[0].set_title('Phillips curve')

axes[1].scatter(jitter_u, jitter_v, s=4, alpha=0.3)
axes[1].set_xlabel('Unemployment (%)')
axes[1].set_ylabel('Vacancies (open positions)')
axes[1].set_title('Beveridge curve')

plt.tight_layout()
out5 = os.path.join(out_dir, 'fig5_phillips_beveridge.png')
plt.savefig(out5, dpi=150)
plt.close(fig)
print(f"Wrote {out5}")

# =========================================================================
# FIGURE 6: Firm size distribution (left) + price-change frequency (right)
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].hist(fs_post['num_workers'].values, bins=np.arange(0, 51), density=True, color='tab:green')
axes[0].set_xlabel('Number of workers')
axes[0].set_ylabel('Density')
axes[0].set_title('Firm size distribution')

price_change_freqs = []
for fid, g in fs_post.sort_values('month').groupby('firm_id'):
    prices = g['price'].values
    if len(prices) > 1:
        price_change_freqs.append(np.mean(np.diff(prices) != 0))
price_change_freqs = np.array(price_change_freqs)

axes[1].hist(price_change_freqs * 100, bins=30, density=True, color='tab:red')
axes[1].axvline(np.median(price_change_freqs) * 100, color='k', linestyle='--',
                label=f'median = {np.median(price_change_freqs)*100:.1f}%')
axes[1].set_xlabel('Price change frequency (% of months)')
axes[1].set_ylabel('Density')
axes[1].set_title('Frequency of price changes')
axes[1].legend()

plt.tight_layout()
out6 = os.path.join(out_dir, 'fig6_firmsize_pricefreq.png')
plt.savefig(out6, dpi=150)
plt.close(fig)
print(f"Wrote {out6}")
print(f"  median price-change frequency: {np.median(price_change_freqs)*100:.2f}% (paper: 9%)")
print(f"  skewness of firm size: {fs_post['num_workers'].skew():.2f} (paper: ~1.88)")
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
