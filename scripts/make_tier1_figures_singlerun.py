#!/usr/bin/env python3
"""
Reproduce Lengnick (2013) Tier 1 figures (Fig 4-7) from a SINGLE run,
matching the paper's own methodology: one 6000-month simulation (plus a
1000-month burn-in), with all distributional figures pooling observations
across months WITHIN that one run, and the time-series panels showing a
subperiod of that same run. Axis ranges, labels and line styles follow the
published figures so the two can be compared side by side.

Usage:
    python scripts/make_tier1_figures_singlerun.py diagnostics/run_seed<SEED>.csv \
        diagnostics/firm_snapshots_seed<SEED>.csv [--burn-in 1000]
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter

parser = argparse.ArgumentParser()
parser.add_argument('run_path')
parser.add_argument('firm_path')
parser.add_argument('--burn-in', type=int, default=1000,
                    help='months to discard before analysis (paper: 1000)')
args = parser.parse_args()

out_dir = os.path.dirname(args.run_path) or '.'

BURN_IN_MONTHS = args.burn_in
REP_WINDOW = 600  # 50-year illustrative window, matching the paper
N_HOUSEHOLDS = 1000

# match the paper's look: black lines, boxed axes, thick step histograms
plt.rcParams.update({
    'font.size': 11,
    'axes.linewidth': 0.8,
    'lines.color': 'black',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})
STEP = dict(histtype='step', color='black', linewidth=2.0)

df = pd.read_csv(args.run_path, index_col=0)
monthly = df.iloc[20::21].reset_index(drop=True)  # last day of each month
if len(monthly) <= BURN_IN_MONTHS:
    raise SystemExit(f"run has only {len(monthly)} months; lower --burn-in (currently {BURN_IN_MONTHS})")
post = monthly.iloc[BURN_IN_MONTHS:].reset_index(drop=True)
print(f"Using single run: {args.run_path} ({len(post)} post-burn-in months)")

fs = pd.read_csv(args.firm_path)
fs_post = fs[fs['month'] > BURN_IN_MONTHS]

# =========================================================================
# FIGURE 4: Excess demand PDF (left) + Employment, 50-yr window (right)
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ED_XMAX = 0.2  # paper's x-range; UnsatisfiedDemandPct is already in percent units
ed = post['UnsatisfiedDemandPct'].values
axes[0].hist(ed, bins=np.linspace(0, ED_XMAX, 81), density=True, **STEP)
axes[0].set_xlim(0, ED_XMAX)
axes[0].set_ylim(bottom=0)
axes[0].set_xlabel('Unsatisfied demand (in %)')
axes[0].set_ylabel('Probability Density Function')
print(f"  fig4 excess demand: {(ed <= ED_XMAX).mean()*100:.1f}% of months within the paper's 0-{ED_XMAX}% axis, "
      f"{(ed == 0).mean()*100:.1f}% of months exactly 0")

window = post.iloc[:REP_WINDOW]
years = window.index / 12
axes[1].plot(years, window['Employment'], color='black', linewidth=0.8)
axes[1].set_xlim(0, REP_WINDOW / 12)
axes[1].set_ylim(min(950, window['Employment'].min() - 5), N_HOUSEHOLDS)
axes[1].set_xlabel('Years')
axes[1].set_ylabel('Employed households')

plt.tight_layout()
out4 = os.path.join(out_dir, 'fig4_excess_demand_employment.pdf')
plt.savefig(out4)
plt.close(fig)
print(f"Wrote {out4}")

# =========================================================================
# FIGURE 5: Phillips curve (left) + Beveridge curve (right)
# =========================================================================
unemployment = (N_HOUSEHOLDS - post['Employment'])  # absolute count, matching the paper (not %)
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
axes[0].scatter(f5['delta_p'].values, jitter_u, s=6, color='black', linewidths=0)
axes[0].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
axes[0].set_ylabel('Unemployment (absolute)')
axes[0].set_xlabel(r'$\Delta$ P')

axes[1].scatter(jitter_v, jitter_u, s=6, color='black', linewidths=0)
axes[1].set_ylabel('Unemployment (absolute)')
axes[1].set_xlabel('Vacancies')

for ax in axes:
    ax.set_ylim(bottom=-2)

plt.tight_layout()
out5 = os.path.join(out_dir, 'fig5_phillips_beveridge.pdf')
plt.savefig(out5)
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
    axes[0].hist(sizes, bins=np.arange(0, max(5000, sizes.max()) + 50, 50), density=True, **STEP)
    axes[0].set_xlim(0, max(5500, sizes.max()))
    axes[0].set_xlabel('Firm Size (in demand)')
else:
    # 'demand' wasn't recorded in this run's firm_snapshots yet -- fall back to
    # worker count as a placeholder until a fresh run with the updated model.py
    print("  NOTE: 'demand' column not found in firm_snapshots -- using num_workers "
          "as a placeholder. Rerun with the updated model.py for the real Fig 6 left panel.")
    sizes = fs_post['num_workers'].values.astype(float)
    axes[0].hist(sizes, bins=50, density=True, **STEP)
    axes[0].set_xlabel('Number of workers (PLACEHOLDER, rerun needed for "in demand")')
axes[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
axes[0].set_ylim(bottom=0)
axes[0].set_ylabel('Probability Density Function')

# share of firms changing price in each month (cross-section), pooled over months.
# 100 firms -> shares are whole percentages, so 1%-wide bins match the paper's histogram.
fs_sorted = fs_post.sort_values(['firm_id', 'month'])
price_diff = fs_sorted.groupby('firm_id')['price'].diff()
changed = (price_diff != 0)[price_diff.notna()]
share_changing = changed.groupby(fs_sorted.loc[changed.index, 'month']).mean() * 100

axes[1].hist(share_changing, bins=np.arange(0, max(35, share_changing.max()) + 1, 1), density=True, **STEP)
axes[1].set_xlim(0, 40)
axes[1].set_ylim(bottom=0)
axes[1].set_xlabel('Firms Changing Price (in %)')
axes[1].set_ylabel('Probability Density Function')

plt.tight_layout()
out6 = os.path.join(out_dir, 'fig6_firmsize_pricefreq.pdf')
plt.savefig(out6)
plt.close(fig)
print(f"Wrote {out6}")
print(f"  median share of firms changing price per month: {share_changing.median():.2f}% (paper: 9%)")
if 'demand' in fs_post.columns:
    print(f"  skewness of firm size (demand): {fs_post['demand'].skew():.2f} (paper: ~1.88)")
print(f"  skewness of price-change share: {share_changing.skew():.2f} (paper: ~0.47)")

# =========================================================================
# FIGURE 7: GDP-price cross-correlation (left) + aggregate liquidity (right)
# =========================================================================
def ccf_at_lag(x, y, lag):
    """corr(x_t, y_{t+lag})"""
    if lag >= 0:
        a = x[:len(x) - lag] if lag > 0 else x
        b = y[lag:]
    else:
        a = x[-lag:]
        b = y[:len(y) + lag]
    if len(a) < 10:
        return np.nan
    return np.corrcoef(a, b)[0, 1]

# GDP proxy: output is ld * employment per day, a linear transform of employment,
# so the HP-filtered cross-correlation is identical to using production directly
n_quarters = len(post) // 3
emp_q = post['Employment'].iloc[:3 * n_quarters].to_numpy().reshape(n_quarters, 3).mean(axis=1)
price_q = post['AvgPrice'].iloc[:3 * n_quarters].to_numpy().reshape(n_quarters, 3).mean(axis=1)

emp_cycle, _ = hpfilter(emp_q, lamb=1600)
price_cycle, _ = hpfilter(price_q, lamb=1600)

lags = list(range(-8, 9))
ccf_vals = [ccf_at_lag(emp_cycle, price_cycle, lag) for lag in lags]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(lags, ccf_vals, color='black', linewidth=1.2, label='model')
axes[0].set_xlim(-8, 8)
axes[0].set_ylim(-1, 0.6)
axes[0].set_xlabel('Lags k (in quarters)')
axes[0].set_ylabel(r'Correlation of GDP$_t$ with Price Index$_{t+k}$')
axes[0].legend(loc='upper left', frameon=True, edgecolor='black', fancybox=False)

# paper shows 4 months of daily data, x-axis labelled in months 1..5
start_day = BURN_IN_MONTHS * 21
daily_window = df.iloc[start_day: start_day + 21 * 4].reset_index(drop=True)
months_axis = 1 + daily_window.index / 21
axes[1].plot(months_axis, daily_window['HHLiquidity'], color='black', linewidth=2.5, label='Households')
axes[1].plot(months_axis, daily_window['FirmLiquidity'], color='black', linewidth=0.8, label='Firms')
axes[1].set_xlim(1, 5)
axes[1].set_xticks(range(1, 6))
# headroom above the sawtooth peaks so the legend sits clear of the lines, as in the paper
axes[1].set_ylim(0, 1.25 * daily_window[['HHLiquidity', 'FirmLiquidity']].max().max())
axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
axes[1].set_xlabel('Months')
axes[1].set_ylabel('Aggr. liquidity')
axes[1].legend(loc='upper right', frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
out7 = os.path.join(out_dir, 'fig7_gdpcorr_liquidity.pdf')
plt.savefig(out7)
plt.close(fig)
print(f"Wrote {out7}")
print(f"  min GDP-price correlation {np.nanmin(ccf_vals):.2f} at lag {lags[int(np.nanargmin(ccf_vals))]} "
      f"(paper: about -0.88 at lag -4)")
