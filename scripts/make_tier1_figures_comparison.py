#!/usr/bin/env python3
"""
Reproduce Lengnick (2013) Tier 1 figures (Fig 4-7), overlaying two runs
(e.g. LLM-pricing vs rule-based pricing) for a like-for-like comparison.

Both runs are truncated to the same number of months (the shorter of the
two) before plotting, so neither series has an unfair data advantage.

Usage:
    python scripts/make_tier1_figures_comparison.py \
        <run_a.csv> <firm_snapshots_a.csv> <label_a> \
        <run_b.csv> <firm_snapshots_b.csv> <label_b> \
        <out_dir>
"""
import sys
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

run_a, firm_a, label_a, run_b, firm_b, label_b, out_dir = sys.argv[1:8]
os.makedirs(out_dir, exist_ok=True)

BURN_IN_MONTHS = 0
REP_WINDOW = 600  # 50-year illustrative window, matching the paper

COLOR_A = '#1f77b4'
COLOR_B = '#d62728'


def load(run_path, firm_path):
    df = pd.read_csv(run_path)
    monthly = df.iloc[20::21].reset_index(drop=True)
    fs = pd.read_csv(firm_path)
    return df, monthly, fs


df_a, monthly_a, fs_a = load(run_a, firm_a)
df_b, monthly_b, fs_b = load(run_b, firm_b)

n_months = min(len(monthly_a), len(monthly_b))
print(f"{label_a}: {len(monthly_a)} months available; {label_b}: {len(monthly_b)} months available")
print(f"Truncating both runs to the first {n_months} months for a like-for-like comparison")

post_a = monthly_a.iloc[BURN_IN_MONTHS:n_months].reset_index(drop=True)
post_b = monthly_b.iloc[BURN_IN_MONTHS:n_months].reset_index(drop=True)

fs_a_post = fs_a[(fs_a['month'] > BURN_IN_MONTHS) & (fs_a['month'] <= n_months)]
fs_b_post = fs_b[(fs_b['month'] > BURN_IN_MONTHS) & (fs_b['month'] <= n_months)]

# day-level data, truncated to the same number of months
days_a = df_a.iloc[:n_months * 21].reset_index(drop=True)
days_b = df_b.iloc[:n_months * 21].reset_index(drop=True)

# =========================================================================
# FIGURE 4: Excess demand (left) + Employment, 50-yr window (right)
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ED_XMAX = 3.0  # UnsatisfiedDemandPct is already in percent units (not a 0-1 fraction);
                # 3% covers ~99.4% of months in both runs, leaving the rare shock tail out of frame
ed_bins = np.linspace(0, ED_XMAX, 200)
ed_a = post_a['UnsatisfiedDemandPct'].values
ed_b = post_b['UnsatisfiedDemandPct'].values
axes[0].hist(ed_a, bins=ed_bins, density=True,
             histtype='step', color=COLOR_A, linewidth=1.0, label=label_a)
axes[0].hist(ed_b, bins=ed_bins, density=True,
             histtype='step', color=COLOR_B, linewidth=1.0, label=label_b)
axes[0].set_xlim(0, ED_XMAX)
axes[0].set_yscale('log')
axes[0].set_xlabel('Unsatisfied demand (in %)')
axes[0].set_ylabel('Probability Density Function (log scale)')
axes[0].set_title('Excess demand')
axes[0].legend(fontsize=8)
frac_a_in = (ed_a <= ED_XMAX).mean() * 100
frac_b_in = (ed_b <= ED_XMAX).mean() * 100
print(f"  fig4 excess-demand panel: {label_a} {frac_a_in:.1f}% of months <= {ED_XMAX}%, "
      f"{label_b} {frac_b_in:.1f}% of months <= {ED_XMAX}% (rest are off-axis in the shock tail)")

window_months = min(REP_WINDOW, n_months)
window_a = post_a.iloc[:window_months]
window_b = post_b.iloc[:window_months]
years = window_a.index / 12
axes[1].plot(years, window_a['Employment'], color=COLOR_A, linewidth=0.7, label=label_a)
axes[1].plot(years, window_b['Employment'], color=COLOR_B, linewidth=0.7, label=label_b)
axes[1].set_xlabel('Years')
axes[1].set_ylabel('Employed households')
axes[1].set_title(f'Employment, {window_months}-month subperiod')
axes[1].legend(fontsize=8)

plt.tight_layout()
out4 = os.path.join(out_dir, 'fig4_excess_demand_employment.pdf')
plt.savefig(out4)
plt.savefig(out4.replace('.pdf', '.png'), dpi=150)
plt.close(fig)
print(f"Wrote {out4}")

# =========================================================================
# FIGURE 5: Phillips curve (left) + Beveridge curve (right)
# =========================================================================
rng = np.random.default_rng(0)


def phillips_beveridge_data(post):
    unemployment = (1000 - post['Employment'])
    delta_p = post['AvgPrice'].diff()
    vacancy = post['NumOpenPositions']
    f5 = pd.DataFrame({'unemployment': unemployment, 'delta_p': delta_p, 'vacancy': vacancy}).dropna()
    n = len(f5)
    jitter_u = f5['unemployment'].values + rng.uniform(-0.5, 0.5, size=n)
    jitter_v = f5['vacancy'].values + rng.uniform(-0.5, 0.5, size=n)
    return f5['delta_p'].values, jitter_u, jitter_v


delta_p_a, jitter_u_a, jitter_v_a = phillips_beveridge_data(post_a)
delta_p_b, jitter_u_b, jitter_v_b = phillips_beveridge_data(post_b)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].scatter(delta_p_a, jitter_u_a, s=4, alpha=0.4, color=COLOR_A, label=label_a)
axes[0].scatter(delta_p_b, jitter_u_b, s=4, alpha=0.4, color=COLOR_B, label=label_b)
axes[0].set_ylabel('Unemployment (absolute)')
axes[0].set_xlabel(r'$\Delta P$')
axes[0].set_title('Phillips curve')
axes[0].legend(fontsize=8)

axes[1].scatter(jitter_v_a, jitter_u_a, s=4, alpha=0.4, color=COLOR_A, label=label_a)
axes[1].scatter(jitter_v_b, jitter_u_b, s=4, alpha=0.4, color=COLOR_B, label=label_b)
axes[1].set_ylabel('Unemployment (absolute)')
axes[1].set_xlabel('Vacancies')
axes[1].set_title('Beveridge curve')
axes[1].legend(fontsize=8)

plt.tight_layout()
out5 = os.path.join(out_dir, 'fig5_phillips_beveridge.pdf')
plt.savefig(out5)
plt.savefig(out5.replace('.pdf', '.png'), dpi=150)
plt.close(fig)
print(f"Wrote {out5}")

# =========================================================================
# FIGURE 6: Firm size distribution (left) + price-change frequency (right)
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))


def firm_sizes(fs_post):
    if 'demand' in fs_post.columns:
        return fs_post['demand'].values.astype(float), 'Firm Size (in demand)'
    return fs_post['num_workers'].values.astype(float), 'Number of workers (PLACEHOLDER)'


sizes_a, xlabel_a = firm_sizes(fs_a_post)
sizes_b, xlabel_b = firm_sizes(fs_b_post)
axes[0].hist(sizes_a, bins=60, density=True, histtype='step', color=COLOR_A, linewidth=1.0, label=label_a)
axes[0].hist(sizes_b, bins=60, density=True, histtype='step', color=COLOR_B, linewidth=1.0, label=label_b)
axes[0].set_xlabel(xlabel_a if xlabel_a == xlabel_b else f'{xlabel_a} / {xlabel_b}')
axes[0].set_ylabel('Probability Density Function')
axes[0].set_title('Firm size distribution')
axes[0].legend(fontsize=8)


def price_change_freqs(fs_post):
    freqs = []
    for fid, g in fs_post.sort_values('month').groupby('firm_id'):
        prices = g['price'].values
        if len(prices) > 1:
            freqs.append(np.mean(np.diff(prices) != 0))
    return np.array(freqs) * 100


freq_a = price_change_freqs(fs_a_post)
freq_b = price_change_freqs(fs_b_post)
axes[1].hist(freq_a, bins=40, density=True, histtype='step', color=COLOR_A, linewidth=1.0, label=label_a)
axes[1].hist(freq_b, bins=40, density=True, histtype='step', color=COLOR_B, linewidth=1.0, label=label_b)
axes[1].set_xlabel('Firms Changing Price (in %)')
axes[1].set_ylabel('Probability Density Function')
axes[1].set_title('Frequency of price changes')
axes[1].legend(fontsize=8)

plt.tight_layout()
out6 = os.path.join(out_dir, 'fig6_firmsize_pricefreq.pdf')
plt.savefig(out6)
plt.savefig(out6.replace('.pdf', '.png'), dpi=150)
plt.close(fig)
print(f"Wrote {out6}")
print(f"  {label_a} median price-change frequency: {np.median(freq_a):.2f}% (paper: 9%)")
print(f"  {label_b} median price-change frequency: {np.median(freq_b):.2f}% (paper: 9%)")

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


def ccf_series(post):
    n_quarters = len(post) // 3
    emp_q = np.array([post['Employment'].iloc[3 * i:3 * i + 3].mean() for i in range(n_quarters)])
    price_q = np.array([post['AvgPrice'].iloc[3 * i:3 * i + 3].mean() for i in range(n_quarters)])
    emp_cycle, _ = hpfilter(emp_q, lamb=1600)
    price_cycle, _ = hpfilter(price_q, lamb=1600)
    lags = list(range(-8, 9))
    return lags, [ccf_at_lag(emp_cycle, price_cycle, lag) for lag in lags]


lags_a, ccf_a = ccf_series(post_a)
lags_b, ccf_b = ccf_series(post_b)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(lags_a, ccf_a, 'o-', color=COLOR_A, label=label_a)
axes[0].plot(lags_b, ccf_b, 'o-', color=COLOR_B, label=label_b)
axes[0].axhline(0, color='k', linewidth=0.5)
axes[0].set_xlabel('lag k (quarters)')
axes[0].set_ylabel('correlation')
axes[0].set_title('GDP (Employment) vs lagged prices')
axes[0].legend(fontsize=8)

liq_months = min(BURN_IN_MONTHS + 6, n_months)
daily_a = days_a.iloc[BURN_IN_MONTHS * 21: liq_months * 21].reset_index(drop=True)
daily_b = days_b.iloc[BURN_IN_MONTHS * 21: liq_months * 21].reset_index(drop=True)
axes[1].plot(daily_a.index, daily_a['HHLiquidity'], color=COLOR_A, linestyle='-', label=f'{label_a} - HH')
axes[1].plot(daily_a.index, daily_a['FirmLiquidity'], color=COLOR_A, linestyle='--', label=f'{label_a} - Firm')
axes[1].plot(daily_b.index, daily_b['HHLiquidity'], color=COLOR_B, linestyle='-', label=f'{label_b} - HH')
axes[1].plot(daily_b.index, daily_b['FirmLiquidity'], color=COLOR_B, linestyle='--', label=f'{label_b} - Firm')
axes[1].set_xlabel('day')
axes[1].set_ylabel('Aggregate liquidity')
axes[1].set_title(f'Liquidity circulation, {liq_months - BURN_IN_MONTHS}-month window')
axes[1].legend(fontsize=7)

plt.tight_layout()
out7 = os.path.join(out_dir, 'fig7_gdpcorr_liquidity.pdf')
plt.savefig(out7)
plt.savefig(out7.replace('.pdf', '.png'), dpi=150)
plt.close(fig)
print(f"Wrote {out7}")
