#!/usr/bin/env python3
"""Make a 4-panel Employment/AvgPrice/AvgWage/TotalInv diagnostic figure,
in the same style as diagnostics/diagnostic_run_20k.png, for any run CSV.

Usage:
    python scripts/make_diagnostic_figure.py <run_csv> [out_path]

If out_path is omitted, writes <run_csv with .csv replaced by _diagnostic.pdf>.
"""
import sys
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

path = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else path.replace('.csv', '_diagnostic.pdf')

df = pd.read_csv(path)

panels = ['Employment', 'AvgPrice', 'AvgWage', 'TotalInv']
titles = ['Employment', 'Average Price', 'Average Wage', 'Total Inventory']

fig, axes = plt.subplots(len(panels), 1, figsize=(15, 15))
for ax, col, title in zip(axes, panels, titles):
    ax.plot(df.index, df[col])
    ax.set_title(title)

plt.tight_layout()
plt.savefig(out)
print(f"Wrote {out}")
