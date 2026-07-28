import sys
import pandas as pd
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else 'diagnostics/firm_snapshots_seed2048561.csv'

fs = pd.read_csv(path)
fs_post = fs[fs['month'] > 1000].sort_values(['firm_id', 'month'])

spell_lengths = []
for fid, g in fs_post.groupby('firm_id'):
    is_zero = (g['num_workers'].values == 0)
    run = 0
    for z in is_zero:
        if z:
            run += 1
        elif run > 0:
            spell_lengths.append(run)
            run = 0
    if run > 0:
        spell_lengths.append(run)

spell_lengths = np.array(spell_lengths)
print(f"{len(spell_lengths)} zero-worker spells found")
if len(spell_lengths) > 0:
    print(f"median spell length: {np.median(spell_lengths):.1f} months")
    print(f"mean: {spell_lengths.mean():.1f}, max: {spell_lengths.max()}")
    print(f"fraction of spells lasting > 12 months: {(spell_lengths > 12).mean():.2%}")
else:
    print("no zero-worker spells found")

import pandas as pd


worst_fid, worst_len, cur_fid, cur_len = None, 0, None, 0
for fid, g in fs_post.groupby('firm_id'):
    run = 0
    for z in (g['num_workers'].values == 0):
        run = run + 1 if z else 0
        if run > worst_len:
            worst_len, worst_fid = run, fid

print(f"Worst firm: {worst_fid}, spell length {worst_len} months")
g = fs_post[fs_post['firm_id'] == worst_fid]
print(g[['month', 'num_workers', 'inventory', 'price']].tail(20))