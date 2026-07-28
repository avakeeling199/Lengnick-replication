import argparse
import os
import time

import pandas as pd

from src.model import LegnickModel
from src.config import PARAMS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--months', type=int, default=7000,
                        help='total months to simulate (paper: 6000 + 1000 burn-in = 7000)')
    parser.add_argument('--out', type=str, required=True,
                        help='path to write the main model-level CSV')
    parser.add_argument('--firm-snapshots', type=str, required=True,
                        help='path to write the per-firm monthly snapshot CSV')
    parser.add_argument('--n-households', type=int, default=1000)
    parser.add_argument('--n-firms', type=int, default=100)
    parser.add_argument('--pricing-mode', choices=['rule', 'llm'], default='rule', 
                        help='whether firms use rule-based or LLM-based pricing')
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    snap_dir = os.path.dirname(args.firm_snapshots)
    if snap_dir:
        os.makedirs(snap_dir, exist_ok=True)

    steps = args.months * 21

    model_params = dict(PARAMS)
    model_params['n_households'] = args.n_households
    model_params['n_firms'] = args.n_firms
    model_params['seed'] = args.seed
    model_params['pricing_mode'] = args.pricing_mode

    model = LegnickModel(**model_params)

    t0 = time.time()
    for i in range(steps):
        model.step()
        if (i + 1) % (21 * 100) == 0:
            elapsed = time.time() - t0
            print(f"[seed {args.seed}] month {(i + 1) // 21}/{args.months} "
                f"({elapsed:.0f}s elapsed)", flush=True)

    data = model.datacollector.get_model_vars_dataframe()
    data.to_csv(args.out)

    pd.DataFrame(model.firm_snapshots).to_csv(args.firm_snapshots, index=False)

    print(f"[seed {args.seed}] DONE in {time.time() - t0:.0f}s, "
        f"wrote {args.out} and {args.firm_snapshots}", flush=True)


if __name__ == '__main__':
    main()