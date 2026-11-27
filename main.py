import argparse
import json
import os
import pickle
import time

import pandas as pd

from src.model import LegnickModel
from src.config import PARAMS
from src.llm.prompts import PROMPT_BUILDERS, WAGE_PROMPT_BUILDERS

def write_prompt_log(path, records):
    with open(path, 'w') as f: 
        for record in records:
            f.write(json.dumps(record, default=float) + "\n")

def save_checkpoint(model, path):
    tmp = path + ".tmp"
    with open (tmp, 'wb') as f:
        pickle.dump(model, f)
    os.replace(tmp, path) # swaps the new file in all at once

def ensure_parent_dir(path):
    if path:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

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
    parser.add_argument('--llm-model', type=str, default=None,
                        help="what model you want used as your LLM firm")
    parser.add_argument('--llm-api-base', type=str, default=None, 
                        help="URL of the LLM server, e.g. http://localhost:11434 for local Ollama " \
                        "or http://<node>:11434 on the cluster (default: LiteLLM picks based on the model)")
    parser.add_argument('--llm-temperature', type=float, default=0.3,
                        help='sampling temperature for LLM calls (default=0.3)')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='path to a checkpoint .pkl to resume from')
    parser.add_argument('--max-concurrent', type=int, default=12, 
                        help='max number of concurrent requests that can be sent to llm (default=12)')
    parser.add_argument('--timings', type=str, default=None,
                        help='PATH you would like the timings to be saved to')
    parser.add_argument('--prompt-log', type=str, default=None,
                        help='where you want the prompt log to save to. If not included \
                        the log wont be saved')
    parser.add_argument('--prompt-variant', choices=sorted(PROMPT_BUILDERS), default="strict", 
                        help="which pricing prompt LLM firms use (default: strict, the original prompt)")
    parser.add_argument('--no-price-band', action='store_true',
                        help='LLM firms price freely: only the >0 and +/-50%% change checks apply '
                        '(default: moves past the 1.025-1.15 x marginal cost band are blocked)')
    parser.add_argument('--wage-mode', choices=['rule', 'llm'], default='rule',
                        help='whether firms use llm or rule-based wage decision function (default: rule)')
    parser.add_argument('--wage-max-change', type=float, default=0.5, 
                        help='the largest fraction a wage can move in one month (rule moves at most 1.9%%)')
    parser.add_argument('--wage-prompt-variant', choices=sorted(WAGE_PROMPT_BUILDERS), default='default',
                        help='what prompt you would like to use for the llm wage decision rule')
    parser.add_argument('--wage-prompt-log', type=str, default=None, 
                        help='whether you would like to log the prompts for the llm wage decision function')

    args = parser.parse_args()
    if args.max_concurrent < 1:
        parser.error('--max-concurrent must be at least 1')
    if args.no_price_band and args.prompt_variant == 'strict':
        parser.error('--no-price-band contradicts the strict prompt, which tells firms moves past the band are blocked')
    if 'llm' in (args.pricing_mode, args.wage_mode) and not args.llm_model:
        parser.error('using llm pricing or wage mode requires --llm-model')
    if not 0 < args.wage_max_change < 1: 
        parser.error('invalid --wage-max-change. please choose a value between 0 and 1')
    if args.wage_prompt_log and args.wage_mode == 'rule' and not args.resume_from:
        parser.error('--wage-prompt-log needs --wage-mode llm')

    for path in (args.out, args.firm_snapshots, args.timings, args.prompt_log, args.wage_prompt_log):
        ensure_parent_dir(path)

    for path in (args.out, args.firm_snapshots):
        open(path, 'a').close() #fail now rather than at the end of run 

    if args.resume_from:
        with open(args.resume_from, 'rb') as f:
            model = pickle.load(f)
        model.max_concurrent = args.max_concurrent
        print(f"Resumed from {args.resume_from} at counter={model.counter}")
        if args.llm_api_base:
            for firm in model.firms:
                if hasattr(firm, "llm"):
                    firm.llm.api_base = args.llm_api_base
            print(f"LLM requests now go to {args.llm_api_base}")
        if not args.prompt_log:
            model.prompt_log = None # stop collecting prompts we wont write out
        elif getattr(model, "prompt_log", None) is None:
            model.prompt_log = []
        if not args.wage_prompt_log:
            model.wage_prompt_log = None
        elif getattr(model, "wage_prompt_log", None) is None:
            model.wage_prompt_log = [] # resuming a run that wasnt logging: start logging now 
    else:
        model_params = dict(PARAMS)
        model_params['n_households'] = args.n_households
        model_params['n_firms'] = args.n_firms
        model_params['seed'] = args.seed
        model_params['pricing_mode'] = args.pricing_mode
        model_params['llm_model'] = args.llm_model
        model_params['llm_temperature'] = args.llm_temperature
        model_params['llm_api_base'] = args.llm_api_base
        model_params['max_concurrent'] = args.max_concurrent
        model_params['log_prompts'] = args.prompt_log is not None
        model_params['prompt_variant'] = args.prompt_variant
        model_params['price_band'] = not args.no_price_band
        model_params['wage_mode'] = args.wage_mode
        model_params['wage_max_change_frac'] = args.wage_max_change        
        model_params['wage_prompt_variant'] = args.wage_prompt_variant
        model_params['log_wage_prompts'] = args.wage_prompt_log is not None
        model = LegnickModel(**model_params)

    checkpoint_path = os.path.splitext(args.out)[0] + "_checkpoint.pkl"
    t0 = time.time()
    target_steps = args.months * 21
    remaining_steps = target_steps - model.counter

    for i in range(remaining_steps):
        model.step()
        if (model.counter) % (21 * 100) == 0:
            elapsed = time.time() - t0
            print(f"[seed {args.seed}] month {model.counter // 21}/{args.months} "
                f"({elapsed:.0f}s elapsed)", flush=True)
            model.datacollector.get_model_vars_dataframe().to_csv(args.out)
            pd.DataFrame(model.firm_snapshots).to_csv(args.firm_snapshots, index=False)
            if args.timings:
                pd.DataFrame(model.timings).to_csv(args.timings, index=False)
            save_checkpoint(model, checkpoint_path)
            if args.prompt_log:
                write_prompt_log(args.prompt_log, model.prompt_log)
            if args.wage_prompt_log:
                write_prompt_log(args.wage_prompt_log, model.wage_prompt_log)

    data = model.datacollector.get_model_vars_dataframe()
    data.to_csv(args.out)

    save_checkpoint(model, checkpoint_path)

    pd.DataFrame(model.firm_snapshots).to_csv(args.firm_snapshots, index=False)
    if args.timings:
        pd.DataFrame(model.timings).to_csv(args.timings, index=False)

    if args.prompt_log:
        write_prompt_log(args.prompt_log, model.prompt_log)
    if args.wage_prompt_log:
        write_prompt_log(args.wage_prompt_log, model.wage_prompt_log)

    print(f"[seed {args.seed}] DONE in {time.time() - t0:.0f}s, "
        f"wrote {args.out} and {args.firm_snapshots}", flush=True)


if __name__ == '__main__':
    main()