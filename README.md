# Lengnick (2013) ABM Replication

A Mesa 3 replication of:

> Lengnick, M. (2013). Agent-based macroeconomics: A baseline model. *Journal of
> Economic Behavior & Organization*, 86, 102-120.

Part of a 10-week project to replicate the baseline model and then replace its
rule-based agent decisions with LLM-based decision-making, to see how the model's
results change. This replication is also intended as a technical foundation for a
separate, longer-term DPhil project on an NHS-focused agent-based model.

## Model overview

Two agent types:
- **Households**: have a reservation wage (`w_h`) and liquidity (`m_h`). Each is
  connected to 7 firms for buying goods (type A connections) and at most one
  employer (type B connection).
- **Firms**: have liquidity (`m_f`), inventory (`i_f`), a goods price (`p_f`), and
  a wage rate (`w_f`).

Time is indexed in days (consumption goods bought daily) and months of 21 days
(labour bought monthly, wages/profits paid monthly).

## Project structure

```
main.py                  # CLI entry point: run a simulation, save outputs
src/
├── agents.py              # Household and Firm agent classes and behaviours
├── model.py                 # LegnickModel: agent creation, network setup, step() orchestration
├── config.py                  # PARAMS: shared model parameter defaults
└── llm/                          # LLM-based decision-making (in progress, see Status)
    ├── client.py                   # Ollama API calls (not yet implemented)
    ├── prompts.py                     # prompt templates (not yet implemented)
    └── parsing.py                       # response parsing/validation (not yet implemented)

scripts/                  # standalone analysis / figure-generation scripts
├── check_all_seed.py       # sanity-checks across seed runs from HPC array jobs
├── check_zombie_firms.py     # diagnostic for firms stuck with no workers/inventory
├── make_tier1_figures_singlerun.py  # reproduces paper's Fig 4-7 from a single run
├── plot_business_cycles.py        # zoomed time series + ACF for business-cycle inspection
├── recreate_figs.sh                  # SLURM array job: runs main.py across seeds on the HPC
└── llm_pricing_first.py                # scratch test script for the Ollama pricing pilot
                                            (to be removed once its logic moves into src/llm/)

tests/
└── test_paper_conformance.py  # tests checking model behaviour against paper's stated assumptions

diagnostics/              # CSV output from simulation runs (not all committed)
figs/                     # generated figures, incl. paper comparison figures (fig4-7)
logs/                     # SLURM job output (created by scripts/recreate_figs.sh)
```

All scripts in `scripts/` and `tests/` assume they are run from the repository
root (e.g. `python scripts/plot_business_cycles.py ...`), since they reference
`diagnostics/`/`figs/` as relative paths.

## Status

- [x] Agent classes with state variables
- [x] Trading network initialisation (type A / type B connections)
- [x] Daily step: household shopping, firm production
- [x] End of month: wage payment, buffer, profit distribution, reservation wage adjustment
- [x] Beginning of month: wage setting, employment/price decisions, job search, connection search
- [x] Model runs for extended periods without crashing
- [x] Data collection (Mesa `DataCollector`)
- [x] Reproduce paper's validation figures (Fig 4-7: employment, unsatisfied
      demand, Phillips/Beveridge curves)
- [ ] Resolve cyclicity bug (model does not yet reproduce the paper's business
      cycles as expected; currently under review)
- [ ] LLM-based decision-making integration — `pricing_mode` is accepted by
      `LegnickModel` and passed through, but `Firm.set_prices` does not yet act on
      it. LLM pricing logic (`src/llm/`) is being built on a separate branch.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires a local [Ollama](https://ollama.com) install for the (in-progress) LLM
decision-making functionality; not required for rule-based runs.

## Running

Run a full simulation and save outputs:

```bash
python main.py --seed 42 --months 7000 \
    --out diagnostics/run_seed42.csv \
    --firm-snapshots diagnostics/firm_snapshots_seed42.csv
```

For a quick local sanity check before committing to a full run:

```bash
python main.py --seed 42 --months 50 \
    --out diagnostics/check_seed42.csv \
    --firm-snapshots diagnostics/check_firm_snapshots_seed42.csv
```

Optional flags: `--n-households`, `--n-firms`, `--pricing-mode {rule,llm}`
(`llm` mode is accepted but not yet functionally different from `rule` — see
Status above).

Model parameter defaults (`alpha`, `theta`, `psi_price`, etc.) live in
`src/config.py`.

To reproduce the paper's Tier 1 comparison figures across multiple seeds on the
HPC: submit `scripts/recreate_figs.sh` (a SLURM array job) from the repo root,
then run `scripts/make_tier1_figures_singlerun.py` on the resulting output.

## Notes

- Initial calibration (e.g. `m_h`, `w_f`, `p_f`) is not specified exactly in the
  paper, since the model relies on a long burn-in period to wash out arbitrary
  starting conditions. Values here were chosen to keep household demand and firm
  production roughly balanced from the start, to make short test runs more
  informative.