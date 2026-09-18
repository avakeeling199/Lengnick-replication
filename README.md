# Lengnick (2013) ABM Replication

A Mesa 3 replication of:

> Lengnick, M. (2013). Agent-based macroeconomics: A baseline model. *Journal of
> Economic Behavior & Organization*, 86, 102-120.

The baseline model's rule-based firm pricing decisions can be swapped for an
LLM-based pricing agent (via a local [Ollama](https://ollama.com) model), so
the two decision-making regimes can be compared directly under the same model
and initial conditions. This replication is also intended as a technical
foundation for a separate, longer-term DPhil project on an NHS-focused
agent-based model.

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
│                             (set_prices_rule / set_prices_llm)
├── model.py                 # LegnickModel: agent creation, network setup, step() orchestration
├── config.py                  # PARAMS: shared model parameter defaults
└── llm/                          # LLM-based pricing decisions
    ├── client.py                   # Ollama API calls (chat endpoint, structured JSON response)
    ├── prompts.py                     # prompt templates for the firm pricing decision
    └── parsing.py                       # response parsing/validation, fallback handling

scripts/                  # standalone analysis / figure-generation scripts
├── check_all_seed.py            # sanity-checks across seed runs from HPC array jobs
├── check_zombie_firms.py          # diagnostic for firms stuck with no workers/inventory
├── make_tier1_figures_singlerun.py  # reproduces paper's Fig 4-7 from a single run
├── make_tier1_figures_comparison.py   # Fig 4-7, overlaying LLM- vs rule-based runs
├── make_diagnostic_figure.py            # ad-hoc diagnostic plots for a single run
├── plot_business_cycles.py                # zoomed time series + ACF for business-cycle inspection
├── recreate_figs.sh                         # SLURM array job: runs main.py across seeds on the HPC
├── llm_full_run.sh                            # SLURM job: full-length run with LLM pricing
├── llm_test_run.sh / llm_smoke_test*.sh          # short LLM-pricing runs for sanity checks
├── llm_prompt_quickcheck.sh                        # quick check of prompt/response behaviour
├── llm_resubmit.sh                                   # resubmits an interrupted LLM run from checkpoint
└── test_llm_gpu.sh                                     # checks Ollama GPU availability on the HPC node

tests/
└── test_paper_conformance.py  # tests checking model behaviour against paper's stated assumptions

diagnostics/              # CSV output from simulation runs (not all committed)
figs/                     # generated figures, incl. paper comparison figures (fig4-7)
logs/                     # SLURM job output (created by scripts/recreate_figs.sh and the LLM run scripts)
```

All scripts in `scripts/` and `tests/` assume they are run from the repository
root (e.g. `python scripts/plot_business_cycles.py ...`), since they reference
`diagnostics/`/`figs/` as relative paths.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

LLM-based pricing runs (`--pricing-mode llm`) require a local
[Ollama](https://ollama.com) install serving the model referenced in
`src/llm/client.py` (default: `llama3.3:70b`); not required for rule-based runs.

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
(`rule` uses the paper's rule-based pricing; `llm` routes each firm's pricing
decision through the local Ollama model instead — see `src/llm/`).

Model parameter defaults (`alpha`, `theta`, `psi_price`, etc.) live in
`src/config.py`.

To reproduce the paper's Tier 1 validation figures (Fig 4-7) from a single
run: `scripts/make_tier1_figures_singlerun.py`. To compare an LLM-pricing run
against a rule-based run side by side (as used in the report):

```bash
python scripts/make_tier1_figures_comparison.py \
    <run_llm.csv> <firm_snapshots_llm.csv> "LLM pricing" \
    <run_rule.csv> <firm_snapshots_rule.csv> "Rule-based pricing" \
    <out_dir>
```

To reproduce runs across multiple seeds on the HPC: submit
`scripts/recreate_figs.sh` (rule-based, SLURM array job) or
`scripts/llm_full_run.sh` (LLM pricing) from the repo root.

## Notes

- Initial calibration (e.g. `m_h`, `w_f`, `p_f`) is not specified exactly in the
  paper, since the model relies on a long burn-in period to wash out arbitrary
  starting conditions. Values here were chosen to keep household demand and firm
  production roughly balanced from the start, to make short test runs more
  informative.