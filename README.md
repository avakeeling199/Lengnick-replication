# Lengnick (2013) ABM Replication

A Mesa 3 replication of:

> Lengnick, M. (2013). Agent-based macroeconomics: A baseline model. *Journal of
> Economic Behavior & Organization*, 86, 102-120.

The firms' rule-based pricing decision can be swapped for an LLM-based one,
built on [mesa-llm](https://github.com/projectmesa/mesa-llm), so the two
decision-making regimes can be compared under the same model and initial
conditions. Any model that [LiteLLM](https://docs.litellm.ai) supports can be
used; development runs use [Ollama](https://ollama.com) locally (`llama3.2:3b`)
and on an HPC cluster (`llama3.3:70b`). This replication is also intended as a
technical foundation for a separate, longer-term DPhil project on an
NHS-focused agent-based model.

## Model overview

Two agent types:
- **Households**: have a reservation wage (`w_h`) and liquidity (`m_h`). Each is
  connected to 7 firms for buying goods (type A connections) and at most one
  employer (type B connection).
- **Firms**: have liquidity (`m_f`), inventory (`i_f`), a goods price (`p_f`), and
  a wage rate (`w_f`).

Time is indexed in days (consumption goods bought daily) and months of 21 days
(labour bought monthly, wages/profits paid monthly).

| When | Firms | Households |
|---|---|---|
| Start of month | set wage, set hiring/firing, **set price** | search for cheaper / better-stocked shops, job search, set consumption budget |
| Every day | produce | buy goods from their type A connections |
| End of month | pay wages, build buffer, distribute profits | adjust reservation wage |

### What the LLM replaces

With `--pricing-mode llm`, every firm's monthly price decision is made by an
LLM instead of the paper's rule; everything else in the table above stays
rule-based.

- **Rule** (`Firm.set_prices_rule`): if inventory is above its upper band, lower
  the price; if below its lower band, raise it. Each change happens with
  probability θ, by a random fraction drawn from U(0, ϑ), and only within the
  marginal-cost price bounds.
- **LLM** (`LLMFirm.aset_price`): the LLM sees the same inputs (inventory,
  demand, current price, marginal cost, inventory bands, price bounds and recent
  history) and returns either "hold" or a new price, with a short
  justification. The same guards as the rule then apply: no cuts once the price
  is at or below its lower bound, no rises once it is at or above its upper
  bound. Unparseable responses, non-positive prices and jumps of more than 50%
  are rejected and the firm keeps its current price. The 100 firms are priced
  concurrently.

Every LLM decision is logged (price, reasoning, validity, latency, inputs) and
written to the per-firm snapshot CSV.

## Project structure

```
main.py                    # CLI entry point: run a simulation, save outputs
src/
├── agents.py              # Household, Firm (rule-based) and LLMFirm (mesa-llm) agents
├── model.py               # LegnickModel: agent creation, network setup, step() orchestration
├── config.py              # PARAMS: shared model parameter defaults
└── llm/                   # LLM pricing support
    ├── sampling_llm.py    # mesa-llm ModuleLLM subclass that passes temperature through to LiteLLM
    ├── prompts.py         # prompt template and JSON response schema for the pricing decision
    └── parsing.py         # response parsing and price validation, with fallback

scripts/                   # standalone analysis / figure-generation scripts
├── make_tier1_figures_singlerun.py   # reproduces the paper's Figs 4-7 from a single run
├── make_tier1_figures_comparison.py  # Figs 4-7, overlaying two runs (e.g. LLM vs rule)
├── make_diagnostic_figure.py         # ad-hoc diagnostic plots for a single run
├── plot_business_cycles.py           # zoomed time series + ACF for business-cycle inspection
├── check_all_seed.py                 # sanity checks across a batch of seed runs
└── check_zombie_firms.py             # diagnostic for firms stuck with no workers/inventory

tests/
└── test_paper_conformance.py  # checks agent behaviour against the paper's stated rules
notes/
└── mesa-llm-issues.md         # friction found while porting to mesa-llm (upstream contribution ideas)

diagnostics/   # CSV output and checkpoints from simulation runs (not all committed)
figs/          # generated figures (git-ignored)
```

All scripts in `scripts/` and `tests/` assume they are run from the repository
root (e.g. `python scripts/plot_business_cycles.py ...`), since they reference
`diagnostics/` and `figs/` as relative paths.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Rule-based runs need nothing else. For LLM-based runs with a local model,
install [Ollama](https://ollama.com) and pull a model:

```bash
ollama pull llama3.2:3b
```

## Running

Run a full rule-based simulation and save outputs:

```bash
python main.py --seed 42 --months 7000 \
    --out diagnostics/run_seed42.csv \
    --firm-snapshots diagnostics/firm_snapshots_seed42.csv
```

For a quick local check, use e.g. `--months 50`.

Run with LLM pricing (here a local Ollama model):

```bash
python main.py --seed 42 --months 3 --pricing-mode llm \
    --llm-model ollama/llama3.2:3b \
    --llm-api-base http://localhost:11434 \
    --out diagnostics/llm_seed42.csv \
    --firm-snapshots diagnostics/llm_firm_snapshots_seed42.csv
```

| Flag | Meaning |
|---|---|
| `--pricing-mode {rule,llm}` | rule-based (paper) or LLM pricing; default `rule` |
| `--llm-model` | LiteLLM model string, e.g. `ollama/llama3.2:3b`; required in `llm` mode |
| `--llm-api-base` | URL of the LLM server, e.g. `http://localhost:11434`; optional |
| `--llm-temperature` | sampling temperature for pricing calls; default `0.3` |
| `--resume-from` | resume from a checkpoint `.pkl` (saved periodically to `diagnostics/`) |
| `--n-households`, `--n-firms` | population sizes; defaults 1000 and 100 |

Model parameter defaults (`alpha`, `theta`, `psi_price`, etc.) live in
`src/config.py`.

To reproduce the paper's validation figures (Figs 4-7) from a single run:

```bash
python scripts/make_tier1_figures_singlerun.py \
    diagnostics/run_seed42.csv diagnostics/firm_snapshots_seed42.csv
```

To compare an LLM-pricing run against a rule-based run side by side:

```bash
python scripts/make_tier1_figures_comparison.py \
    <run_llm.csv> <firm_snapshots_llm.csv> "LLM pricing" \
    <run_rule.csv> <firm_snapshots_rule.csv> "Rule-based pricing" \
    <out_dir>
```

## Tests

```bash
pytest
```

## Reproducibility

- **Rule-based runs** are fully reproducible: the same `--seed` gives identical
  output.
- **LLM runs** are not, even with a fixed seed: the seed controls the model's
  own randomness but not the LLM's sampling. mesa-llm 0.3.2 does not pass
  sampling parameters to LiteLLM, so `src/llm/sampling_llm.py` adds temperature;
  passing a sampling seed is a planned addition.

## Notes

- Initial calibration (e.g. `m_h`, `w_f`, `p_f`) is not specified exactly in the
  paper, since the model relies on a long burn-in period to wash out arbitrary
  starting conditions. Values here were chosen to keep household demand and firm
  production roughly balanced from the start, to make short test runs more
  informative.
- A resumed run keeps the settings it was started with; LLM flags passed
  alongside `--resume-from` are ignored.
- Checkpoints saved before the mesa-llm port contain plain `Firm` agents and
  cannot be resumed in LLM mode; start a fresh run instead.
