# Lengnick (2013) ABM Replication

A Mesa 3 replication of:

> Lengnick, M. (2013). Agent-based macroeconomics: A baseline model. *Journal of Economic Behavior & Organization*, 86, 102-120.

Part of a 10-week project to replicate the baseline model and then
replace its rule-based agent decisions with LLM-based decision-making, to see how
the model's results change.

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
agents.py    # Household and Firm agent classes and behaviours
model.py     # LengnickModel: agent creation, network setup, step() orchestration
test.py      # scratch script for running and sanity-checking the model
```

## Status

- [x] Agent classes with state variables
- [x] Trading network initialisation (type A / type B connections)
- [x] Daily step: household shopping, firm production
- [x] End of month: wage payment, buffer, profit distribution, reservation wage adjustment
- [x] Beginning of month: wage setting, employment/price decisions, job search, connection search
- [x] Model runs for extended periods without crashing
- [ ] Data collection (Mesa `DataCollector`)
- [ ] Reproduce paper's validation figures (employment, unsatisfied demand, Phillips/Beveridge curves)
- [ ] LLM-based decision-making integration

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install mesa networkx
```

## Running

```bash
python test.py
```

## Notes

- Initial calibration (e.g. `m_h`, `w_f`, `p_f`) is not specified exactly in the
  paper, since the model relies on a long burn-in period to wash out arbitrary
  starting conditions. Values here were chosen to keep household demand and firm
  production roughly balanced from the start, to make short test runs more
  informative.
