"""
Regression guard: the rule-based model must behave exactly as it did before
the LLM wage work, so rule-vs-rule and rule-vs-LLM comparisons stay valid.

The expected values were produced on `main` at commit 87c2434 (before the
wage-prompt branch) with the same parameters as below. They are compared with
`==`, not approximately: any change to rule behaviour, including an extra or
missing random draw, should fail this test. If a later change alters the rule
model on purpose, regenerate the numbers and say why in the commit message.
"""
from src.agents import Firm
from src.config import PARAMS
from src.model import LegnickModel

MONTHS = 24

# from main @ 87c2434: n_households=200, n_firms=20, seed=7, 24 months
EXPECTED_AVG_PRICE = 26.32972192210978
EXPECTED_AVG_WAGE = 1406.5590466046222
EXPECTED_EMPLOYMENT = 198
EXPECTED_NEXT_RANDOM = 0.6790838518769762


def _rule_model():
    p = dict(PARAMS)
    p.update(n_households=200, n_firms=20, seed=7, pricing_mode="rule")
    return LegnickModel(**p)


def test_rule_model_matches_main_fingerprint():
    m = _rule_model()
    for _ in range(21 * MONTHS):
        m.step()
    last = m.datacollector.get_model_vars_dataframe().iloc[-1]

    assert float(last.AvgPrice) == EXPECTED_AVG_PRICE
    assert float(last.AvgWage) == EXPECTED_AVG_WAGE
    assert int(last.Employment) == EXPECTED_EMPLOYMENT
    # same RNG state => same sequence of random draws over the whole run
    assert m.random.random() == EXPECTED_NEXT_RANDOM


def test_rule_mode_uses_plain_firms():
    m = _rule_model()
    assert m.wage_mode == "rule"
    assert all(type(f) is Firm for f in m.firms)
    assert m.wage_prompt_log is None
