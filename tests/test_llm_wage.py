"""
Tests for LLMFirm.aset_wage and the wage_mode="llm" path through the model.

No real LLM is called: each firm's .llm is swapped for FakeLLM, which returns a
fixed reply and records the prompts it was sent. Building the model does not
contact Ollama; only agenerate would.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest  # type: ignore[import]

from src.agents import LLMFirm
from src.config import PARAMS
from src.model import LegnickModel


class FakeLLM:
    """Stands in for SamplingModuleLLM: returns a fixed reply, records prompts, no network."""

    def __init__(self, content):
        self.content = content
        self.prompts = []

    async def agenerate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        msg = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)

    def _build_messages(self, prompt):
        return [{"role": "user", "content": prompt}]


def reply(action, new_wage):
    return json.dumps({"action": action, "new_wage": new_wage, "reasoning": "test"})


@pytest.fixture
def m():
    """Fresh model per test so months_full etc. never leak between tests."""
    p = dict(PARAMS)
    # n_firms must be > 7: each household shops at 7 firms and, when searching
    # for a cheaper one, picks from the firms it is NOT connected to
    p.update(n_households=20, n_firms=10, seed=1, pricing_mode="rule",
             wage_mode="llm", llm_model="ollama/llama3.2:3b", log_wage_prompts=True)
    return LegnickModel(**p)


@pytest.fixture
def firm(m):
    return m.firms[0]


def set_wage(firm, content, max_change_frac=0.5):
    firm.llm = FakeLLM(content)
    return asyncio.run(firm.aset_wage(max_change_frac=max_change_frac))


# --- setup -------------------------------------------------------------------------

def test_llm_wage_mode_creates_llm_firms(m):
    assert all(isinstance(f, LLMFirm) for f in m.firms)
    assert m.wage_prompt_log == []


# --- the decision itself ---------------------------------------------------------

def test_raise_is_applied_and_logged(firm):
    firm.w_f = 1428
    assert set_wage(firm, reply("raise", 1450)) == (1450.0, True)
    assert firm.w_f == 1450.0
    log = firm.wage_log[-1]
    assert (log["action"], log["ok"], log["w_f"], log["new_wage"]) == ("raise", True, 1428, 1450.0)


def test_bad_reply_keeps_the_wage(firm):
    firm.w_f = 1428
    new_wage, ok = set_wage(firm, "nonsense")
    assert (new_wage, ok, firm.w_f) == (1428, False, 1428)
    assert firm.wage_log[-1]["action"] is None


def test_max_change_frac_is_passed_through(firm):
    firm.w_f = 1428
    new_wage, ok = set_wage(firm, reply("raise", 1600), max_change_frac=0.1)
    assert (new_wage, ok) == (1428, False)


# --- zero-wage floor (same as Firm.set_wages) ------------------------------------

def test_zero_wage_is_floored_before_the_prompt(firm):
    firm.w_f = 0
    set_wage(firm, reply("hold", 0))
    assert firm.w_f == 1.0
    assert "Current wage: 1.00" in firm.llm.prompts[-1]
    assert firm.wage_log[-1]["w_f"] == 1.0


def test_floored_wage_can_be_raised(firm):
    firm.w_f = 0
    assert set_wage(firm, reply("raise", 1.4)) == (1.4, True)


# --- months_full bookkeeping -------------------------------------------------------

def test_unfilled_vacancy_resets_months_full(firm):
    firm.open_position, firm.months_full = True, 5
    set_wage(firm, reply("raise", 1450))
    assert firm.months_full == 0


def test_no_vacancy_increments_months_full(firm):
    firm.open_position, firm.months_full = False, 5
    set_wage(firm, reply("lower", 1420))
    assert firm.months_full == 6
    assert firm.wage_log[-1]["months_full"] == 5  # what the LLM was shown


def test_wage_cut_does_not_reset_months_full(firm):
    """Unlike the rule, only a vacancy resets the counter, so the prompt line stays true."""
    firm.open_position, firm.months_full = False, 30
    set_wage(firm, reply("lower", 1400))
    assert firm.months_full == 31


def test_prompt_shows_vacancy_state(firm):
    firm.open_position, firm.months_full = True, 0
    set_wage(firm, reply("hold", 1428))
    assert "no job seeker accepted it" in firm.llm.prompts[-1]

    firm.open_position, firm.months_full = False, 7
    set_wage(firm, reply("hold", 1428))
    prompt = firm.llm.prompts[-1]
    assert "You have no unfilled vacancy" in prompt
    assert "Months in a row without an unfilled vacancy: 7" in prompt


def test_wage_decision_does_not_change_hiring(firm):
    for flag in (True, False):
        firm.open_position = flag
        set_wage(firm, reply("raise", firm.w_f * 1.01))
        assert firm.open_position is flag


# --- prompt log --------------------------------------------------------------------

def test_wage_prompt_log_records_the_decision(m, firm):
    firm.w_f = 1428
    set_wage(firm, reply("raise", 1450))
    rec = m.wage_prompt_log[-1]
    assert rec["firm_id"] == firm.unique_id
    assert (rec["w_f"], rec["new_wage"], rec["action"], rec["ok"]) == (1428, 1450.0, "raise", True)
    assert rec["prompt_variant"] == "default"
    assert m.prompt_log is None  # wage records never go to the price log


# --- whole month through the model -----------------------------------------------

def test_one_month_runs_without_ollama(m):
    for f in m.firms:
        f.llm = FakeLLM(reply("hold", 1428))
    for _ in range(21):
        m.step()

    assert len(m.wage_prompt_log) == len(m.firms)
    assert all(s["llm_wage_ok"] is True for s in m.firm_snapshots)
    assert all(s["llm_wage_action"] == "hold" for s in m.firm_snapshots)
    assert "wage_wall_s" in m.timings[-1]


def test_wage_is_set_before_employment(m, monkeypatch):
    """The LLM must see last month's open_position, before set_employment_and_fire resets it."""
    seen = []
    for f in m.firms:
        f.llm = FakeLLM(reply("hold", 1428))
        f.open_position = True
    original = LLMFirm.aset_wage

    async def spy(self, **kwargs):
        seen.append(self.open_position)
        return await original(self, **kwargs)

    monkeypatch.setattr(LLMFirm, "aset_wage", spy)
    m.step()  # first step of the month runs the wage decision
    assert seen == [True] * len(m.firms)
