"""
Paper-conformance tests for the Legnick replication.

Reference: Lengnick, M. (2013). "Agent-based macroeconomics: A baseline
model." Journal of Economic Behavior & Organization 86, 102-120.

Unlike test_rule_conformance.py (which only checks the code is internally
consistent), every expected value/threshold here is derived from the paper
itself (section/equation/table cited in each test's docstring), not from
reading agents.py/model.py. This file does not modify agents.py or model.py.

Run with:
    pytest test_paper_conformance.py -v
"""
import inspect
import mesa
import pytest

from agents import Household, Firm
from model import distribute_all_profits, LegnickModel


@pytest.fixture
def model():
    return mesa.Model(rng=1)


# ---------------------------------------------------------------------------
# Table 1: Calibration of the model
# ---------------------------------------------------------------------------
# Households: psi_price=0.25, xi=0.01, beta=5, pie=0.1, alpha=0.9, n=7,
#             psi_price=0.25, psi_quant=0.25
# Firms:      gamma=24, delta=0.019, phi_emp_upper=1, phi_emp_lower=0.25,
#             phi_price_upper=1.15, phi_price_lower=1.025, vartheta=0.02,
#             theta=0.75, ld(=lambda, production tech)=3, chi=0.1

def test_default_calibration_matches_paper_table_1():
    defaults = {
        k: v.default
        for k, v in inspect.signature(LegnickModel.__init__).parameters.items()
        if v.default is not inspect._empty
    }
    expected = dict(
        alpha=0.9, n=7, ld=3, chi=0.1, gamma=24, delta=0.019,
        phi_emp_upper=1, phi_emp_lower=0.25,
        phi_price_lower=1.025, phi_price_upper=1.15,
        vartheta=0.02, theta=0.75,
        psi_price=0.25, xi=0.01, psi_quant=0.25, beta=5, pie=0.1,
    )
    for key, value in expected.items():
        assert defaults[key] == pytest.approx(value), (
            f"{key}: default={defaults[key]!r}, paper Table 1 value={value!r}"
        )


# ---------------------------------------------------------------------------
# Reservation wage adjustment - Section 2.4, penultimate paragraph:
# "If the labor income exceeds a household's reservation wage, w_h is raised
# to the level of the received labor income. If the labor income is lower
# than w_h, the reservation wage is not changed... If a household has been
# unemployed during the last month, his reservation wage for the next month
# is reduced by 10 percent."
# ---------------------------------------------------------------------------

def test_income_above_reservation_wage_raises_it_to_income(model):
    h = Household(model)
    h.type_b_connection = Firm(model)  # employed
    h.w_h, h.income = 10.0, 15.0

    h.adjust_reservation_wage()

    assert h.w_h == 15.0


def test_income_below_reservation_wage_leaves_it_unchanged(model):
    h = Household(model)
    h.type_b_connection = Firm(model)  # employed
    h.w_h, h.income = 10.0, 5.0

    h.adjust_reservation_wage()

    assert h.w_h == 10.0


def test_unemployed_reservation_wage_falls_10_percent(model):
    h = Household(model)
    h.type_b_connection = None  # unemployed last month
    h.w_h = 10.0

    h.adjust_reservation_wage()

    assert h.w_h == pytest.approx(10.0 * 0.9)


# ---------------------------------------------------------------------------
# Unemployed job search - Section 2.2:
# "If the household is unemployed he visits a randomly chosen firm to check
# whether there is an open position. If the firm indeed offers an open
# position and pays a wage [...] the position is accepted [...] If the firm
# offers no vacancy or the wage it pays is too small, the search process is
# repeated until a total of beta firms have been visited."
#
# NOTE on an ambiguity in the paper: the acceptance wage is described as
# exceeding "the household's currently received wage" - for an unemployed
# household that phrase has no obvious value. Footnote 25, however, states
# a model invariant: "at the point in time when the household accepts a
# given job offer, the received wage rate is always above his reservation
# wage." That only holds in general if the acceptance threshold here is the
# reservation wage w_h, which is what agents.py compares against. The tests
# below encode that reading - worth you double-checking against your own
# reading of the paper since the literal wording is genuinely ambiguous.
# ---------------------------------------------------------------------------

def test_unemployed_accepts_open_position_meeting_reservation_wage(model, monkeypatch):
    h = Household(model)
    f = Firm(model)
    h.w_h, f.w_f, f.n_positions = 5.0, 5.0, 1  # open position, wage >= w_h

    monkeypatch.setattr(model.random, "sample", lambda pop, k: list(pop)[:k])
    h.job_search(beta=1, pie=0.1)

    assert h.type_b_connection is f
    assert h in f.workers


def test_unemployed_rejects_wage_below_reservation_wage(model, monkeypatch):
    h = Household(model)
    f = Firm(model)
    h.w_h, f.w_f, f.n_positions = 5.0, 4.0, 1  # wage too small

    monkeypatch.setattr(model.random, "sample", lambda pop, k: list(pop)[:k])
    h.job_search(beta=1, pie=0.1)

    assert h.type_b_connection is None


def test_unemployed_skips_firm_with_no_vacancy(model, monkeypatch):
    h = Household(model)
    f = Firm(model)
    h.w_h, f.w_f, f.n_positions = 5.0, 10.0, 0  # great wage, no vacancy

    monkeypatch.setattr(model.random, "sample", lambda pop, k: list(pop)[:k])
    h.job_search(beta=1, pie=0.1)

    assert h.type_b_connection is None


# ---------------------------------------------------------------------------
# Satisfied-employee search - Section 2.2:
# "employees who are satisfied with their job (wf >= w_h) show the least
# search effort [...] With a probability of pi < 1, they visit one randomly
# determined firm per month [...] The position is accepted if the offered
# wage payment exceeds that of their current position."
# ---------------------------------------------------------------------------

def test_satisfied_worker_switches_only_if_new_wage_beats_current(model, monkeypatch):
    h = Household(model)
    f_current, f_better = Firm(model), Firm(model)
    f_current.w_f, f_better.w_f, f_better.n_positions = 6.0, 8.0, 1
    h.type_b_connection = f_current
    f_current.workers.append(h)
    h.w_h = 5.0  # satisfied: f_current.w_f (6) >= w_h (5)

    monkeypatch.setattr(model.random, "random", lambda: 0.0)  # forces < pie
    monkeypatch.setattr(model.random, "choice", lambda seq: f_better)
    h.job_search(beta=1, pie=0.5)

    assert h.type_b_connection is f_better


def test_satisfied_worker_does_not_switch_to_a_worse_wage(model, monkeypatch):
    h = Household(model)
    f_current, f_worse = Firm(model), Firm(model)
    f_current.w_f, f_worse.w_f, f_worse.n_positions = 6.0, 5.0, 1
    h.type_b_connection = f_current
    f_current.workers.append(h)
    h.w_h = 5.0  # satisfied

    monkeypatch.setattr(model.random, "random", lambda: 0.0)
    monkeypatch.setattr(model.random, "choice", lambda seq: f_worse)
    h.job_search(beta=1, pie=0.5)

    assert h.type_b_connection is f_current


# ---------------------------------------------------------------------------
# Dissatisfied (unsatisfied) worker search - Section 2.2:
# "An employee who is unsatisfied (wf < w_h) shows higher search effort. He
# performs the same searching mechanism with a probability of 1."
# Per the footnote-25 invariant discussed above, the acceptance threshold
# here is w_h (the household's reservation wage), and the search happens
# unconditionally (no probability gate), unlike the satisfied branch.
# ---------------------------------------------------------------------------

def test_dissatisfied_worker_switches_to_firm_meeting_reservation_wage(model, monkeypatch):
    h = Household(model)
    f_low, f_high = Firm(model), Firm(model)
    f_low.w_f, f_high.w_f, f_high.n_positions = 3.0, 8.0, 1
    h.type_b_connection = f_low
    f_low.workers.append(h)
    h.w_h = 6.0  # f_low.w_f (3) < w_h (6) -> unsatisfied

    monkeypatch.setattr(model.random, "choice", lambda seq: f_high)
    h.job_search(beta=1, pie=0.1)

    assert h.type_b_connection is f_high
    assert h not in f_low.workers
    assert h in f_high.workers


def test_dissatisfied_worker_search_is_not_gated_by_a_probability(model, monkeypatch):
    """The paper says the unsatisfied search happens 'with a probability of
    1' - i.e. every month, unconditionally. Force model.random.random() to
    always return 1.0 (which would fail any '< pie'-style gate) and confirm
    the switch still happens, proving no probability check is applied."""
    h = Household(model)
    f_low, f_high = Firm(model), Firm(model)
    f_low.w_f, f_high.w_f, f_high.n_positions = 3.0, 8.0, 1
    h.type_b_connection = f_low
    f_low.workers.append(h)
    h.w_h = 6.0

    monkeypatch.setattr(model.random, "random", lambda: 1.0)
    monkeypatch.setattr(model.random, "choice", lambda seq: f_high)
    h.job_search(beta=1, pie=0.1)

    assert h.type_b_connection is f_high


def test_dissatisfied_worker_stays_if_alternative_firm_has_no_vacancy(model, monkeypatch):
    h = Household(model)
    f_low, f_full = Firm(model), Firm(model)
    f_low.w_f, f_full.w_f, f_full.n_positions = 3.0, 8.0, 0
    h.type_b_connection = f_low
    f_low.workers.append(h)
    h.w_h = 6.0

    monkeypatch.setattr(model.random, "choice", lambda seq: f_full)
    h.job_search(beta=1, pie=0.1)

    assert h.type_b_connection is f_low
    assert h in f_low.workers


# ---------------------------------------------------------------------------
# Wage adjustment - Section 2.2, Eq. (5) + surrounding text:
# "The firm increases wf if a free position was offered during the last
# month, but no worker was found to accept it. It is decreased if all
# positions have been filled with workers throughout the last gamma months."
# w_f_new := w_f_old * (1 +/- eps), eps ~ U[0, delta]
# Table 1: gamma=24, delta=0.019
# ---------------------------------------------------------------------------

def test_wage_rises_when_a_position_stays_unfilled(model, monkeypatch):
    f = Firm(model)
    f.n_positions, f.workers, f.w_f, f.months_full = 5, [], 3.0, 5  # vacancy exists

    monkeypatch.setattr(model.random, "uniform", lambda a, b: 0.1)
    f.set_wages(gamma=24, delta=0.019)

    assert f.w_f == pytest.approx(3.0 * 1.1)
    assert f.months_full == 0


def test_wage_falls_after_gamma_months_fully_staffed(model, monkeypatch):
    worker = object()
    f = Firm(model)
    f.n_positions, f.workers, f.months_full, f.w_f = 1, [worker], 25, 3.0  # > gamma=24

    monkeypatch.setattr(model.random, "uniform", lambda a, b: 0.1)
    f.set_wages(gamma=24, delta=0.019)

    assert f.w_f == pytest.approx(3.0 * 0.9)
    assert f.months_full == 0


def test_wage_unchanged_while_fully_staffed_but_under_gamma(model):
    worker = object()
    f = Firm(model)
    f.n_positions, f.workers, f.months_full, f.w_f = 1, [worker], 5, 3.0  # < gamma=24

    f.set_wages(gamma=24, delta=0.019)

    assert f.w_f == 3.0
    assert f.months_full == 6


# ---------------------------------------------------------------------------
# Price adjustment - Section 2.2, Eqs. (6)-(10):
# if_lower = phi_emp_lower * demand, if_upper = phi_emp_upper * demand
# p_upper = phi_price_upper * mc_f, p_lower = phi_price_lower * mc_f
# "If current inventories are below the critical lower bound, the firm
# considers to increase its price... In the opposite case of high
# inventories a decrease of pf is considered... Prices are raised as long as
# they are not exceeding p_upper and decreased as long as they are above
# p_lower... firms set the newly determined price only with probability
# theta."
# Table 1: phi_price_upper=1.15, phi_price_lower=1.025, vartheta=0.02, theta=0.75
# ---------------------------------------------------------------------------

def test_price_falls_when_inventory_above_upperbar(model, monkeypatch):
    f = Firm(model)
    f.demand, f.i_f, f.w_f, f.p_f = 100, 150, 6.0, 2.2  # upperbar=100, mc_f=2.0

    monkeypatch.setattr(model.random, "uniform", lambda a, b: 0.01)
    monkeypatch.setattr(model.random, "random", lambda: 0.0)  # forces < theta
    f.set_prices(phi_price_upper=1.15, phi_price_lower=1.025, ld=3,
                 theta=0.75, phi_emp_upper=1, vartheta=0.02, phi_emp_lower=0.25)

    assert f.p_f == pytest.approx(2.2 * 0.99)


def test_price_rises_when_inventory_below_lowerbar(model, monkeypatch):
    f = Firm(model)
    f.demand, f.i_f, f.w_f, f.p_f = 100, 10, 6.0, 2.2  # lowerbar=25

    monkeypatch.setattr(model.random, "uniform", lambda a, b: 0.01)
    monkeypatch.setattr(model.random, "random", lambda: 0.0)
    f.set_prices(phi_price_upper=1.15, phi_price_lower=1.025, ld=3,
                 theta=0.75, phi_emp_upper=1, vartheta=0.02, phi_emp_lower=0.25)

    assert f.p_f == pytest.approx(2.2 * 1.01)


def test_price_unchanged_when_inventory_within_band(model, monkeypatch):
    f = Firm(model)
    f.demand, f.i_f, f.w_f, f.p_f = 100, 50, 6.0, 2.2  # between 25 and 100

    monkeypatch.setattr(model.random, "uniform", lambda a, b: 0.01)
    monkeypatch.setattr(model.random, "random", lambda: 0.0)
    f.set_prices(phi_price_upper=1.15, phi_price_lower=1.025, ld=3,
                 theta=0.75, phi_emp_upper=1, vartheta=0.02, phi_emp_lower=0.25)

    assert f.p_f == 2.2


def test_price_change_is_gated_by_theta_probability(model, monkeypatch):
    """Even with high inventory (would-be decrease), if the theta draw
    fails (random() >= theta) the price must not move."""
    f = Firm(model)
    f.demand, f.i_f, f.w_f, f.p_f = 100, 150, 6.0, 2.2

    monkeypatch.setattr(model.random, "uniform", lambda a, b: 0.01)
    monkeypatch.setattr(model.random, "random", lambda: 0.99)  # fails < theta
    f.set_prices(phi_price_upper=1.15, phi_price_lower=1.025, ld=3,
                 theta=0.75, phi_emp_upper=1, vartheta=0.02, phi_emp_lower=0.25)

    assert f.p_f == 2.2


# ---------------------------------------------------------------------------
# Inventory thresholds / hiring & firing - Section 2.2, Eqs. (6)-(7):
# "If the inventory has fallen below if_lower a new open position is
# created [...] If, vice versa, inventories are above if_upper, a randomly
# chosen worker is fired [...] hiring decisions lead to an immediate
# offering of a new position, while firing decisions are implemented with a
# lag of one month." (Footnote 18: at most one worker hired/fired per firm
# per month.)
# ---------------------------------------------------------------------------

def test_hires_immediately_when_inventory_below_lowerbar(model):
    f = Firm(model)
    f.demand, f.i_f, f.n_positions = 100, 5, 3  # lowerbar = 25

    f.set_employment(phi_emp_upper=1, phi_emp_lower=0.25)

    assert f.n_positions == 4  # one new position, immediately


def test_marks_one_worker_to_fire_when_inventory_above_upperbar(model):
    worker = object()
    f = Firm(model)
    f.demand, f.i_f, f.n_positions, f.workers = 100, 150, 3, [worker]  # upperbar = 100

    f.set_employment(phi_emp_upper=1, phi_emp_lower=0.25)

    assert f.to_fire == [worker]     # queued, not yet removed (one-month lag)
    assert worker in f.workers       # still employed this month
    assert f.n_positions == 2


def test_firing_takes_effect_only_after_one_month_lag(model):
    h = Household(model)
    f = Firm(model)
    h.type_b_connection = f
    f.demand, f.i_f, f.n_positions, f.workers = 100, 150, 3, [h]

    f.set_employment(phi_emp_upper=1, phi_emp_lower=0.25)
    assert h in f.workers  # not fired yet this month

    f.fire_workers()  # next month's processing
    assert h not in f.workers
    assert h.type_b_connection is None


def test_no_hire_or_fire_when_inventory_within_band(model):
    worker = object()
    f = Firm(model)
    f.demand, f.i_f, f.n_positions, f.workers = 100, 50, 3, [worker]

    f.set_employment(phi_emp_upper=1, phi_emp_lower=0.25)

    assert f.n_positions == 3
    assert f.to_fire == []


# ---------------------------------------------------------------------------
# Profit distribution - Section 2.4:
# "all remaining liquidity of the firm is distributed as profit among all
# households... each household receives a share of aggregate profits that
# is proportional to his current liquidity." (Footnote 22: if h_A owns
# twice as much money as h_B, A's profit share is twice as large as B's.)
#
# Note: "aggregate profits" (singular pool) implies profits from ALL firms
# are pooled before distribution - which is what distribute_all_profits
# does (as opposed to the old, now-unused Firm.distribute_profits, which
# distributed each firm's profit separately/sequentially). These tests
# target distribute_all_profits, since that's what model.py actually calls.
# ---------------------------------------------------------------------------

def test_profits_pooled_across_firms_then_split_by_liquidity_share(model):
    f1, f2 = Firm(model), Firm(model)
    f1.m_f, f2.m_f = 10.0, 30.0  # aggregate profit = 40
    h1, h2 = Household(model), Household(model)
    h1.m_h, h2.m_h = 100.0, 300.0  # h2 has 3x h1's liquidity

    model.firms = [f1, f2]
    model.Households = [h1, h2]
    distribute_all_profits(model)

    assert h1.m_h == pytest.approx(100.0 + 40 * 100 / 400)
    assert h2.m_h == pytest.approx(300.0 + 40 * 300 / 400)
    # footnote 22 check: h2's share is exactly 3x h1's share
    h1_share = h1.m_h - 100.0
    h2_share = h2.m_h - 300.0
    assert h2_share == pytest.approx(3 * h1_share)
    assert f1.m_f == 0.0 and f2.m_f == 0.0


def test_no_distribution_when_aggregate_profit_is_zero_or_negative(model):
    f = Firm(model)
    f.m_f = 0.0
    h = Household(model)
    h.m_h = 100.0

    model.firms = [f]
    model.Households = [h]
    distribute_all_profits(model)

    assert h.m_h == 100.0
