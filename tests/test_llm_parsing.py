"""
Unit tests for src/llm/parsing.py: parse_llm_response and validate_decision.

parse_llm_response returns (new_value, reasoning, ok, action). On any failure it
falls back to the current value with ok=False; action is None only when the
response could not be parsed at all.
"""
import json

from src.llm.parsing import parse_llm_response


def wage_reply(action, new_wage, reasoning="x"):
    return json.dumps({"action": action, "new_wage": new_wage, "reasoning": reasoning})


def price_reply(action, new_price, reasoning="x"):
    return json.dumps({"action": action, "new_price": new_price, "reasoning": reasoning})


# --- reading the right field -------------------------------------------------

def test_wage_field_is_read():
    assert parse_llm_response(wage_reply("raise", 1450), 1428, field="new_wage") == (1450.0, "x", True, "raise")


def test_price_field_is_read():
    assert parse_llm_response(price_reply("lower", 24), 25, field="new_price") == (24.0, "x", True, "lower")


def test_wrong_field_falls_back_cleanly():
    """A price reply parsed as a wage has no new_wage key: keep the wage, flag it, no action."""
    value, reasoning, ok, action = parse_llm_response(price_reply("raise", 26), 1428, field="new_wage")
    assert (value, ok, action) == (1428, False, None)
    assert reasoning.startswith("PARSE_FAILURE")


def test_action_is_lowercased():
    assert parse_llm_response(wage_reply("RAISE", 1450), 1428, field="new_wage")[3] == "raise"


# --- failures ------------------------------------------------------------------

def test_malformed_json_falls_back():
    value, reasoning, ok, action = parse_llm_response("not json", 1428, field="new_wage")
    assert (value, ok, action) == (1428, False, None)
    assert reasoning.startswith("PARSE_FAILURE")


def test_non_numeric_value_falls_back():
    value, _, ok, action = parse_llm_response(wage_reply("raise", "lots"), 1428, field="new_wage")
    assert (value, ok, action) == (1428, False, None)


def test_non_positive_value_is_rejected():
    value, reasoning, ok, action = parse_llm_response(wage_reply("lower", 0), 1428, field="new_wage")
    assert (value, ok, action) == (1428, False, "lower")
    assert reasoning.startswith("REJECTED (non-positive")


# --- hold --------------------------------------------------------------------------

def test_hold_keeps_current_value_even_if_number_differs():
    assert parse_llm_response(wage_reply("hold", 9999), 1428, field="new_wage") == (1428, "x", True, "hold")


# --- jump limit ----------------------------------------------------------------

def test_jump_beyond_max_change_is_rejected():
    value, reasoning, ok, action = parse_llm_response(
        wage_reply("raise", 1600), 1428, field="new_wage", max_change_frac=0.1)
    assert (value, ok, action) == (1428, False, "raise")
    assert reasoning.startswith("REJECTED (implausible jump")


def test_jump_within_max_change_is_accepted():
    value, _, ok, _ = parse_llm_response(wage_reply("raise", 1500), 1428, field="new_wage", max_change_frac=0.1)
    assert (value, ok) == (1500.0, True)


def test_jump_limit_is_inclusive_at_the_edge():
    value, _, ok, _ = parse_llm_response(wage_reply("raise", 150), 100, field="new_wage", max_change_frac=0.5)
    assert (value, ok) == (150.0, True)


# --- bands (prices only) -----------------------------------------------------------

def test_no_bands_never_block_a_cut():
    value, _, ok, _ = parse_llm_response(wage_reply("lower", 10), 15, field="new_wage")
    assert (value, ok) == (10.0, True)


def test_cut_below_floor_already_reached_is_held():
    value, reasoning, ok, action = parse_llm_response(
        price_reply("lower", 19), 20, field="new_price", lowerbar=21, upperbar=30)
    assert (value, ok, action) == (20, True, "lower")
    assert "blocked further cut below floor" in reasoning


def test_rise_above_ceiling_already_reached_is_held():
    value, reasoning, ok, _ = parse_llm_response(
        price_reply("raise", 32), 31, field="new_price", lowerbar=21, upperbar=30)
    assert (value, ok) == (31, True)
    assert "blocked further rise above ceiling" in reasoning


def test_move_back_toward_band_is_allowed():
    """Price above the ceiling may still be cut: only moves further out are blocked."""
    value, _, ok, _ = parse_llm_response(price_reply("lower", 29), 31, field="new_price", lowerbar=21, upperbar=30)
    assert (value, ok) == (29.0, True)


# --- known behaviour ---------------------------------------------------------------

def test_direction_mismatch_passes_through():
    """
    Known and deliberate (for now): the parser does not check that the number
    moves the way the action says. llama3.2:3b sometimes says "raise" and gives
    a lower number. The stated action is logged so mismatches can be counted.
    If a direction guard is added later, update this test.
    """
    assert parse_llm_response(wage_reply("raise", 1424), 1428, field="new_wage") == (1424.0, "x", True, "raise")
