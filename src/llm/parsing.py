import json 

def parse_price_response(raw_text, current_price, max_change_frac=0.5,
                          p_f_lowerbar=None, p_f_upperbar=None):
    """
    Parse and validate an LLm price response.

    Falls back to 'current_price' (no change) on any failure: malformed JSON
    missing keys, wrong types, non-positive prices, or implausible jumps.
    If p_f_lowerbar/p_f_upperbar are given, the resulting price is clamped
    into that band (mirrors the hard marginal-cost band the rule-based
    pricer enforces) so the LLM can't walk price arbitrarily far from cost
    over many consecutive months.
    Returns (new_price, reasoning, ok) so callers can log what happened
    """
    try:
        parsed = json.loads(raw_text)
        action = str(parsed.get("action", "")).lower()
        new_price = float(parsed["new_price"])
        reasoning = str(parsed.get("reasoning", ""))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return current_price, f"PARSE_FAILURE: {raw_text!r}", False

    if action == "hold":
        return current_price, reasoning, True

    if new_price <= 0:
        return current_price, f"REJECTED (non-positive price): {new_price}", False

    lower = current_price * (1 - max_change_frac)
    upper = current_price * (1 + max_change_frac)
    if not (lower <= new_price <= upper):
        return current_price, f"REJECTED (implausible jump to {new_price}): {reasoning}", False

    if p_f_lowerbar is not None and new_price < p_f_lowerbar:
        reasoning = f"{reasoning} [clamped {new_price:.2f} -> floor {p_f_lowerbar:.2f}]"
        new_price = p_f_lowerbar
    elif p_f_upperbar is not None and new_price > p_f_upperbar:
        reasoning = f"{reasoning} [clamped {new_price:.2f} -> ceiling {p_f_upperbar:.2f}]"
        new_price = p_f_upperbar

    return new_price, reasoning, True