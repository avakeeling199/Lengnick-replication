import json 

def parse_price_response(raw_text, current_price, max_change_frac=0.5,
                          p_f_lowerbar=None, p_f_upperbar=None):
    """
    Parse and validate an LLm price response.

    Falls back to 'current_price' (no change) on any failure: malformed JSON
    missing keys, wrong types, non-positive prices, or implausible jumps.
    If p_f_lowerbar/p_f_upperbar are given, a move is only blocked when it
    would push price further past a boundary the current price has already
    reached (mirrors the directional gate the rule-based pricer enforces:
    only cut further once price is still >= the floor, only raise further
    once price is still <= the ceiling). This still stops runaway drift far
    from cost, but -- unlike a hard clamp -- lets price overshoot the band
    and later correct, instead of pinning it to the edge every month.
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

    if p_f_lowerbar is not None and new_price < current_price and current_price <= p_f_lowerbar:
        reasoning = f"{reasoning} [blocked further cut below floor {p_f_lowerbar:.2f}; held at {current_price:.2f}]"
        new_price = current_price
    elif p_f_upperbar is not None and new_price > current_price and current_price >= p_f_upperbar:
        reasoning = f"{reasoning} [blocked further rise above ceiling {p_f_upperbar:.2f}; held at {current_price:.2f}]"
        new_price = current_price

    return new_price, reasoning, True