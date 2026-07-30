import json 

def parse_price_response(raw_text, current_price, max_change_frac=0.5):
    """
    Parse and validate an LLm price response.

    Falls back to 'current_price' (no change) on any failure: malformed JSON
    missing keys, wrong types, non-positive prices, or implausible jumps.
    Returns (new_price, reasoning, ok) so callers can log what happened
    """
    try: 
        parsed = json.loads(raw_text)
        new_price = float(parsed["new_price"])
        reasoning = str(parsed.get("reasoning", ""))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return current_price, f"PARSE_FAILURE: {raw_text!r}", False

    if new_price <= 0:
        return current_price, f"REJECTED (non-positive price): {new_price}", False

    lower = current_price * (1 - max_change_frac)
    upper = current_price * (1 + max_change_frac)
    if not (lower <= new_price <= upper):
        return current_price, f"REJECTED (implausible jump to {new_price}): {reasoning}", False

    return new_price, reasoning, True