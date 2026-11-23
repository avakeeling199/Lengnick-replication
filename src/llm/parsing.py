import json 

def parse_llm_response(raw_text, current_value, field, max_change_frac=0.5, lowerbar=None, upperbar=None):
    """
    Parse and validate an LLM response
    """
    try:
        parsed = json.loads(raw_text)
        action = str(parsed.get("action", "")).lower()
        new_value = float(parsed[field])
        reasoning = str(parsed.get("reasoning", ""))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return current_value, f"PARSE_FAILURE: {raw_text!r}", False

    return validate_decision(action, new_value, reasoning, current_value, max_change_frac, 
                        lowerbar, upperbar)

def validate_decision(action, new_value, reasoning, current_value, max_change_frac=0.5,
                    lowerbar=None, upperbar=None):
    """ 
    applies the pricing guards to an already parsed decision and returns (new_value, reasoning, ok)
    """
    if action == "hold":
        return current_value, reasoning, True

    if new_value <= 0:
        return current_value, f"REJECTED (non-positive value): {new_value}", False

    lower = current_value * (1 - max_change_frac)
    upper = current_value * (1 + max_change_frac)
    if not (lower <= new_value <= upper):
        return current_value, f"REJECTED (implausible jump to {new_value}): {reasoning}", False

    if lowerbar is not None and new_value < current_value and current_value <= lowerbar:
        reasoning = f"{reasoning} [blocked further cut below floor {lowerbar:.2f}; held at {current_value:.2f}]"
        new_value = current_value
    elif upperbar is not None and new_value > current_value and current_value >= upperbar:
        reasoning = f"{reasoning} [blocked further rise above ceiling {upperbar:.2f}; held at {current_value:.2f}]"
        new_value = current_value

    return new_value, reasoning, True
