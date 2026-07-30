def build_price_prompt(i_f, p_f, mc_f, demand):
    return (
        "You are the pricing manager for a firm in a simple closed economy. "
        "Each month you decide whether to raise, lower, or hold your goods "
        "price, based on your inventory levels relative to recent demand.\n\n"
        f"Current state:\n"
        f"- Inventory: {i_f}\n"
        f"- Current price: {p_f}\n"
        f"- Marginal cost this month: {mc_f}\n"
        f"- Realised demand last month: {demand}\n\n"
        "Decide your price for next month. Respond ONLY with JSON in this "
        "exact schema:\n"
        '{"new_price": <float>, "reasoning": "<one_sentence>"}'
    )