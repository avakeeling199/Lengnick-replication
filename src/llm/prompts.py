def build_price_prompt(i_f, p_f, mc_f, demand, price_history=None, demand_history=None):
    history_section = ""
    if price_history or demand_history:
        price_history = price_history or []
        demand_history = demand_history or []
        history_section = (
            "\nRecent history (oldest to most recent, before this month):\n"
            f"- Prices: {list(price_history)}\n"
            f"- Realised demand: {list(demand_history)}\n"
            "Check whether your recent price changes actually grew demand or "
            "shrank it -- if demand has been falling as you raised price, that's "
            "a sign the increase isn't sustainable and you may want to hold or "
            "lower instead.\n"
        )

    return (
        "You are the pricing manager for a firm in a simple closed economy. "
        "Each month you decide whether to raise, lower, or hold your goods "
        "price, based on your inventory levels relative to recent demand.\n\n"
        f"Current state:\n"
        f"- Inventory: {i_f}\n"
        f"- Current price: {p_f}\n"
        f"- Marginal cost this month: {mc_f}\n"
        f"- Realised demand last month: {demand}\n"
        f"{history_section}\n"
        "Decide your price for next month. Respond ONLY with JSON in this "
        "exact schema:\n"
        '{"new_price": <float>, "reasoning": "<one_sentence>"}'
    )