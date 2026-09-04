def build_price_prompt(i_f, p_f, mc_f, demand, price_history=None, demand_history=None, inventory_history=None):
    history_section = ""
    if price_history or demand_history or inventory_history:
        price_history = price_history or []
        demand_history = demand_history or []
        inventory_history = inventory_history or []
        history_section = (
            "\nRecent history (oldest to most recent, before this month):\n"
            f"- Prices: {list(price_history)}\n"
            f"- Realised demand: {list(demand_history)}\n"
            f"- Inventory: {list(inventory_history)}\n"
            "Use the trend in your price, demand, and inventory together to "
            "judge whether your current price is too high, too low, or about "
            "right.\n"
        )

    return (
        "You are the pricing manager for a firm in a simple closed economy. "
        "Each month you decide whether to raise, lower, or hold your goods "
        "price, based on your inventory levels relative to recent demand.\n\n"
        "How to read inventory: inventory is UNSOLD stock sitting in your "
        "warehouse. High or rising inventory relative to demand means goods "
        "are NOT selling fast enough -- that is a signal to LOWER your "
        "price, not raise it. Low or falling inventory relative to demand "
        "means customers are buying faster than you can restock -- that is "
        "a signal to RAISE your price. Do not confuse rising inventory with "
        "rising demand; they point in opposite directions for pricing.\n\n"
        "Changing your price is not free in practice: relabeling and "
        "catalogue updates take effort, and customers who see your price "
        "move around too much lose trust and shop elsewhere. Only raise or "
        "lower when the evidence clearly justifies it. If the trend in "
        "demand and inventory is mixed or inconclusive, hold your price.\n\n"
        "Size the change to match the evidence: a small, borderline "
        "mismatch between demand and inventory calls for a small price "
        "adjustment; a large, clear-cut mismatch justifies a larger one. "
        "Don't default to the same fixed step every time -- think about "
        "how big a change the current situation actually warrants, as a "
        "proportion of your current price.\n\n"
        f"Current state:\n"
        f"- Inventory: {i_f}\n"
        f"- Current price: {p_f}\n"
        f"- Marginal cost this month: {mc_f}\n"
        f"- Realised demand last month: {demand}\n"
        f"{history_section}\n"
        "Decide your action for next month. Respond ONLY with JSON in this "
        "exact schema:\n"
        '{"action": "hold" | "raise" | "lower", "new_price": <float>, '
        '"reasoning": "<one_sentence>"}\n'
        'If action is "hold", set new_price equal to the current price.'
    )