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