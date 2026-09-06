def build_price_prompt(i_f, p_f, mc_f, demand, price_history=None, demand_history=None,
                        inventory_history=None, i_f_lowerbar=None, i_f_upperbar=None,
                        p_f_lowerbar=None, p_f_upperbar=None):
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
            "Check whether your last price change actually worked: did "
            "inventory move back toward its target band afterwards? If it "
            "did and you are still outside the band, a similar move may "
            "still be justified. If inventory is now back within its band, "
            "that is evidence to hold or reverse -- do not keep moving the "
            "price in the same direction just because it has been moving "
            "that way recently.\n"
        )

    bands_section = ""
    if i_f_lowerbar is not None and i_f_upperbar is not None:
        bands_section += (
            f"- Inventory target band: {i_f_lowerbar:.1f} to {i_f_upperbar:.1f}. "
            "Below this band you are understocked (a signal to raise price); "
            "above it you are overstocked (a signal to lower price); inside "
            "it, inventory alone does not justify a change.\n"
        )
    if p_f_lowerbar is not None and p_f_upperbar is not None:
        bands_section += (
            f"- Price band relative to marginal cost: {p_f_lowerbar:.2f} to "
            f"{p_f_upperbar:.2f}. Any price you set outside this band will "
            "be clamped back into it automatically, so there is no benefit "
            "to proposing a price beyond it.\n"
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
        "How to read marginal cost: this is what it costs you to produce "
        "one more unit this month. Selling at or below marginal cost means "
        "you lose money on every unit you sell, so it is not a sustainable "
        "response to weak demand or a glut of inventory -- if a price cut "
        "would take you to or below marginal cost, that is a sign the cut "
        "is too large, not a reason to make it anyway. Weigh this against "
        "the inventory and demand evidence rather than treating it as a "
        "hard rule.\n\n"
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
        f"{bands_section}"
        f"{history_section}\n"
        "Decide your action for next month. Respond ONLY with JSON in this "
        "exact schema:\n"
        '{"action": "hold" | "raise" | "lower", "new_price": <float>, '
        '"reasoning": "<one_sentence>"}\n'
        'If action is "hold", set new_price equal to the current price.'
    )