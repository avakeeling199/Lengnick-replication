import requests
import json
import time

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b" 

def build_price_prompt(i_f, p_f, mc_f, demand):
    return f"""You are the pricing manager for a firm in a simple closed economy. Each month you decide whether to raise, lower, or hold your goods price, based on your inventory levels relative to recent demand.
    
    Current state:
    - Inventory: {i_f}
    - Current price: {p_f}
    - Marginal cost this month: {mc_f}
    - Realised demand last month: {demand}

    Decide your price for next month. Respond ONLY with JSON in this exact schema:
    {{"new_price": <float>, "reasoning": "<one_sentence>"}}"""

def call_ollama_price(i_f, p_f, mc_f, demand):
    prompt = build_price_prompt(i_f, p_f, mc_f, demand)
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": {
            "type": "object",
            "properties": {
                "new_price": {"type": "number"},
                "reasoning": {"type": "string"}
            },
            "required": ["new_price", "reasoning"]
        },
        "options": {"temperature": 0.7},
    }
    start = time.time()
    response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(response.text)
    raw_text = response.json()["message"]["content"].strip()
    return raw_text, time.time() - start


# test with plausible fake state 
raw_text, elapsed = call_ollama_price(i_f = 500, p_f = 25.0, mc_f = 20.0, demand=100)
print("Raw response:", raw_text)
print("elapsed:", elapsed, "seconds")

try:
    parsed = json.loads(raw_text)
    print("Parsed new_price:", parsed.get("new_price"))
    print("Reasoning:", parsed.get("reasoning"))
except json.JSONDecodeError:
    print("FAILED TO PARSE - got:", raw_text)