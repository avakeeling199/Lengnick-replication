import requests
import time

from src.llm.prompts import build_price_prompt

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.3:70b"

def call_ollama_price(i_f, p_f, mc_f, demand, price_history=None, demand_history=None, inventory_history=None):
    prompt = build_price_prompt(i_f, p_f, mc_f, demand, price_history, demand_history, inventory_history)
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["hold", "raise", "lower"]},
                "new_price": {"type": "number"},
                "reasoning": {"type": "string"}
            },
            "required": ["action", "new_price", "reasoning"]
        },
        "options": {"temperature": 0.3},
    }
    start = time.time()
    response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=600)
    if response.status_code != 200:
        raise RuntimeError(response.text)
    raw_text = response.json()["message"]["content"].strip()
    return raw_text, time.time() - start