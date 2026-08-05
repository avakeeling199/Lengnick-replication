import requests
import time

from src.llm.prompts import build_price_prompt

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.3:70b"

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
    response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=600)
    if response.status_code != 200:
        raise RuntimeError(response.text)
    raw_text = response.json()["message"]["content"].strip()
    return raw_text, time.time() - start