# llm.py
import os, json, requests
from typing import Optional

# Env flags:
# USE_LLM=1 enables the LLM step
#   EITHER:
#     USE_OLLAMA=1 and OLLAMA_MODEL=llama3 (and OLLAMA_HOST=http://localhost:11434 optional)
#   OR:
#     OPENAI_API_KEY=<key> and OPENAI_MODEL=gpt-4o-mini (or similar)



def _openai_generate(prompt: str) -> Optional[str]:
    # Minimal dependency; requires `openai` package and env OPENAI_API_KEY
    try:
        import openai
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if not api_key:
            return None
        openai.api_key = api_key
        # Chat Completions
        resp = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None

def generate_answer(prompt: str) -> Optional[str]:
    if os.getenv("USE_OLLAMA") == "1":
        text = _ollama_generate(prompt)
        if text: return text
    # Try OpenAI next
    if os.getenv("OPENAI_API_KEY"):
        text = _openai_generate(prompt)
        if text: return text
    return None
