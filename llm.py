import os

def generate_answer(prompt: str) -> str | None:
    """
    Lightweight stub. Returns None unless explicitly enabled.
    To enable later, add your API code here and set USE_LLM=1.
    """
    if os.getenv("USE_LLM") != "1":
        return None
    # No external calls on Render free tier — return None gracefully.
    return None
