# logger.py
from datetime import datetime
import json, os

LOG_DIR = "data"
LOG_FILE = os.path.join(LOG_DIR, "chat_logs.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

def log_event(ev: dict):
    ev = {**ev, "ts": datetime.utcnow().isoformat()}
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")
