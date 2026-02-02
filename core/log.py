import json
import os
from datetime import datetime, timezone


def log(level: str, event: str, **fields: object) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event": event,
        **fields,
    }
    line = json.dumps(payload, separators=(",", ":"))
    print(line)
    log_file = os.getenv("HIPPO_LOG_FILE", "")
    if log_file:
        with open(log_file, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
