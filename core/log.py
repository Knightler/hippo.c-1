import json
import os
from datetime import datetime, timezone


_SENSITIVE_KEY_PARTS = (
    "authorization",
    "api_key",
    "apikey",
    "token",
    "password",
    "secret",
    "dsn",
)


def _sanitize(value: object) -> object:
    if isinstance(value, dict):
        masked: dict[object, object] = {}
        for key, item in value.items():
            if isinstance(key, str) and any(part in key.lower() for part in _SENSITIVE_KEY_PARTS):
                masked[key] = "***redacted***"
            else:
                masked[key] = _sanitize(item)
        return masked
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize(item) for item in value)
    if isinstance(value, str):
        lowered = value.lower()
        if "bearer " in lowered:
            return "***redacted***"
    return value


def log(level: str, event: str, **fields: object) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event": event,
        **_sanitize(fields),
    }
    line = json.dumps(payload, separators=(",", ":"))
    print(line)
    log_file = os.getenv("HIPPO_LOG_FILE", ".hippo.log")
    if log_file and log_file.lower() not in {"off", "none", "0"}:
        with open(log_file, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
