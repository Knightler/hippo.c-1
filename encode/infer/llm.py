import json
import os
import urllib.request

from core import log


class LLMExtractor:
    """Optional LLM extraction using an OpenAI-compatible API.

    If API key is missing, extractor is disabled and returns empty.
    """

    def __init__(
        self,
        api_key_env: str = "DEEPSEEK_API_KEY",
        base_env: str = "DEEPSEEK_API_BASE",
        model_env: str = "DEEPSEEK_MODEL",
    ):
        self.api_key = os.getenv(api_key_env, "")
        self.base = os.getenv(base_env, "https://api.deepseek.com/v1")
        self.model = os.getenv(model_env, "deepseek-chat")

    def enabled(self) -> bool:
        return bool(self.api_key)

    def extract(self, text: str, context: list[dict]) -> list[dict]:
        if not self.enabled():
            return []
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": _user_prompt(text, context)},
            ],
            "temperature": 0.1,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            return _validate_items(_safe_json_loads(content))
        except Exception as exc:
            log("error", "llm_extract_failed", error=type(exc).__name__, message=str(exc))
            return []


def _system_prompt() -> str:
    return (
        "Extract user memory facts from the message. "
        "Return JSON array with fields: content, category, label. "
        "Only include meaningful, compact facts."
    )


def _user_prompt(text: str, context: list[dict]) -> str:
    return json.dumps({"text": text, "candidates": context})


def _safe_json_loads(raw: str) -> object:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        log("error", "llm_parse_failed", error=type(exc).__name__, message=str(exc))
        return []


def _validate_items(data: object) -> list[dict]:
    if not isinstance(data, list):
        log("error", "llm_invalid_payload", reason="not_list")
        return []
    valid: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        category = item.get("category")
        label = item.get("label")
        if isinstance(content, str) and isinstance(category, str) and isinstance(label, str):
            valid.append(item)
    if not valid and data:
        log("error", "llm_invalid_payload", reason="no_valid_items")
    return valid
