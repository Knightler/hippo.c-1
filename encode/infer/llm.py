import json
import os
import urllib.request


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
            return json.loads(content)
        except Exception:
            return []


def _system_prompt() -> str:
    return (
        "Extract user memory facts from the message. "
        "Return JSON array with fields: content, category, label. "
        "Only include meaningful, compact facts."
    )


def _user_prompt(text: str, context: list[dict]) -> str:
    return json.dumps({"text": text, "candidates": context})
