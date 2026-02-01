import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Pattern:
    regex: str
    category: str
    template: str
    weight: float = 0.5
    uses: int = 0
    successes: int = 0
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def record(self, success: bool) -> None:
        self.uses += 1
        if success:
            self.successes += 1
        self.updated_at = datetime.utcnow()

    def compiled(self) -> re.Pattern:
        return re.compile(self.regex, re.IGNORECASE)


class PatternLibrary:
    """Persistent pattern storage for cheap extraction."""

    def __init__(self, path: str = "ingest/triage/patterns.json"):
        self.path = path
        self.patterns: list[Pattern] = []
        self._ensure_dir()
        self.load()
        if not self.patterns:
            self._load_defaults()

    def _ensure_dir(self) -> None:
        directory = os.path.dirname(self.path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def _load_defaults(self) -> None:
        self.patterns = [
            Pattern(r"i (?:really )?(?:like|love|enjoy|prefer) (.+?)(?:\.|$)", "preference", "likes {0}", 0.6),
            Pattern(r"i (?:really )?(?:hate|dislike|detest|can't stand) (.+?)(?:\.|$)", "preference", "dislikes {0}", 0.6),
            Pattern(r"i (?:want|plan|aim) to (.+?)(?:\.|$)", "goal", "wants to {0}", 0.55),
            Pattern(r"i (?:feel|am feeling|i'm feeling) (.+?)(?:\.|$)", "emotion", "feels {0}", 0.5),
            Pattern(r"my (.+?) (?:is|are) (.+?)(?:\.|$)", "relationship", "{0} is {1}", 0.55),
            Pattern(r"i (?:live in|am from|stay in) (.+?)(?:\.|$)", "location", "lives in {0}", 0.6),
        ]

    def match(self, text: str) -> list[dict]:
        results = []
        for pattern in self.patterns:
            for match in pattern.compiled().finditer(text):
                content = self._format(pattern, match)
                if content:
                    results.append(
                        {
                            "content": content,
                            "category": pattern.category,
                            "confidence": pattern.weight,
                            "pattern": pattern.regex,
                        }
                    )
        return results

    def record(self, regex: str, success: bool) -> None:
        for pattern in self.patterns:
            if pattern.regex == regex:
                pattern.record(success)
                break
        self.save()

    def save(self) -> None:
        data = {
            "patterns": [self._to_dict(p) for p in self.patterns],
            "updated_at": datetime.utcnow().isoformat(),
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r") as f:
            data = json.load(f)
        self.patterns = [self._from_dict(p) for p in data.get("patterns", [])]

    def _format(self, pattern: Pattern, match: re.Match) -> str | None:
        groups = match.groups()
        try:
            content = pattern.template.format(*groups)
        except IndexError:
            return None
        return content.strip() if content.strip() else None

    def _to_dict(self, pattern: Pattern) -> dict:
        return {
            "regex": pattern.regex,
            "category": pattern.category,
            "template": pattern.template,
            "weight": pattern.weight,
            "uses": pattern.uses,
            "successes": pattern.successes,
            "updated_at": pattern.updated_at.isoformat(),
        }

    def _from_dict(self, raw: dict) -> Pattern:
        pattern = Pattern(
            regex=raw["regex"],
            category=raw["category"],
            template=raw["template"],
            weight=raw.get("weight", 0.5),
            uses=raw.get("uses", 0),
            successes=raw.get("successes", 0),
        )
        if "updated_at" in raw:
            try:
                pattern.updated_at = datetime.fromisoformat(raw["updated_at"])
            except ValueError:
                pass
        return pattern
