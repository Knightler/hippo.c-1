import re
from dataclasses import dataclass

from memory import MemoryClient


@dataclass
class Pattern:
    regex: str
    category: str
    template: str
    weight: float = 0.5
    label_mode: str = "group"

    def compiled(self) -> re.Pattern:
        return re.compile(self.regex, re.IGNORECASE)


class PatternLibrary:
    """Pattern registry backed by memory DB."""

    def __init__(self, memory: MemoryClient):
        self.memory = memory
        self.patterns: list[Pattern] = []
        self._load()
        if not self.patterns:
            self._seed_defaults()

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
                            "groups": list(match.groups()),
                            "label_mode": pattern.label_mode,
                        }
                    )
        return results

    def record(self, regex: str, success: bool) -> None:
        self.memory.record_pattern_use(regex, success)

    def _load(self) -> None:
        rows = self.memory.get_patterns()
        self.patterns = [
            Pattern(r["regex"], r["category"], r["template"], r["weight"])
            for r in rows
        ]

    def _seed_defaults(self) -> None:
        defaults = [
            Pattern(r"i (?:really )?(?:like|love|loved|enjoy|prefer) (.+?)(?:\.|$)", "preference", "likes {0}", 0.6, "object"),
            Pattern(r"i (?:really )?(?:hate|hated|dislike|detest|can't stand) (.+?)(?:\.|$)", "preference", "dislikes {0}", 0.6, "object"),
            Pattern(r"i (?:want|plan|aim) to (.+?)(?:\.|$)", "goal", "wants to {0}", 0.55, "object"),
            Pattern(r"i (?:feel|am feeling|i'm feeling) (.+?)(?:\.|$)", "emotion", "feels {0}", 0.5, "fixed"),
            Pattern(r"my (.+?) (?:is|are) (.+?)(?:\.|$)", "relationship", "{0} is {1}", 0.55, "subject"),
            Pattern(r"i (?:live in|am from|stay in) (.+?)(?:\.|$)", "location", "lives in {0}", 0.6, "object"),
            Pattern(r"i am (\d{1,3}) (?:years? old)?(?:\.|$)", "identity", "age is {0}", 0.55, "fixed"),
            Pattern(r"i am (?:a|an) (.+?)(?:\.|$)", "identity", "is {0}", 0.5, "fixed"),
            Pattern(r"i usually (.+?)(?:\.|$)", "habit", "usually {0}", 0.45, "fixed"),
            Pattern(r"i (?:need|have) to (.+?)(?:\.|$)", "goal", "needs to {0}", 0.5, "fixed"),
            Pattern(r"i (?:prefer) (.+?)(?:\.|$)", "preference", "prefers {0}", 0.6, "object"),
            Pattern(r"i (?:write|work) in (.+?)(?:\.|$)", "style", "writes in {0}", 0.5, "object"),
            Pattern(r"i like when (.+?)(?:\.|$)", "preference", "likes when {0}", 0.45, "object"),
        ]
        for p in defaults:
            self.memory.upsert_pattern(p.regex, p.category, p.template, p.weight)
        self.patterns = defaults

    def _format(self, pattern: Pattern, match: re.Match) -> str | None:
        groups = match.groups()
        try:
            content = pattern.template.format(*groups)
        except IndexError:
            return None
        return content.strip() if content.strip() else None
