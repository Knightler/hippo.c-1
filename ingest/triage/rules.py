import re
from dataclasses import dataclass

from ingest.models import Prompt
from ingest.triage.patterns import PatternLibrary


@dataclass
class TriageResult:
    score: float
    terms: list[str]
    should_infer: bool
    reason: str


SIGNAL_KEYWORDS = {
    "like",
    "love",
    "hate",
    "prefer",
    "feel",
    "want",
    "plan",
    "believe",
    "think",
    "my",
    "relationship",
}


def triage(prompt: Prompt, patterns: PatternLibrary, threshold: float = 0.55) -> TriageResult:
    text = prompt.text.lower()
    tokens = _tokenize(text)
    keyword_hits = sum(1 for t in tokens if t in SIGNAL_KEYWORDS)
    pattern_hits = len(patterns.match(prompt.text))

    score = min(1.0, (keyword_hits * 0.15) + (pattern_hits * 0.2))
    terms = _extract_terms(prompt.text)
    should_infer = score >= threshold
    reason = "signal" if should_infer else "low_signal"

    return TriageResult(score=score, terms=terms, should_infer=should_infer, reason=reason)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text)


def _extract_terms(text: str) -> list[str]:
    tokens = _tokenize(text)
    terms = [t for t in tokens if len(t) > 3]
    return list(dict.fromkeys(terms))[:8]
