from typing import Iterable

from encode.infer import LLMExtractor
from encode.match import HashEmbedder, retrieve_candidates
from encode.models import MemoryUpdate, Prompt
from encode.triage import PatternLibrary, triage
from encode.update import MemoryUpdater
from memory import MemoryClient
from core import log
import re


class EncodeEngine:
    """Selective encode engine with incremental learning."""

    def __init__(self):
        self.memory = MemoryClient()
        self.patterns = PatternLibrary(self.memory)
        self.embedder = HashEmbedder()
        self.llm = LLMExtractor()
        self.updater = MemoryUpdater(self.embedder, self.memory)

    def process(self, prompts: Iterable[Prompt]) -> list[MemoryUpdate]:
        updates: list[MemoryUpdate] = []
        count = 0
        for prompt in prompts:
            count += 1
            self.memory.upsert_prompt(prompt)
            updates.extend(self._process_prompt(prompt))
        log("info", "encode_done", prompts=count, updates=len(updates))
        return updates

    def _process_prompt(self, prompt: Prompt) -> list[MemoryUpdate]:
        triage_result = triage(prompt, self.patterns)
        candidates = retrieve_candidates(prompt.text, self.memory, self.embedder)

        cheap = self._cheap_extract(prompt)
        llm_facts = []

        if triage_result.should_infer and self.llm.enabled():
            context = _build_context(candidates, self.memory)
            llm_facts = self.llm.extract(prompt.text, context)

        extracted = _dedupe(cheap + llm_facts)
        updates = self.updater.apply(prompt, extracted, self.memory)

        self._learn_patterns(prompt.text, extracted)

        for item in extracted:
            if item.get("pattern"):
                self.patterns.record(item["pattern"], success=True)

        return updates

    def _cheap_extract(self, prompt: Prompt) -> list[dict]:
        items: list[dict] = []
        for clause in _split_clauses(prompt.text):
            items.extend(_semantic_extract_clause(clause))
            items.extend(self.patterns.match(clause))
        normalized_items: list[dict] = []
        for item in items:
            normalized = _normalize_fact(item.get("content", ""))
            if not normalized:
                continue
            base_label = _derive_label(item)
            for fact in _expand_compound(normalized, item.get("category", "fact")):
                if not _is_compact_fact(fact["content"]):
                    continue
                fact["source"] = item.get("source", "pattern")
                fact["pattern"] = item.get("pattern")
                fact["confidence"] = item.get("confidence", 0.45)
                fact["label"] = _derive_label(fact, fallback=base_label)
                normalized_items.append(fact)
        return normalized_items

    def _learn_patterns(self, text: str, facts: list[dict]) -> None:
        sentences = _split_sentences(text)
        for sentence in sentences:
            candidate = _prompt_pattern_candidate(sentence)
            if candidate:
                self.memory.upsert_learned_pattern(
                    signature=candidate["signature"],
                    template=candidate["template"],
                    category=candidate["category"],
                    confidence=0.2,
                    success=True,
                    metadata={"source": "prompt", "example": sentence},
                )
        for fact in facts:
            candidate = _fact_pattern_candidate(fact)
            if candidate:
                self.memory.upsert_learned_pattern(
                    signature=candidate["signature"],
                    template=candidate["template"],
                    category=candidate["category"],
                    confidence=0.5,
                    success=True,
                    metadata={"source": "fact", "example": fact.get("content", "")},
                )


def _dedupe(items: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for item in items:
        key = (item.get("content", "").lower(), item.get("category", "fact"))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _build_context(candidates, memory: MemoryClient) -> list[dict]:
    context = []
    for cand in candidates:
        facts = memory.get_facts_by_label(cand.label_id)
        context.append(
            {
                "label": cand.label_name,
                "facts": [f.content for f in facts[:3]],
            }
        )
    return context


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"[.!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def _split_clauses(text: str) -> list[str]:
    chunks = re.split(r"[.!?]+", text)
    clauses: list[str] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = re.split(
            r"\s+(?:and|but|because|so|which|that)\s+(?=(?:i|my|i\s+am|i\s+was|i\s+have|i\s+do|i'm|im)\b)",
            chunk,
            flags=re.IGNORECASE,
        )
        for part in parts:
            part = part.strip(" ,;:\t\n\r")
            if part:
                clauses.append(part)
    return clauses


def _derive_label(item: dict, fallback: str | None = None) -> str:
    mode = item.get("label_mode", "group")
    groups = item.get("groups", [])
    category = item.get("category", "")
    if mode == "fixed":
        return category or "general"
    if mode == "subject" and groups:
        return _normalize_label(groups[0])
    if mode == "object" and groups:
        return _normalize_label(groups[0])
    if groups:
        return _normalize_label(groups[0])
    content = str(item.get("content", "")).lower().strip()
    label = _label_from_content(content)
    if label:
        return label
    if fallback:
        return fallback
    return "general"


def _normalize_fact(content: str, max_words: int = 16, min_words: int = 2) -> str | None:
    text = content.strip().lower()
    if not text:
        return None
    text = re.split(r"\s+(?:but|because|so|which|that)\s+", text)[0]
    text = text.strip(" ,;:\t\n\r")
    text = re.sub(r"\b(really|just|actually|basically|literally)\b", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    words = text.split()
    if len(words) < min_words:
        return None
    if len(words) > max_words:
        words = words[:max_words]
        text = " ".join(words)
    return text


def _semantic_extract_clause(clause: str) -> list[dict]:
    text = clause.strip()
    if not text:
        return []
    lowered = text.lower()
    lowered = lowered.replace("i'm", "i am").replace("im", "i am")
    patterns = [
        (r"^i\s+am\s+(\d{1,3})\b", "identity", "age is {0}", "fixed"),
        (r"^i\s+am\s+(?:a|an)\s+(.+)$", "identity", "is {0}", "fixed"),
        (r"^i\s+am\s+always\s+(.+)$", "habit", "always {0}", "object"),
        (r"^i\s+always\s+(.+)$", "habit", "always {0}", "object"),
        (r"^i\s+feel\s+(.+)$", "emotion", "feels {0}", "fixed"),
        (r"^i\s+(?:really\s+)?like\s+(.+)$", "preference", "likes {0}", "object"),
        (r"^i\s+(?:really\s+)?love\s+(.+)$", "preference", "likes {0}", "object"),
        (r"^i\s+(?:really\s+)?loved\s+(.+)$", "preference", "likes {0}", "object"),
        (r"^i\s+(?:really\s+)?enjoy\s+(.+)$", "preference", "likes {0}", "object"),
        (r"^i\s+(?:really\s+)?prefer\s+(.+)$", "preference", "prefers {0}", "object"),
        (r"^i\s+(?:really\s+)?dislike\s+(.+)$", "preference", "dislikes {0}", "object"),
        (r"^i\s+(?:really\s+)?hate\s+(.+)$", "preference", "dislikes {0}", "object"),
        (r"^i\s+(?:really\s+)?hated\s+(.+)$", "preference", "dislikes {0}", "object"),
        (r"^i\s+want\s+to\s+(.+)$", "goal", "wants to {0}", "object"),
        (r"^i\s+plan\s+to\s+(.+)$", "goal", "wants to {0}", "object"),
        (r"^i\s+aim\s+to\s+(.+)$", "goal", "wants to {0}", "object"),
        (r"^i\s+need\s+to\s+(.+)$", "goal", "needs to {0}", "object"),
        (r"^i\s+live\s+in\s+(.+)$", "location", "lives in {0}", "object"),
        (r"^i\s+am\s+from\s+(.+)$", "location", "from {0}", "object"),
        (r"^i\s+work\s+in\s+(.+)$", "work", "works in {0}", "object"),
        (r"^i\s+write\s+in\s+(.+)$", "style", "writes in {0}", "object"),
        (r"^my\s+(.+?)\s+(?:is|are)\s+(.+)$", "relationship", "{0} is {1}", "subject"),
    ]
    for pattern, category, template, label_mode in patterns:
        match = re.match(pattern, lowered)
        if match:
            content = template.format(*match.groups())
            return [
                {
                    "content": content,
                    "category": category,
                    "confidence": 0.55,
                    "pattern": pattern,
                    "groups": list(match.groups()),
                    "label_mode": label_mode,
                    "source": "semantic",
                }
            ]
    return []


def _expand_compound(content: str, category: str) -> list[dict]:
    if " " not in content:
        return [{"content": content, "category": category}]
    verb, rest = content.split(" ", 1)
    if not rest:
        return [{"content": content, "category": category}]
    items = _split_list_items(rest)
    if len(items) <= 1:
        return [{"content": content, "category": category}]
    expanded = []
    for item in items:
        expanded.append({"content": f"{verb} {item}", "category": category})
    return expanded


def _is_compact_fact(content: str, max_words: int = 12, min_words: int = 2) -> bool:
    words = content.split()
    return min_words <= len(words) <= max_words


def _split_list_items(text: str) -> list[str]:
    parts = re.split(r",|\s+and\s+", text)
    items = [p.strip(" ,;:\t\n\r") for p in parts if p.strip(" ,;:\t\n\r")]
    return items


def _label_from_content(content: str) -> str | None:
    patterns = [
        r"^(?:likes|dislikes|prefers)\s+(.+)$",
        r"^(?:wants to|needs to)\s+(.+)$",
        r"^(?:feels)\s+(.+)$",
        r"^(?:lives in|from|works in|writes in)\s+(.+)$",
        r"^(?:age is)\s+(.+)$",
        r"^(.+?)\s+is\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, content)
        if match:
            value = match.group(1)
            return _normalize_label(value)
    return None


def _normalize_label(text: str, max_words: int = 4) -> str:
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    if not tokens:
        return "general"
    stop = {
        "i",
        "my",
        "the",
        "a",
        "an",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "and",
        "but",
    }
    cleaned = [t for t in tokens if t not in stop]
    if not cleaned:
        cleaned = tokens
    tail = cleaned[-max_words:]
    return " ".join(tail)


def _prompt_pattern_candidate(sentence: str) -> dict | None:
    text = sentence.strip().lower()
    match = re.match(r"^i\s+(\w+)\s+(.+)$", text)
    if not match:
        return None
    verb = match.group(1)
    if len(verb) < 3:
        return None
    template = f"i {verb} {{object}}"
    signature = f"prompt:{verb}"
    return {"signature": signature, "template": template, "category": "custom"}


def _fact_pattern_candidate(fact: dict) -> dict | None:
    content = str(fact.get("content", "")).strip().lower()
    category = fact.get("category", "fact")
    if not content:
        return None
    template = None
    if content.startswith("likes "):
        template = "likes {thing}"
    elif content.startswith("dislikes "):
        template = "dislikes {thing}"
    elif content.startswith("prefers "):
        template = "prefers {thing}"
    elif content.startswith("wants to "):
        template = "wants to {action}"
    elif content.startswith("needs to "):
        template = "needs to {action}"
    elif content.startswith("feels "):
        template = "feels {emotion}"
    elif " is " in content:
        template = "{subject} is {value}"
    elif content.startswith("lives in "):
        template = "lives in {place}"
    if not template:
        return None
    signature = f"fact:{category}:{template}"
    return {"signature": signature, "template": template, "category": category}
