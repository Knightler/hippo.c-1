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
            items.extend(self.patterns.match(clause))
        for item in items:
            normalized = _normalize_fact(item.get("content", ""))
            if not normalized:
                item["skip"] = True
                continue
            item["content"] = normalized
            label = _derive_label(item)
            item["label"] = label
            item["source"] = "pattern"
        return [item for item in items if not item.get("skip")]

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
        parts = re.split(r"\s+(?:and|but|because|so|which|that)\s+", chunk, flags=re.IGNORECASE)
        for part in parts:
            part = part.strip(" ,;:\t\n\r")
            if part:
                clauses.append(part)
    return clauses


def _derive_label(item: dict) -> str:
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
    return "general"


def _normalize_fact(content: str, max_words: int = 16, min_words: int = 2) -> str | None:
    text = content.strip().lower()
    if not text:
        return None
    text = re.split(r"\s+(?:and|but|because|so|which|that)\s+", text)[0]
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
