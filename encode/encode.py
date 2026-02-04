from typing import Iterable

from core import log
from encode.infer import LLMExtractor
from encode.match import HashEmbedder, retrieve_candidates
from encode.models import Fact, MemoryUpdate, Prompt
from memory import MemoryClient


class EncodeEngine:
    """LLM-first encode engine with two-layer decisions."""

    def __init__(self):
        self.memory = MemoryClient()
        self.embedder = HashEmbedder()
        self.llm = LLMExtractor()

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
        if not self.llm.enabled():
            log("error", "llm_disabled")
            return []

        context = _build_context(prompt.text, self.memory, self.embedder)
        llm_facts = self.llm.extract(prompt.text, context)
        log("info", "llm_extract", count=len(llm_facts))

        llm_facts = _normalize_llm_facts(llm_facts)
        llm_facts = _dedupe(llm_facts)
        if not llm_facts:
            return []

        decision_context = _build_fact_context(llm_facts, self.memory, self.embedder)
        decisions = self.llm.decide(prompt.text, llm_facts, decision_context)
        log("info", "llm_decide", count=len(decisions))

        updates = _apply_decisions(self.memory, self.embedder, prompt, llm_facts, decisions)
        _learn_patterns(self.memory, llm_facts)
        return updates


def _build_context(text: str, memory: MemoryClient, embedder: HashEmbedder) -> list[dict]:
    candidates = retrieve_candidates(text, memory, embedder)
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


def _build_fact_context(
    facts: list[dict], memory: MemoryClient, embedder: HashEmbedder
) -> list[dict]:
    context = []
    for fact in facts:
        candidates = retrieve_candidates(fact.get("content", ""), memory, embedder)
        existing: list[str] = []
        for cand in candidates:
            for row in memory.get_facts_by_label(cand.label_id, limit=3):
                existing.append(row.content)
        context.append(
            {
                "content": fact.get("content"),
                "label": fact.get("label"),
                "category": fact.get("category"),
                "existing": existing[:5],
            }
        )
    return context


def _normalize_llm_facts(items: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip().lower()
        category = str(item.get("category", "fact")).strip().lower() or "fact"
        label = str(item.get("label", "")).strip().lower()
        if not content or not label:
            continue
        normalized.append(
            {
                "content": content,
                "category": category,
                "label": label,
                "confidence": float(item.get("confidence", 0.6)),
                "source": "llm",
            }
        )
    return normalized


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


def _apply_decisions(
    memory: MemoryClient,
    embedder: HashEmbedder,
    prompt: Prompt,
    facts: list[dict],
    decisions: list[dict],
) -> list[MemoryUpdate]:
    updates: list[MemoryUpdate] = []
    decisions_map = {d.get("content"): d for d in decisions if isinstance(d, dict)}
    for fact in facts:
        content = fact.get("content")
        decision = decisions_map.get(content, {})
        action = decision.get("action", "skip")
        if action == "skip":
            continue
        label = memory.upsert_label(
            name=fact.get("label", "general"),
            kind="topic",
            category="topic",
            embedding=embedder.embed(fact.get("label", "general")),
        )
        if action == "reinforce":
            match = memory.find_best_fact(label.id, embedder.embed(content))
            if match:
                memory.reinforce_fact(match.id, boost=0.05)
                updates.append(
                    MemoryUpdate(
                        action="update",
                        entity="fact",
                        entity_id=match.id,
                        confidence=match.confidence,
                        reason="reinforced",
                        payload={"content": match.content},
                    )
                )
                continue
        fact_row = Fact(
            label_id=label.id,
            content=content,
            category=fact.get("category", "fact"),
            confidence=fact.get("confidence", 0.5),
            source_role=prompt.role.value,
            embedding=embedder.embed(content),
            metadata={"source": fact.get("source", "llm")},
        )
        memory.insert_fact(fact_row)
        updates.append(
            MemoryUpdate(
                action="create",
                entity="fact",
                entity_id=fact_row.id,
                confidence=fact_row.confidence,
                reason="llm_create",
                payload={"content": fact_row.content},
            )
        )
    return updates


def _learn_patterns(memory: MemoryClient, facts: list[dict]) -> None:
    for fact in facts:
        candidate = _fact_pattern_candidate(fact)
        if candidate:
            memory.upsert_learned_pattern(
                signature=candidate["signature"],
                template=candidate["template"],
                category=candidate["category"],
                confidence=0.5,
                success=True,
                metadata={"source": "fact", "example": fact.get("content", "")},
            )


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
