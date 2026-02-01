from typing import Iterable

from ingest.infer import LLMExtractor
from ingest.match import HashEmbedder, retrieve_candidates
from ingest.memory import MemoryStore
from ingest.models import MemoryUpdate, Prompt
from ingest.triage import PatternLibrary, triage
from ingest.update import MemoryUpdater


class IngestEngine:
    """Selective ingest engine with incremental learning."""

    def __init__(self):
        self.memory = MemoryStore()
        self.patterns = PatternLibrary()
        self.embedder = HashEmbedder()
        self.llm = LLMExtractor()
        self.updater = MemoryUpdater(self.embedder)

    def process(self, prompts: Iterable[Prompt]) -> list[MemoryUpdate]:
        updates: list[MemoryUpdate] = []
        for prompt in prompts:
            updates.extend(self._process_prompt(prompt))
        self.memory.save()
        return updates

    def _process_prompt(self, prompt: Prompt) -> list[MemoryUpdate]:
        triage_result = triage(prompt, self.patterns)
        candidates = retrieve_candidates(
            prompt.text, triage_result.terms, self.memory, self.embedder
        )

        cheap = self._cheap_extract(prompt, triage_result)
        llm_facts = []

        if triage_result.should_infer and self.llm.enabled():
            context = _build_context(candidates, self.memory)
            llm_facts = self.llm.extract(prompt.text, context)

        extracted = _dedupe(cheap + llm_facts)
        updates = self.updater.apply(prompt, extracted, self.memory)

        for item in extracted:
            if item.get("pattern"):
                self.patterns.record(item["pattern"], success=True)

        return updates

    def _cheap_extract(self, prompt: Prompt, triage_result) -> list[dict]:
        items = self.patterns.match(prompt.text)
        label_name = triage_result.terms[0] if triage_result.terms else "general"
        for item in items:
            item["label"] = label_name
            item["source"] = "pattern"
        return items


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


def _build_context(candidates, memory: MemoryStore) -> list[dict]:
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
