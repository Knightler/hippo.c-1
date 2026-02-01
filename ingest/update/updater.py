from ingest.match.embedder import HashEmbedder
from ingest.memory import MemoryStore
from ingest.models import Fact, MemoryUpdate, Prompt


class MemoryUpdater:
    """Applies extracted facts to memory with incremental updates."""

    def __init__(self, embedder: HashEmbedder, match_threshold: float = 0.85):
        self.embedder = embedder
        self.match_threshold = match_threshold

    def apply(
        self,
        prompt: Prompt,
        extracted: list[dict],
        memory: MemoryStore,
    ) -> list[MemoryUpdate]:
        updates: list[MemoryUpdate] = []

        for item in extracted:
            label = memory.upsert_label(
                name=item.get("label", "general"),
                kind=item.get("category", "fact"),
                embedding=self.embedder.embed(item.get("label", "general")),
            )
            match = self._find_best_match(item["content"], label.id, memory)
            if match:
                match.reinforce(boost=0.05)
                updates.append(
                    MemoryUpdate(
                        action="update",
                        entity="fact",
                        entity_id=match.id,
                        confidence=match.confidence,
                        reason="matched_existing",
                        payload={"content": match.content},
                    )
                )
            else:
                fact = Fact(
                    label_id=label.id,
                    content=item["content"],
                    category=item.get("category", "fact"),
                    confidence=item.get("confidence", 0.45),
                    source_role=prompt.role.value,
                    metadata={"source": item.get("source", "ingest")},
                )
                memory.upsert_fact(fact)
                updates.append(
                    MemoryUpdate(
                        action="create",
                        entity="fact",
                        entity_id=fact.id,
                        confidence=fact.confidence,
                        reason="new_fact",
                        payload={"content": fact.content},
                    )
                )

        return updates

    def _find_best_match(self, content: str, label_id: str, memory: MemoryStore) -> Fact | None:
        best = None
        best_score = 0.0
        target_vec = self.embedder.embed(content)
        for fact in memory.get_facts_by_label(label_id):
            score = self.embedder.similarity(target_vec, self.embedder.embed(fact.content))
            if score > best_score:
                best_score = score
                best = fact
        if best_score >= self.match_threshold:
            return best
        return None
