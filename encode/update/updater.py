from encode.match.embedder import HashEmbedder
from encode.models import Fact, MemoryUpdate, Prompt
from memory import MemoryClient


class MemoryUpdater:
    """Applies extracted facts to memory with incremental updates."""

    def __init__(
        self, embedder: HashEmbedder, memory: MemoryClient, match_threshold: float = 0.85
    ):
        self.embedder = embedder
        self.memory = memory
        self.match_threshold = match_threshold

    def apply(
        self,
        prompt: Prompt,
        extracted: list[dict],
        memory: MemoryClient,
    ) -> list[MemoryUpdate]:
        updates: list[MemoryUpdate] = []

        for item in extracted:
            label = memory.upsert_label(
                name=item.get("label", "general"),
                kind="topic",
                category="topic",
                embedding=self.embedder.embed(item.get("label", "general")),
            )
            content_embedding = self.embedder.embed(item["content"])
            match = memory.find_best_fact(label.id, content_embedding)
            if match:
                similarity = self.embedder.similarity(
                    content_embedding, self.embedder.embed(match.content)
                )
                if similarity < self.match_threshold:
                    match = None
                else:
                    memory.reinforce_fact(match.id, boost=0.05)
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
                    embedding=content_embedding,
                    metadata={"source": item.get("source", "encode")},
                )
                memory.insert_fact(fact)
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
