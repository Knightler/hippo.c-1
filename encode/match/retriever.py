from dataclasses import dataclass

from encode.match.embedder import HashEmbedder
from memory import MemoryClient


@dataclass
class Candidate:
    label_id: str
    label_name: str
    score: float


def retrieve_candidates(
    text: str,
    memory: MemoryClient,
    embedder: HashEmbedder,
    top_k: int = 5,
    min_score: float = 0.2,
) -> list[Candidate]:
    query_vec = embedder.embed(text)
    labels = memory.search_labels(query_vec, top_k=top_k)
    candidates: list[Candidate] = []
    for label in labels:
        score = embedder.similarity(query_vec, label.embedding)
        if score < min_score:
            continue
        candidates.append(
            Candidate(label_id=label.id, label_name=label.name, score=score)
        )
    return candidates
