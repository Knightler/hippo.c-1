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
) -> list[Candidate]:
    query_vec = embedder.embed(text)
    labels = memory.search_labels(query_vec, top_k=top_k)
    return [
        Candidate(label_id=l.id, label_name=l.name, score=1.0) for l in labels
    ]
