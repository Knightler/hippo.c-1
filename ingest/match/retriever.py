from dataclasses import dataclass

from ingest.match.embedder import HashEmbedder
from ingest.memory import MemoryStore


@dataclass
class Candidate:
    label_id: str
    label_name: str
    score: float


def retrieve_candidates(
    text: str,
    terms: list[str],
    memory: MemoryStore,
    embedder: HashEmbedder,
    top_k: int = 5,
) -> list[Candidate]:
    query_vec = embedder.embed(text)
    results: list[Candidate] = []

    for label in memory.get_labels():
        lexical = _lexical_score(label.name, terms)
        semantic = 0.0
        if label.embedding:
            semantic = embedder.similarity(query_vec, label.embedding)
        score = (0.7 * semantic) + (0.3 * lexical)
        if score > 0:
            results.append(Candidate(label_id=label.id, label_name=label.name, score=score))

    results.sort(key=lambda c: c.score, reverse=True)
    return results[:top_k]


def _lexical_score(name: str, terms: list[str]) -> float:
    if not terms:
        return 0.0
    name_lower = name.lower()
    hits = sum(1 for t in terms if t.lower() in name_lower)
    return hits / max(1, len(terms))
