from typing import List

from ingest.models import Fact
from ingest.store import StoreClient


class InMemoryStore(StoreClient):
    """Minimal in-memory store for local testing"""

    def __init__(self):
        self.data: dict[str, Fact] = {}

    def save_facts(self, facts: List[Fact]) -> List[str]:
        ids = []
        for fact in facts:
            self.data[fact.id] = fact
            ids.append(fact.id)
        return ids

    def update_fact(self, fact_id: str, updates: dict) -> bool:
        if fact_id not in self.data:
            return False
        for key, value in updates.items():
            setattr(self.data[fact_id], key, value)
        return True

    def batch_save_facts(self, batches: List[List[Fact]]) -> List[List[str]]:
        return [self.save_facts(batch) for batch in batches]
