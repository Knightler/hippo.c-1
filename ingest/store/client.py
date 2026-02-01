from abc import ABC, abstractmethod
from typing import List

from ingest.models import Fact


class StoreClient(ABC):
    """Abstract interface for the storage layer

    This defines the contract for storing extracted facts in
    PostgreSQL. The concrete implementation will live in the
    store module, but ingest uses this interface to stay decoupled.
    """

    @abstractmethod
    def save_facts(self, facts: List[Fact]) -> List[str]:
        """Save facts to the store

        Args:
            facts: List of facts to store

        Returns:
            List of fact IDs assigned by the store
        """

    @abstractmethod
    def update_fact(self, fact_id: str, updates: dict) -> bool:
        """Update a stored fact

        Args:
            fact_id: ID of the fact to update
            updates: Dictionary of updates to apply

        Returns:
            True if updated, False if not found
        """

    @abstractmethod
    def batch_save_facts(self, batches: List[List[Fact]]) -> List[List[str]]:
        """Save multiple batches of facts

        Args:
            batches: List of fact batches

        Returns:
            List of ID lists (one per batch)
        """
