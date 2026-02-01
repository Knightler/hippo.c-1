"""Store interfaces for ingest module"""

from .client import StoreClient
from .memory import InMemoryStore

__all__ = ["StoreClient", "InMemoryStore"]
