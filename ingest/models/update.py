from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryUpdate:
    action: str = "skip"  # create | update | skip
    entity: str = "fact"  # label | fact
    entity_id: str = ""
    confidence: float = 0.0
    reason: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
