from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import uuid


@dataclass
class Label:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    kind: str = "topic"
    embedding: list[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    usage_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = datetime.utcnow()
        self.usage_count += 1
