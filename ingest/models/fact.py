from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import uuid


@dataclass
class Fact:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    label_id: str = ""
    content: str = ""
    category: str = "fact"
    confidence: float = 0.5
    source_role: str = "user"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_seen_at: datetime = field(default_factory=datetime.utcnow)
    evidence_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def reinforce(self, boost: float = 0.05) -> None:
        self.evidence_count += 1
        self.last_seen_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.confidence = min(1.0, self.confidence + boost)
