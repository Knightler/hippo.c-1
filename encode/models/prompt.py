from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import uuid


class Role(str, Enum):
    USER = "user"
    AI = "ai"
    SYSTEM = "system"


@dataclass(frozen=True)
class Prompt:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    role: Role = Role.USER
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
