from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field
import uuid


class FactCategory(str, Enum):
    """Supported fact categories for classification"""

    PREFERENCE = "preference"
    EVENT = "event"
    RELATIONSHIP = "relationship"
    BELIEF = "belief"
    FACT = "fact"
    LOCATION = "location"
    GOAL = "goal"
    EMOTION = "emotion"


class FactSource(str, Enum):
    """Source of the fact - user or AI"""

    USER = "user"
    AI = "ai"


class Fact(BaseModel):
    """High-dimensional fact representation with rich metadata

    Attributes:
        id: Unique identifier (auto-generated if not provided)
        content: The extracted fact/insight text
        category: Type classification (preference, event, etc.)
        confidence: Extraction confidence score (0.0-1.0)
        source: Where this fact came from (user or AI)
        timestamp: When this fact was extracted
        source_timestamp: When original content was created
        source_id: Reference to original prompt/response
        embedding_id: For future vector indexing (optional)
        metadata: Extensible JSON field for additional attributes
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    category: FactCategory
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source: FactSource
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_timestamp: datetime
    source_id: str
    embedding_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert fact to dictionary for storage"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Fact":
        """Create Fact from dictionary"""
        return cls(**data)
