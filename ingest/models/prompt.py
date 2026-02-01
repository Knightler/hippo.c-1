from datetime import datetime
from enum import Enum
from typing import List
import uuid

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Role of the message sender"""

    USER = "user"
    AI = "ai"
    SYSTEM = "system"


class Prompt(BaseModel):
    """Single prompt/message representation

    Attributes:
        id: Unique identifier (auto-generated if not provided)
        text: The actual message content
        role: Who sent this message (user/ai/system)
        timestamp: When this message was created
        metadata: Optional additional context about this prompt
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    role: MessageRole
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, object] = Field(default_factory=dict)


class PromptBatch(BaseModel):
    """Batch of prompts for efficient processing

    Attributes:
        prompts: List of prompts to process
        batch_id: Unique identifier for this batch
        created_at: When this batch was created
        metadata: Optional batch-level metadata
    """

    prompts: List[Prompt] = Field(default_factory=list)
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, object] = Field(default_factory=dict)

    def add_prompt(self, prompt: Prompt) -> None:
        """Add a single prompt to the batch"""
        self.prompts.append(prompt)

    def size(self) -> int:
        """Return the number of prompts in this batch"""
        return len(self.prompts)
