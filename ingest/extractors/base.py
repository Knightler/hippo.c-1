from abc import ABC, abstractmethod
from typing import List

from ingest.models import Fact, Prompt, PromptBatch


class BaseExtractor(ABC):
    """Abstract base class for fact extractors

    All extractor implementations must inherit from this class
    and implement the required methods. This enables dynamic
    dispatch and polymorphic behavior for the hybrid orchestrator.
    """

    @abstractmethod
    def extract(self, prompts: PromptBatch) -> List[Fact]:
        """Extract facts from a batch of prompts

        Args:
            prompts: Batch of prompts to process

        Returns:
            List of extracted facts with metadata

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """

    @abstractmethod
    def can_handle(self, prompt: Prompt) -> float:
        """Determine if this extractor can handle the given prompt

        Returns a confidence score (0.0-1.0) indicating how well
        this extractor can process the prompt. Higher scores indicate
        better suitability.

        Args:
            prompt: Single prompt to evaluate

        Returns:
            Confidence score from 0.0 to 1.0

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """

    def get_name(self) -> str:
        """Return the name of this extractor

        Returns:
            Extractor name for logging and debugging
        """
        return self.__class__.__name__
