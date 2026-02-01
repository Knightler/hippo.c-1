from typing import List, Optional

from ingest.extractors import HybridExtractor, LLMExtractor, RulesExtractor
from ingest.learning import FeedbackManager, PatternManager
from ingest.models import Fact, PromptBatch
from ingest.store import StoreClient


class IngestEngine:
    """Main entry point for the ingest system

    This engine coordinates the extraction pipeline and storage layer:
    - Runs hybrid extraction (rules + LLM)
    - Applies learning patterns
    - Sends facts to store
    - Handles feedback to improve extraction
    """

    def __init__(
        self,
        store_client: StoreClient,
        learning_enabled: bool = True,
        llm_model: str = "gpt-3.5-turbo",
        llm_threshold: float = 0.6,
    ):
        """Initialize the ingest engine

        Args:
            store_client: Store interface for persistence
            learning_enabled: Enable pattern learning system
            llm_model: LLM model name for extraction
            llm_threshold: Confidence threshold for LLM fallback
        """
        self.store = store_client
        self.learning_enabled = learning_enabled

        self.rules_extractor = RulesExtractor()
        self.llm_extractor = LLMExtractor(model_name=llm_model)
        self.hybrid_extractor = HybridExtractor(
            rules_extractor=self.rules_extractor,
            llm_extractor=self.llm_extractor,
            llm_threshold=llm_threshold,
        )

        self.pattern_manager: Optional[PatternManager] = None
        self.feedback_manager: Optional[FeedbackManager] = None

        if learning_enabled:
            self.pattern_manager = PatternManager()
            self.pattern_manager.apply_to_extractor(self.rules_extractor)
            self.feedback_manager = FeedbackManager(self.pattern_manager)

    def process(self, prompts: PromptBatch) -> List[str]:
        """Process a batch of prompts and store extracted facts

        Args:
            prompts: Batch of prompts to process

        Returns:
            List of stored fact IDs
        """
        facts = self.hybrid_extractor.extract(prompts)
        return self.store.save_facts(facts)

    def provide_feedback(
        self,
        fact: Fact,
        correction: Optional[str] = None,
        positive: bool = False,
        negative: bool = False,
    ) -> Optional[dict]:
        """Provide feedback to improve extraction

        Args:
            fact: The fact being reviewed
            correction: Corrected content if applicable
            positive: True if fact is confirmed correct
            negative: True if fact is incorrect (no correction)

        Returns:
            Feedback result or None if learning disabled
        """
        if not self.feedback_manager:
            return None

        if correction:
            return self.feedback_manager.add_correction(fact, correction)
        if positive:
            return self.feedback_manager.add_positive_feedback(fact)
        if negative:
            return self.feedback_manager.add_negative_feedback(fact)

        return None

    def get_stats(self) -> dict:
        """Get ingest system statistics

        Returns:
            Dictionary with extractor and learning stats
        """
        stats = {
            "llm_threshold": self.hybrid_extractor.get_threshold(),
            "learning_enabled": self.learning_enabled,
        }

        if self.pattern_manager:
            stats["patterns"] = self.pattern_manager.get_stats()

        return stats
