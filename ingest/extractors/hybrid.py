from typing import List

from ingest.extractors.base import BaseExtractor
from ingest.extractors.rules import RulesExtractor
from ingest.extractors.llm import LLMExtractor
from ingest.models import Fact, Prompt, PromptBatch


class HybridExtractor(BaseExtractor):
    """Smart orchestrator that combines rules and LLM extractors

    This extractor uses a hybrid approach to balance speed and accuracy:
    1. First, apply rules extractor (fast, sub-millisecond)
    2. If confidence is below threshold, use LLM extractor
    3. Combine results and deduplicate similar facts
    4. Batch LLM calls for efficiency

    The confidence threshold can be adjusted dynamically based
    on feedback from the learning system.
    """

    def __init__(
        self,
        rules_extractor: RulesExtractor,
        llm_extractor: LLMExtractor,
        llm_threshold: float = 0.6,
    ):
        """Initialize hybrid extractor

        Args:
            rules_extractor: Fast rule-based extractor
            llm_extractor: Nuanced LLM-based extractor
            llm_threshold: Confidence below which LLM is used (0.0-1.0)
        """
        self.rules = rules_extractor
        self.llm = llm_extractor
        self.llm_threshold = llm_threshold

    def extract(self, prompts: PromptBatch) -> List[Fact]:
        """Extract facts from a batch of prompts

        Processes each prompt, using LLM only when needed based
        on confidence threshold.

        Args:
            prompts: Batch of prompts to process

        Returns:
            List of deduplicated facts with metadata
        """
        all_facts = []
        low_confidence_prompts = []

        for prompt in prompts.prompts:
            rules_conf = self.rules.can_handle(prompt)
            llm_conf = self.llm.can_handle(prompt)

            if rules_conf >= self.llm_threshold:
                batch = PromptBatch(prompts=[prompt])
                facts = self.rules.extract(batch)
                all_facts.extend(facts)
            else:
                low_confidence_prompts.append(prompt)

        if low_confidence_prompts:
            llm_batch = PromptBatch(prompts=low_confidence_prompts)
            llm_facts = self.llm.extract(llm_batch)
            all_facts.extend(llm_facts)

        deduplicated = self._deduplicate(all_facts)
        return deduplicated

    def _deduplicate(self, facts: List[Fact]) -> List[Fact]:
        """Remove duplicate or very similar facts

        Uses simple text similarity to deduplicate. Facts with
        identical or very similar content are merged, keeping
        the one with higher confidence.

        Args:
            facts: List of facts to deduplicate

        Returns:
            Deduplicated list of facts
        """
        unique_facts = []
        seen_contents = set()

        for fact in sorted(facts, key=lambda f: -f.confidence):
            normalized = self._normalize_content(fact.content)

            if normalized not in seen_contents:
                seen_contents.add(normalized)
                unique_facts.append(fact)

        return unique_facts

    def _normalize_content(self, content: str) -> str:
        """Normalize content for comparison

        Args:
            content: Fact content to normalize

        Returns:
            Normalized content string
        """
        return content.lower().strip()

    def can_handle(self, prompt: Prompt) -> float:
        """Determine confidence of handling this prompt

        Returns the maximum confidence between rules and LLM.

        Args:
            prompt: Single prompt to evaluate

        Returns:
            Confidence score from 0.0 to 1.0
        """
        rules_conf = self.rules.can_handle(prompt)
        llm_conf = self.llm.can_handle(prompt)
        return max(rules_conf, llm_conf)

    def set_llm_threshold(self, threshold: float) -> None:
        """Update the LLM confidence threshold

        This allows dynamic adjustment based on learning system
        feedback. Lower threshold means more LLM usage (slower
        but more accurate), higher threshold means more rules
        usage (faster but potentially less accurate).

        Args:
            threshold: New threshold value (0.0-1.0)
        """
        self.llm_threshold = max(0.0, min(1.0, threshold))

    def get_threshold(self) -> float:
        """Get current LLM confidence threshold

        Returns:
            Current threshold value
        """
        return self.llm_threshold
