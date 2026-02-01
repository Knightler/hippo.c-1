from typing import Optional

from ingest.learning.patterns import PatternManager
from ingest.models import Fact


class FeedbackManager:
    """Handles user feedback and learning from corrections

    This system allows the ingest module to learn from user feedback:
    - Corrections: Update incorrect facts and learn from mistakes
    - Positive feedback: Boost confidence in patterns that extracted correctly
    - Negative feedback: Lower confidence in patterns that extracted incorrectly

    Feedback is applied to the pattern manager, allowing the system
    to evolve and improve over time.
    """

    def __init__(self, pattern_manager: PatternManager):
        """Initialize feedback manager

        Args:
            pattern_manager: PatternManager to update with feedback
        """
        self.pattern_manager = pattern_manager

    def add_correction(self, fact: Fact, correct_content: str) -> Optional[dict]:
        """Handle a correction to an extracted fact

        When a user corrects a fact, we:
        1. Identify which pattern extracted this fact
        2. Lower that pattern's confidence
        3. Optionally generate a new pattern from the correction

        Args:
            fact: The incorrect fact that was extracted
            correct_content: The corrected content from user

        Returns:
            Dictionary with feedback results or None if no pattern found
        """
        rule_pattern = fact.metadata.get("rule_pattern")

        if not rule_pattern:
            return None

        pattern_index = self._find_pattern_by_regex(rule_pattern)
        if pattern_index is None:
            return None

        self.pattern_manager.record_pattern_use(pattern_index, successful=False)

        result = {
            "pattern_index": pattern_index,
            "original_content": fact.content,
            "corrected_content": correct_content,
            "action": "lowered_pattern_confidence",
        }

        return result

    def add_positive_feedback(self, fact: Fact) -> Optional[dict]:
        """Handle positive feedback on an extracted fact

        When a user confirms a fact is correct, we boost the
        confidence of the pattern that extracted it.

        Args:
            fact: The fact that was confirmed correct

        Returns:
            Dictionary with feedback results or None if no pattern found
        """
        rule_pattern = fact.metadata.get("rule_pattern")

        if not rule_pattern:
            return None

        pattern_index = self._find_pattern_by_regex(rule_pattern)
        if pattern_index is None:
            return None

        self.pattern_manager.record_pattern_use(pattern_index, successful=True)

        result = {
            "pattern_index": pattern_index,
            "fact_content": fact.content,
            "action": "boosted_pattern_confidence",
        }

        return result

    def add_negative_feedback(self, fact: Fact) -> Optional[dict]:
        """Handle negative feedback on an extracted fact

        When a user indicates a fact is incorrect (without correction),
        we lower the confidence of the pattern that extracted it.

        Args:
            fact: The fact that was marked incorrect

        Returns:
            Dictionary with feedback results or None if no pattern found
        """
        rule_pattern = fact.metadata.get("rule_pattern")

        if not rule_pattern:
            return None

        pattern_index = self._find_pattern_by_regex(rule_pattern)
        if pattern_index is None:
            return None

        self.pattern_manager.record_pattern_use(pattern_index, successful=False)

        result = {
            "pattern_index": pattern_index,
            "fact_content": fact.content,
            "action": "lowered_pattern_confidence",
        }

        return result

    def _find_pattern_by_regex(self, regex_pattern: str) -> Optional[int]:
        """Find a pattern index by its regex string

        Args:
            regex_pattern: The regex pattern string to find

        Returns:
            Pattern index if found, None otherwise
        """
        for i, pattern in enumerate(self.pattern_manager.patterns):
            if pattern.pattern.pattern == regex_pattern:
                return i
        return None

    def suggest_new_pattern(
        self,
        source_text: str,
        extracted_fact: str,
        confidence: float = 0.6,
    ) -> Optional[dict]:
        """Suggest a new pattern based on successful extraction

        This can be used to generate patterns from LLM extractions
        that were successful, allowing the rules extractor to learn
        from LLM successes.

        Args:
            source_text: Original text that contained the fact
            extracted_fact: The fact that was extracted
            confidence: Confidence to assign to new pattern

        Returns:
            Dictionary with suggested pattern or None if can't suggest
        """
        if not source_text or not extracted_fact:
            return None

        simple_pattern = self._generate_simple_pattern(source_text, extracted_fact)

        if not simple_pattern:
            return None

        return {
            "pattern": simple_pattern,
            "confidence": confidence,
            "source_text": source_text,
            "extracted_fact": extracted_fact,
        }

    def _generate_simple_pattern(
        self, source_text: str, extracted_fact: str
    ) -> Optional[str]:
        """Generate a simple regex pattern from source and extracted fact

        This is a basic pattern generation. In a production system,
        this could use more sophisticated NLP techniques.

        Args:
            source_text: Original text
            extracted_fact: Extracted fact text

        Returns:
            Generated regex pattern or None if can't generate
        """
        if extracted_fact.lower() not in source_text.lower():
            return None

        words = extracted_fact.lower().split()

        if len(words) < 2:
            return None

        if source_text.lower().startswith("i "):
            pattern = f"i (.+?) {words[-1]}"
            return pattern

        return None
