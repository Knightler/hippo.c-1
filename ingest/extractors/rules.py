import re
from typing import Dict, List

from ingest.extractors.base import BaseExtractor
from ingest.models import Fact, FactCategory, FactSource, MessageRole, Prompt, PromptBatch


class RulePattern:
    """Single extraction rule pattern

    Attributes:
        pattern: Regex pattern to match
        category: Category to assign if matched
        confidence: Base confidence for this rule
        template: Template to extract fact content (optional)
    """

    def __init__(
        self,
        pattern: str,
        category: FactCategory,
        confidence: float,
        template: str | None = None,
    ):
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.category = category
        self.confidence = confidence
        self.template = template


class RulesExtractor(BaseExtractor):
    """Fast rule-based fact extractor using regex patterns

    This extractor uses predefined patterns to quickly extract facts
    from text. It's designed for speed (sub-millisecond per prompt)
    and provides a baseline confidence that the hybrid orchestrator
    can use to decide if LLM extraction is needed.

    Patterns can evolve through the learning system, allowing this
    extractor to improve over time without manual intervention.
    """

    def __init__(self, patterns: List[RulePattern] | None = None):
        """Initialize with patterns (uses defaults if none provided)

        Args:
            patterns: Custom list of rule patterns
        """
        self.patterns = patterns or self._get_default_patterns()

    def _get_default_patterns(self) -> List[RulePattern]:
        """Get default extraction patterns

        Returns:
            List of predefined RulePattern instances
        """
        return [
            RulePattern(
                r"i (?:really )?(?:like|love|enjoy|prefer) (.+?)(?:\.|$)",
                FactCategory.PREFERENCE,
                0.7,
                "likes {0}",
            ),
            RulePattern(
                r"i (?:really )?(?:hate|dislike|can't stand|detest) (.+?)(?:\.|$)",
                FactCategory.PREFERENCE,
                0.7,
                "dislikes {0}",
            ),
            RulePattern(
                r"my (.+?) (?:is|are) (.+?)(?:\.|$)",
                FactCategory.RELATIONSHIP,
                0.6,
                "{0} is {1}",
            ),
            RulePattern(
                r"i (?:went|visited|traveled to) (.+?)(?:\.|$)",
                FactCategory.LOCATION,
                0.65,
                "visited {0}",
            ),
            RulePattern(
                r"i (?:live|live in|am from|stay in) (.+?)(?:\.|$)",
                FactCategory.LOCATION,
                0.7,
                "lives in {0}",
            ),
            RulePattern(
                r"(?:yesterday|today|last week) i (.+?)(?:\.|$)",
                FactCategory.EVENT,
                0.6,
                "{0}",
            ),
            RulePattern(
                r"i (.+?)(?:\.|$)",
                FactCategory.EVENT,
                0.4,
                "{0}",
            ),
        ]

    def extract(self, prompts: PromptBatch) -> List[Fact]:
        """Extract facts from a batch of prompts

        Args:
            prompts: Batch of prompts to process

        Returns:
            List of extracted facts with metadata
        """
        facts = []

        for prompt in prompts.prompts:
            extracted = self._extract_from_prompt(prompt)
            facts.extend(extracted)

        return facts

    def _extract_from_prompt(self, prompt: Prompt) -> List[Fact]:
        """Extract facts from a single prompt

        Args:
            prompt: Single prompt to process

        Returns:
            List of facts extracted from this prompt
        """
        facts = []

        for rule in self.patterns:
            matches = rule.pattern.finditer(prompt.text)

            for match in matches:
                content = self._build_content(rule, match)
                if content:
                    fact = Fact(
                        content=content,
                        category=rule.category,
                        confidence=rule.confidence,
                        source=FactSource.USER
                        if prompt.role == MessageRole.USER
                        else FactSource.AI,
                        source_timestamp=prompt.timestamp,
                        source_id=prompt.id,
                        metadata={"rule_pattern": rule.pattern.pattern},
                    )
                    facts.append(fact)

        return facts

    def _build_content(self, rule: RulePattern, match: re.Match) -> str | None:
        """Build fact content from match and template

        Args:
            rule: The rule pattern that matched
            match: The regex match object

        Returns:
            Formatted fact content or None if invalid
        """
        if rule.template:
            groups = match.groups()
            try:
                content = rule.template.format(*groups)
            except IndexError:
                return None
        else:
            content = match.group(0)

        if len(content.strip()) < 3:
            return None

        return content.strip()

    def can_handle(self, prompt: Prompt) -> float:
        """Determine confidence of handling this prompt

        Returns the maximum confidence among all matching patterns.

        Args:
            prompt: Single prompt to evaluate

        Returns:
            Confidence score from 0.0 to 1.0
        """
        max_confidence = 0.0

        for rule in self.patterns:
            if rule.pattern.search(prompt.text):
                max_confidence = max(max_confidence, rule.confidence)

        return max_confidence

    def add_pattern(self, pattern: RulePattern) -> None:
        """Add a new extraction pattern

        Args:
            pattern: RulePattern to add to the extractor
        """
        self.patterns.append(pattern)

    def remove_pattern(self, index: int) -> None:
        """Remove a pattern by index

        Args:
            index: Index of pattern to remove
        """
        if 0 <= index < len(self.patterns):
            self.patterns.pop(index)

    def get_patterns(self) -> List[Dict]:
        """Export patterns for storage/learning

        Returns:
            List of pattern dictionaries
        """
        return [
            {
                "pattern": p.pattern.pattern,
                "category": p.category.value,
                "confidence": p.confidence,
                "template": p.template,
            }
            for p in self.patterns
        ]

    def load_patterns(self, patterns_data: List[Dict]) -> None:
        """Load patterns from exported data

        Args:
            patterns_data: List of pattern dictionaries
        """
        self.patterns = [
            RulePattern(
                pattern=p["pattern"],
                category=FactCategory(p["category"]),
                confidence=p["confidence"],
                template=p["template"],
            )
            for p in patterns_data
        ]
