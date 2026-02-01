import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from ingest.extractors.rules import RulePattern, RulesExtractor
from ingest.models import FactCategory


class PatternStats:
    """Statistics for a single extraction pattern

    Attributes:
        total_uses: Total number of times this pattern was used
        successful_extractions: Number of successful extractions
        last_used: Timestamp of last use
        confidence_score: Calculated confidence based on performance
    """

    def __init__(self):
        self.total_uses: int = 0
        self.successful_extractions: int = 0
        self.last_used: datetime = datetime.utcnow()

    def get_confidence(self) -> float:
        """Calculate confidence based on performance

        Returns:
            Confidence score (0.0-1.0)
        """
        if self.total_uses == 0:
            return 0.5

        base_confidence = self.successful_extractions / self.total_uses
        recency_boost = self._get_recency_boost()
        return min(1.0, base_confidence + recency_boost)

    def _get_recency_boost(self) -> float:
        """Calculate recency boost for frequently used patterns

        Returns:
            Boost amount (0.0-0.2)
        """
        hours_since_use = (datetime.utcnow() - self.last_used).total_seconds() / 3600

        if hours_since_use < 1:
            return 0.2
        elif hours_since_use < 24:
            return 0.1
        elif hours_since_use < 168:  # 1 week
            return 0.05
        else:
            return 0.0

    def record_use(self, successful: bool) -> None:
        """Record a use of this pattern

        Args:
            successful: Whether this extraction was successful
        """
        self.total_uses += 1
        if successful:
            self.successful_extractions += 1
        self.last_used = datetime.utcnow()


class PatternManager:
    """Manages extraction patterns with learning capabilities

    This system stores patterns both in-memory and persistently,
    tracks their performance, and allows dynamic evolution.

    Patterns can be:
    - Added from user feedback or LLM suggestions
    - Updated with new confidence scores
    - Pruned when they perform poorly
    - Saved/loaded for persistence
    """

    DEFAULT_PATTERNS_PATH = "ingest/learning/patterns.json"

    def __init__(self, patterns_path: str | None = None):
        """Initialize pattern manager

        Args:
            patterns_path: Path to patterns file (default: ./patterns.json)
        """
        self.patterns_path = patterns_path or self.DEFAULT_PATTERNS_PATH
        self.patterns: List[RulePattern] = []
        self.stats: Dict[int, PatternStats] = {}
        self._ensure_patterns_directory()
        self._load_patterns()

    def _ensure_patterns_directory(self) -> None:
        """Ensure the patterns directory exists"""
        directory = os.path.dirname(self.patterns_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def add_pattern(self, pattern: RulePattern) -> None:
        """Add a new pattern to the manager

        Args:
            pattern: Pattern to add
        """
        self.patterns.append(pattern)
        index = len(self.patterns) - 1
        self.stats[index] = PatternStats()
        self._save_patterns()

    def update_pattern(self, index: int, pattern: RulePattern) -> bool:
        """Update an existing pattern

        Args:
            index: Index of pattern to update
            pattern: New pattern data

        Returns:
            True if updated, False if index invalid
        """
        if 0 <= index < len(self.patterns):
            self.patterns[index] = pattern
            self._save_patterns()
            return True
        return False

    def remove_pattern(self, index: int) -> bool:
        """Remove a pattern by index

        Args:
            index: Index of pattern to remove

        Returns:
            True if removed, False if index invalid
        """
        if 0 <= index < len(self.patterns):
            self.patterns.pop(index)
            del self.stats[index]
            self._save_patterns()
            return True
        return False

    def record_pattern_use(self, index: int, successful: bool) -> None:
        """Record a use of a pattern for learning

        Args:
            index: Index of pattern used
            successful: Whether extraction was successful
        """
        if index in self.stats:
            self.stats[index].record_use(successful)
            self._save_patterns()

    def get_pattern_confidence(self, index: int) -> float:
        """Get the learned confidence for a pattern

        Args:
            index: Index of pattern

        Returns:
            Confidence score (0.0-1.0) or 0.0 if invalid
        """
        if index in self.stats:
            return self.stats[index].get_confidence()
        return 0.0

    def prune_low_confidence_patterns(self, threshold: float = 0.3) -> int:
        """Remove patterns with low learned confidence

        Args:
            threshold: Confidence threshold below which patterns are pruned

        Returns:
            Number of patterns pruned
        """
        to_remove = []

        for index, stats in self.stats.items():
            if stats.get_confidence() < threshold and stats.total_uses > 10:
                to_remove.append(index)

        for index in sorted(to_remove, reverse=True):
            self.remove_pattern(index)

        return len(to_remove)

    def apply_to_extractor(self, extractor: RulesExtractor) -> None:
        """Apply managed patterns to a RulesExtractor instance

        Args:
            extractor: RulesExtractor to update with patterns
        """
        if self.patterns:
            extractor.load_patterns(self.get_patterns())

    def get_patterns(self) -> List[Dict]:
        """Export patterns for external use

        Returns:
            List of pattern dictionaries
        """
        return [
            {
                "pattern": p.pattern.pattern,
                "category": p.category.value,
                "confidence": p.confidence,
                "template": p.template,
                "total_uses": self.stats.get(i, PatternStats()).total_uses,
                "successful_extractions": self.stats.get(i, PatternStats()).successful_extractions,
                "last_used": self.stats.get(i, PatternStats()).last_used.isoformat(),
            }
            for i, p in enumerate(self.patterns)
        ]

    def _save_patterns(self) -> None:
        """Save patterns to persistent storage"""
        data = {
            "patterns": self.get_patterns(),
            "last_updated": datetime.utcnow().isoformat(),
        }

        with open(self.patterns_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_patterns(self) -> None:
        """Load patterns from persistent storage"""
        if not os.path.exists(self.patterns_path):
            return

        try:
            with open(self.patterns_path, "r") as f:
                data = json.load(f)

            for pattern_data in data.get("patterns", []):
                pattern = RulePattern(
                    pattern=pattern_data["pattern"],
                    category=FactCategory(pattern_data["category"]),
                    confidence=pattern_data["confidence"],
                    template=pattern_data.get("template"),
                )
                self.patterns.append(pattern)
                index = len(self.patterns) - 1

                stats = PatternStats()
                if "total_uses" in pattern_data:
                    stats.total_uses = int(pattern_data["total_uses"])
                if "successful_extractions" in pattern_data:
                    stats.successful_extractions = int(pattern_data["successful_extractions"])
                if "last_used" in pattern_data:
                    stats.last_used = datetime.fromisoformat(pattern_data["last_used"])
                self.stats[index] = stats

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error loading patterns: {e}")

    def get_stats(self) -> Dict:
        """Get statistics about pattern manager

        Returns:
            Dictionary with pattern statistics
        """
        total_confidence = sum(s.get_confidence() for s in self.stats.values())
        avg_confidence = total_confidence / len(self.stats) if self.stats else 0.0

        return {
            "total_patterns": len(self.patterns),
            "total_uses": sum(s.total_uses for s in self.stats.values()),
            "successful_extractions": sum(s.successful_extractions for s in self.stats.values()),
            "average_confidence": avg_confidence,
        }
