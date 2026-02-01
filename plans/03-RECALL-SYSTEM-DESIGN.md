# Recall System Design: Complete Architecture

## Overview

The Recall System is the missing piece that allows stored memories to be retrieved and used by an LLM. This document provides a complete, production-ready design.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            RECALL SYSTEM                                     │
│                                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │  Query   │ → │ Strategy │ → │  Search  │ → │  Rank &  │ → │  Format  │ │
│  │ Parser   │   │ Selector │   │  Engine  │   │  Filter  │   │ Response │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│       ↓              ↓              ↓              ↓              ↓        │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ Extract  │   │ Choose   │   │ Vector + │   │ Rerank   │   │ Build    │ │
│  │ Intent   │   │ Search   │   │ Keyword  │   │ Dedupe   │   │ Context  │ │
│  │ Entities │   │ Method   │   │ Hybrid   │   │ Explain  │   │ Summary  │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MEMORY STORE                                       │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                   │
│  │    Labels     │  │    Facts      │  │   Prompts     │                   │
│  │  (pgvector)   │  │  (pgvector)   │  │   (history)   │                   │
│  └───────────────┘  └───────────────┘  └───────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## API Specification

### Recall Endpoint

```
POST /api/v1/recall
```

#### Request Schema

```json
{
    "query": "string",              // Natural language query (required)
    "context": "string",            // Current conversation context (optional)
    "user_id": "uuid",              // User identifier (required in multi-user mode)
    "options": {
        "limit": 10,                // Max results (default: 10, max: 50)
        "min_confidence": 0.3,      // Minimum confidence threshold (default: 0.3)
        "min_relevance": 0.5,       // Minimum relevance score (default: 0.5)
        "categories": ["preference", "fact"],  // Filter by category (optional)
        "labels": ["food", "work"], // Filter by label names (optional)
        "time_range": {             // Filter by time (optional)
            "after": "2024-01-01T00:00:00Z",
            "before": "2024-12-31T23:59:59Z"
        },
        "include_metadata": false,  // Include full metadata (default: false)
        "include_explanation": true // Explain why each result matched (default: true)
    }
}
```

#### Response Schema

```json
{
    "success": true,
    "query_id": "uuid",
    "latency_ms": 45,
    "results": [
        {
            "fact_id": "uuid",
            "content": "likes Italian food, especially pasta dishes",
            "category": "preference",
            "label": "food_preferences",
            "confidence": 0.85,
            "relevance_score": 0.92,
            "last_seen": "2024-01-15T10:30:00Z",
            "evidence_count": 5,
            "explanation": "Matched query 'food preferences' with high semantic similarity"
        }
    ],
    "summary": "User has strong preferences for Italian cuisine, particularly pasta. They dislike spicy food.",
    "total_matches": 15,
    "search_strategy": "hybrid_vector_keyword"
}
```

#### Error Response

```json
{
    "success": false,
    "error": {
        "code": "INVALID_QUERY",
        "message": "Query cannot be empty",
        "details": {}
    },
    "query_id": "uuid",
    "latency_ms": 2
}
```

---

## Component Design

### 1. Query Parser

**Purpose**: Extract intent, entities, and search parameters from natural language queries.

```python
# recall/query/parser.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryIntent(Enum):
    FACT_LOOKUP = "fact_lookup"       # "What is user's favorite color?"
    LIST_ALL = "list_all"             # "What do I know about user's hobbies?"
    COMPARE = "compare"               # "Does user prefer X or Y?"
    TEMPORAL = "temporal"             # "What did user say last week?"
    RELATIONSHIP = "relationship"     # "Who is user's friend?"
    PREFERENCE = "preference"         # "What foods does user like?"
    SUMMARY = "summary"               # "Summarize user's interests"


@dataclass
class ParsedQuery:
    raw_query: str
    intent: QueryIntent
    entities: list[str]               # Extracted named entities
    keywords: list[str]               # Important keywords
    categories: list[str]             # Inferred categories
    temporal_hints: dict[str, Any]    # Time-related filters
    negations: list[str]              # Negated terms (things NOT to match)
    confidence: float                 # Parser confidence
    

class QueryParser:
    """Parse natural language queries into structured search parameters."""
    
    # Intent detection patterns
    INTENT_PATTERNS = {
        QueryIntent.FACT_LOOKUP: [
            r"what is (user's|my|the user's) (.+)",
            r"tell me (user's|my|the user's) (.+)",
            r"what does (user|the user) (.+)",
        ],
        QueryIntent.LIST_ALL: [
            r"what do (i|you) know about (.+)",
            r"list (all|everything) (.+)",
            r"show me (.+)",
        ],
        QueryIntent.PREFERENCE: [
            r"what (does user|do i) (like|prefer|enjoy|love|hate)",
            r"(user's|my) (favorite|preferred|favourite)",
        ],
        QueryIntent.RELATIONSHIP: [
            r"who is (user's|my) (.+)",
            r"(user's|my) (family|friend|colleague)",
        ],
        QueryIntent.TEMPORAL: [
            r"(yesterday|last week|recently|lately)",
            r"(this|last) (week|month|year)",
        ],
        QueryIntent.SUMMARY: [
            r"summarize|summary|overview",
            r"tell me (about|everything)",
        ],
    }
    
    # Category inference keywords
    CATEGORY_KEYWORDS = {
        "preference": ["like", "love", "prefer", "favorite", "enjoy", "hate", "dislike"],
        "relationship": ["friend", "family", "colleague", "partner", "spouse", "parent", "child"],
        "location": ["live", "from", "located", "city", "country", "address", "home"],
        "goal": ["want", "plan", "hope", "aim", "goal", "aspire"],
        "emotion": ["feel", "feeling", "mood", "happy", "sad", "anxious"],
        "habit": ["always", "usually", "often", "every day", "routine"],
        "fact": ["is", "are", "has", "have", "born", "age", "name"],
    }
    
    def __init__(self, embedder=None):
        self.embedder = embedder
    
    def parse(self, query: str, context: str = "") -> ParsedQuery:
        """Parse query into structured format."""
        query_lower = query.lower()
        
        # Detect intent
        intent = self._detect_intent(query_lower)
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Extract keywords
        keywords = self._extract_keywords(query_lower)
        
        # Infer categories
        categories = self._infer_categories(query_lower)
        
        # Extract temporal hints
        temporal_hints = self._extract_temporal(query_lower)
        
        # Extract negations
        negations = self._extract_negations(query_lower)
        
        return ParsedQuery(
            raw_query=query,
            intent=intent,
            entities=entities,
            keywords=keywords,
            categories=categories,
            temporal_hints=temporal_hints,
            negations=negations,
            confidence=self._calculate_confidence(intent, keywords),
        )
    
    def _detect_intent(self, query: str) -> QueryIntent:
        """Detect query intent from patterns."""
        import re
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return intent
        
        # Default to fact lookup
        return QueryIntent.FACT_LOOKUP
    
    def _extract_entities(self, query: str) -> list[str]:
        """Extract named entities (simple implementation)."""
        # In production, use spaCy or similar NER
        import re
        
        entities = []
        
        # Capitalized words (potential names)
        caps = re.findall(r'\b[A-Z][a-z]+\b', query)
        entities.extend(caps)
        
        # Quoted strings
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)
        
        return list(set(entities))
    
    def _extract_keywords(self, query: str) -> list[str]:
        """Extract important keywords."""
        import re
        
        # Remove stop words
        STOP_WORDS = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "what",
            "who", "where", "when", "why", "how", "which", "that", "this",
            "these", "those", "i", "me", "my", "user", "user's", "about",
            "tell", "show", "know", "all", "everything"
        }
        
        tokens = re.findall(r"[a-z']+", query)
        keywords = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
        
        return keywords
    
    def _infer_categories(self, query: str) -> list[str]:
        """Infer relevant categories from query."""
        categories = []
        
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query:
                    categories.append(category)
                    break
        
        return categories if categories else ["fact"]  # Default
    
    def _extract_temporal(self, query: str) -> dict:
        """Extract time-related filters."""
        from datetime import datetime, timedelta
        import re
        
        temporal = {}
        now = datetime.utcnow()
        
        if "yesterday" in query:
            temporal["after"] = now - timedelta(days=1)
        elif "last week" in query:
            temporal["after"] = now - timedelta(weeks=1)
        elif "last month" in query:
            temporal["after"] = now - timedelta(days=30)
        elif "recently" in query or "lately" in query:
            temporal["after"] = now - timedelta(days=7)
        
        return temporal
    
    def _extract_negations(self, query: str) -> list[str]:
        """Extract negated terms."""
        import re
        
        negations = []
        
        # "not X", "don't X", "doesn't X"
        patterns = [
            r"not (\w+)",
            r"don't (\w+)",
            r"doesn't (\w+)",
            r"never (\w+)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            negations.extend(matches)
        
        return negations
    
    def _calculate_confidence(self, intent: QueryIntent, keywords: list[str]) -> float:
        """Calculate parsing confidence."""
        base = 0.5
        
        # More keywords = more confident
        base += min(0.3, len(keywords) * 0.1)
        
        # Specific intent = more confident
        if intent != QueryIntent.FACT_LOOKUP:
            base += 0.1
        
        return min(1.0, base)
```

---

### 2. Search Strategy Selector

**Purpose**: Choose optimal search method based on query characteristics.

```python
# recall/strategy/selector.py

from dataclasses import dataclass
from enum import Enum
from recall.query.parser import ParsedQuery, QueryIntent


class SearchStrategy(Enum):
    VECTOR_ONLY = "vector_only"           # Pure semantic search
    KEYWORD_ONLY = "keyword_only"         # Full-text search
    HYBRID = "hybrid"                     # Vector + keyword fusion
    LABEL_FILTER = "label_filter"         # Filter by label, then search
    CATEGORY_FILTER = "category_filter"   # Filter by category, then search
    EXACT_MATCH = "exact_match"           # Look for exact phrase
    

@dataclass
class SearchPlan:
    primary_strategy: SearchStrategy
    fallback_strategies: list[SearchStrategy]
    vector_weight: float           # Weight for vector results in hybrid
    keyword_weight: float          # Weight for keyword results in hybrid
    label_filter: list[str] | None
    category_filter: list[str] | None
    confidence: float
    

class StrategySelector:
    """Select optimal search strategy based on query."""
    
    def select(self, parsed_query: ParsedQuery) -> SearchPlan:
        """Select best search strategy for query."""
        
        # Determine primary strategy
        if parsed_query.intent == QueryIntent.LIST_ALL:
            primary = SearchStrategy.LABEL_FILTER
            vector_weight, keyword_weight = 0.3, 0.7
        
        elif parsed_query.intent == QueryIntent.FACT_LOOKUP and len(parsed_query.entities) > 0:
            primary = SearchStrategy.EXACT_MATCH
            vector_weight, keyword_weight = 0.2, 0.8
        
        elif parsed_query.intent == QueryIntent.PREFERENCE:
            primary = SearchStrategy.CATEGORY_FILTER
            vector_weight, keyword_weight = 0.6, 0.4
        
        elif len(parsed_query.keywords) <= 2:
            primary = SearchStrategy.VECTOR_ONLY
            vector_weight, keyword_weight = 1.0, 0.0
        
        elif len(parsed_query.keywords) >= 5:
            primary = SearchStrategy.KEYWORD_ONLY
            vector_weight, keyword_weight = 0.0, 1.0
        
        else:
            primary = SearchStrategy.HYBRID
            vector_weight, keyword_weight = 0.5, 0.5
        
        # Determine filters
        label_filter = None
        category_filter = None
        
        if parsed_query.categories:
            category_filter = parsed_query.categories
        
        # Build fallback chain
        fallbacks = self._build_fallbacks(primary)
        
        return SearchPlan(
            primary_strategy=primary,
            fallback_strategies=fallbacks,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            label_filter=label_filter,
            category_filter=category_filter,
            confidence=parsed_query.confidence,
        )
    
    def _build_fallbacks(self, primary: SearchStrategy) -> list[SearchStrategy]:
        """Build fallback strategy chain."""
        all_strategies = [
            SearchStrategy.HYBRID,
            SearchStrategy.VECTOR_ONLY,
            SearchStrategy.KEYWORD_ONLY,
        ]
        return [s for s in all_strategies if s != primary]
```

---

### 3. Search Engine

**Purpose**: Execute searches against the memory store.

```python
# recall/search/engine.py

from dataclasses import dataclass
from typing import Any

from encode.match.embedder import BaseEmbedder
from memory import MemoryClient
from recall.strategy.selector import SearchPlan, SearchStrategy


@dataclass
class SearchResult:
    fact_id: str
    content: str
    category: str
    label_id: str
    label_name: str
    confidence: float
    vector_score: float
    keyword_score: float
    combined_score: float


class SearchEngine:
    """Execute memory searches using various strategies."""
    
    def __init__(self, memory: MemoryClient, embedder: BaseEmbedder):
        self.memory = memory
        self.embedder = embedder
    
    def search(
        self,
        query: str,
        plan: SearchPlan,
        limit: int = 10,
        min_confidence: float = 0.3,
    ) -> list[SearchResult]:
        """Execute search based on plan."""
        
        results = []
        
        if plan.primary_strategy == SearchStrategy.VECTOR_ONLY:
            results = self._vector_search(query, limit * 2, min_confidence)
        
        elif plan.primary_strategy == SearchStrategy.KEYWORD_ONLY:
            results = self._keyword_search(query, limit * 2, min_confidence)
        
        elif plan.primary_strategy == SearchStrategy.HYBRID:
            results = self._hybrid_search(
                query, limit * 2, min_confidence,
                plan.vector_weight, plan.keyword_weight
            )
        
        elif plan.primary_strategy == SearchStrategy.CATEGORY_FILTER:
            results = self._category_filtered_search(
                query, plan.category_filter, limit * 2, min_confidence
            )
        
        elif plan.primary_strategy == SearchStrategy.LABEL_FILTER:
            results = self._label_filtered_search(
                query, plan.label_filter, limit * 2, min_confidence
            )
        
        elif plan.primary_strategy == SearchStrategy.EXACT_MATCH:
            results = self._exact_match_search(query, limit * 2, min_confidence)
        
        # Apply fallbacks if insufficient results
        if len(results) < limit // 2:
            for fallback in plan.fallback_strategies:
                if fallback == SearchStrategy.HYBRID:
                    more = self._hybrid_search(query, limit, min_confidence, 0.5, 0.5)
                elif fallback == SearchStrategy.VECTOR_ONLY:
                    more = self._vector_search(query, limit, min_confidence)
                elif fallback == SearchStrategy.KEYWORD_ONLY:
                    more = self._keyword_search(query, limit, min_confidence)
                else:
                    continue
                
                # Merge results
                existing_ids = {r.fact_id for r in results}
                for r in more:
                    if r.fact_id not in existing_ids:
                        results.append(r)
                
                if len(results) >= limit:
                    break
        
        # Sort by combined score and return top results
        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results[:limit]
    
    def _vector_search(
        self,
        query: str,
        limit: int,
        min_confidence: float,
    ) -> list[SearchResult]:
        """Pure vector similarity search."""
        query_embedding = self.embedder.embed(query)
        
        with self.memory._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        f.id, f.content, f.category, f.confidence,
                        f.label_id, l.name as label_name,
                        1 - (f.embedding <=> %s) as vector_score
                    FROM facts f
                    JOIN labels l ON f.label_id = l.id
                    WHERE f.confidence >= %s
                      AND f.deleted_at IS NULL
                    ORDER BY f.embedding <=> %s
                    LIMIT %s
                    """,
                    (query_embedding, min_confidence, query_embedding, limit),
                )
                rows = cur.fetchall()
        
        return [
            SearchResult(
                fact_id=str(r[0]),
                content=r[1],
                category=r[2],
                confidence=r[3],
                label_id=str(r[4]),
                label_name=r[5],
                vector_score=r[6],
                keyword_score=0.0,
                combined_score=r[6],
            )
            for r in rows
        ]
    
    def _keyword_search(
        self,
        query: str,
        limit: int,
        min_confidence: float,
    ) -> list[SearchResult]:
        """Full-text keyword search."""
        # Convert query to tsquery format
        search_terms = query.lower().split()
        tsquery = " & ".join(search_terms)
        
        with self.memory._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        f.id, f.content, f.category, f.confidence,
                        f.label_id, l.name as label_name,
                        ts_rank(to_tsvector('english', f.content), plainto_tsquery('english', %s)) as keyword_score
                    FROM facts f
                    JOIN labels l ON f.label_id = l.id
                    WHERE f.confidence >= %s
                      AND f.deleted_at IS NULL
                      AND to_tsvector('english', f.content) @@ plainto_tsquery('english', %s)
                    ORDER BY keyword_score DESC
                    LIMIT %s
                    """,
                    (query, min_confidence, query, limit),
                )
                rows = cur.fetchall()
        
        return [
            SearchResult(
                fact_id=str(r[0]),
                content=r[1],
                category=r[2],
                confidence=r[3],
                label_id=str(r[4]),
                label_name=r[5],
                vector_score=0.0,
                keyword_score=r[6],
                combined_score=r[6],
            )
            for r in rows
        ]
    
    def _hybrid_search(
        self,
        query: str,
        limit: int,
        min_confidence: float,
        vector_weight: float,
        keyword_weight: float,
    ) -> list[SearchResult]:
        """Hybrid vector + keyword search with RRF fusion."""
        
        # Get both result sets
        vector_results = self._vector_search(query, limit * 2, min_confidence)
        keyword_results = self._keyword_search(query, limit * 2, min_confidence)
        
        # Reciprocal Rank Fusion
        K = 60  # RRF constant
        scores = {}
        
        for rank, r in enumerate(vector_results):
            scores[r.fact_id] = scores.get(r.fact_id, 0) + vector_weight / (K + rank + 1)
            if r.fact_id not in scores:
                scores[r.fact_id] = {"result": r, "rrf": 0}
        
        for rank, r in enumerate(keyword_results):
            scores[r.fact_id] = scores.get(r.fact_id, 0) + keyword_weight / (K + rank + 1)
        
        # Merge results
        all_results = {r.fact_id: r for r in vector_results + keyword_results}
        
        merged = []
        for fact_id, rrf_score in scores.items():
            result = all_results[fact_id]
            result.combined_score = rrf_score
            merged.append(result)
        
        merged.sort(key=lambda x: x.combined_score, reverse=True)
        return merged[:limit]
    
    def _category_filtered_search(
        self,
        query: str,
        categories: list[str],
        limit: int,
        min_confidence: float,
    ) -> list[SearchResult]:
        """Search within specific categories."""
        query_embedding = self.embedder.embed(query)
        
        with self.memory._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        f.id, f.content, f.category, f.confidence,
                        f.label_id, l.name as label_name,
                        1 - (f.embedding <=> %s) as vector_score
                    FROM facts f
                    JOIN labels l ON f.label_id = l.id
                    WHERE f.confidence >= %s
                      AND f.category = ANY(%s)
                      AND f.deleted_at IS NULL
                    ORDER BY f.embedding <=> %s
                    LIMIT %s
                    """,
                    (query_embedding, min_confidence, categories, query_embedding, limit),
                )
                rows = cur.fetchall()
        
        return [
            SearchResult(
                fact_id=str(r[0]),
                content=r[1],
                category=r[2],
                confidence=r[3],
                label_id=str(r[4]),
                label_name=r[5],
                vector_score=r[6],
                keyword_score=0.0,
                combined_score=r[6],
            )
            for r in rows
        ]
    
    def _label_filtered_search(
        self,
        query: str,
        labels: list[str] | None,
        limit: int,
        min_confidence: float,
    ) -> list[SearchResult]:
        """Search within specific labels."""
        query_embedding = self.embedder.embed(query)
        
        with self.memory._connect() as conn:
            with conn.cursor() as cur:
                if labels:
                    cur.execute(
                        """
                        SELECT 
                            f.id, f.content, f.category, f.confidence,
                            f.label_id, l.name as label_name,
                            1 - (f.embedding <=> %s) as vector_score
                        FROM facts f
                        JOIN labels l ON f.label_id = l.id
                        WHERE f.confidence >= %s
                          AND l.name = ANY(%s)
                          AND f.deleted_at IS NULL
                        ORDER BY f.embedding <=> %s
                        LIMIT %s
                        """,
                        (query_embedding, min_confidence, labels, query_embedding, limit),
                    )
                else:
                    # No label filter, search all
                    return self._vector_search(query, limit, min_confidence)
                
                rows = cur.fetchall()
        
        return [
            SearchResult(
                fact_id=str(r[0]),
                content=r[1],
                category=r[2],
                confidence=r[3],
                label_id=str(r[4]),
                label_name=r[5],
                vector_score=r[6],
                keyword_score=0.0,
                combined_score=r[6],
            )
            for r in rows
        ]
    
    def _exact_match_search(
        self,
        query: str,
        limit: int,
        min_confidence: float,
    ) -> list[SearchResult]:
        """Search for exact or near-exact phrase matches."""
        # First try exact substring match
        with self.memory._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        f.id, f.content, f.category, f.confidence,
                        f.label_id, l.name as label_name,
                        CASE 
                            WHEN lower(f.content) = lower(%s) THEN 1.0
                            WHEN lower(f.content) LIKE lower(%s) THEN 0.9
                            ELSE 0.8
                        END as match_score
                    FROM facts f
                    JOIN labels l ON f.label_id = l.id
                    WHERE f.confidence >= %s
                      AND f.deleted_at IS NULL
                      AND (
                          lower(f.content) = lower(%s)
                          OR lower(f.content) LIKE lower(%s)
                      )
                    ORDER BY match_score DESC
                    LIMIT %s
                    """,
                    (query, f"%{query}%", min_confidence, query, f"%{query}%", limit),
                )
                rows = cur.fetchall()
        
        return [
            SearchResult(
                fact_id=str(r[0]),
                content=r[1],
                category=r[2],
                confidence=r[3],
                label_id=str(r[4]),
                label_name=r[5],
                vector_score=0.0,
                keyword_score=r[6],
                combined_score=r[6],
            )
            for r in rows
        ]
```

---

### 4. Ranker & Filter

**Purpose**: Rerank results for relevance and apply final filters.

```python
# recall/rank/ranker.py

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from recall.search.engine import SearchResult


@dataclass
class RankedResult:
    fact_id: str
    content: str
    category: str
    label: str
    confidence: float
    relevance_score: float
    last_seen: datetime
    evidence_count: int
    explanation: str
    metadata: dict[str, Any] | None


class ResultRanker:
    """Rerank and filter search results."""
    
    def __init__(
        self,
        recency_weight: float = 0.2,
        confidence_weight: float = 0.3,
        relevance_weight: float = 0.5,
    ):
        self.recency_weight = recency_weight
        self.confidence_weight = confidence_weight
        self.relevance_weight = relevance_weight
    
    def rank_and_filter(
        self,
        results: list[SearchResult],
        query_keywords: list[str],
        min_relevance: float = 0.5,
        dedupe: bool = True,
        include_explanation: bool = True,
    ) -> list[RankedResult]:
        """Rerank results with multiple signals."""
        
        # Deduplicate by content similarity
        if dedupe:
            results = self._deduplicate(results)
        
        ranked = []
        now = datetime.utcnow()
        
        for result in results:
            # Calculate recency score (1.0 for today, decays over 30 days)
            # Note: Would need to fetch last_seen from DB in real implementation
            recency_score = 0.8  # Placeholder
            
            # Combined score
            final_score = (
                self.relevance_weight * result.combined_score +
                self.confidence_weight * result.confidence +
                self.recency_weight * recency_score
            )
            
            if final_score < min_relevance:
                continue
            
            # Generate explanation
            explanation = ""
            if include_explanation:
                explanation = self._generate_explanation(result, query_keywords)
            
            ranked.append(RankedResult(
                fact_id=result.fact_id,
                content=result.content,
                category=result.category,
                label=result.label_name,
                confidence=result.confidence,
                relevance_score=final_score,
                last_seen=now,  # Would be from DB
                evidence_count=1,  # Would be from DB
                explanation=explanation,
                metadata=None,
            ))
        
        # Sort by final relevance
        ranked.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return ranked
    
    def _deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
        """Remove near-duplicate results."""
        seen_content = set()
        deduped = []
        
        for result in results:
            # Normalize content for comparison
            normalized = result.content.lower().strip()
            
            # Check if similar content already seen
            is_duplicate = False
            for seen in seen_content:
                if self._is_similar(normalized, seen):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_content.add(normalized)
                deduped.append(result)
        
        return deduped
    
    def _is_similar(self, a: str, b: str, threshold: float = 0.8) -> bool:
        """Check if two strings are similar."""
        # Simple word overlap similarity
        words_a = set(a.split())
        words_b = set(b.split())
        
        if not words_a or not words_b:
            return False
        
        overlap = len(words_a & words_b)
        similarity = overlap / max(len(words_a), len(words_b))
        
        return similarity >= threshold
    
    def _generate_explanation(
        self,
        result: SearchResult,
        query_keywords: list[str],
    ) -> str:
        """Generate human-readable explanation for why result matched."""
        parts = []
        
        # Explain vector match
        if result.vector_score > 0.7:
            parts.append("high semantic similarity to query")
        elif result.vector_score > 0.5:
            parts.append("moderate semantic relevance")
        
        # Explain keyword match
        if result.keyword_score > 0.5:
            matched_keywords = [k for k in query_keywords if k in result.content.lower()]
            if matched_keywords:
                parts.append(f"matched keywords: {', '.join(matched_keywords[:3])}")
        
        # Explain confidence
        if result.confidence > 0.8:
            parts.append("high confidence fact")
        elif result.confidence < 0.4:
            parts.append("low confidence - may need verification")
        
        return "; ".join(parts) if parts else "matched query"
```

---

### 5. Response Formatter

**Purpose**: Build the final response including summary generation.

```python
# recall/format/formatter.py

from dataclasses import dataclass
from typing import Any
import json

from recall.rank.ranker import RankedResult


@dataclass
class RecallResponse:
    success: bool
    query_id: str
    latency_ms: int
    results: list[dict]
    summary: str | None
    total_matches: int
    search_strategy: str
    error: dict | None = None


class ResponseFormatter:
    """Format recall results into API response."""
    
    def __init__(self, llm_summarizer=None):
        self.llm_summarizer = llm_summarizer
    
    def format(
        self,
        results: list[RankedResult],
        query_id: str,
        latency_ms: int,
        strategy: str,
        include_summary: bool = True,
        include_metadata: bool = False,
    ) -> RecallResponse:
        """Format results into response."""
        
        formatted_results = []
        for r in results:
            item = {
                "fact_id": r.fact_id,
                "content": r.content,
                "category": r.category,
                "label": r.label,
                "confidence": round(r.confidence, 3),
                "relevance_score": round(r.relevance_score, 3),
                "last_seen": r.last_seen.isoformat() if r.last_seen else None,
                "evidence_count": r.evidence_count,
                "explanation": r.explanation,
            }
            if include_metadata and r.metadata:
                item["metadata"] = r.metadata
            formatted_results.append(item)
        
        # Generate summary
        summary = None
        if include_summary and results:
            summary = self._generate_summary(results)
        
        return RecallResponse(
            success=True,
            query_id=query_id,
            latency_ms=latency_ms,
            results=formatted_results,
            summary=summary,
            total_matches=len(results),
            search_strategy=strategy,
        )
    
    def format_error(
        self,
        error_code: str,
        message: str,
        query_id: str,
        latency_ms: int,
        details: dict | None = None,
    ) -> RecallResponse:
        """Format error response."""
        return RecallResponse(
            success=False,
            query_id=query_id,
            latency_ms=latency_ms,
            results=[],
            summary=None,
            total_matches=0,
            search_strategy="none",
            error={
                "code": error_code,
                "message": message,
                "details": details or {},
            },
        )
    
    def _generate_summary(self, results: list[RankedResult]) -> str:
        """Generate a natural language summary of results."""
        if not results:
            return "No relevant memories found."
        
        # Group by category
        by_category = {}
        for r in results:
            cat = r.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(r.content)
        
        # Build summary
        parts = []
        
        if "preference" in by_category:
            prefs = by_category["preference"][:3]
            parts.append(f"User preferences: {'; '.join(prefs)}")
        
        if "relationship" in by_category:
            rels = by_category["relationship"][:3]
            parts.append(f"Relationships: {'; '.join(rels)}")
        
        if "goal" in by_category:
            goals = by_category["goal"][:2]
            parts.append(f"Goals: {'; '.join(goals)}")
        
        if "fact" in by_category:
            facts = by_category["fact"][:3]
            parts.append(f"Key facts: {'; '.join(facts)}")
        
        # Add remaining categories
        for cat, items in by_category.items():
            if cat not in {"preference", "relationship", "goal", "fact"}:
                parts.append(f"{cat.title()}: {items[0]}")
        
        return " | ".join(parts) if parts else f"Found {len(results)} relevant memories."
    
    def to_dict(self, response: RecallResponse) -> dict:
        """Convert response to dictionary."""
        return {
            "success": response.success,
            "query_id": response.query_id,
            "latency_ms": response.latency_ms,
            "results": response.results,
            "summary": response.summary,
            "total_matches": response.total_matches,
            "search_strategy": response.search_strategy,
            "error": response.error,
        }
    
    def to_json(self, response: RecallResponse) -> str:
        """Convert response to JSON string."""
        return json.dumps(self.to_dict(response), default=str)
```

---

### 6. Main Recall Engine

**Purpose**: Orchestrate all components.

```python
# recall/engine.py

import time
import uuid
from dataclasses import dataclass
from typing import Any

from encode.match.embedder import BaseEmbedder, get_embedder
from memory import MemoryClient
from recall.query.parser import QueryParser, ParsedQuery
from recall.strategy.selector import StrategySelector, SearchPlan
from recall.search.engine import SearchEngine
from recall.rank.ranker import ResultRanker
from recall.format.formatter import ResponseFormatter, RecallResponse


@dataclass
class RecallOptions:
    limit: int = 10
    min_confidence: float = 0.3
    min_relevance: float = 0.5
    categories: list[str] | None = None
    labels: list[str] | None = None
    time_range: dict | None = None
    include_metadata: bool = False
    include_explanation: bool = True
    include_summary: bool = True


class RecallEngine:
    """Main recall engine orchestrating all components."""
    
    def __init__(
        self,
        memory: MemoryClient | None = None,
        embedder: BaseEmbedder | None = None,
    ):
        self.memory = memory or MemoryClient()
        self.embedder = embedder or get_embedder()
        
        self.parser = QueryParser(self.embedder)
        self.selector = StrategySelector()
        self.search = SearchEngine(self.memory, self.embedder)
        self.ranker = ResultRanker()
        self.formatter = ResponseFormatter()
    
    def recall(
        self,
        query: str,
        context: str = "",
        options: RecallOptions | None = None,
    ) -> RecallResponse:
        """Execute a recall query.
        
        Args:
            query: Natural language query
            context: Current conversation context
            options: Search options
        
        Returns:
            RecallResponse with results and metadata
        """
        start_time = time.time()
        query_id = str(uuid.uuid4())
        options = options or RecallOptions()
        
        try:
            # Validate input
            if not query or not query.strip():
                return self.formatter.format_error(
                    "INVALID_QUERY",
                    "Query cannot be empty",
                    query_id,
                    int((time.time() - start_time) * 1000),
                )
            
            # Parse query
            parsed = self.parser.parse(query, context)
            
            # Override categories/labels from options
            if options.categories:
                parsed.categories = options.categories
            
            # Select search strategy
            plan = self.selector.select(parsed)
            
            # Override label filter from options
            if options.labels:
                plan.label_filter = options.labels
            
            # Execute search
            results = self.search.search(
                query,
                plan,
                limit=options.limit * 2,  # Get extra for filtering
                min_confidence=options.min_confidence,
            )
            
            # Rank and filter
            ranked = self.ranker.rank_and_filter(
                results,
                query_keywords=parsed.keywords,
                min_relevance=options.min_relevance,
                dedupe=True,
                include_explanation=options.include_explanation,
            )
            
            # Limit results
            ranked = ranked[:options.limit]
            
            # Format response
            latency_ms = int((time.time() - start_time) * 1000)
            
            return self.formatter.format(
                ranked,
                query_id,
                latency_ms,
                plan.primary_strategy.value,
                include_summary=options.include_summary,
                include_metadata=options.include_metadata,
            )
        
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return self.formatter.format_error(
                "INTERNAL_ERROR",
                str(e),
                query_id,
                latency_ms,
            )
    
    def recall_for_llm(
        self,
        query: str,
        context: str = "",
        max_tokens: int = 500,
    ) -> str:
        """Recall formatted specifically for LLM context injection.
        
        Returns a compact string suitable for adding to LLM system prompt.
        """
        response = self.recall(
            query,
            context,
            RecallOptions(
                limit=10,
                include_metadata=False,
                include_explanation=False,
                include_summary=True,
            ),
        )
        
        if not response.success or not response.results:
            return ""
        
        # Build compact representation
        lines = ["## User Memory Context"]
        
        if response.summary:
            lines.append(f"Summary: {response.summary}")
            lines.append("")
        
        lines.append("Specific facts:")
        for r in response.results[:5]:  # Top 5 only
            confidence_str = "high" if r["confidence"] > 0.7 else "medium" if r["confidence"] > 0.4 else "low"
            lines.append(f"- [{r['category']}] {r['content']} ({confidence_str} confidence)")
        
        result = "\n".join(lines)
        
        # Truncate if too long (rough token estimate)
        if len(result) > max_tokens * 4:
            result = result[:max_tokens * 4] + "..."
        
        return result


# Convenience function for simple usage
def recall(query: str, context: str = "") -> RecallResponse:
    """Simple recall function."""
    engine = RecallEngine()
    return engine.recall(query, context)


def recall_for_llm(query: str, context: str = "", max_tokens: int = 500) -> str:
    """Get recall results formatted for LLM context."""
    engine = RecallEngine()
    return engine.recall_for_llm(query, context, max_tokens)
```

---

## Usage Examples

### Basic Recall

```python
from recall import RecallEngine, RecallOptions

engine = RecallEngine()

# Simple query
response = engine.recall("What food does the user like?")
print(response.summary)
# "User preferences: likes Italian food; enjoys pasta; prefers home cooking"

for result in response.results:
    print(f"- {result['content']} (confidence: {result['confidence']})")
```

### LLM Integration

```python
from recall import recall_for_llm

# Get memory context for LLM
memory_context = recall_for_llm(
    query="food preferences and dietary restrictions",
    context="User is asking for dinner recommendations",
    max_tokens=500,
)

# Inject into LLM prompt
system_prompt = f"""You are a helpful assistant.

{memory_context}

Use the above context to personalize your responses."""

# Send to LLM...
```

### Filtered Search

```python
from recall import RecallEngine, RecallOptions

engine = RecallEngine()

# Search only preferences
response = engine.recall(
    "What are the user's interests?",
    options=RecallOptions(
        categories=["preference", "hobby"],
        min_confidence=0.5,
        limit=20,
    ),
)
```

### API Endpoint

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from recall import RecallEngine, RecallOptions

app = FastAPI()
engine = RecallEngine()


class RecallRequest(BaseModel):
    query: str
    context: str = ""
    limit: int = 10
    min_confidence: float = 0.3
    categories: list[str] | None = None


@app.post("/api/v1/recall")
def recall_endpoint(req: RecallRequest):
    response = engine.recall(
        req.query,
        req.context,
        RecallOptions(
            limit=req.limit,
            min_confidence=req.min_confidence,
            categories=req.categories,
        ),
    )
    
    if not response.success:
        raise HTTPException(status_code=400, detail=response.error)
    
    return response
```

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| P50 Latency | <50ms | For simple vector search |
| P95 Latency | <200ms | With hybrid search |
| P99 Latency | <500ms | With summary generation |
| Throughput | >100 req/s | Per instance |
| Recall Accuracy | >80% | Relevant facts in top 10 |
| Precision | >70% | Relevant facts / returned facts |

---

## Next Steps

1. **Read**: `04-PRODUCTION-ROADMAP.md` for implementation timeline
2. **Read**: `05-IMPLEMENTATION-GUIDE.md` for step-by-step coding guide
3. **Read**: `06-API-SPECIFICATION.md` for full API documentation
