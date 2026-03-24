# Implementation Guide: Step-by-Step Coding Instructions

## Purpose

This document provides exact, copy-paste-ready code for implementing all components. Any AI can read this and build the complete system.

---

## Prerequisites

Before starting:

```bash
# Clone repository
git clone <repo-url>
cd hippo.c-1

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install base dependencies
pip install -e .
```

---

## Step 1: Update Dependencies

### File: `pyproject.toml`

Replace the entire file:

```toml
[project]
name = "hippo-c-1"
version = "0.3.0"
description = "AI memory system with encode, store, and recall layers"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    # Database
    "psycopg[binary,pool]>=3.1.18",
    "pgvector>=0.2.4",
    
    # Embeddings
    "sentence-transformers>=2.2.0",
    "torch>=2.0.0",
    
    # API
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.23.0",
    "pydantic>=2.0.0",
    
    # Utilities
    "python-dotenv>=1.0.0",
    "structlog>=23.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "httpx>=0.24.0",
    "ruff>=0.1.0",
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

Then install:

```bash
pip install -e ".[dev]"
```

---

## Step 2: Implement Connection Pooling

### File: `memory/client.py`

Replace entirely with:

```python
"""PostgreSQL client with connection pooling for memory storage."""

import os
from contextlib import contextmanager
from typing import Generator
import logging

import psycopg
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

from encode.models import Fact, Label, Prompt

logger = logging.getLogger(__name__)


class MemoryClient:
    """Postgres client for memory storage with connection pooling."""
    
    _pool: ConnectionPool | None = None
    _pool_dsn: str | None = None
    
    def __init__(self, dsn_env: str = "SUPABASE_DATABASE_URL"):
        self.dsn = os.getenv(dsn_env, "")
        if not self.dsn:
            raise ValueError(f"{dsn_env} environment variable is not set")
        self._ensure_pool()
    
    def _ensure_pool(self) -> None:
        """Initialize or reuse connection pool."""
        if MemoryClient._pool is None or MemoryClient._pool_dsn != self.dsn:
            if MemoryClient._pool is not None:
                MemoryClient._pool.close()
            
            logger.info("Initializing database connection pool")
            MemoryClient._pool = ConnectionPool(
                self.dsn,
                min_size=2,
                max_size=10,
                max_idle=300,
                max_lifetime=3600,
                configure=self._configure_connection,
                open=True,
            )
            MemoryClient._pool_dsn = self.dsn
    
    @staticmethod
    def _configure_connection(conn: psycopg.Connection) -> None:
        """Configure each connection with pgvector support."""
        register_vector(conn)
    
    @contextmanager
    def _connect(self) -> Generator[psycopg.Connection, None, None]:
        """Get a connection from the pool."""
        if self._pool is None:
            self._ensure_pool()
        with self._pool.connection() as conn:
            yield conn
    
    def ping(self) -> bool:
        """Test database connectivity."""
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return cur.fetchone() == (1,)
        except Exception as e:
            logger.error(f"Database ping failed: {e}")
            return False
    
    def close(self) -> None:
        """Close the connection pool."""
        if MemoryClient._pool is not None:
            MemoryClient._pool.close()
            MemoryClient._pool = None
            MemoryClient._pool_dsn = None
    
    # === PROMPT OPERATIONS ===
    
    def upsert_prompt(self, prompt: Prompt) -> str:
        """Insert or update a prompt."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO prompts (id, role, text, timestamp, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    RETURNING id
                    """,
                    (prompt.id, prompt.role.value, prompt.text, prompt.timestamp, prompt.metadata),
                )
                row = cur.fetchone()
                conn.commit()
                return str(row[0]) if row else prompt.id
    
    # === LABEL OPERATIONS ===
    
    def upsert_label(
        self,
        name: str,
        kind: str,
        category: str,
        embedding: list[float],
    ) -> Label:
        """Insert or update a label."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO labels (name, kind, category, embedding, usage_count, updated_at)
                    VALUES (%s, %s, %s, %s, 1, now())
                    ON CONFLICT (name, kind, category)
                    DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        usage_count = labels.usage_count + 1,
                        updated_at = now()
                    RETURNING id, name, kind, category, embedding, usage_count, 
                              created_at, updated_at, metadata
                    """,
                    (name, kind, category, embedding),
                )
                row = cur.fetchone()
                conn.commit()
        
        return Label(
            id=str(row[0]),
            name=row[1],
            kind=row[2],
            category=row[3],
            embedding=list(row[4]) if row[4] else [],
            usage_count=row[5],
            created_at=row[6],
            updated_at=row[7],
            metadata=row[8],
        )
    
    def get_label(self, label_id: str) -> Label | None:
        """Get a label by ID."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, kind, category, embedding, usage_count,
                           created_at, updated_at, metadata
                    FROM labels
                    WHERE id = %s AND deleted_at IS NULL
                    """,
                    (label_id,),
                )
                row = cur.fetchone()
        
        if not row:
            return None
        
        return Label(
            id=str(row[0]),
            name=row[1],
            kind=row[2],
            category=row[3],
            embedding=list(row[4]) if row[4] else [],
            usage_count=row[5],
            created_at=row[6],
            updated_at=row[7],
            metadata=row[8],
        )
    
    def search_labels(
        self,
        embedding: list[float],
        top_k: int = 5,
    ) -> list[Label]:
        """Search labels by embedding similarity."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, kind, category, embedding, usage_count,
                           created_at, updated_at, metadata,
                           1 - (embedding <=> %s) as similarity
                    FROM labels
                    WHERE deleted_at IS NULL
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (embedding, embedding, top_k),
                )
                rows = cur.fetchall()
        
        return [
            Label(
                id=str(r[0]),
                name=r[1],
                kind=r[2],
                category=r[3],
                embedding=list(r[4]) if r[4] else [],
                usage_count=r[5],
                created_at=r[6],
                updated_at=r[7],
                metadata=r[8],
            )
            for r in rows
        ]
    
    # === FACT OPERATIONS ===
    
    def insert_fact(self, fact: Fact) -> Fact:
        """Insert a new fact."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO facts (
                        id, label_id, content, category, confidence, source_role,
                        embedding, created_at, updated_at, last_seen_at,
                        evidence_count, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (
                        fact.id, fact.label_id, fact.content, fact.category,
                        fact.confidence, fact.source_role, fact.embedding,
                        fact.created_at, fact.updated_at, fact.last_seen_at,
                        fact.evidence_count, fact.metadata,
                    ),
                )
                conn.commit()
        return fact
    
    def find_best_fact(
        self,
        label_id: str,
        embedding: list[float],
    ) -> Fact | None:
        """Find the most similar fact for a label."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, label_id, content, category, confidence, source_role,
                           embedding, created_at, updated_at, last_seen_at,
                           evidence_count, metadata
                    FROM facts
                    WHERE label_id = %s AND deleted_at IS NULL
                    ORDER BY embedding <=> %s
                    LIMIT 1
                    """,
                    (label_id, embedding),
                )
                row = cur.fetchone()
        
        if not row:
            return None
        
        return Fact(
            id=str(row[0]),
            label_id=str(row[1]),
            content=row[2],
            category=row[3],
            confidence=row[4],
            source_role=row[5],
            embedding=list(row[6]) if row[6] else [],
            created_at=row[7],
            updated_at=row[8],
            last_seen_at=row[9],
            evidence_count=row[10],
            metadata=row[11],
        )
    
    def get_facts_by_label(
        self,
        label_id: str,
        limit: int = 10,
    ) -> list[Fact]:
        """Get facts for a label ordered by confidence."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, label_id, content, category, confidence, source_role,
                           embedding, created_at, updated_at, last_seen_at,
                           evidence_count, metadata
                    FROM facts
                    WHERE label_id = %s AND deleted_at IS NULL
                    ORDER BY confidence DESC, last_seen_at DESC
                    LIMIT %s
                    """,
                    (label_id, limit),
                )
                rows = cur.fetchall()
        
        return [
            Fact(
                id=str(r[0]),
                label_id=str(r[1]),
                content=r[2],
                category=r[3],
                confidence=r[4],
                source_role=r[5],
                embedding=list(r[6]) if r[6] else [],
                created_at=r[7],
                updated_at=r[8],
                last_seen_at=r[9],
                evidence_count=r[10],
                metadata=r[11],
            )
            for r in rows
        ]
    
    def reinforce_fact(self, fact_id: str, boost: float = 0.05) -> None:
        """Reinforce a fact by boosting confidence."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE facts
                    SET confidence = LEAST(1.0, confidence + %s),
                        evidence_count = evidence_count + 1,
                        last_seen_at = now(),
                        updated_at = now()
                    WHERE id = %s
                    """,
                    (boost, fact_id),
                )
                conn.commit()
    
    def search_facts(
        self,
        embedding: list[float],
        top_k: int = 10,
        min_confidence: float = 0.3,
        categories: list[str] | None = None,
    ) -> list[tuple[Fact, float]]:
        """Search facts by embedding similarity with filters."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                if categories:
                    cur.execute(
                        """
                        SELECT f.id, f.label_id, f.content, f.category, f.confidence,
                               f.source_role, f.embedding, f.created_at, f.updated_at,
                               f.last_seen_at, f.evidence_count, f.metadata,
                               l.name as label_name,
                               1 - (f.embedding <=> %s) as similarity
                        FROM facts f
                        JOIN labels l ON f.label_id = l.id
                        WHERE f.confidence >= %s
                          AND f.deleted_at IS NULL
                          AND f.category = ANY(%s)
                        ORDER BY f.embedding <=> %s
                        LIMIT %s
                        """,
                        (embedding, min_confidence, categories, embedding, top_k),
                    )
                else:
                    cur.execute(
                        """
                        SELECT f.id, f.label_id, f.content, f.category, f.confidence,
                               f.source_role, f.embedding, f.created_at, f.updated_at,
                               f.last_seen_at, f.evidence_count, f.metadata,
                               l.name as label_name,
                               1 - (f.embedding <=> %s) as similarity
                        FROM facts f
                        JOIN labels l ON f.label_id = l.id
                        WHERE f.confidence >= %s
                          AND f.deleted_at IS NULL
                        ORDER BY f.embedding <=> %s
                        LIMIT %s
                        """,
                        (embedding, min_confidence, embedding, top_k),
                    )
                rows = cur.fetchall()
        
        results = []
        for r in rows:
            fact = Fact(
                id=str(r[0]),
                label_id=str(r[1]),
                content=r[2],
                category=r[3],
                confidence=r[4],
                source_role=r[5],
                embedding=list(r[6]) if r[6] else [],
                created_at=r[7],
                updated_at=r[8],
                last_seen_at=r[9],
                evidence_count=r[10],
                metadata={**r[11], "label_name": r[12]},
            )
            similarity = r[13]
            results.append((fact, similarity))
        
        return results
    
    def keyword_search_facts(
        self,
        query: str,
        top_k: int = 10,
        min_confidence: float = 0.3,
    ) -> list[tuple[Fact, float]]:
        """Full-text search on facts."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT f.id, f.label_id, f.content, f.category, f.confidence,
                           f.source_role, f.embedding, f.created_at, f.updated_at,
                           f.last_seen_at, f.evidence_count, f.metadata,
                           l.name as label_name,
                           ts_rank(to_tsvector('english', f.content), 
                                   plainto_tsquery('english', %s)) as rank
                    FROM facts f
                    JOIN labels l ON f.label_id = l.id
                    WHERE f.confidence >= %s
                      AND f.deleted_at IS NULL
                      AND to_tsvector('english', f.content) @@ plainto_tsquery('english', %s)
                    ORDER BY rank DESC
                    LIMIT %s
                    """,
                    (query, min_confidence, query, top_k),
                )
                rows = cur.fetchall()
        
        results = []
        for r in rows:
            fact = Fact(
                id=str(r[0]),
                label_id=str(r[1]),
                content=r[2],
                category=r[3],
                confidence=r[4],
                source_role=r[5],
                embedding=list(r[6]) if r[6] else [],
                created_at=r[7],
                updated_at=r[8],
                last_seen_at=r[9],
                evidence_count=r[10],
                metadata={**r[11], "label_name": r[12]},
            )
            rank = r[13]
            results.append((fact, rank))
        
        return results
    
    # === PATTERN OPERATIONS ===
    
    def get_patterns(self) -> list[dict]:
        """Get all patterns."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, regex, category, template, weight, uses, successes,
                           updated_at, metadata
                    FROM patterns
                    ORDER BY weight DESC, updated_at DESC
                    """
                )
                rows = cur.fetchall()
        
        return [
            {
                "id": str(r[0]),
                "regex": r[1],
                "category": r[2],
                "template": r[3],
                "weight": r[4],
                "uses": r[5],
                "successes": r[6],
                "updated_at": r[7],
                "metadata": r[8],
            }
            for r in rows
        ]
    
    def upsert_pattern(
        self,
        regex: str,
        category: str,
        template: str,
        weight: float,
    ) -> None:
        """Insert or update a pattern."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO patterns (regex, category, template, weight, updated_at)
                    VALUES (%s, %s, %s, %s, now())
                    ON CONFLICT (regex)
                    DO UPDATE SET
                        category = EXCLUDED.category,
                        template = EXCLUDED.template,
                        weight = EXCLUDED.weight,
                        updated_at = now()
                    """,
                    (regex, category, template, weight),
                )
                conn.commit()
    
    def record_pattern_use(self, regex: str, success: bool) -> None:
        """Record pattern usage."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE patterns
                    SET uses = uses + 1,
                        successes = successes + %s,
                        updated_at = now()
                    WHERE regex = %s
                    """,
                    (1 if success else 0, regex),
                )
                conn.commit()
    
    # === MAINTENANCE OPERATIONS ===
    
    def decay_confidence(
        self,
        decay_factor: float = 0.99,
        min_confidence: float = 0.1,
        days_threshold: int = 7,
    ) -> int:
        """Apply confidence decay to old facts."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE facts
                    SET confidence = GREATEST(%s, confidence * %s),
                        updated_at = now()
                    WHERE last_seen_at < now() - interval '%s days'
                      AND confidence > %s
                      AND deleted_at IS NULL
                    """,
                    (min_confidence, decay_factor, days_threshold, min_confidence),
                )
                count = cur.rowcount
                conn.commit()
        return count
    
    def soft_delete_fact(self, fact_id: str) -> bool:
        """Soft delete a fact."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE facts
                    SET deleted_at = now()
                    WHERE id = %s AND deleted_at IS NULL
                    """,
                    (fact_id,),
                )
                deleted = cur.rowcount > 0
                conn.commit()
        return deleted
```

---

## Step 3: Implement Quality Embeddings

### File: `encode/match/embedder.py`

Replace entirely with:

```python
"""Embedding implementations for semantic similarity."""

import math
import os
import re
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Sequence
import logging

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract base class for embedders."""
    
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @property
    @abstractmethod
    def dims(self) -> int:
        """Return embedding dimensions."""
        pass
    
    def similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)


class HashEmbedder(BaseEmbedder):
    """Fast hash-based embedder for quick comparisons."""
    
    def __init__(self, dims: int = 256):
        self._dims = dims
    
    @property
    def dims(self) -> int:
        return self._dims
    
    def embed(self, text: str) -> list[float]:
        """Generate hash-based embedding."""
        vec = [0.0] * self._dims
        for token in self._tokenize(text):
            idx = hash(token) % self._dims
            vec[idx] += 1.0
        return self._normalize(vec)
    
    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Batch embedding (just maps over single embed)."""
        return [self.embed(t) for t in texts]
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        return re.findall(r"[a-zA-Z']+", text.lower())
    
    def _normalize(self, vec: list[float]) -> list[float]:
        """L2 normalize vector."""
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]


class SentenceTransformerEmbedder(BaseEmbedder):
    """High-quality embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._dims_cache = None
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            logger.info(f"Loading sentence-transformer model: {self._model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self._dims_cache = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded with {self._dims_cache} dimensions")
    
    @property
    def dims(self) -> int:
        if self._dims_cache is None:
            self._load_model()
        return self._dims_cache
    
    def embed(self, text: str) -> list[float]:
        """Generate high-quality embedding."""
        self._load_model()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Batch embedding for efficiency."""
        self._load_model()
        embeddings = self._model.encode(list(texts), convert_to_numpy=True)
        return [e.tolist() for e in embeddings]
    
    @lru_cache(maxsize=1000)
    def embed_cached(self, text: str) -> tuple[float, ...]:
        """Cached embedding lookup."""
        return tuple(self.embed(text))


class HybridEmbedder(BaseEmbedder):
    """Combines fast hash embeddings with quality transformer embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.fast = HashEmbedder(dims=256)
        self.quality = SentenceTransformerEmbedder(model_name)
    
    @property
    def dims(self) -> int:
        return self.quality.dims
    
    def embed(self, text: str) -> list[float]:
        """Use quality embeddings for storage."""
        return self.quality.embed(text)
    
    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Batch embedding using quality model."""
        return self.quality.embed_batch(texts)
    
    def embed_fast(self, text: str) -> list[float]:
        """Fast embedding for pre-filtering."""
        return self.fast.embed(text)


def get_embedder(quality: str = "auto") -> BaseEmbedder:
    """Factory function to get appropriate embedder.
    
    Args:
        quality: "fast", "quality", "hybrid", or "auto"
                 "auto" uses env var EMBEDDING_QUALITY or defaults to "quality"
    
    Returns:
        Appropriate embedder instance
    """
    if quality == "auto":
        quality = os.getenv("EMBEDDING_QUALITY", "quality")
    
    if quality == "fast":
        return HashEmbedder()
    elif quality == "quality":
        return SentenceTransformerEmbedder()
    elif quality == "hybrid":
        return HybridEmbedder()
    else:
        raise ValueError(f"Unknown embedding quality: {quality}")


# Keep backward compatibility
class HashEmbedder(HashEmbedder):
    """Alias for backward compatibility."""
    pass
```

### File: `encode/match/__init__.py`

Update to:

```python
from .embedder import (
    BaseEmbedder,
    HashEmbedder,
    SentenceTransformerEmbedder,
    HybridEmbedder,
    get_embedder,
)
from .retriever import retrieve_candidates, Candidate

__all__ = [
    "BaseEmbedder",
    "HashEmbedder",
    "SentenceTransformerEmbedder",
    "HybridEmbedder",
    "get_embedder",
    "retrieve_candidates",
    "Candidate",
]
```

---

## Step 4: Create Recall System

### Create Directory Structure

```bash
mkdir -p recall/query recall/strategy recall/search recall/rank recall/format
touch recall/__init__.py recall/query/__init__.py recall/strategy/__init__.py
touch recall/search/__init__.py recall/rank/__init__.py recall/format/__init__.py
```

### File: `recall/query/parser.py`

```python
"""Query parsing and intent detection."""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryIntent(Enum):
    """Types of recall queries."""
    FACT_LOOKUP = "fact_lookup"
    LIST_ALL = "list_all"
    PREFERENCE = "preference"
    RELATIONSHIP = "relationship"
    TEMPORAL = "temporal"
    SUMMARY = "summary"


@dataclass
class ParsedQuery:
    """Structured representation of a query."""
    raw_query: str
    intent: QueryIntent
    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    temporal_hints: dict[str, Any] = field(default_factory=dict)
    negations: list[str] = field(default_factory=list)
    confidence: float = 0.5


class QueryParser:
    """Parse natural language queries into structured format."""
    
    INTENT_PATTERNS = {
        QueryIntent.FACT_LOOKUP: [
            r"what is (the user's|my|user's) (.+)",
            r"tell me (the user's|my|user's) (.+)",
        ],
        QueryIntent.LIST_ALL: [
            r"what do (i|you) know about (.+)",
            r"list (all|everything) (.+)",
            r"show me (.+)",
        ],
        QueryIntent.PREFERENCE: [
            r"what (does user|do i) (like|prefer|enjoy|love|hate)",
            r"(user's|my) (favorite|preferred|favourite)",
            r"preferences",
        ],
        QueryIntent.RELATIONSHIP: [
            r"who is (user's|my) (.+)",
            r"(user's|my) (family|friend|colleague|partner)",
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
    
    CATEGORY_KEYWORDS = {
        "preference": ["like", "love", "prefer", "favorite", "enjoy", "hate", "dislike"],
        "relationship": ["friend", "family", "colleague", "partner", "spouse"],
        "location": ["live", "from", "located", "city", "country", "home"],
        "goal": ["want", "plan", "hope", "aim", "goal", "aspire"],
        "emotion": ["feel", "feeling", "mood", "happy", "sad"],
        "habit": ["always", "usually", "often", "every", "routine"],
        "fact": ["is", "are", "has", "have", "born", "age", "name"],
    }
    
    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "what", "who", "where", "when",
        "why", "how", "which", "that", "this", "these", "those", "i", "me",
        "my", "user", "user's", "about", "tell", "show", "know", "all",
    }
    
    def parse(self, query: str, context: str = "") -> ParsedQuery:
        """Parse a query into structured format."""
        query_lower = query.lower()
        
        intent = self._detect_intent(query_lower)
        entities = self._extract_entities(query)
        keywords = self._extract_keywords(query_lower)
        categories = self._infer_categories(query_lower)
        temporal_hints = self._extract_temporal(query_lower)
        negations = self._extract_negations(query_lower)
        confidence = self._calculate_confidence(intent, keywords)
        
        return ParsedQuery(
            raw_query=query,
            intent=intent,
            entities=entities,
            keywords=keywords,
            categories=categories,
            temporal_hints=temporal_hints,
            negations=negations,
            confidence=confidence,
        )
    
    def _detect_intent(self, query: str) -> QueryIntent:
        """Detect query intent from patterns."""
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return intent
        return QueryIntent.FACT_LOOKUP
    
    def _extract_entities(self, query: str) -> list[str]:
        """Extract named entities."""
        entities = []
        # Capitalized words
        caps = re.findall(r'\b[A-Z][a-z]+\b', query)
        entities.extend(caps)
        # Quoted strings
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)
        return list(set(entities))
    
    def _extract_keywords(self, query: str) -> list[str]:
        """Extract important keywords."""
        tokens = re.findall(r"[a-z']+", query)
        keywords = [t for t in tokens if t not in self.STOP_WORDS and len(t) > 2]
        return keywords
    
    def _infer_categories(self, query: str) -> list[str]:
        """Infer relevant categories from query."""
        categories = []
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query:
                    categories.append(category)
                    break
        return categories if categories else ["fact"]
    
    def _extract_temporal(self, query: str) -> dict:
        """Extract time-related filters."""
        from datetime import datetime, timedelta
        
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
        negations = []
        patterns = [r"not (\w+)", r"don't (\w+)", r"doesn't (\w+)", r"never (\w+)"]
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            negations.extend(matches)
        return negations
    
    def _calculate_confidence(self, intent: QueryIntent, keywords: list[str]) -> float:
        """Calculate parsing confidence."""
        base = 0.5
        base += min(0.3, len(keywords) * 0.1)
        if intent != QueryIntent.FACT_LOOKUP:
            base += 0.1
        return min(1.0, base)
```

### File: `recall/query/__init__.py`

```python
from .parser import QueryParser, ParsedQuery, QueryIntent

__all__ = ["QueryParser", "ParsedQuery", "QueryIntent"]
```

### File: `recall/search/engine.py`

```python
"""Search engine for memory recall."""

from dataclasses import dataclass
from typing import Sequence

from encode.match.embedder import BaseEmbedder
from memory import MemoryClient


@dataclass
class SearchResult:
    """A single search result."""
    fact_id: str
    content: str
    category: str
    label_id: str
    label_name: str
    confidence: float
    vector_score: float
    keyword_score: float
    combined_score: float
    evidence_count: int = 1
    last_seen: str = ""


class SearchEngine:
    """Execute memory searches."""
    
    def __init__(self, memory: MemoryClient, embedder: BaseEmbedder):
        self.memory = memory
        self.embedder = embedder
    
    def search(
        self,
        query: str,
        limit: int = 10,
        min_confidence: float = 0.3,
        categories: list[str] | None = None,
        use_hybrid: bool = True,
    ) -> list[SearchResult]:
        """Execute search with optional hybrid mode."""
        if use_hybrid:
            return self._hybrid_search(query, limit, min_confidence, categories)
        else:
            return self._vector_search(query, limit, min_confidence, categories)
    
    def _vector_search(
        self,
        query: str,
        limit: int,
        min_confidence: float,
        categories: list[str] | None,
    ) -> list[SearchResult]:
        """Pure vector similarity search."""
        query_embedding = self.embedder.embed(query)
        results = self.memory.search_facts(
            query_embedding,
            top_k=limit,
            min_confidence=min_confidence,
            categories=categories,
        )
        
        return [
            SearchResult(
                fact_id=fact.id,
                content=fact.content,
                category=fact.category,
                label_id=fact.label_id,
                label_name=fact.metadata.get("label_name", ""),
                confidence=fact.confidence,
                vector_score=score,
                keyword_score=0.0,
                combined_score=score,
                evidence_count=fact.evidence_count,
                last_seen=fact.last_seen_at.isoformat() if fact.last_seen_at else "",
            )
            for fact, score in results
        ]
    
    def _keyword_search(
        self,
        query: str,
        limit: int,
        min_confidence: float,
    ) -> list[SearchResult]:
        """Full-text keyword search."""
        results = self.memory.keyword_search_facts(
            query,
            top_k=limit,
            min_confidence=min_confidence,
        )
        
        return [
            SearchResult(
                fact_id=fact.id,
                content=fact.content,
                category=fact.category,
                label_id=fact.label_id,
                label_name=fact.metadata.get("label_name", ""),
                confidence=fact.confidence,
                vector_score=0.0,
                keyword_score=score,
                combined_score=score,
                evidence_count=fact.evidence_count,
                last_seen=fact.last_seen_at.isoformat() if fact.last_seen_at else "",
            )
            for fact, score in results
        ]
    
    def _hybrid_search(
        self,
        query: str,
        limit: int,
        min_confidence: float,
        categories: list[str] | None,
    ) -> list[SearchResult]:
        """Hybrid vector + keyword search with RRF fusion."""
        # Get both result sets
        vector_results = self._vector_search(query, limit * 2, min_confidence, categories)
        keyword_results = self._keyword_search(query, limit * 2, min_confidence)
        
        # Reciprocal Rank Fusion
        K = 60
        rrf_scores: dict[str, float] = {}
        all_results: dict[str, SearchResult] = {}
        
        for rank, r in enumerate(vector_results):
            rrf_scores[r.fact_id] = rrf_scores.get(r.fact_id, 0) + 0.5 / (K + rank + 1)
            all_results[r.fact_id] = r
        
        for rank, r in enumerate(keyword_results):
            rrf_scores[r.fact_id] = rrf_scores.get(r.fact_id, 0) + 0.5 / (K + rank + 1)
            if r.fact_id not in all_results:
                all_results[r.fact_id] = r
        
        # Update combined scores and sort
        merged = []
        for fact_id, rrf_score in rrf_scores.items():
            result = all_results[fact_id]
            result.combined_score = rrf_score
            merged.append(result)
        
        merged.sort(key=lambda x: x.combined_score, reverse=True)
        return merged[:limit]
```

### File: `recall/search/__init__.py`

```python
from .engine import SearchEngine, SearchResult

__all__ = ["SearchEngine", "SearchResult"]
```

### File: `recall/engine.py`

```python
"""Main recall engine orchestrating all components."""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any
import logging

from encode.match.embedder import BaseEmbedder, get_embedder
from memory import MemoryClient
from recall.query.parser import QueryParser, ParsedQuery
from recall.search.engine import SearchEngine, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RecallOptions:
    """Options for recall queries."""
    limit: int = 10
    min_confidence: float = 0.3
    min_relevance: float = 0.5
    categories: list[str] | None = None
    labels: list[str] | None = None
    include_metadata: bool = False
    include_explanation: bool = True
    include_summary: bool = True


@dataclass
class RecallResult:
    """A single recall result."""
    fact_id: str
    content: str
    category: str
    label: str
    confidence: float
    relevance_score: float
    last_seen: str
    evidence_count: int
    explanation: str = ""


@dataclass
class RecallResponse:
    """Complete recall response."""
    success: bool
    query_id: str
    latency_ms: int
    results: list[RecallResult]
    summary: str | None = None
    total_matches: int = 0
    error: dict[str, Any] | None = None


class RecallEngine:
    """Main recall engine for memory retrieval."""
    
    def __init__(
        self,
        memory: MemoryClient | None = None,
        embedder: BaseEmbedder | None = None,
    ):
        self.memory = memory or MemoryClient()
        self.embedder = embedder or get_embedder()
        self.parser = QueryParser()
        self.search = SearchEngine(self.memory, self.embedder)
    
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
            RecallResponse with results
        """
        start_time = time.time()
        query_id = str(uuid.uuid4())
        options = options or RecallOptions()
        
        try:
            # Validate input
            if not query or not query.strip():
                return RecallResponse(
                    success=False,
                    query_id=query_id,
                    latency_ms=int((time.time() - start_time) * 1000),
                    results=[],
                    error={"code": "INVALID_QUERY", "message": "Query cannot be empty"},
                )
            
            # Parse query
            parsed = self.parser.parse(query, context)
            
            # Override categories from options
            if options.categories:
                parsed.categories = options.categories
            
            # Execute search
            search_results = self.search.search(
                query,
                limit=options.limit * 2,
                min_confidence=options.min_confidence,
                categories=parsed.categories if parsed.categories else None,
                use_hybrid=True,
            )
            
            # Convert to recall results
            results = self._convert_results(
                search_results,
                parsed,
                options.min_relevance,
                options.include_explanation,
            )
            
            # Limit results
            results = results[:options.limit]
            
            # Generate summary
            summary = None
            if options.include_summary and results:
                summary = self._generate_summary(results)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            return RecallResponse(
                success=True,
                query_id=query_id,
                latency_ms=latency_ms,
                results=results,
                summary=summary,
                total_matches=len(results),
            )
        
        except Exception as e:
            logger.exception(f"Recall error: {e}")
            return RecallResponse(
                success=False,
                query_id=query_id,
                latency_ms=int((time.time() - start_time) * 1000),
                results=[],
                error={"code": "INTERNAL_ERROR", "message": str(e)},
            )
    
    def recall_for_llm(
        self,
        query: str,
        context: str = "",
        max_tokens: int = 500,
    ) -> str:
        """Recall formatted for LLM context injection.
        
        Returns a compact string for adding to LLM system prompt.
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
        
        lines = ["## User Memory Context"]
        
        if response.summary:
            lines.append(f"Summary: {response.summary}")
            lines.append("")
        
        lines.append("Specific facts:")
        for r in response.results[:5]:
            conf_str = "high" if r.confidence > 0.7 else "medium" if r.confidence > 0.4 else "low"
            lines.append(f"- [{r.category}] {r.content} ({conf_str} confidence)")
        
        result = "\n".join(lines)
        
        # Rough token limit (4 chars per token estimate)
        if len(result) > max_tokens * 4:
            result = result[:max_tokens * 4] + "..."
        
        return result
    
    def _convert_results(
        self,
        search_results: list[SearchResult],
        parsed: ParsedQuery,
        min_relevance: float,
        include_explanation: bool,
    ) -> list[RecallResult]:
        """Convert search results to recall results."""
        results = []
        seen_content = set()
        
        for sr in search_results:
            # Deduplicate
            normalized = sr.content.lower().strip()
            if normalized in seen_content:
                continue
            seen_content.add(normalized)
            
            # Calculate final relevance
            relevance = (
                0.5 * sr.combined_score +
                0.3 * sr.confidence +
                0.2 * min(1.0, sr.evidence_count / 5)
            )
            
            if relevance < min_relevance:
                continue
            
            # Generate explanation
            explanation = ""
            if include_explanation:
                explanation = self._generate_explanation(sr, parsed.keywords)
            
            results.append(RecallResult(
                fact_id=sr.fact_id,
                content=sr.content,
                category=sr.category,
                label=sr.label_name,
                confidence=sr.confidence,
                relevance_score=relevance,
                last_seen=sr.last_seen,
                evidence_count=sr.evidence_count,
                explanation=explanation,
            ))
        
        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results
    
    def _generate_explanation(
        self,
        result: SearchResult,
        keywords: list[str],
    ) -> str:
        """Generate explanation for match."""
        parts = []
        
        if result.vector_score > 0.7:
            parts.append("high semantic similarity")
        elif result.vector_score > 0.5:
            parts.append("moderate semantic relevance")
        
        if result.keyword_score > 0.5:
            matched = [k for k in keywords if k in result.content.lower()]
            if matched:
                parts.append(f"matched: {', '.join(matched[:3])}")
        
        if result.confidence > 0.8:
            parts.append("high confidence")
        
        return "; ".join(parts) if parts else "matched query"
    
    def _generate_summary(self, results: list[RecallResult]) -> str:
        """Generate natural language summary."""
        if not results:
            return "No relevant memories found."
        
        by_category: dict[str, list[str]] = {}
        for r in results:
            if r.category not in by_category:
                by_category[r.category] = []
            by_category[r.category].append(r.content)
        
        parts = []
        
        if "preference" in by_category:
            prefs = by_category["preference"][:3]
            parts.append(f"Preferences: {'; '.join(prefs)}")
        
        if "relationship" in by_category:
            rels = by_category["relationship"][:2]
            parts.append(f"Relationships: {'; '.join(rels)}")
        
        if "goal" in by_category:
            goals = by_category["goal"][:2]
            parts.append(f"Goals: {'; '.join(goals)}")
        
        if "fact" in by_category:
            facts = by_category["fact"][:3]
            parts.append(f"Facts: {'; '.join(facts)}")
        
        return " | ".join(parts) if parts else f"Found {len(results)} relevant memories."


# Convenience functions
def recall(query: str, context: str = "") -> RecallResponse:
    """Simple recall function."""
    engine = RecallEngine()
    return engine.recall(query, context)


def recall_for_llm(query: str, context: str = "", max_tokens: int = 500) -> str:
    """Get recall formatted for LLM context."""
    engine = RecallEngine()
    return engine.recall_for_llm(query, context, max_tokens)
```

### File: `recall/__init__.py`

```python
from .engine import (
    RecallEngine,
    RecallOptions,
    RecallResult,
    RecallResponse,
    recall,
    recall_for_llm,
)
from .query import QueryParser, ParsedQuery, QueryIntent
from .search import SearchEngine, SearchResult

__all__ = [
    "RecallEngine",
    "RecallOptions",
    "RecallResult",
    "RecallResponse",
    "recall",
    "recall_for_llm",
    "QueryParser",
    "ParsedQuery",
    "QueryIntent",
    "SearchEngine",
    "SearchResult",
]
```

---

## Step 5: Create API Server

### Create Directory

```bash
mkdir -p api/routes
touch api/__init__.py api/routes/__init__.py
```

### File: `api/main.py`

```python
"""FastAPI application for Hippo.c memory system."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from encode import EncodeEngine
from encode.models import Prompt, Role
from memory import MemoryClient
from recall import RecallEngine, RecallOptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Hippo.c API")
    yield
    logger.info("Shutting down Hippo.c API")


app = FastAPI(
    title="Hippo.c Memory API",
    description="Personal memory system for AI assistants",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Request/Response Models ===

class HealthResponse(BaseModel):
    status: str
    database: bool
    version: str


class RecallRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language query")
    context: str = Field("", description="Conversation context")
    limit: int = Field(10, ge=1, le=50, description="Max results")
    min_confidence: float = Field(0.3, ge=0, le=1, description="Min confidence")
    categories: list[str] | None = Field(None, description="Filter categories")


class RecallResultItem(BaseModel):
    fact_id: str
    content: str
    category: str
    label: str
    confidence: float
    relevance_score: float
    last_seen: str
    evidence_count: int
    explanation: str


class RecallResponse(BaseModel):
    success: bool
    query_id: str
    latency_ms: int
    results: list[RecallResultItem]
    summary: str | None
    total_matches: int


class EncodeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to encode")
    role: str = Field("user", description="Message role")


class EncodeResponse(BaseModel):
    success: bool
    updates: list[dict]
    latency_ms: int


class RecallForLLMRequest(BaseModel):
    query: str = Field(..., min_length=1)
    context: str = ""
    max_tokens: int = Field(500, ge=100, le=2000)


class RecallForLLMResponse(BaseModel):
    context: str


# === Endpoints ===

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check system health."""
    try:
        client = MemoryClient()
        db_ok = client.ping()
    except Exception:
        db_ok = False
    
    return HealthResponse(
        status="healthy" if db_ok else "degraded",
        database=db_ok,
        version="0.3.0",
    )


@app.post("/api/v1/recall", response_model=RecallResponse)
def recall_endpoint(request: RecallRequest):
    """Recall memories based on query."""
    engine = RecallEngine()
    
    response = engine.recall(
        request.query,
        request.context,
        RecallOptions(
            limit=request.limit,
            min_confidence=request.min_confidence,
            categories=request.categories,
        ),
    )
    
    if not response.success:
        raise HTTPException(
            status_code=400,
            detail=response.error or {"message": "Recall failed"},
        )
    
    return RecallResponse(
        success=True,
        query_id=response.query_id,
        latency_ms=response.latency_ms,
        results=[
            RecallResultItem(
                fact_id=r.fact_id,
                content=r.content,
                category=r.category,
                label=r.label,
                confidence=r.confidence,
                relevance_score=r.relevance_score,
                last_seen=r.last_seen,
                evidence_count=r.evidence_count,
                explanation=r.explanation,
            )
            for r in response.results
        ],
        summary=response.summary,
        total_matches=response.total_matches,
    )


@app.post("/api/v1/recall/llm", response_model=RecallForLLMResponse)
def recall_for_llm_endpoint(request: RecallForLLMRequest):
    """Get recall formatted for LLM context injection."""
    engine = RecallEngine()
    context = engine.recall_for_llm(
        request.query,
        request.context,
        request.max_tokens,
    )
    return RecallForLLMResponse(context=context)


@app.post("/api/v1/encode", response_model=EncodeResponse)
def encode_endpoint(request: EncodeRequest):
    """Encode text into memory."""
    import time
    
    start = time.time()
    
    try:
        engine = EncodeEngine()
        
        role = Role.USER
        if request.role.lower() == "ai":
            role = Role.AI
        elif request.role.lower() == "system":
            role = Role.SYSTEM
        
        prompt = Prompt(text=request.text, role=role)
        updates = engine.process([prompt])
        
        latency_ms = int((time.time() - start) * 1000)
        
        return EncodeResponse(
            success=True,
            updates=[
                {
                    "action": u.action,
                    "entity": u.entity,
                    "entity_id": u.entity_id,
                    "confidence": u.confidence,
                    "reason": u.reason,
                }
                for u in updates
            ],
            latency_ms=latency_ms,
        )
    
    except Exception as e:
        logger.exception(f"Encode error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### File: `api/__init__.py`

```python
from .main import app

__all__ = ["app"]
```

---

## Step 6: Run and Test

### Start the API Server

```bash
# Set environment variables
export SUPABASE_DATABASE_URL="postgresql://..."
export EMBEDDING_QUALITY="quality"  # or "fast" for development

# Run the server
python -m uvicorn api.main:app --reload --port 8000
```

### Test Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Encode a message
curl -X POST http://localhost:8000/api/v1/encode \
  -H "Content-Type: application/json" \
  -d '{"text": "I really love Italian food, especially pasta."}'

# Recall memories
curl -X POST http://localhost:8000/api/v1/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "What food does the user like?"}'

# Recall for LLM
curl -X POST http://localhost:8000/api/v1/recall/llm \
  -H "Content-Type: application/json" \
  -d '{"query": "food preferences", "max_tokens": 300}'
```

---

## Next Steps

After completing these steps:

1. Run the test suite: `pytest tests/ -v`
2. Review `06-API-SPECIFICATION.md` for full API documentation
3. Review `07-PERFORMANCE-OPTIMIZATION.md` for tuning
4. Review `08-SECURITY-GUIDE.md` for security hardening
