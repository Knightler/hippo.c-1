# Detailed Recommendations: Hippo.c Memory System

## Priority Matrix

| Priority | Issue | Impact | Effort | ROI |
|----------|-------|--------|--------|-----|
| P0 | Build Recall System | Critical | High | ∞ |
| P0 | Connection Pooling | High | Low | Very High |
| P0 | Quality Embeddings | Critical | Medium | Very High |
| P1 | Error Handling | High | Low | High |
| P1 | Confidence Decay | Medium | Low | High |
| P1 | Contradiction Detection | Medium | Medium | High |
| P2 | Batch Processing | Medium | Medium | Medium |
| P2 | Caching Layer | Medium | Medium | Medium |
| P2 | User Authentication | High | High | Medium |
| P3 | Temporal Awareness | Low | High | Low |
| P3 | GDPR Compliance | High | High | Low |

---

## P0: Critical Issues

### 1. Build Recall System

**Problem**: The entire retrieval layer is missing. Without it, stored memories cannot be used.

**Impact**: System is currently write-only. 100% blocker for any use case.

**Solution**: See `03-RECALL-SYSTEM-DESIGN.md` for complete design.

**Quick Summary**:
```python
# Recall API endpoint
POST /recall
{
    "query": "What food preferences does the user have?",
    "context": "User is asking about dinner recommendations",
    "limit": 10,
    "filters": {
        "categories": ["preference", "food"],
        "min_confidence": 0.5
    }
}

# Response
{
    "memories": [
        {
            "fact_id": "uuid",
            "content": "likes Italian food",
            "category": "preference",
            "confidence": 0.85,
            "relevance_score": 0.92,
            "label": "food_preferences",
            "last_seen": "2024-01-15T10:30:00Z"
        }
    ],
    "context_summary": "User prefers Italian food, especially pasta..."
}
```

---

### 2. Connection Pooling

**Problem**: Every database operation creates a new connection.

**Current Code** (`memory/client.py`):
```python
def _connect(self):
    conn = psycopg.connect(self.dsn)  # 50-100ms EVERY call!
    register_vector(conn)
    return conn
```

**Impact**:
- 50-100ms overhead per operation
- Connection storms under load
- Database resource exhaustion

**Solution**:

```python
# memory/client.py - UPDATED

import os
from contextlib import contextmanager
from typing import Generator

import psycopg
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector


class MemoryClient:
    """Postgres client for memory storage with connection pooling."""
    
    _pool: ConnectionPool | None = None
    
    def __init__(self, dsn_env: str = "SUPABASE_DATABASE_URL"):
        self.dsn = os.getenv(dsn_env, "")
        if not self.dsn:
            raise ValueError("SUPABASE_DATABASE_URL is not set")
        self._ensure_pool()
    
    def _ensure_pool(self) -> None:
        """Initialize connection pool if not exists."""
        if MemoryClient._pool is None:
            MemoryClient._pool = ConnectionPool(
                self.dsn,
                min_size=2,
                max_size=10,
                max_idle=300,  # Close idle connections after 5 minutes
                max_lifetime=3600,  # Recycle connections every hour
                configure=self._configure_connection,
            )
    
    def _configure_connection(self, conn: psycopg.Connection) -> None:
        """Configure each connection in the pool."""
        register_vector(conn)
    
    @contextmanager
    def _connect(self) -> Generator[psycopg.Connection, None, None]:
        """Get connection from pool."""
        with self._pool.connection() as conn:
            yield conn
    
    def ping(self) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("select 1")
                return cur.fetchone() == (1,)
    
    # ... rest of methods stay the same but use context manager
```

**Dependencies to Add**:
```toml
# pyproject.toml
dependencies = [
    "psycopg[binary,pool]>=3.1.18",  # Add pool extra
    "pgvector>=0.2.4",
]
```

**Expected Improvement**:
| Metric | Before | After |
|--------|--------|-------|
| Per-operation latency | 50-100ms | 1-5ms |
| Max concurrent ops | ~50 | ~1000 |
| DB connections used | Unlimited | 10 max |

---

### 3. Quality Embeddings

**Problem**: Hash-based embeddings have no semantic understanding.

**Current Code** (`encode/match/embedder.py`):
```python
def embed(self, text: str) -> list[float]:
    vec = [0.0] * 256
    for token in self._tokenize(text):
        idx = hash(token) % 256  # Massive collision rate!
        vec[idx] += 1.0
    return self._normalize(vec)
```

**Impact**:
- "I love cats" ≠ "I adore felines" (should be similar)
- "Bank (river)" = "Bank (money)" (should be different)
- Recall accuracy will be <50%

**Solution Options**:

#### Option A: Local Embedding Model (Recommended)

```python
# encode/match/embedder.py - NEW IMPLEMENTATION

import os
from abc import ABC, abstractmethod
from functools import lru_cache
import math
import re


class BaseEmbedder(ABC):
    """Abstract base for embedding implementations."""
    
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        pass
    
    @property
    @abstractmethod
    def dims(self) -> int:
        pass
    
    def similarity(self, a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors."""
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class HashEmbedder(BaseEmbedder):
    """Cheap, fast hash-based embedder for pre-filtering."""
    
    def __init__(self, dims: int = 256):
        self._dims = dims
    
    @property
    def dims(self) -> int:
        return self._dims
    
    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dims
        for token in self._tokenize(text):
            idx = hash(token) % self._dims
            vec[idx] += 1.0
        return self._normalize(vec)
    
    def _normalize(self, vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]
    
    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z']+", text.lower())


class SentenceTransformerEmbedder(BaseEmbedder):
    """Quality embedder using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Lazy load to avoid import overhead
        self._model_name = model_name
        self._model = None
        self._dims_cache = None
    
    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self._dims_cache = self._model.get_sentence_embedding_dimension()
    
    @property
    def dims(self) -> int:
        if self._dims_cache is None:
            self._load_model()
        return self._dims_cache
    
    def embed(self, text: str) -> list[float]:
        self._load_model()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    @lru_cache(maxsize=1000)
    def embed_cached(self, text: str) -> tuple[float, ...]:
        """Cached version for repeated lookups."""
        return tuple(self.embed(text))


class HybridEmbedder(BaseEmbedder):
    """Two-stage embedder: fast hash for filtering, quality for ranking."""
    
    def __init__(self):
        self.fast = HashEmbedder(dims=256)
        self.quality = SentenceTransformerEmbedder()
    
    @property
    def dims(self) -> int:
        return self.quality.dims
    
    def embed(self, text: str) -> list[float]:
        """Use quality embeddings for storage."""
        return self.quality.embed(text)
    
    def embed_fast(self, text: str) -> list[float]:
        """Use fast embeddings for pre-filtering."""
        return self.fast.embed(text)


def get_embedder(quality: str = "auto") -> BaseEmbedder:
    """Factory function to get appropriate embedder.
    
    Args:
        quality: "fast", "quality", "hybrid", or "auto"
                 "auto" uses env var EMBEDDING_QUALITY or defaults to "hybrid"
    """
    if quality == "auto":
        quality = os.getenv("EMBEDDING_QUALITY", "hybrid")
    
    if quality == "fast":
        return HashEmbedder()
    elif quality == "quality":
        return SentenceTransformerEmbedder()
    elif quality == "hybrid":
        return HybridEmbedder()
    else:
        raise ValueError(f"Unknown embedding quality: {quality}")
```

**Dependencies**:
```toml
# pyproject.toml
dependencies = [
    "psycopg[binary,pool]>=3.1.18",
    "pgvector>=0.2.4",
    "sentence-transformers>=2.2.0",  # ~400MB model download
]
```

**Schema Update** (for different embedding dimensions):
```sql
-- Update to 384 dims for all-MiniLM-L6-v2
ALTER TABLE labels ALTER COLUMN embedding TYPE vector(384);
ALTER TABLE facts ALTER COLUMN embedding TYPE vector(384);

-- Recreate indexes
DROP INDEX IF EXISTS idx_labels_embedding;
DROP INDEX IF EXISTS idx_facts_embedding;
CREATE INDEX idx_labels_embedding ON labels USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_facts_embedding ON facts USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

#### Option B: API-Based Embeddings (Alternative)

```python
# encode/match/embedder_api.py

import json
import os
import urllib.request


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI API-based embedder."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model = "text-embedding-3-small"  # 1536 dims, $0.02/1M tokens
        self._dims = 1536
    
    @property
    def dims(self) -> int:
        return self._dims
    
    def embed(self, text: str) -> list[float]:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        payload = {"input": text, "model": self.model}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        
        return body["data"][0]["embedding"]
```

**Recommendation**: Use **Option A (SentenceTransformers)** for:
- Zero ongoing cost
- Lower latency (no network)
- Works offline
- Good enough quality for personal memory

Use **Option B (OpenAI)** if:
- Quality is paramount
- Cost is not a concern
- Local compute is limited

---

## P1: High Priority Issues

### 4. Error Handling

**Problem**: Errors are silently swallowed, making debugging impossible.

**Current Code** (`encode/infer/llm.py`):
```python
try:
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    return json.loads(content)
except Exception:
    return []  # Silent failure!
```

**Solution**:

```python
# encode/infer/llm.py - IMPROVED

import json
import logging
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    facts: list[dict]
    success: bool
    error: str | None = None
    latency_ms: int = 0
    tokens_used: int = 0


class LLMExtractor:
    """LLM extraction with proper error handling and retry."""
    
    def __init__(
        self,
        api_key_env: str = "DEEPSEEK_API_KEY",
        base_env: str = "DEEPSEEK_API_BASE",
        model_env: str = "DEEPSEEK_MODEL",
        max_retries: int = 3,
        timeout: int = 20,
    ):
        self.api_key = os.getenv(api_key_env, "")
        self.base = os.getenv(base_env, "https://api.deepseek.com/v1")
        self.model = os.getenv(model_env, "deepseek-chat")
        self.max_retries = max_retries
        self.timeout = timeout
    
    def enabled(self) -> bool:
        return bool(self.api_key)
    
    def extract(self, text: str, context: list[dict]) -> list[dict]:
        """Extract facts with retry logic."""
        result = self.extract_with_metadata(text, context)
        return result.facts
    
    def extract_with_metadata(self, text: str, context: list[dict]) -> ExtractionResult:
        """Extract facts with full metadata."""
        if not self.enabled():
            return ExtractionResult(facts=[], success=True, error="LLM disabled")
        
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                facts = self._do_extract(text, context)
                latency_ms = int((time.time() - start_time) * 1000)
                return ExtractionResult(
                    facts=facts,
                    success=True,
                    latency_ms=latency_ms,
                )
            except urllib.error.HTTPError as e:
                last_error = f"HTTP {e.code}: {e.reason}"
                if e.code == 429:  # Rate limited
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                elif e.code >= 500:  # Server error
                    wait_time = 2 ** attempt
                    logger.warning(f"Server error, retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    break  # Client error, don't retry
            except urllib.error.URLError as e:
                last_error = f"Network error: {e.reason}"
                wait_time = 2 ** attempt
                logger.warning(f"Network error, retrying in {wait_time}s")
                time.sleep(wait_time)
            except json.JSONDecodeError as e:
                last_error = f"Invalid JSON response: {e}"
                logger.error(f"LLM returned invalid JSON: {e}")
                break  # Don't retry JSON errors
            except Exception as e:
                last_error = str(e)
                logger.exception(f"Unexpected error in LLM extraction")
                break
        
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"LLM extraction failed after {self.max_retries} attempts: {last_error}")
        return ExtractionResult(
            facts=[],
            success=False,
            error=last_error,
            latency_ms=latency_ms,
        )
    
    def _do_extract(self, text: str, context: list[dict]) -> list[dict]:
        """Single extraction attempt."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": _user_prompt(text, context)},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},  # Force JSON output
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        
        content = body["choices"][0]["message"]["content"]
        result = json.loads(content)
        
        # Validate structure
        if isinstance(result, dict) and "facts" in result:
            return self._validate_facts(result["facts"])
        elif isinstance(result, list):
            return self._validate_facts(result)
        else:
            logger.warning(f"Unexpected response structure: {type(result)}")
            return []
    
    def _validate_facts(self, facts: list[Any]) -> list[dict]:
        """Validate and clean extracted facts."""
        valid = []
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            if "content" not in fact:
                continue
            if not fact["content"].strip():
                continue
            valid.append({
                "content": fact["content"].strip(),
                "category": fact.get("category", "fact"),
                "label": fact.get("label", "general"),
                "confidence": min(1.0, max(0.0, fact.get("confidence", 0.5))),
            })
        return valid


def _system_prompt() -> str:
    return """You are a memory extraction system. Extract personal facts from user messages.

Rules:
1. Only extract factual information about the user
2. Each fact should be atomic (one piece of information)
3. Use present tense for current states, past tense for events
4. Categorize facts: preference, relationship, emotion, goal, location, habit, opinion, fact
5. Assign confidence 0.0-1.0 based on how explicit the statement is

Return JSON with format:
{
    "facts": [
        {
            "content": "likes Italian food",
            "category": "preference",
            "label": "food_preferences",
            "confidence": 0.8
        }
    ]
}

If no extractable facts, return {"facts": []}"""


def _user_prompt(text: str, context: list[dict]) -> str:
    return json.dumps({
        "message": text,
        "existing_context": context,
        "instruction": "Extract new facts from the message. Don't repeat existing context."
    })
```

---

### 5. Confidence Decay

**Problem**: Fact confidence only increases, never decreases.

**Impact**: Old, potentially outdated facts maintain high confidence forever.

**Solution**:

```python
# memory/client.py - ADD METHOD

def decay_confidence(self, decay_factor: float = 0.99, min_confidence: float = 0.1) -> int:
    """Apply time-based decay to all facts not seen recently.
    
    Args:
        decay_factor: Multiply confidence by this (0.99 = 1% decay)
        min_confidence: Floor for confidence values
    
    Returns:
        Number of facts updated
    """
    with self._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE facts
                SET confidence = GREATEST(%s, confidence * %s),
                    updated_at = now()
                WHERE last_seen_at < now() - interval '7 days'
                  AND confidence > %s
                RETURNING id
                """,
                (min_confidence, decay_factor, min_confidence),
            )
            return cur.rowcount
```

**Scheduled Job** (run daily):
```python
# scripts/maintenance.py

from memory import MemoryClient

def run_daily_maintenance():
    client = MemoryClient()
    
    # Decay old facts
    decayed = client.decay_confidence(decay_factor=0.99)
    print(f"Decayed {decayed} facts")
    
    # Archive very old, low-confidence facts
    archived = client.archive_stale_facts(
        min_age_days=90,
        max_confidence=0.2,
    )
    print(f"Archived {archived} facts")

if __name__ == "__main__":
    run_daily_maintenance()
```

---

### 6. Contradiction Detection

**Problem**: Conflicting facts can coexist ("likes coffee" + "hates coffee").

**Solution**:

```python
# encode/update/conflict.py

from dataclasses import dataclass
from encode.match.embedder import BaseEmbedder
from memory import MemoryClient


@dataclass
class ConflictResult:
    has_conflict: bool
    conflicting_fact_id: str | None = None
    conflicting_content: str | None = None
    resolution: str = "none"  # none, supersede, merge, reject


class ConflictDetector:
    """Detect and resolve contradictory facts."""
    
    # Words that indicate opposite meanings
    NEGATION_PAIRS = [
        ({"like", "love", "enjoy", "prefer"}, {"hate", "dislike", "detest", "avoid"}),
        ({"am", "is", "are"}, {"not", "isn't", "aren't", "am not"}),
        ({"want", "need", "wish"}, {"don't want", "don't need"}),
        ({"always", "often"}, {"never", "rarely"}),
    ]
    
    def __init__(self, embedder: BaseEmbedder, memory: MemoryClient, threshold: float = 0.7):
        self.embedder = embedder
        self.memory = memory
        self.threshold = threshold
    
    def check_conflict(self, new_content: str, label_id: str) -> ConflictResult:
        """Check if new fact conflicts with existing facts."""
        existing_facts = self.memory.get_facts_by_label(label_id, limit=20)
        
        new_embedding = self.embedder.embed(new_content)
        new_tokens = set(new_content.lower().split())
        
        for fact in existing_facts:
            # Check semantic similarity (similar topic)
            similarity = self.embedder.similarity(
                new_embedding,
                self.embedder.embed(fact.content)
            )
            
            if similarity < self.threshold:
                continue  # Not related enough to conflict
            
            # Check for negation patterns
            fact_tokens = set(fact.content.lower().split())
            
            if self._has_negation_conflict(new_tokens, fact_tokens):
                return ConflictResult(
                    has_conflict=True,
                    conflicting_fact_id=fact.id,
                    conflicting_content=fact.content,
                    resolution="supersede" if self._is_newer_more_reliable(new_content, fact) else "reject",
                )
        
        return ConflictResult(has_conflict=False)
    
    def _has_negation_conflict(self, tokens1: set, tokens2: set) -> bool:
        """Check if token sets have negation conflict."""
        for positive, negative in self.NEGATION_PAIRS:
            if (tokens1 & positive and tokens2 & negative) or \
               (tokens1 & negative and tokens2 & positive):
                return True
        return False
    
    def _is_newer_more_reliable(self, new_content: str, existing_fact) -> bool:
        """Determine if new fact should supersede existing."""
        # Simple heuristic: newer always wins for preferences/emotions
        # Could be made more sophisticated with explicit temporal signals
        return existing_fact.category in {"preference", "emotion", "goal"}


# Integration in updater.py
class MemoryUpdater:
    def __init__(self, embedder, memory, conflict_detector=None):
        self.embedder = embedder
        self.memory = memory
        self.conflict_detector = conflict_detector or ConflictDetector(embedder, memory)
    
    def apply(self, prompt, extracted, memory):
        updates = []
        
        for item in extracted:
            label = memory.upsert_label(...)
            
            # Check for conflicts BEFORE inserting
            conflict = self.conflict_detector.check_conflict(
                item["content"],
                label.id
            )
            
            if conflict.has_conflict:
                if conflict.resolution == "supersede":
                    # Mark old fact as superseded
                    memory.supersede_fact(
                        conflict.conflicting_fact_id,
                        new_content=item["content"]
                    )
                    # Continue with insert
                elif conflict.resolution == "reject":
                    updates.append(MemoryUpdate(
                        action="skip",
                        reason="conflicting_fact_exists",
                        payload={"conflict": conflict.conflicting_content},
                    ))
                    continue
            
            # Normal insert/update logic...
```

---

## P2: Medium Priority Issues

### 7. Batch Processing

**Problem**: Prompts processed one at a time, sequential DB operations.

**Solution**:

```python
# encode/encode.py - BATCH VERSION

class EncodeEngine:
    def process_batch(self, prompts: list[Prompt], batch_size: int = 10) -> list[MemoryUpdate]:
        """Process prompts in batches for efficiency."""
        updates = []
        
        # Pre-compute all embeddings in batch
        texts = [p.text for p in prompts]
        embeddings = self.embedder.embed_batch(texts)
        
        # Batch triage
        triage_results = [triage(p, self.patterns) for p in prompts]
        
        # Batch candidate retrieval
        all_candidates = self.memory.batch_search_labels(
            embeddings, 
            top_k=5
        )
        
        # Process each prompt with pre-computed data
        for i, prompt in enumerate(prompts):
            updates.extend(self._process_with_cache(
                prompt,
                triage_results[i],
                all_candidates[i],
                embeddings[i]
            ))
        
        return updates
```

---

### 8. Caching Layer

**Problem**: Same labels/facts queried repeatedly.

**Solution**:

```python
# memory/cache.py

from functools import lru_cache
from typing import Any
import time


class MemoryCache:
    """In-memory cache for frequently accessed data."""
    
    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self._label_cache: dict[str, tuple[Any, float]] = {}
        self._fact_cache: dict[str, tuple[Any, float]] = {}
    
    def get_label(self, label_id: str) -> Any | None:
        if label_id in self._label_cache:
            value, timestamp = self._label_cache[label_id]
            if time.time() - timestamp < self.ttl:
                return value
            del self._label_cache[label_id]
        return None
    
    def set_label(self, label_id: str, value: Any) -> None:
        self._label_cache[label_id] = (value, time.time())
    
    def invalidate_label(self, label_id: str) -> None:
        self._label_cache.pop(label_id, None)
    
    # Similar for facts...
    
    def clear(self) -> None:
        self._label_cache.clear()
        self._fact_cache.clear()


# Integration
class MemoryClient:
    def __init__(self, dsn_env: str = "SUPABASE_DATABASE_URL"):
        self.dsn = os.getenv(dsn_env, "")
        self.cache = MemoryCache(ttl_seconds=300)
    
    def get_label(self, label_id: str) -> Label | None:
        # Try cache first
        cached = self.cache.get_label(label_id)
        if cached:
            return cached
        
        # Fetch from DB
        with self._connect() as conn:
            # ... query
            pass
        
        # Cache result
        self.cache.set_label(label_id, label)
        return label
```

---

### 9. User Authentication

**Problem**: No user isolation; all data in single namespace.

**Solution**: See `08-SECURITY-GUIDE.md` for full implementation.

Quick Schema Change:
```sql
-- Add user_id to all tables
ALTER TABLE labels ADD COLUMN user_id uuid NOT NULL;
ALTER TABLE facts ADD COLUMN user_id uuid NOT NULL;
ALTER TABLE prompts ADD COLUMN user_id uuid NOT NULL;
ALTER TABLE patterns ADD COLUMN user_id uuid;  -- nullable for global patterns

-- Create indexes
CREATE INDEX idx_labels_user ON labels(user_id);
CREATE INDEX idx_facts_user ON facts(user_id);
CREATE INDEX idx_prompts_user ON prompts(user_id);

-- Update queries to always filter by user
-- Example:
SELECT * FROM facts 
WHERE user_id = $1 
  AND label_id = $2 
ORDER BY confidence DESC;
```

---

## P3: Lower Priority Issues

### 10. Temporal Awareness

**Problem**: No concept of when facts were true.

**Solution**:
```sql
ALTER TABLE facts ADD COLUMN valid_from timestamptz;
ALTER TABLE facts ADD COLUMN valid_until timestamptz;
ALTER TABLE facts ADD COLUMN temporal_type text DEFAULT 'permanent';
-- Types: permanent, current_state, past_event, future_plan
```

### 11. GDPR Compliance

**Problem**: No data export/deletion capabilities.

**Solution**: See `08-SECURITY-GUIDE.md`

---

## Quick Wins (Implement Today)

1. **Connection Pooling**: 30 min implementation, 10-50x latency improvement
2. **Logging**: Add logging to all components, 1 hour
3. **Error Handling**: Add try/catch with logging, 2 hours
4. **Health Check Endpoint**: 30 min, critical for production

```python
# Simple health check
def health_check() -> dict:
    checks = {
        "database": False,
        "llm": False,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    try:
        client = MemoryClient()
        checks["database"] = client.ping()
    except Exception as e:
        checks["database_error"] = str(e)
    
    try:
        llm = LLMExtractor()
        checks["llm"] = llm.enabled()
    except Exception as e:
        checks["llm_error"] = str(e)
    
    return checks
```

---

## Next Document

Continue to `03-RECALL-SYSTEM-DESIGN.md` for the complete recall system architecture.
