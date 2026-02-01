# Performance Optimization Guide

## Overview

This document provides detailed guidance for optimizing Hippo.c for production performance.

---

## Performance Targets

| Metric | Target | Acceptable | Current* |
|--------|--------|------------|----------|
| Encode P50 | <100ms | <200ms | ~150ms |
| Encode P95 | <500ms | <1000ms | ~5000ms |
| Recall P50 | <50ms | <100ms | ~200ms |
| Recall P95 | <200ms | <500ms | N/A |
| Throughput | >100 req/s | >50 req/s | ~20 req/s |
| Embedding | <100ms | <200ms | ~80ms |
| DB Query | <20ms | <50ms | ~100ms |

*Estimated based on prototype analysis

---

## Optimization Layers

```
┌────────────────────────────────────────────────────────────┐
│ Layer 1: Application Code                                   │
│ - Algorithm optimization                                    │
│ - Memory management                                         │
│ - Async processing                                          │
├────────────────────────────────────────────────────────────┤
│ Layer 2: Database                                           │
│ - Index optimization                                        │
│ - Query optimization                                        │
│ - Connection pooling                                        │
├────────────────────────────────────────────────────────────┤
│ Layer 3: Embeddings                                         │
│ - Model selection                                           │
│ - Batching                                                  │
│ - Caching                                                   │
├────────────────────────────────────────────────────────────┤
│ Layer 4: Infrastructure                                     │
│ - Hardware resources                                        │
│ - Network optimization                                      │
│ - Load balancing                                            │
└────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Application Code

### 1.1 Connection Pooling (Critical)

**Impact**: 10-50x latency reduction

**Problem**: Creating a new connection per operation.

**Solution**: Already implemented in 05-IMPLEMENTATION-GUIDE.md

```python
# Pool configuration tuning
ConnectionPool(
    dsn,
    min_size=2,         # Keep 2 warm connections
    max_size=10,        # Scale up to 10 under load
    max_idle=300,       # Close idle after 5 min
    max_lifetime=3600,  # Recycle every hour
)
```

**Tuning Guidelines**:

| Workload | min_size | max_size | max_idle |
|----------|----------|----------|----------|
| Light (<10 req/s) | 1 | 5 | 600 |
| Medium (10-50 req/s) | 2 | 10 | 300 |
| Heavy (>50 req/s) | 5 | 20 | 120 |

### 1.2 Async Processing

**Impact**: 2-3x throughput improvement

**Current**: Synchronous processing
**Target**: Async where beneficial

```python
# api/main.py - Async version

from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Create executor for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=4)


async def run_in_executor(func, *args):
    """Run sync function in thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)


@app.post("/api/v1/recall")
async def recall_endpoint(request: RecallRequest):
    """Async recall endpoint."""
    engine = RecallEngine()
    
    # Run recall in thread pool (embeddings are CPU-bound)
    response = await run_in_executor(
        engine.recall,
        request.query,
        request.context,
        RecallOptions(limit=request.limit),
    )
    
    return response
```

### 1.3 Batch Processing

**Impact**: 2-5x throughput for bulk operations

```python
# encode/match/embedder.py

class SentenceTransformerEmbedder(BaseEmbedder):
    def embed_batch(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        """Efficient batch embedding."""
        self._load_model()
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self._model.encode(
                batch,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=False,
            )
            all_embeddings.extend([e.tolist() for e in embeddings])
        
        return all_embeddings
```

### 1.4 Memory Management

**Impact**: Reduced memory pressure, faster GC

```python
# Use __slots__ for frequently created objects
@dataclass(slots=True)
class SearchResult:
    fact_id: str
    content: str
    category: str
    # ... rest of fields

# Use generators for large result sets
def search_generator(self, query: str) -> Generator[SearchResult, None, None]:
    """Stream results instead of collecting all."""
    # ... yield results one at a time
```

---

## Layer 2: Database

### 2.1 Index Optimization

**Current Indexes**:
```sql
-- Vector indexes (IVFFlat)
CREATE INDEX idx_labels_embedding ON labels USING ivfflat (embedding vector_l2_ops);
CREATE INDEX idx_facts_embedding ON facts USING ivfflat (embedding vector_l2_ops);

-- Full-text search
CREATE INDEX idx_facts_content_fts ON facts USING gin (to_tsvector('english', content));
```

**Optimized Indexes**:
```sql
-- Use cosine similarity (better for normalized vectors)
DROP INDEX IF EXISTS idx_labels_embedding;
DROP INDEX IF EXISTS idx_facts_embedding;

CREATE INDEX idx_labels_embedding ON labels 
    USING ivfflat (embedding vector_cosine_ops) 
    WITH (lists = 100);

CREATE INDEX idx_facts_embedding ON facts 
    USING ivfflat (embedding vector_cosine_ops) 
    WITH (lists = 100);

-- Add partial indexes for common queries
CREATE INDEX idx_facts_high_confidence ON facts (confidence DESC)
    WHERE confidence >= 0.5 AND deleted_at IS NULL;

CREATE INDEX idx_facts_by_category ON facts (category, confidence DESC)
    WHERE deleted_at IS NULL;

-- Composite index for label lookup
CREATE INDEX idx_labels_name_kind ON labels (name, kind, category)
    WHERE deleted_at IS NULL;
```

**IVFFlat Tuning**:

| Total Rows | lists | probes | Trade-off |
|------------|-------|--------|-----------|
| <1,000 | 10 | 5 | More accurate |
| 1,000-10,000 | 50 | 10 | Balanced |
| 10,000-100,000 | 100 | 20 | Balanced |
| >100,000 | 200 | 40 | More speed |

```sql
-- Set probes at session level for accuracy
SET ivfflat.probes = 20;

-- Or per-query (for critical searches)
SELECT * FROM facts 
ORDER BY embedding <=> $1 
LIMIT 10;  -- Uses index.probes setting
```

### 2.2 Query Optimization

**Before** (slow):
```sql
SELECT f.*, l.name, 1 - (f.embedding <=> $1) as similarity
FROM facts f
JOIN labels l ON f.label_id = l.id
WHERE f.confidence >= $2
ORDER BY f.embedding <=> $1
LIMIT $3;
```

**After** (optimized):
```sql
-- Use subquery to limit vector search first
WITH vector_matches AS (
    SELECT id, embedding <=> $1 as distance
    FROM facts
    WHERE confidence >= $2 AND deleted_at IS NULL
    ORDER BY embedding <=> $1
    LIMIT $3 * 2  -- Get extra for filtering
)
SELECT f.id, f.content, f.category, f.confidence,
       l.name as label_name,
       1 - vm.distance as similarity
FROM vector_matches vm
JOIN facts f ON vm.id = f.id
JOIN labels l ON f.label_id = l.id
ORDER BY vm.distance
LIMIT $3;
```

### 2.3 Query Analysis

Use `EXPLAIN ANALYZE` to identify slow queries:

```sql
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT * FROM facts
WHERE confidence >= 0.5
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 10;
```

Look for:
- Sequential scans on large tables
- High buffer cache misses
- Index not being used

### 2.4 Connection Settings

```sql
-- For Supabase/cloud DB, these are usually set
-- For self-hosted, tune these:

-- Memory
shared_buffers = '256MB'           -- 25% of RAM
effective_cache_size = '768MB'     -- 75% of RAM
work_mem = '16MB'                  -- Per operation

-- Parallelism
max_parallel_workers_per_gather = 2
parallel_tuple_cost = 0.01
parallel_setup_cost = 10

-- Write performance
wal_buffers = '16MB'
checkpoint_completion_target = 0.9
```

---

## Layer 3: Embeddings

### 3.1 Model Selection

| Model | Dims | Speed | Quality | Memory |
|-------|------|-------|---------|--------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | 90MB |
| all-mpnet-base-v2 | 768 | Medium | Better | 420MB |
| text-embedding-3-small | 1536 | API | Best | N/A |

**Recommendation**: Start with `all-MiniLM-L6-v2` for:
- 80ms average latency
- Good quality for personal memory
- Small memory footprint
- Works on CPU

### 3.2 Embedding Caching

**In-Memory Cache**:

```python
from functools import lru_cache
import hashlib

class CachedEmbedder:
    def __init__(self, embedder: BaseEmbedder, max_cache: int = 10000):
        self._embedder = embedder
        self._cache = {}
        self._max_cache = max_cache
    
    def embed(self, text: str) -> list[float]:
        # Create cache key
        key = hashlib.md5(text.encode()).hexdigest()
        
        if key in self._cache:
            return self._cache[key]
        
        # Compute embedding
        embedding = self._embedder.embed(text)
        
        # Cache with LRU eviction
        if len(self._cache) >= self._max_cache:
            # Remove oldest entry
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        
        self._cache[key] = embedding
        return embedding
```

**Redis Cache** (for distributed systems):

```python
import redis
import json

class RedisEmbeddingCache:
    def __init__(self, embedder: BaseEmbedder, redis_url: str):
        self._embedder = embedder
        self._redis = redis.from_url(redis_url)
        self._ttl = 3600 * 24  # 24 hour TTL
    
    def embed(self, text: str) -> list[float]:
        key = f"embed:{hashlib.md5(text.encode()).hexdigest()}"
        
        # Try cache
        cached = self._redis.get(key)
        if cached:
            return json.loads(cached)
        
        # Compute and cache
        embedding = self._embedder.embed(text)
        self._redis.setex(key, self._ttl, json.dumps(embedding))
        
        return embedding
```

### 3.3 Batch Optimization

```python
# Process multiple texts efficiently
def encode_documents(texts: list[str], embedder: BaseEmbedder) -> list[list[float]]:
    """Optimized batch encoding."""
    # Deduplicate to avoid redundant computation
    unique_texts = list(set(texts))
    
    # Batch embed
    unique_embeddings = embedder.embed_batch(unique_texts)
    
    # Build lookup
    lookup = dict(zip(unique_texts, unique_embeddings))
    
    # Return in original order
    return [lookup[t] for t in texts]
```

---

## Layer 4: Infrastructure

### 4.1 Hardware Recommendations

**Minimum (Development)**:
- CPU: 2 cores
- RAM: 4GB
- Storage: SSD

**Production (Single Instance)**:
- CPU: 4+ cores
- RAM: 8-16GB
- Storage: NVMe SSD
- Network: Low latency to DB

**Production (Scaled)**:
- Multiple API instances (stateless)
- Dedicated DB (managed or self-hosted)
- Redis for caching
- Load balancer

### 4.2 Container Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (caching)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Copy application
COPY . .

# Pre-download embedding model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Run with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - SUPABASE_DATABASE_URL=${SUPABASE_DATABASE_URL}
      - EMBEDDING_QUALITY=quality
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 4.3 Load Balancing

```nginx
# nginx.conf
upstream hippo_api {
    least_conn;
    server api1:8000 weight=1;
    server api2:8000 weight=1;
    server api3:8000 weight=1;
    keepalive 32;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://hippo_api;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_connect_timeout 5s;
        proxy_read_timeout 30s;
    }
    
    location /health {
        proxy_pass http://hippo_api;
        proxy_connect_timeout 2s;
        proxy_read_timeout 5s;
    }
}
```

---

## Monitoring & Profiling

### 5.1 Key Metrics

```python
# monitoring/metrics.py

from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUEST_COUNT = Counter(
    'hippo_requests_total',
    'Total requests',
    ['endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'hippo_request_latency_seconds',
    'Request latency',
    ['endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# Embedding metrics
EMBEDDING_LATENCY = Histogram(
    'hippo_embedding_latency_seconds',
    'Embedding generation latency',
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5]
)

EMBEDDING_CACHE_HITS = Counter(
    'hippo_embedding_cache_hits_total',
    'Embedding cache hits'
)

# Database metrics
DB_QUERY_LATENCY = Histogram(
    'hippo_db_query_latency_seconds',
    'Database query latency',
    ['query_type'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25]
)

DB_POOL_SIZE = Gauge(
    'hippo_db_pool_size',
    'Database connection pool size'
)

# Memory metrics
FACTS_TOTAL = Gauge(
    'hippo_facts_total',
    'Total facts in memory'
)

LABELS_TOTAL = Gauge(
    'hippo_labels_total',
    'Total labels in memory'
)
```

### 5.2 Profiling

```python
# Profile a specific function
import cProfile
import pstats

def profile_recall():
    engine = RecallEngine()
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    for _ in range(100):
        engine.recall("What food does the user like?")
    
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)


# Profile embedding
import time

def benchmark_embeddings(texts: list[str], iterations: int = 10):
    embedder = SentenceTransformerEmbedder()
    
    # Warm up
    embedder.embed("warm up")
    
    # Single text
    times = []
    for _ in range(iterations):
        start = time.time()
        for text in texts:
            embedder.embed(text)
        times.append(time.time() - start)
    
    print(f"Single: avg={sum(times)/len(times):.3f}s, texts={len(texts)}")
    
    # Batch
    times = []
    for _ in range(iterations):
        start = time.time()
        embedder.embed_batch(texts)
        times.append(time.time() - start)
    
    print(f"Batch: avg={sum(times)/len(times):.3f}s, texts={len(texts)}")
```

---

## Performance Checklist

### Quick Wins (Do First)
- [ ] Connection pooling implemented
- [ ] Embedding model pre-loaded
- [ ] Basic caching enabled
- [ ] Index probes tuned

### Medium Term
- [ ] Async endpoints
- [ ] Batch processing
- [ ] Redis caching
- [ ] Query optimization

### Long Term
- [ ] Horizontal scaling
- [ ] Read replicas
- [ ] CDN for static content
- [ ] Edge caching

---

## Troubleshooting

### High Latency

1. **Check database connection pool**
   ```python
   print(f"Pool size: {client._pool.get_stats()}")
   ```

2. **Check query execution plan**
   ```sql
   EXPLAIN ANALYZE <your query>;
   ```

3. **Check embedding model loading**
   ```python
   import time
   start = time.time()
   embedder = SentenceTransformerEmbedder()
   embedder.embed("test")  # Forces load
   print(f"Model load time: {time.time() - start:.2f}s")
   ```

### Memory Issues

1. **Check embedding cache size**
   ```python
   import sys
   print(f"Cache memory: {sys.getsizeof(embedder._cache) / 1024 / 1024:.2f}MB")
   ```

2. **Profile memory usage**
   ```python
   import tracemalloc
   tracemalloc.start()
   # ... run operations
   current, peak = tracemalloc.get_traced_memory()
   print(f"Current: {current / 1024 / 1024:.2f}MB, Peak: {peak / 1024 / 1024:.2f}MB")
   ```

### Throughput Issues

1. **Check connection pool saturation**
2. **Enable request queuing metrics**
3. **Check for blocking operations**
4. **Profile CPU usage per endpoint**

---

## Next Steps

Continue to `08-SECURITY-GUIDE.md` for security implementation.
