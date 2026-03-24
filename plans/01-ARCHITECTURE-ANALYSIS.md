# Architecture Analysis: Hippo.c Personal Memory System

## Executive Summary

Hippo.c is a personal memory system prototype designed to help LLMs maintain persistent knowledge about users. The current implementation provides **encode** (learning) and **memory storage** capabilities, but lacks the critical **recall** (retrieval) layer needed for production use.

---

## System Overview

### Core Purpose
Build a system where an AI can:
1. **Learn** everything about a user over time
2. **Remember** facts, preferences, relationships, goals, emotions
3. **Recall** relevant information instantly during conversations

### Current Components

```
┌─────────────────────────────────────────────────────────────┐
│                    ENCODE ENGINE                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  Triage  │→ │  Match   │→ │  Infer   │→ │  Update  │    │
│  │ (rules)  │  │(embedder)│  │  (LLM)   │  │(writer)  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    MEMORY STORE                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  Labels  │  │  Facts   │  │ Patterns │  │ Prompts  │    │
│  │(pgvector)│  │(pgvector)│  │ (regex)  │  │ (raw)    │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    RECALL (MISSING)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  Query   │  │  Search  │  │  Rank    │  │  Format  │    │
│  │ Parser   │  │  Engine  │  │  Rerank  │  │ Response │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Analysis

### 1. Encode Engine (`encode/encode.py`)

**Purpose**: Process user prompts and extract meaningful facts for storage.

**Current Flow**:
```python
process(prompts) → foreach prompt:
    1. upsert_prompt(prompt)           # Store raw prompt
    2. triage(prompt)                  # Determine if worth processing
    3. retrieve_candidates(text)       # Find related labels
    4. cheap_extract(prompt)           # Pattern-based extraction
    5. llm_extract(text, context)      # Optional LLM extraction
    6. dedupe(cheap + llm_facts)       # Remove duplicates
    7. updater.apply(extracted)        # Write to memory
```

**Analysis**:

| Aspect | Assessment | Rating |
|--------|------------|--------|
| **Efficiency** | Good layered approach with cheap→expensive fallback | ✅ Good |
| **Latency** | Sync processing, sequential operations | ⚠️ Needs work |
| **Throughput** | Single-threaded, no batching | ⚠️ Needs work |
| **Cost** | LLM calls gated by triage | ✅ Good |
| **Accuracy** | Pattern + LLM hybrid is solid | ✅ Good |

---

### 2. Triage System (`encode/triage/`)

**Purpose**: Decide if a prompt is worth deep processing.

**Components**:
- `rules.py`: Keyword scoring + pattern matching
- `patterns.py`: Regex-based pattern library with persistence

**Current Scoring**:
```python
score = (keyword_hits * 0.15) + (pattern_hits * 0.2)
should_infer = score >= 0.55  # threshold
```

**Analysis**:

| Aspect | Assessment | Issue |
|--------|------------|-------|
| **Signal Keywords** | Only 11 keywords | Too limited |
| **Scoring Weights** | Hardcoded | Should be learned |
| **Threshold** | Fixed 0.55 | Should adapt |
| **Pattern Weighting** | DB-backed weights exist but unused in triage | Inconsistent |

**Issues Found**:

1. **Limited Signal Detection**: Only 11 keywords for all personal information
2. **Static Threshold**: 0.55 is arbitrary, should be adaptive
3. **Missing Signal Types**: No entity detection, no temporal signals
4. **Pattern Weight Unused**: `pattern.weight` stored but not used in scoring

---

### 3. Embedding System (`encode/match/embedder.py`)

**Purpose**: Create vector representations for similarity search.

**Current Implementation**: Hash-based embedder (no ML model)

```python
def embed(self, text: str) -> list[float]:
    vec = [0.0] * 256
    for token in tokenize(text):
        idx = hash(token) % 256
        vec[idx] += 1.0
    return normalize(vec)
```

**Analysis**:

| Aspect | Assessment | Rating |
|--------|------------|--------|
| **Cost** | Zero (no API calls) | ✅ Excellent |
| **Speed** | Microseconds | ✅ Excellent |
| **Dependencies** | None | ✅ Excellent |
| **Quality** | Poor semantic understanding | ❌ Critical issue |
| **Collision Rate** | High (256 dims + hash collisions) | ❌ Critical issue |

**Critical Issues**:

1. **No Semantic Understanding**: "I love cats" and "I adore felines" have different embeddings
2. **Hash Collisions**: 256 dimensions is very small; many words will collide
3. **No Context**: Word order completely lost
4. **Quality vs Speed Tradeoff**: Currently prioritizes speed over quality

**Recommendation**: This is the SINGLE BIGGEST blocker for production. Options:
- **Local Embedding Model**: Use sentence-transformers (~50-100ms)
- **API Embedding**: OpenAI/Cohere embeddings (~100-300ms, $0.0001/1K tokens)
- **Hybrid**: Keep hash for pre-filtering, use quality embeddings for final ranking

---

### 4. LLM Extraction (`encode/infer/llm.py`)

**Purpose**: Extract structured facts from text using LLM.

**Current Implementation**:
- Uses DeepSeek API (OpenAI-compatible)
- Simple system prompt for fact extraction
- Returns JSON array of facts

**Analysis**:

| Aspect | Assessment | Rating |
|--------|------------|--------|
| **Cost** | DeepSeek is cheap (~$0.14/M tokens) | ✅ Good |
| **Quality** | Depends on prompt engineering | ⚠️ Needs work |
| **Latency** | 2-5 seconds typical | ⚠️ Acceptable |
| **Reliability** | No retry, no fallback | ❌ Critical |
| **Error Handling** | Swallows all exceptions | ❌ Critical |

**Issues**:

1. **No Retry Logic**: Network failures silently return `[]`
2. **No Rate Limiting**: Could hit API limits
3. **Weak Prompt**: System prompt is too generic
4. **No Structured Output**: Uses `json.loads()` on raw LLM output (fragile)
5. **No Validation**: Extracted facts not validated

---

### 5. Memory Storage (`memory/client.py`)

**Purpose**: PostgreSQL + pgvector persistence layer.

**Schema Analysis**:

| Table | Purpose | Index Strategy | Issues |
|-------|---------|----------------|--------|
| `labels` | Topic/category grouping | IVFFlat vector index | ✅ Good |
| `facts` | Individual memory items | IVFFlat + FTS | ✅ Good |
| `patterns` | Regex extraction rules | None | ⚠️ Missing |
| `prompts` | Raw conversation history | None | ⚠️ Missing |

**Connection Issues**:

```python
def _connect(self):
    conn = psycopg.connect(self.dsn)  # New connection EVERY call
    register_vector(conn)
    return conn
```

**Critical Issue**: Creates a new database connection for EVERY operation!

- **Impact**: Massive latency overhead (50-100ms per connection)
- **Fix**: Connection pooling (pgbouncer or psycopg pool)

---

### 6. Update System (`encode/update/updater.py`)

**Purpose**: Write extracted facts to memory with deduplication.

**Current Flow**:
```python
for item in extracted:
    1. upsert_label(name, embedding)       # Create/update label
    2. find_best_fact(label_id, embedding) # Find similar fact
    3. if similarity >= 0.85:
         reinforce_fact(fact_id)           # Boost confidence
       else:
         insert_fact(new_fact)             # Create new fact
```

**Analysis**:

| Aspect | Assessment | Rating |
|--------|------------|--------|
| **Deduplication** | Uses embedding similarity | ⚠️ Weak (hash embeddings) |
| **Threshold** | 0.85 fixed | ⚠️ Should be adaptive |
| **Confidence Decay** | Not implemented | ❌ Missing |
| **Conflict Resolution** | None | ❌ Missing |

**Issues**:

1. **Confidence Only Goes Up**: No decay for old facts
2. **No Contradiction Detection**: "I like coffee" and "I hate coffee" can coexist
3. **Temporal Blindness**: Recent facts should trump old ones
4. **Batch Writes**: Could batch multiple inserts for efficiency

---

## Database Schema Analysis

### Current Schema Strengths

1. **pgvector Integration**: Vector similarity search built-in
2. **Proper Foreign Keys**: facts → labels relationship
3. **Audit Fields**: created_at, updated_at, last_seen_at
4. **Evidence Tracking**: evidence_count for reinforcement
5. **Full-Text Search**: GIN index on facts.content

### Schema Issues

1. **No User Isolation**: Schema assumes single user
2. **No Soft Delete**: Hard delete only
3. **IVFFlat Index Issues**: 
   - Requires `lists` parameter tuning
   - Needs periodic reindexing
   - Poor performance with <1000 vectors
4. **No Temporal Partitioning**: Old facts should be archived

### Recommended Schema Additions

```sql
-- Add user support
ALTER TABLE labels ADD COLUMN user_id uuid;
ALTER TABLE facts ADD COLUMN user_id uuid;
CREATE INDEX idx_labels_user ON labels(user_id);
CREATE INDEX idx_facts_user ON facts(user_id);

-- Add soft delete
ALTER TABLE facts ADD COLUMN deleted_at timestamptz;
ALTER TABLE labels ADD COLUMN deleted_at timestamptz;

-- Add version tracking
ALTER TABLE facts ADD COLUMN version int default 1;
ALTER TABLE facts ADD COLUMN superseded_by uuid references facts(id);
```

---

## Performance Characteristics

### Current Latency Breakdown (Estimated)

| Operation | Current | Target | Gap |
|-----------|---------|--------|-----|
| DB Connection | 50-100ms | 1-5ms | 10-50x |
| Hash Embedding | 0.1ms | 0.1ms | ✅ OK |
| Quality Embedding | N/A | 50-100ms | Needed |
| Label Search (pgvector) | 10-50ms | 10-50ms | ✅ OK |
| Fact Search (pgvector) | 10-50ms | 10-50ms | ✅ OK |
| LLM Extraction | 2000-5000ms | 2000-5000ms | Acceptable |
| Pattern Matching | 1-5ms | 1-5ms | ✅ OK |

### Total Latency Scenarios

| Scenario | Current | With Fixes |
|----------|---------|------------|
| Simple pattern match | ~150ms | ~20ms |
| Pattern + vector search | ~200ms | ~70ms |
| Full LLM extraction | ~5500ms | ~5200ms |

---

## Cost Analysis

### Current Costs (Per 1000 Operations)

| Component | Cost | Notes |
|-----------|------|-------|
| Supabase DB | ~$0.01 | Compute + storage |
| DeepSeek LLM | ~$0.14 | Only when triage passes |
| Hash Embedding | $0.00 | No external calls |

### Projected Production Costs (Per 1000 Operations)

| Component | Low | Medium | High | Notes |
|-----------|-----|--------|------|-------|
| DB (managed) | $0.02 | $0.05 | $0.10 | Depends on scale |
| Quality Embeddings | $0.01 | $0.05 | $0.10 | API vs local |
| LLM Extraction | $0.10 | $0.20 | $0.50 | Depends on model |
| **Total** | **$0.13** | **$0.30** | **$0.70** | Per 1K prompts |

---

## Security Assessment

### Current Vulnerabilities

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| SQL Injection | ✅ Safe | All queries parameterized | N/A |
| API Key Exposure | ⚠️ Medium | .env file | Use secrets manager |
| No Input Sanitization | ⚠️ Medium | Prompt text stored raw | Sanitize before storage |
| No Rate Limiting | ⚠️ Medium | All endpoints | Add rate limiter |
| No Auth | ❌ High | No auth system | Add auth layer |
| No Encryption at Rest | ⚠️ Medium | Personal data in DB | Enable DB encryption |

### Privacy Concerns

1. **Data Retention**: No automatic cleanup of old data
2. **Export/Delete**: No GDPR-compliant data export/deletion
3. **Access Logging**: No audit trail for data access
4. **PII Detection**: No automatic PII flagging/masking

---

## What's Missing (Critical)

### 1. RECALL SYSTEM (Most Critical)
The entire retrieval layer is missing. Without it, stored memories are useless.

### 2. Connection Pooling
Every DB operation creates a new connection. This is a 10-50x latency overhead.

### 3. Quality Embeddings
Hash-based embeddings cannot capture semantic meaning. Critical for recall accuracy.

### 4. Confidence Decay
Facts should decay over time if not reinforced. Currently only increases.

### 5. Contradiction Detection
System can store conflicting facts without resolution.

### 6. Temporal Awareness
No concept of when facts were true, or if they've changed.

### 7. User Authentication
No user isolation or access control.

### 8. Error Handling
LLM errors silently swallowed. DB errors not handled gracefully.

---

## Summary Ratings

| Component | Functionality | Efficiency | Production Ready |
|-----------|--------------|------------|------------------|
| Encode Engine | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Triage System | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Hash Embedder | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| LLM Extractor | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Memory Client | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| Update System | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Recall (Missing)** | N/A | N/A | N/A |

**Overall Prototype Rating**: ⭐⭐⭐ (3/5)

The architecture is sound, but execution needs significant work for production.

---

## Next Steps

1. **Read**: `02-RECOMMENDATIONS.md` for detailed fix recommendations
2. **Read**: `03-RECALL-SYSTEM-DESIGN.md` for recall architecture
3. **Read**: `04-PRODUCTION-ROADMAP.md` for implementation timeline
