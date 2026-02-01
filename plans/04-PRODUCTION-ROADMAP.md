# Production Roadmap: From Prototype to Production

## Executive Summary

This document outlines the complete roadmap for transforming Hippo.c from a prototype into a production-ready personal memory system.

**Timeline**: 8-12 weeks for MVP, 16-20 weeks for production-grade

**Effort Estimate**: ~400-600 hours of development

---

## Phase Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Phase 1: Foundation (Weeks 1-2)                                            │
│  ├── Connection pooling                                                     │
│  ├── Error handling & logging                                               │
│  ├── Quality embeddings                                                     │
│  └── Basic tests                                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  Phase 2: Recall System (Weeks 3-5)                                         │
│  ├── Query parser                                                           │
│  ├── Search engine                                                          │
│  ├── Ranker & formatter                                                     │
│  └── Integration tests                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  Phase 3: API & Integration (Weeks 6-7)                                     │
│  ├── REST API with FastAPI                                                  │
│  ├── Authentication                                                         │
│  ├── Rate limiting                                                          │
│  └── API documentation                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  Phase 4: Hardening (Weeks 8-10)                                            │
│  ├── Confidence decay                                                       │
│  ├── Contradiction detection                                                │
│  ├── Batch processing                                                       │
│  └── Caching layer                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  Phase 5: Production (Weeks 11-14)                                          │
│  ├── Performance optimization                                               │
│  ├── Monitoring & alerting                                                  │
│  ├── Security audit                                                         │
│  └── Documentation                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  Phase 6: Scale (Weeks 15-20)                                               │
│  ├── Multi-user support                                                     │
│  ├── Horizontal scaling                                                     │
│  ├── GDPR compliance                                                        │
│  └── Advanced features                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Foundation (Weeks 1-2)

### Goal
Fix critical infrastructure issues that would block all future work.

### Tasks

#### 1.1 Connection Pooling (Day 1-2)
**Effort**: 4-6 hours

```
Files to modify:
- memory/client.py
- pyproject.toml

Steps:
1. Add psycopg[pool] to dependencies
2. Implement connection pool in MemoryClient
3. Update all methods to use pool context manager
4. Test connection reuse
5. Add pool health monitoring
```

**Success Criteria**:
- [ ] Connection pool initialized once
- [ ] All methods reuse connections
- [ ] Latency reduced by 10-50x per operation
- [ ] No connection leaks under load

#### 1.2 Error Handling (Day 2-3)
**Effort**: 6-8 hours

```
Files to modify:
- encode/infer/llm.py
- encode/encode.py
- memory/client.py

Steps:
1. Add logging infrastructure
2. Implement retry logic for LLM calls
3. Add proper exception types
4. Add graceful degradation
5. Implement health check endpoint
```

**Success Criteria**:
- [ ] All errors logged with context
- [ ] LLM retries with exponential backoff
- [ ] DB errors caught and reported
- [ ] System continues on partial failures

#### 1.3 Quality Embeddings (Day 3-5)
**Effort**: 8-12 hours

```
Files to create/modify:
- encode/match/embedder.py (rewrite)
- memory/schema.sql (update dimensions)
- migrations/001_update_embedding_dims.sql

Steps:
1. Add sentence-transformers dependency
2. Implement SentenceTransformerEmbedder class
3. Implement HybridEmbedder class
4. Update schema for 384 dimensions
5. Write migration script
6. Create embedding factory function
7. Update tests
```

**Success Criteria**:
- [ ] Quality embeddings working
- [ ] Schema migrated to new dimensions
- [ ] Backward compatibility with fast embeddings
- [ ] <100ms embedding latency

#### 1.4 Basic Test Suite (Day 5-7)
**Effort**: 8-10 hours

```
Files to create:
- tests/conftest.py
- tests/test_memory_client.py
- tests/test_embedder.py
- tests/test_encode_engine.py

Steps:
1. Set up pytest infrastructure
2. Create test database fixtures
3. Write unit tests for memory client
4. Write unit tests for embedder
5. Write integration tests for encode
6. Set up CI pipeline
```

**Success Criteria**:
- [ ] >70% code coverage
- [ ] All tests pass
- [ ] CI runs on every PR
- [ ] Test database isolation

### Phase 1 Deliverables
- [ ] Connection pooling implemented
- [ ] Error handling with logging
- [ ] Quality embeddings option
- [ ] Basic test suite
- [ ] CI/CD pipeline

---

## Phase 2: Recall System (Weeks 3-5)

### Goal
Build the complete recall system for memory retrieval.

### Tasks

#### 2.1 Query Parser (Week 3, Day 1-2)
**Effort**: 8-10 hours

```
Files to create:
- recall/__init__.py
- recall/query/__init__.py
- recall/query/parser.py

Steps:
1. Create recall package structure
2. Implement QueryIntent enum
3. Implement ParsedQuery dataclass
4. Implement QueryParser class
5. Add intent detection patterns
6. Add entity extraction
7. Add keyword extraction
8. Add temporal hint extraction
9. Write comprehensive tests
```

**Success Criteria**:
- [ ] Intent detection >80% accuracy
- [ ] Entity extraction working
- [ ] Keyword extraction working
- [ ] <5ms parsing latency

#### 2.2 Strategy Selector (Week 3, Day 3)
**Effort**: 4-6 hours

```
Files to create:
- recall/strategy/__init__.py
- recall/strategy/selector.py

Steps:
1. Implement SearchStrategy enum
2. Implement SearchPlan dataclass
3. Implement StrategySelector class
4. Add strategy selection logic
5. Add fallback chain logic
6. Write tests
```

**Success Criteria**:
- [ ] Correct strategy for each intent type
- [ ] Fallback chains working
- [ ] Weight tuning possible

#### 2.3 Search Engine (Week 3-4)
**Effort**: 16-20 hours

```
Files to create:
- recall/search/__init__.py
- recall/search/engine.py

Steps:
1. Implement SearchResult dataclass
2. Implement SearchEngine class
3. Implement vector search
4. Implement keyword search
5. Implement hybrid search (RRF fusion)
6. Implement filtered search
7. Implement exact match search
8. Add search index optimizations
9. Write comprehensive tests
```

**Success Criteria**:
- [ ] Vector search <50ms
- [ ] Keyword search <50ms
- [ ] Hybrid search <100ms
- [ ] >80% recall accuracy

#### 2.4 Ranker & Filter (Week 4)
**Effort**: 8-10 hours

```
Files to create:
- recall/rank/__init__.py
- recall/rank/ranker.py

Steps:
1. Implement RankedResult dataclass
2. Implement ResultRanker class
3. Implement multi-signal ranking
4. Implement deduplication
5. Implement explanation generation
6. Tune ranking weights
7. Write tests
```

**Success Criteria**:
- [ ] Deduplication working
- [ ] Ranking improves result quality
- [ ] Explanations are helpful

#### 2.5 Response Formatter (Week 4)
**Effort**: 4-6 hours

```
Files to create:
- recall/format/__init__.py
- recall/format/formatter.py

Steps:
1. Implement RecallResponse dataclass
2. Implement ResponseFormatter class
3. Implement summary generation
4. Implement LLM context formatting
5. Write tests
```

**Success Criteria**:
- [ ] JSON serializable responses
- [ ] Summaries are accurate
- [ ] LLM format is useful

#### 2.6 Recall Engine (Week 5)
**Effort**: 8-10 hours

```
Files to create:
- recall/engine.py

Steps:
1. Implement RecallOptions dataclass
2. Implement RecallEngine class
3. Wire all components together
4. Add recall_for_llm helper
5. Write integration tests
6. Benchmark performance
7. Tune parameters
```

**Success Criteria**:
- [ ] End-to-end recall working
- [ ] <200ms P95 latency
- [ ] LLM integration working

### Phase 2 Deliverables
- [ ] Complete recall system
- [ ] Query parser with intent detection
- [ ] Hybrid search engine
- [ ] Smart ranking with explanations
- [ ] LLM context formatting
- [ ] Integration tests passing

---

## Phase 3: API & Integration (Weeks 6-7)

### Goal
Build REST API for external access with proper security.

### Tasks

#### 3.1 FastAPI Setup (Week 6, Day 1-2)
**Effort**: 6-8 hours

```
Files to create:
- api/__init__.py
- api/main.py
- api/routes/__init__.py
- api/routes/recall.py
- api/routes/encode.py
- api/routes/health.py

Steps:
1. Add FastAPI dependency
2. Create API app structure
3. Implement health endpoint
4. Implement recall endpoint
5. Implement encode endpoint
6. Add request validation
7. Add response models
```

**Success Criteria**:
- [ ] API server starts
- [ ] Health check works
- [ ] Recall endpoint works
- [ ] Encode endpoint works

#### 3.2 Authentication (Week 6, Day 3-5)
**Effort**: 10-12 hours

```
Files to create:
- api/auth/__init__.py
- api/auth/middleware.py
- api/auth/tokens.py

Steps:
1. Choose auth strategy (API keys vs JWT)
2. Implement API key validation
3. Add auth middleware
4. Add user context injection
5. Update all routes to require auth
6. Write auth tests
```

**Success Criteria**:
- [ ] All endpoints require auth
- [ ] Invalid auth rejected
- [ ] User context available in routes

#### 3.3 Rate Limiting (Week 7, Day 1-2)
**Effort**: 4-6 hours

```
Files to modify:
- api/main.py
- api/auth/middleware.py

Steps:
1. Add rate limiting dependency
2. Configure per-user limits
3. Add rate limit headers
4. Implement graceful limit handling
5. Write rate limit tests
```

**Success Criteria**:
- [ ] Rate limits enforced
- [ ] Headers show remaining quota
- [ ] Graceful 429 responses

#### 3.4 API Documentation (Week 7, Day 3-5)
**Effort**: 6-8 hours

```
Files to create:
- api/docs/openapi.yaml
- docs/api-guide.md

Steps:
1. Configure OpenAPI spec
2. Add detailed endpoint docs
3. Add request/response examples
4. Generate client SDKs
5. Create API usage guide
```

**Success Criteria**:
- [ ] OpenAPI spec complete
- [ ] Swagger UI accessible
- [ ] Examples for all endpoints

### Phase 3 Deliverables
- [ ] REST API with FastAPI
- [ ] Authentication system
- [ ] Rate limiting
- [ ] API documentation
- [ ] OpenAPI specification

---

## Phase 4: Hardening (Weeks 8-10)

### Goal
Add robustness features for production reliability.

### Tasks

#### 4.1 Confidence Decay (Week 8, Day 1-2)
**Effort**: 6-8 hours

```
Files to modify/create:
- memory/client.py
- scripts/maintenance.py

Steps:
1. Add decay_confidence method
2. Add archive_stale_facts method
3. Create maintenance script
4. Set up scheduled job
5. Write tests
```

**Success Criteria**:
- [ ] Old facts decay over time
- [ ] Stale facts archived
- [ ] Maintenance runs daily

#### 4.2 Contradiction Detection (Week 8, Day 3-5)
**Effort**: 10-12 hours

```
Files to create:
- encode/update/conflict.py

Files to modify:
- encode/update/updater.py

Steps:
1. Implement ConflictDetector class
2. Define negation patterns
3. Implement conflict check logic
4. Add supersede_fact method
5. Integrate with updater
6. Write comprehensive tests
```

**Success Criteria**:
- [ ] Contradictions detected
- [ ] Resolution strategy applied
- [ ] Old facts superseded

#### 4.3 Batch Processing (Week 9)
**Effort**: 10-12 hours

```
Files to modify:
- encode/encode.py
- encode/match/embedder.py
- memory/client.py

Steps:
1. Add embed_batch method to embedder
2. Add batch_search_labels to memory
3. Add batch_insert_facts to memory
4. Implement process_batch in EncodeEngine
5. Benchmark improvements
6. Write tests
```

**Success Criteria**:
- [ ] Batch embedding working
- [ ] Batch DB operations working
- [ ] 2-5x throughput improvement

#### 4.4 Caching Layer (Week 10)
**Effort**: 10-12 hours

```
Files to create:
- memory/cache.py

Files to modify:
- memory/client.py

Steps:
1. Implement MemoryCache class
2. Add label caching
3. Add fact caching
4. Add pattern caching
5. Implement cache invalidation
6. Add cache metrics
7. Write tests
```

**Success Criteria**:
- [ ] Hot data cached
- [ ] Cache hit rate >50%
- [ ] Cache invalidation working

### Phase 4 Deliverables
- [ ] Confidence decay system
- [ ] Contradiction detection
- [ ] Batch processing
- [ ] Caching layer
- [ ] All tests passing

---

## Phase 5: Production (Weeks 11-14)

### Goal
Performance optimization, monitoring, and security hardening.

### Tasks

#### 5.1 Performance Optimization (Week 11-12)
**Effort**: 16-20 hours

```
Steps:
1. Profile all operations
2. Optimize hot paths
3. Add query plan analysis
4. Tune pgvector indexes
5. Add async operations where beneficial
6. Benchmark all endpoints
7. Document performance characteristics
```

**Success Criteria**:
- [ ] P50 latency <50ms
- [ ] P95 latency <200ms
- [ ] Throughput >100 req/s

#### 5.2 Monitoring & Alerting (Week 12-13)
**Effort**: 12-16 hours

```
Files to create:
- monitoring/metrics.py
- monitoring/alerts.py

Steps:
1. Add Prometheus metrics
2. Add latency histograms
3. Add error counters
4. Add business metrics
5. Configure alerting rules
6. Create dashboards
```

**Success Criteria**:
- [ ] Key metrics exposed
- [ ] Dashboards created
- [ ] Alerts configured

#### 5.3 Security Audit (Week 13)
**Effort**: 8-12 hours

```
Steps:
1. Input validation review
2. SQL injection audit
3. API security review
4. Dependency vulnerability scan
5. Secrets management review
6. Access control review
7. Fix all identified issues
8. Document security posture
```

**Success Criteria**:
- [ ] No SQL injection vectors
- [ ] All inputs validated
- [ ] Dependencies secure
- [ ] Secrets properly managed

#### 5.4 Documentation (Week 14)
**Effort**: 12-16 hours

```
Files to create:
- docs/getting-started.md
- docs/architecture.md
- docs/deployment.md
- docs/configuration.md
- docs/troubleshooting.md

Steps:
1. Write getting started guide
2. Document architecture
3. Document deployment process
4. Document configuration options
5. Create troubleshooting guide
6. Add inline code comments
```

**Success Criteria**:
- [ ] New users can onboard
- [ ] Architecture documented
- [ ] Deployment documented
- [ ] Common issues documented

### Phase 5 Deliverables
- [ ] Performance targets met
- [ ] Monitoring & alerting
- [ ] Security audit passed
- [ ] Complete documentation

---

## Phase 6: Scale (Weeks 15-20)

### Goal
Multi-user support, horizontal scaling, and advanced features.

### Tasks

#### 6.1 Multi-User Support (Week 15-16)
**Effort**: 20-24 hours

```
Steps:
1. Add user_id to all tables
2. Create migration script
3. Update all queries with user filter
4. Add user management API
5. Implement user isolation
6. Test multi-user scenarios
```

**Success Criteria**:
- [ ] Users completely isolated
- [ ] Queries always filtered
- [ ] User management working

#### 6.2 Horizontal Scaling (Week 17-18)
**Effort**: 16-20 hours

```
Steps:
1. Make services stateless
2. External session storage
3. Implement read replicas
4. Add load balancing
5. Document scaling procedures
6. Load test at scale
```

**Success Criteria**:
- [ ] Multiple instances work
- [ ] No shared state issues
- [ ] Linear scaling achieved

#### 6.3 GDPR Compliance (Week 18-19)
**Effort**: 12-16 hours

```
Steps:
1. Implement data export
2. Implement data deletion
3. Add consent management
4. Add audit logging
5. Document privacy practices
6. Review with legal
```

**Success Criteria**:
- [ ] Data export works
- [ ] Data deletion complete
- [ ] Audit trail exists
- [ ] Consent managed

#### 6.4 Advanced Features (Week 19-20)
**Effort**: 16-20 hours

```
Features:
1. Temporal awareness (valid_from/valid_until)
2. Graph-based relationships
3. Smart summarization with LLM
4. Learning from corrections
5. Export/import memory
```

**Success Criteria**:
- [ ] Temporal queries working
- [ ] Relationship graph working
- [ ] LLM summarization working

### Phase 6 Deliverables
- [ ] Multi-user support
- [ ] Horizontal scalability
- [ ] GDPR compliance
- [ ] Advanced features

---

## Dependency Installation Order

```bash
# Phase 1 dependencies
pip install "psycopg[binary,pool]>=3.1.18"
pip install sentence-transformers>=2.2.0
pip install pytest pytest-asyncio pytest-cov

# Phase 3 dependencies
pip install fastapi>=0.100.0
pip install uvicorn[standard]>=0.23.0
pip install pydantic>=2.0.0
pip install python-jose[cryptography]  # For JWT

# Phase 5 dependencies
pip install prometheus-client
pip install structlog  # Better logging
```

---

## Database Migrations

### Migration 001: Update Embedding Dimensions
```sql
-- migrations/001_update_embedding_dims.sql
-- Run before deploying quality embeddings

-- Temporarily disable index
DROP INDEX IF EXISTS idx_labels_embedding;
DROP INDEX IF EXISTS idx_facts_embedding;

-- Update column types
ALTER TABLE labels ALTER COLUMN embedding TYPE vector(384);
ALTER TABLE facts ALTER COLUMN embedding TYPE vector(384);

-- Recreate indexes with cosine similarity
CREATE INDEX idx_labels_embedding ON labels 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_facts_embedding ON facts 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Note: Existing data needs re-embedding after this migration!
```

### Migration 002: Add Soft Delete
```sql
-- migrations/002_add_soft_delete.sql

ALTER TABLE facts ADD COLUMN deleted_at timestamptz;
ALTER TABLE labels ADD COLUMN deleted_at timestamptz;

CREATE INDEX idx_facts_deleted ON facts(deleted_at) WHERE deleted_at IS NOT NULL;
CREATE INDEX idx_labels_deleted ON labels(deleted_at) WHERE deleted_at IS NOT NULL;
```

### Migration 003: Add User Support
```sql
-- migrations/003_add_user_support.sql

-- Add user_id columns
ALTER TABLE labels ADD COLUMN user_id uuid;
ALTER TABLE facts ADD COLUMN user_id uuid;
ALTER TABLE prompts ADD COLUMN user_id uuid;

-- Create indexes
CREATE INDEX idx_labels_user ON labels(user_id);
CREATE INDEX idx_facts_user ON facts(user_id);
CREATE INDEX idx_prompts_user ON prompts(user_id);

-- Update existing data (set to default user)
UPDATE labels SET user_id = '00000000-0000-0000-0000-000000000000' WHERE user_id IS NULL;
UPDATE facts SET user_id = '00000000-0000-0000-0000-000000000000' WHERE user_id IS NULL;
UPDATE prompts SET user_id = '00000000-0000-0000-0000-000000000000' WHERE user_id IS NULL;

-- Make not nullable
ALTER TABLE labels ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE facts ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE prompts ALTER COLUMN user_id SET NOT NULL;
```

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Embedding model too slow | Medium | High | Use batch processing, caching |
| DB connection exhaustion | High | High | Connection pooling (Phase 1) |
| LLM API rate limits | Medium | Medium | Implement backoff, queue |
| Search quality poor | Medium | High | Tune ranking, add more signals |
| Scaling bottlenecks | Medium | Medium | Profile early, design for scale |
| Security vulnerability | Low | Critical | Audit, penetration testing |
| Cost overrun on LLM | Medium | Medium | Aggressive caching, triage |

---

## Success Metrics

### Phase 1 Success
- [ ] <10ms connection latency
- [ ] >70% code coverage
- [ ] CI pipeline green

### Phase 2 Success
- [ ] <200ms recall latency
- [ ] >80% recall accuracy
- [ ] LLM integration working

### Phase 3 Success
- [ ] API server stable
- [ ] Auth working
- [ ] Rate limits enforced

### Phase 4 Success
- [ ] Contradictions resolved
- [ ] Cache hit rate >50%
- [ ] Batch throughput 2x

### Phase 5 Success
- [ ] P95 latency <200ms
- [ ] No security vulnerabilities
- [ ] Documentation complete

### Phase 6 Success
- [ ] Multi-user isolated
- [ ] 1000+ concurrent users
- [ ] GDPR compliant

---

## Next Steps

Continue to:
1. `05-IMPLEMENTATION-GUIDE.md` - Step-by-step coding guide
2. `06-API-SPECIFICATION.md` - Full API documentation
3. `07-PERFORMANCE-OPTIMIZATION.md` - Performance tuning guide
4. `08-SECURITY-GUIDE.md` - Security implementation guide
