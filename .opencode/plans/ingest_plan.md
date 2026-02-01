# Hippo.c-1 Ingest System Plan

## Overview
The ingest module extracts facts, insights, and information from both user prompts and AI responses, using a hybrid approach that learns and evolves over time.

## Architecture Philosophy
- **Ultra minimal**: Only necessary components
- **Ultra modular**: Each file has single responsibility
- **Clear naming**: Self-documenting code
- **Batch processing**: Handle multiple prompts efficiently
- **Learning capable**: Improves through feedback/evolution

## Directory Structure

```
ingest/
├── models/
│   ├── fact.py           # Fact data model with high-dim metadata
│   └── prompt.py         # Input prompt models (single & batch)
├── extractors/
│   ├── base.py           # Base extractor interface
│   ├── rules.py          # Rule-based extraction (fast, simple patterns)
│   ├── llm.py            # LLM-based extraction (nuanced, complex)
│   └── hybrid.py         # Orchestrator: selects best extractor
├── learning/
│   ├── patterns.py       # Pattern storage and matching
│   └── feedback.py       # Handle user corrections/learning
├── store/
│   └── client.py         # PostgreSQL store interface
└── ingest.py             # Main entry point
```

## Component Specifications

### 1. Models (`ingest/models/`)

#### `fact.py`
- **Purpose**: High-dimensional fact representation
- **Fields**:
  - `id`: Unique identifier
  - `content`: The fact/insight text
  - `category`: Type (preference, event, relationship, belief, fact, etc.)
  - `confidence`: 0.0-1.0 score
  - `source`: "user" or "ai"
  - `timestamp`: When extracted
  - `source_timestamp`: When original prompt/response was created
  - `source_id`: Reference to original prompt/response
  - `embedding_id`: For future vector indexing
  - `metadata`: JSON field for extensible attributes

#### `prompt.py`
- **Purpose**: Input data models
- **Classes**:
  - `Prompt`: Single prompt with text, role (user/ai), timestamp
  - `PromptBatch`: List of prompts with batch metadata

### 2. Extractors (`ingest/extractors/`)

#### `base.py`
- **Purpose**: Abstract base class
- **Methods**:
  - `extract(prompts: PromptBatch) -> List[Fact]`
  - `can_handle(prompt: Prompt) -> bool` (confidence score)

#### `rules.py`
- **Purpose**: Fast pattern matching
- **Patterns** (evolving via learning):
  - "I like/hate X" → preferences
  - "X is my Y" → relationships
  - "I did X" → events
  - Named entities (people, places)
- **Performance**: Sub-millisecond per prompt
- **Accuracy**: Lower, but evolves

#### `llm.py`
- **Purpose**: Contextual extraction
- **Implementation**: LangChain with lightweight local model or API
- **Capabilities**:
  - Understands context and nuance
  - Detects implicit facts
  - Categorizes automatically
- **Performance**: 100-500ms per prompt
- **Accuracy**: High

#### `hybrid.py`
- **Purpose**: Smart orchestration
- **Strategy**:
  1. Apply rules first (fast, low confidence)
  2. If confidence < threshold, use LLM
  3. Combine results, deduplicate
  4. Batch LLM calls for efficiency
- **Evolution**: Adjusts thresholds based on learning

### 3. Learning (`ingest/learning/`)

#### `patterns.py`
- **Purpose**: Store and manage evolving extraction patterns
- **Storage**: In-memory + persistent file (patterns.json)
- **Fields**:
  - `patterns`: List of extraction rules
  - `success_rate`: Track performance
  - `last_updated`: Timestamp
- **Evolution**:
  - New patterns added from LLM extractions
  - Confidence scores updated via feedback
  - Prune low-confidence patterns

#### `feedback.py`
- **Purpose**: Handle corrections and learning
- **Methods**:
  - `add_correction(fact_id: str, correct_content: str)`
  - `add_positive_feedback(fact_id: str)`
  - `add_negative_feedback(fact_id: str)`
- **Effect**:
  - Updates pattern confidence
  - May retire or promote patterns
  - Triggers pattern regeneration if needed

### 4. Store Interface (`ingest/store/`)

#### `client.py`
- **Purpose**: Bridge to PostgreSQL store
- **Methods**:
  - `save_facts(facts: List[Fact]) -> List[str]` (returns IDs)
  - `update_fact(fact_id: str, updates: dict)`
  - `batch_save_facts(batches: List[List[Fact]])`
- **Note**: Implementation deferred to store module, just interface now

### 5. Main Entry Point (`ingest/ingest.py`)

- **Purpose**: Public API for ingest module
- **Interface**:
```python
class IngestEngine:
    def __init__(self, store_client, learning_enabled=True)
    
    def process(self, prompts: PromptBatch) -> List[str]
        # Returns fact IDs
    
    def provide_feedback(self, fact_id: str, correction: str = None, 
                        positive: bool = False, negative: bool = False)
    
    def get_stats(self) -> dict
```

## Workflow

### Ingestion Flow
```
1. User calls ingest.process(batch of prompts)
   ↓
2. Hybrid extractor analyzes each prompt
   - Rules extractor runs first (fast)
   - LLM extractor for low-confidence or complex
   - Batch LLM calls for efficiency
   ↓
3. Results combined, deduplicated
   - High-dimensional metadata attached
   - Timestamps generated
   ↓
4. Facts sent to store.client.save_facts()
   ↓
5. IDs returned to caller
```

### Learning Flow
```
1. User calls ingest.provide_feedback()
   - Correction: Update fact, learn from mistake
   - Positive: Boost pattern confidence
   - Negative: Lower pattern confidence
   ↓
2. Learning module updates patterns
   - Adjust confidence scores
   - Generate new patterns from LLM corrections
   - Prune/retire as needed
   ↓
3. Future extractions use updated patterns
   - Evolves over time
   - Becomes more accurate
```

## Fact Categories

Initial categories (extensible):
- `preference`: Likes/dislikes
- `event`: Things that happened
- `relationship`: People/connections
- `belief`: Opinions/beliefs
- `fact`: Objective statements
- `location`: Places mentioned
- `goal`: Intentions/goals
- `emotion`: Emotional states

## Performance Targets

- **Single prompt**: <100ms (rules) or <500ms (LLM)
- **Batch of 100**: <5s total (parallel processing)
- **Pattern load**: <100ms
- **Store save**: Async, non-blocking

## Tech Stack

- **Python**: Primary language for ingest
- **LangChain**: LLM orchestration (lightweight)
- **PostgreSQL**: Data store (interface only for now)
- **AsyncIO**: Concurrent processing
- **Pydantic**: Data validation

## Next Steps

1. Implement models (fact, prompt)
2. Implement base extractor interface
3. Implement rules extractor (simple patterns)
4. Implement LLM extractor (LangChain)
5. Implement hybrid orchestrator
6. Implement learning patterns
7. Implement feedback system
8. Implement store client interface
9. Create main ingest engine
10. Add example usage and tests

## Key Design Decisions

1. **Hybrid approach**: Balances speed and accuracy
2. **Learning-enabled**: Evolves without manual intervention
3. **Batch-first**: Optimized for high-throughput
4. **Metadata-rich**: High-dimensional for future index layer
5. **Error-tolerant**: Mistakes don't break the system
6. **Modular**: Easy to swap components (e.g., change LLM provider)
