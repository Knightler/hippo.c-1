# hippo.c-1 Project Notes

## Vision
Build a lightweight, fast, and private memory layer for AI that can:
- Encode user and assistant interactions into compact, normalized facts.
- Learn reusable patterns over time with confidence and timestamps.
- Store memory for ultra-fast search and relevance ranking.
- Recall only the most relevant memory for future prompts.

This is a personal memory system first. It should be useful without any LLM, and improved when an LLM is available.

## What We Have Built (Current State)

### Core Pipeline
- **Encode**: splits prompts into clauses, extracts compact facts, normalizes them, assigns labels, and stores them.
- **Memory**: Postgres + pgvector storage with labels, facts, patterns, and learned patterns.
- **Patterns**: static regex patterns (cheap) + learned patterns (from prompts + facts).
- **Observability**: live logs, live memory watch, and pretty inspection views.

### Key Behavior (Now Working)
- Multi-fact extraction per prompt.
- Compact normalized facts (not full sentences).
- Per-fact labels derived from the object/subject.
- Learned patterns stored separately with confidence + timestamps.

### Local Setup
1) Start local DB and load schema:
```bash
./scripts/local_db.sh
```

2) Create `.env`:
```bash
cp .env.example .env
```

3) Install deps:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

4) Run interactive chat:
```bash
./hippo chat
```

5) Watch live logs:
```bash
./hippo logs --follow
```

6) Watch live memory:
```bash
./hippo watch --pretty
```

7) Inspect facts/patterns snapshots:
```bash
./hippo inspect facts --latest 20 --pretty
./hippo inspect patterns --pretty
```

### Optional LLM (DeepSeek)
Enable if you want LLM inference for extraction:
```
DEEPSEEK_API_KEY=your_key
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```
The system runs fine without LLM.

## Extraction Rules (Current)

### Clause Splitting
Splits on punctuation, then only splits on conjunctions when a new subject starts (e.g., "and I ...").

### Semantic Extraction
Lightweight parser for:
- preferences (likes, dislikes, prefers)
- habits (always)
- goals (wants to, needs to, plans to)
- emotions (feels)
- identity (age is, is ...)
- relationships (my X is Y)
- location (lives in, from)

### Normalization
- Lowercase
- Remove filler words
- Trim to max word count
- Enforce compact fact length

### Labels
Derived from the object/subject (e.g., "likes jazz" → label: "jazz").

### Learned Patterns
- Prompt sentences generate low-confidence candidates.
- Facts promote patterns with higher confidence.
- Stored in `learned_patterns` table with `confidence`, `uses`, `successes`.

## Memory Schema (Key Tables)
- `labels`: topic anchors + embeddings + usage_count
- `facts`: compact facts + confidence + timestamps
- `patterns`: static regex patterns
- `learned_patterns`: learned templates + confidence
- `prompts`: raw prompt provenance

## Known Tradeoffs
- Current extraction is rule-based; accuracy depends on patterns.
- No semantic embeddings yet (only hash embeddings).
- Learned patterns are early-stage; still need stronger validation.

## Next Work: Encode Improvements (Short Term)
1) Expand semantic patterns for more verbs (need, have, use, study, work, etc).
2) Better label normalization (noun phrase extraction).
3) Confidence tuning based on evidence_count + recency.
4) Cleanly separate assistant-derived facts (source_role=ai) with metadata markers.

## Recall Plan (Detailed)

### Goal
Return a short, high-quality memory summary relevant to the current prompt.

### Inputs
- Current prompt text
- Recent conversation context (optional)

### Steps
1) **Topic detection**: derive candidate labels from the prompt using the same compact extractor.
2) **Search**:
   - label embedding search (top K)
   - category filter (optional)
   - recency/usage filter
3) **Scoring**:
   - score = confidence * recency_weight * usage_weight
   - promote facts seen recently but not over-repeating
4) **Selection**:
   - select top N facts (e.g., 5–10)
   - group by label to avoid redundancy
5) **Output format**:
   - short list of compact facts + labels
   - provide a "memory summary" string for LLM consumption

### Output Example
```
memory:
- prefers concise, structured writing (label: writing style)
- likes jazz (label: jazz)
- dad lives in Toronto (label: dad)
```

### Recall API (Proposed CLI)
```
./hippo recall --text "..."
```
Returns a compact memory block.

## Project Decisions
- Keep it lean and modular (no heavy frameworks).
- Prefer deterministic extraction + compact normalization over raw prompt storage.
- Use optional LLM inference only as a refiner, not a dependency.
- Learned patterns are stored and updated, not just fixed regex.

## What Success Looks Like
- The system accurately captures important user facts in compact form.
- Repeated signals reinforce memory instead of duplicating.
- Recall returns only relevant facts, not noise.
- Everything runs locally with clear logs and inspect commands.
