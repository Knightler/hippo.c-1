# hippo.c-1

Minimal memory system (encode + memory). Local Postgres by default.

## Local setup

1) Start local Postgres and load schema:

```bash
./scripts/local_db.sh
```

2) Create `.env`:

```bash
cp .env.example .env
```

3) Create a venv and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

4) Run the interactive prompt:

```bash
./hippo chat
```

5) Watch live logs:

```bash
./hippo logs --follow
```

6) Watch live memory changes:

```bash
./hippo watch
```

## Optional LLM extraction

### OpenRouter (recommended)
```bash
OPENROUTER_API_KEY=your_key
OPENROUTER_API_BASE=https://openrouter.ai/api/v1
OPENROUTER_MODEL=google/gemini-2.0-flash-001
OPENROUTER_APP_URL=http://localhost
OPENROUTER_APP_NAME=hippo.c-1
```

### DeepSeek
```bash
DEEPSEEK_API_KEY=your_key
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

The extractor uses OpenRouter if `OPENROUTER_API_KEY` is set; otherwise it falls back to DeepSeek.
