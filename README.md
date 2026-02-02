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
python main.py chat
```

5) Watch live logs:

```bash
python main.py logs --follow
```

6) Watch live memory changes:

```bash
python main.py watch
```
