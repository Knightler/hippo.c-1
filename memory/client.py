import os

import psycopg
from psycopg_pool import ConnectionPool
from contextlib import contextmanager
from psycopg.types.json import Json
from pgvector.psycopg import register_vector, Vector

from encode.models import Fact, Label, Prompt


class MemoryClient:
    """Postgres client for memory storage."""

    def __init__(self, dsn_env: str = "SUPABASE_DATABASE_URL"):
        self.dsn = os.getenv(dsn_env, "")
        if not self.dsn:
            raise ValueError("SUPABASE_DATABASE_URL is not set")
        self.pool = ConnectionPool(self.dsn, min_size=0, max_size=5, timeout=10)

    @contextmanager
    def _connect(self):
        with self.pool.connection() as conn:
            register_vector(conn)
            yield conn

    def _json(self, value: dict) -> Json:
        return Json(value)

    def close(self) -> None:
        self.pool.close()

    def ping(self) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("select 1")
                return cur.fetchone() == (1,)

    def upsert_prompt(self, prompt: Prompt) -> str:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into prompts (id, role, text, timestamp, metadata)
                    values (%s, %s, %s, %s, %s)
                    on conflict (id) do nothing
                    returning id
                    """,
                    (
                        prompt.id,
                        prompt.role.value,
                        prompt.text,
                        prompt.timestamp,
                        self._json(prompt.metadata),
                    ),
                )
                row = cur.fetchone()
                return str(row[0]) if row else prompt.id

    def upsert_label(self, name: str, kind: str, category: str, embedding: list[float]) -> Label:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into labels (name, kind, category, embedding, usage_count, updated_at)
                    values (%s, %s, %s, %s, 1, now())
                    on conflict (name, kind, category)
                    do update set
                        embedding = excluded.embedding,
                        usage_count = labels.usage_count + 1,
                        updated_at = now()
                    returning id, name, kind, category, embedding, usage_count, created_at, updated_at, metadata
                    """,
                    (name, kind, category, embedding),
                )
                row = cur.fetchone()
        return Label(
            id=str(row[0]),
            name=row[1],
            kind=row[2],
            category=row[3],
            embedding=list(row[4]) if row[4] is not None else [],
            usage_count=row[5],
            created_at=row[6],
            updated_at=row[7],
            metadata=row[8],
        )

    def search_labels(self, embedding: list[float], top_k: int = 5) -> list[Label]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, name, kind, category, embedding, usage_count, created_at, updated_at, metadata
                    from labels
                    order by embedding <-> %s
                    limit %s
                    """,
                    (Vector(embedding), top_k),
                )
                rows = cur.fetchall()
        return [
            Label(
                id=str(r[0]),
                name=r[1],
                kind=r[2],
                category=r[3],
                embedding=list(r[4]) if r[4] is not None else [],
                usage_count=r[5],
                created_at=r[6],
                updated_at=r[7],
                metadata=r[8],
            )
            for r in rows
        ]

    def find_best_fact(self, label_id: str, embedding: list[float]) -> Fact | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, label_id, content, category, confidence, source_role,
                           embedding, created_at, updated_at, last_seen_at, evidence_count, metadata
                    from facts
                    where label_id = %s
                    order by embedding <-> %s
                    limit 1
                    """,
                    (label_id, Vector(embedding)),
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
            embedding=list(row[6]) if row[6] is not None else [],
            created_at=row[7],
            updated_at=row[8],
            last_seen_at=row[9],
            evidence_count=row[10],
            metadata=row[11],
        )

    def get_facts_by_label(self, label_id: str, limit: int = 3) -> list[Fact]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, label_id, content, category, confidence, source_role,
                           embedding, created_at, updated_at, last_seen_at, evidence_count, metadata
                    from facts
                    where label_id = %s
                    order by confidence desc, last_seen_at desc
                    limit %s
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
                embedding=list(r[6]) if r[6] is not None else [],
                created_at=r[7],
                updated_at=r[8],
                last_seen_at=r[9],
                evidence_count=r[10],
                metadata=r[11],
            )
            for r in rows
        ]

    def list_labels(self, limit: int = 50) -> list[Label]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, name, kind, category, embedding, usage_count, created_at, updated_at, metadata
                    from labels
                    order by updated_at desc
                    limit %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
        return [
            Label(
                id=str(r[0]),
                name=r[1],
                kind=r[2],
                category=r[3],
                embedding=list(r[4]) if r[4] is not None else [],
                usage_count=r[5],
                created_at=r[6],
                updated_at=r[7],
                metadata=r[8],
            )
            for r in rows
        ]

    def list_latest_facts(self, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select f.id, f.content, f.category, f.confidence, f.last_seen_at, f.created_at,
                           l.name as label_name
                    from facts f
                    join labels l on l.id = f.label_id
                    order by f.last_seen_at desc
                    limit %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
        return [
            {
                "id": str(r[0]),
                "content": r[1],
                "category": r[2],
                "confidence": r[3],
                "last_seen_at": r[4],
                "created_at": r[5],
                "label": r[6],
            }
            for r in rows
        ]

    def list_facts_by_label(self, label_name: str, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select f.id, f.content, f.category, f.confidence, f.last_seen_at, f.created_at,
                           l.name as label_name
                    from facts f
                    join labels l on l.id = f.label_id
                    where l.name = %s
                    order by f.confidence desc, f.last_seen_at desc
                    limit %s
                    """,
                    (label_name, limit),
                )
                rows = cur.fetchall()
        return [
            {
                "id": str(r[0]),
                "content": r[1],
                "category": r[2],
                "confidence": r[3],
                "last_seen_at": r[4],
                "created_at": r[5],
                "label": r[6],
            }
            for r in rows
        ]

    def insert_fact(self, fact: Fact) -> Fact:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into facts (
                        id, label_id, content, category, confidence, source_role, embedding,
                        created_at, updated_at, last_seen_at, evidence_count, metadata
                    )
                    values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    on conflict (id) do nothing
                    """,
                    (
                        fact.id,
                        fact.label_id,
                        fact.content,
                        fact.category,
                        fact.confidence,
                        fact.source_role,
                        fact.embedding,
                        fact.created_at,
                        fact.updated_at,
                        fact.last_seen_at,
                        fact.evidence_count,
                        self._json(fact.metadata),
                    ),
                )
        return fact

    def reinforce_fact(self, fact_id: str, boost: float) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    update facts
                    set confidence = least(1.0, confidence + %s),
                        evidence_count = evidence_count + 1,
                        last_seen_at = now(),
                        updated_at = now()
                    where id = %s
                    """,
                    (boost, fact_id),
                )

    def get_patterns(self) -> list[dict]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, regex, category, template, weight, uses, successes, updated_at, metadata
                    from patterns
                    order by updated_at desc
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

    def list_patterns(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, regex, category, template, weight, uses, successes, updated_at, metadata
                    from patterns
                    order by updated_at desc
                    limit %s
                    """,
                    (limit,),
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

    def upsert_pattern(self, regex: str, category: str, template: str, weight: float) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into patterns (regex, category, template, weight, updated_at)
                    values (%s, %s, %s, %s, now())
                    on conflict (regex)
                    do update set
                        category = excluded.category,
                        template = excluded.template,
                        weight = excluded.weight,
                        updated_at = now()
                    """,
                    (regex, category, template, weight),
                )

    def record_pattern_use(self, regex: str, success: bool) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    update patterns
                    set uses = uses + 1,
                        successes = successes + %s,
                        updated_at = now()
                    where regex = %s
                    """,
                    (1 if success else 0, regex),
                )

    def upsert_learned_pattern(
        self,
        signature: str,
        template: str,
        category: str,
        confidence: float,
        success: bool,
        metadata: dict,
    ) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into learned_patterns (
                        signature, template, category, confidence, uses, successes, updated_at, metadata
                    )
                    values (%s, %s, %s, %s, 1, %s, now(), %s)
                    on conflict (signature)
                    do update set
                        confidence = least(1.0, learned_patterns.confidence + %s),
                        uses = learned_patterns.uses + 1,
                        successes = learned_patterns.successes + %s,
                        updated_at = now(),
                        metadata = learned_patterns.metadata || excluded.metadata
                    """,
                    (
                        signature,
                        template,
                        category,
                        confidence,
                        1 if success else 0,
                        self._json(metadata),
                        confidence * 0.1,
                        1 if success else 0,
                    ),
                )

    def list_learned_patterns(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, signature, template, category, confidence, uses, successes, updated_at, metadata
                    from learned_patterns
                    order by updated_at desc
                    limit %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
        return [
            {
                "id": str(r[0]),
                "signature": r[1],
                "template": r[2],
                "category": r[3],
                "confidence": r[4],
                "uses": r[5],
                "successes": r[6],
                "updated_at": r[7],
                "metadata": r[8],
            }
            for r in rows
        ]
