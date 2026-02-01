import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Iterable

from ingest.models import Fact, Label


class MemoryStore:
    """Local memory state for ingest (labels + facts).

    This is not the long-term DB. It is the ingest-side state
    used for learning and incremental updates.
    """

    def __init__(self, path: str = "ingest/memory/state.json"):
        self.path = path
        self.labels: dict[str, Label] = {}
        self.facts: dict[str, Fact] = {}
        self._ensure_dir()
        self.load()

    def _ensure_dir(self) -> None:
        directory = os.path.dirname(self.path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r") as f:
            data = json.load(f)
        for raw in data.get("labels", []):
            label = Label(**self._decode_datetimes(raw))
            self.labels[label.id] = label
        for raw in data.get("facts", []):
            fact = Fact(**self._decode_datetimes(raw))
            self.facts[fact.id] = fact

    def save(self) -> None:
        data = {
            "labels": [self._encode_datetimes(asdict(l)) for l in self.labels.values()],
            "facts": [self._encode_datetimes(asdict(f)) for f in self.facts.values()],
            "updated_at": datetime.utcnow().isoformat(),
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def upsert_label(self, name: str, kind: str, embedding: list[float]) -> Label:
        existing = self.find_label_by_name(name)
        if existing:
            existing.embedding = embedding or existing.embedding
            existing.kind = kind or existing.kind
            existing.touch()
            return existing
        label = Label(name=name, kind=kind, embedding=embedding)
        self.labels[label.id] = label
        return label

    def upsert_fact(self, fact: Fact) -> Fact:
        existing = self.facts.get(fact.id)
        if existing:
            self.facts[fact.id] = fact
            return fact
        self.facts[fact.id] = fact
        return fact

    def find_label_by_name(self, name: str) -> Label | None:
        name_norm = name.strip().lower()
        for label in self.labels.values():
            if label.name.strip().lower() == name_norm:
                return label
        return None

    def get_labels(self) -> Iterable[Label]:
        return self.labels.values()

    def get_facts_by_label(self, label_id: str) -> list[Fact]:
        return [f for f in self.facts.values() if f.label_id == label_id]

    def _encode_datetimes(self, payload: dict) -> dict:
        for key, value in payload.items():
            if isinstance(value, datetime):
                payload[key] = value.isoformat()
        return payload

    def _decode_datetimes(self, payload: dict) -> dict:
        for key, value in payload.items():
            if isinstance(value, str) and self._looks_like_dt(value):
                try:
                    payload[key] = datetime.fromisoformat(value)
                except ValueError:
                    pass
        return payload

    def _looks_like_dt(self, value: str) -> bool:
        return "T" in value and ":" in value
