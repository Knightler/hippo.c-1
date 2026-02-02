import argparse
import json
import os
import sys

from encode import EncodeEngine
from encode.models import Prompt, Role
from memory import MemoryClient


def _read_stdin() -> str:
    data = sys.stdin.read()
    return data.strip()


def _print_json(item: object) -> None:
    print(json.dumps(item, default=str, separators=(",", ":")))


def _encode(texts: list[str], role: Role) -> None:
    prompts = [Prompt(text=t, role=role) for t in texts]
    engine = EncodeEngine()
    engine.process(prompts)


def _labels(limit: int) -> None:
    client = MemoryClient()
    for label in client.list_labels(limit=limit):
        _print_json(
            {
                "id": label.id,
                "name": label.name,
                "kind": label.kind,
                "category": label.category,
                "usage_count": label.usage_count,
                "updated_at": label.updated_at,
                "created_at": label.created_at,
            }
        )


def _facts_latest(limit: int) -> None:
    client = MemoryClient()
    for fact in client.list_latest_facts(limit=limit):
        _print_json(fact)


def _facts_by_label(label: str, limit: int) -> None:
    client = MemoryClient()
    for fact in client.list_facts_by_label(label, limit=limit):
        _print_json(fact)


def _show_logs(limit: int) -> None:
    log_file = os.getenv("HIPPO_LOG_FILE", "")
    if not log_file:
        print("HIPPO_LOG_FILE is not set")
        return
    try:
        with open(log_file, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except FileNotFoundError:
        print("log file not found")
        return
    for line in lines[-limit:]:
        print(line.rstrip("\n"))


def main() -> None:
    parser = argparse.ArgumentParser(description="hippo.c-1")
    sub = parser.add_subparsers(dest="cmd", required=True)

    encode_parser = sub.add_parser("encode")
    encode_parser.add_argument("--text", action="append", help="prompt text")
    encode_parser.add_argument("--role", default="user", choices=["user", "ai", "system"])

    inspect_parser = sub.add_parser("inspect")
    inspect_sub = inspect_parser.add_subparsers(dest="inspect_cmd", required=True)

    labels_parser = inspect_sub.add_parser("labels")
    labels_parser.add_argument("--limit", type=int, default=50)

    facts_parser = inspect_sub.add_parser("facts")
    facts_parser.add_argument("--latest", type=int, default=0)
    facts_parser.add_argument("--label", type=str, default="")
    facts_parser.add_argument("--limit", type=int, default=20)

    logs_parser = sub.add_parser("logs")
    logs_parser.add_argument("--limit", type=int, default=200)

    args = parser.parse_args()

    if args.cmd == "encode":
        texts = args.text or []
        if not texts:
            stdin_text = _read_stdin()
            if stdin_text:
                texts = [stdin_text]
        if not texts:
            encode_parser.print_usage()
            return
        _encode(texts=texts, role=Role(args.role))
        return

    if args.cmd == "inspect":
        if args.inspect_cmd == "labels":
            _labels(limit=args.limit)
            return
        if args.inspect_cmd == "facts":
            if args.latest:
                _facts_latest(limit=args.latest)
                return
            if args.label:
                _facts_by_label(label=args.label, limit=args.limit)
                return
            facts_parser.error("facts requires --latest or --label")
        return

    if args.cmd == "logs":
        _show_logs(limit=args.limit)
        return


if __name__ == "__main__":
    main()
