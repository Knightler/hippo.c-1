import argparse
import json

from memory import MemoryClient


def _print_json(item: object) -> None:
    print(json.dumps(item, default=str, separators=(",", ":")))


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


def main() -> None:
    parser = argparse.ArgumentParser(description="hippo.c-1 memory inspect")
    sub = parser.add_subparsers(dest="cmd", required=True)

    labels_parser = sub.add_parser("labels")
    labels_parser.add_argument("--limit", type=int, default=50)

    facts_parser = sub.add_parser("facts")
    facts_parser.add_argument("--latest", type=int, default=0)
    facts_parser.add_argument("--label", type=str, default="")
    facts_parser.add_argument("--limit", type=int, default=20)

    args = parser.parse_args()

    if args.cmd == "labels":
        _labels(limit=args.limit)
        return

    if args.cmd == "facts":
        if args.latest:
            _facts_latest(limit=args.latest)
            return
        if args.label:
            _facts_by_label(label=args.label, limit=args.limit)
            return
        parser.error("facts requires --latest or --label")


if __name__ == "__main__":
    main()
