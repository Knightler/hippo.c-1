import argparse
import json
import os
import sys
import time

from core import log


def _handle_missing_dep(exc: Exception) -> bool:
    if isinstance(exc, ModuleNotFoundError):
        name = exc.name or "dependency"
        print(f"missing dependency: {name}")
        print("activate venv and run: python -m pip install -e .")
        return True
    return False


def _read_stdin() -> str:
    data = sys.stdin.read()
    return data.strip()


def _load_env() -> None:
    if os.getenv("SUPABASE_DATABASE_URL"):
        return
    if not os.path.exists(".env"):
        return
    with open(".env", "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"")
            if key and key not in os.environ:
                os.environ[key] = value


def _print_json(item: object) -> None:
    print(json.dumps(item, default=str, separators=(",", ":")))


def _encode(texts: list[str], role) -> None:
    try:
        from encode.encode import EncodeEngine
        from encode.models import Prompt
        prompts = [Prompt(text=t, role=role) for t in texts]
        engine = EncodeEngine()
        updates = engine.process(prompts)
        print(f"encoded updates={len(updates)}")
    except Exception as exc:
        if _handle_missing_dep(exc):
            return
        log("error", "encode_failed", error=type(exc).__name__, message=str(exc))
        print(f"encode failed: {exc}")


def _labels(limit: int) -> None:
    client = None
    try:
        from memory import MemoryClient
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
    except Exception as exc:
        if _handle_missing_dep(exc):
            return
        print(f"inspect labels failed: {exc}")
    finally:
        if client:
            client.close()


def _facts_latest(limit: int) -> None:
    client = None
    try:
        from memory import MemoryClient
        client = MemoryClient()
        for fact in client.list_latest_facts(limit=limit):
            _print_json(fact)
    except Exception as exc:
        if _handle_missing_dep(exc):
            return
        print(f"inspect facts failed: {exc}")
    finally:
        if client:
            client.close()


def _facts_by_label(label: str, limit: int) -> None:
    client = None
    try:
        from memory import MemoryClient
        client = MemoryClient()
        for fact in client.list_facts_by_label(label, limit=limit):
            _print_json(fact)
    except Exception as exc:
        if _handle_missing_dep(exc):
            return
        print(f"inspect facts failed: {exc}")
    finally:
        if client:
            client.close()


def _show_logs(limit: int) -> None:
    log_file = os.getenv("HIPPO_LOG_FILE", ".hippo.log")
    if not log_file or log_file.lower() in {"off", "none", "0"}:
        print("logging is disabled")
        return
    if not os.path.exists(log_file):
        with open(log_file, "a", encoding="utf-8"):
            pass
    with open(log_file, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    for line in lines[-limit:]:
        print(line.rstrip("\n"))


def _follow_logs(limit: int) -> None:
    log_file = os.getenv("HIPPO_LOG_FILE", ".hippo.log")
    if not log_file or log_file.lower() in {"off", "none", "0"}:
        print("logging is disabled")
        return
    if not os.path.exists(log_file):
        with open(log_file, "a", encoding="utf-8"):
            pass
    with open(log_file, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
        for line in lines[-limit:]:
            print(line.rstrip("\n"))
        while True:
            line = handle.readline()
            if line:
                print(line.rstrip("\n"))
            else:
                time.sleep(0.2)


def _chat() -> None:
    print("hippo.c-1 interactive")
    print("Type :exit to quit")
    print("Type :paste for multi-line, end with :end")
    while True:
        try:
            text = input("hippo> ").strip()
        except KeyboardInterrupt:
            print("\nchat stopped")
            break
        except EOFError:
            print()
            break
        if not text:
            continue
        if text == ":exit":
            break
        if text == ":paste":
            lines: list[str] = []
            while True:
                line = input()
                if line.strip() == ":end":
                    break
                lines.append(line)
            text = "\n".join(lines).strip()
            if not text:
                continue
        from encode.models import Role
        _encode([text], role=Role.USER)


def _watch_memory(
    watch_facts: bool,
    watch_labels: bool,
    watch_patterns: bool,
    interval: float,
    initial: bool,
) -> None:
    client = None
    last_error = ""
    try:
        from memory import MemoryClient
        client = MemoryClient()
    except Exception as exc:
        if _handle_missing_dep(exc):
            return
        print(f"watch failed: {exc}")
        return
    seen_facts: set[tuple[str, object]] = set()
    seen_labels: set[tuple[str, object]] = set()
    seen_patterns: set[tuple[str, object]] = set()

    def load_initial() -> None:
        if watch_facts:
            for fact in client.list_latest_facts(limit=200):
                seen_facts.add((fact["id"], fact["last_seen_at"]))
                if initial:
                    _print_json(fact)
        if watch_labels:
            for label in client.list_labels(limit=200):
                seen_labels.add((label.id, label.updated_at))
                if initial:
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
        if watch_patterns:
            for pattern in client.list_patterns(limit=200):
                seen_patterns.add((pattern["id"], pattern["updated_at"]))
                if initial:
                    _print_json(pattern)

    try:
        try:
            load_initial()
        except Exception as exc:
            last_error = f"watch error: {exc}"
            print(last_error)
        while True:
            try:
                if watch_facts:
                    for fact in client.list_latest_facts(limit=200):
                        key = (fact["id"], fact["last_seen_at"])
                        if key not in seen_facts:
                            seen_facts.add(key)
                            _print_json(fact)
                if watch_labels:
                    for label in client.list_labels(limit=200):
                        key = (label.id, label.updated_at)
                        if key not in seen_labels:
                            seen_labels.add(key)
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
                if watch_patterns:
                    for pattern in client.list_patterns(limit=200):
                        key = (pattern["id"], pattern["updated_at"])
                        if key not in seen_patterns:
                            seen_patterns.add(key)
                            _print_json(pattern)
                if last_error:
                    last_error = ""
            except Exception as exc:
                message = f"watch error: {exc}"
                if message != last_error:
                    print(message)
                    last_error = message
            time.sleep(interval)
    except KeyboardInterrupt:
        print("watch stopped")
    finally:
        if client:
            client.close()


def main() -> None:
    _load_env()
    parser = argparse.ArgumentParser(description="hippo.c-1")
    sub = parser.add_subparsers(dest="cmd", required=True)

    encode_parser = sub.add_parser("encode")
    encode_parser.add_argument("--text", action="append", help="prompt text")
    encode_parser.add_argument("--role", default="user", choices=["user", "ai", "system"])

    sub.add_parser("chat")

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
    logs_parser.add_argument("--follow", action="store_true")

    watch_parser = sub.add_parser("watch")
    watch_parser.add_argument("--facts", action="store_true")
    watch_parser.add_argument("--labels", action="store_true")
    watch_parser.add_argument("--patterns", action="store_true")
    watch_parser.add_argument("--interval", type=float, default=0.5)
    watch_parser.add_argument("--initial", action="store_true")

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
        from encode.models import Role
        _encode(texts=texts, role=Role(args.role))
        return

    if args.cmd == "chat":
        _chat()
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
        if args.follow:
            _follow_logs(limit=args.limit)
        else:
            _show_logs(limit=args.limit)
        return

    if args.cmd == "watch":
        watch_facts = args.facts or not (args.facts or args.labels or args.patterns)
        watch_labels = args.labels or not (args.facts or args.labels or args.patterns)
        watch_patterns = args.patterns or not (args.facts or args.labels or args.patterns)
        _watch_memory(
            watch_facts=watch_facts,
            watch_labels=watch_labels,
            watch_patterns=watch_patterns,
            interval=args.interval,
            initial=args.initial,
        )
        return


if __name__ == "__main__":
    main()
