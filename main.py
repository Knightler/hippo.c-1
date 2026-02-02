import argparse
import sys

from encode import EncodeEngine
from encode.models import Prompt, Role


def _read_stdin() -> str:
    data = sys.stdin.read()
    return data.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="hippo.c-1 encode runner")
    parser.add_argument("--text", action="append", help="prompt text")
    parser.add_argument("--role", default="user", choices=["user", "ai", "system"])
    args = parser.parse_args()

    texts = args.text or []
    if not texts:
        stdin_text = _read_stdin()
        if stdin_text:
            texts = [stdin_text]

    if not texts:
        parser.print_usage()
        return

    role = Role(args.role)
    prompts = [Prompt(text=t, role=role) for t in texts]
    engine = EncodeEngine()
    engine.process(prompts)


if __name__ == "__main__":
    main()
