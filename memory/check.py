from memory import MemoryClient


def main() -> None:
    client = MemoryClient()
    ok = client.ping()
    if ok:
        print("memory: connection ok")
    else:
        print("memory: connection failed")


if __name__ == "__main__":
    main()
