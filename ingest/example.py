from ingest.ingest import IngestEngine
from ingest.models import MessageRole, Prompt, PromptBatch
from ingest.store import InMemoryStore


def main():
    store = InMemoryStore()
    engine = IngestEngine(store_client=store, learning_enabled=True)

    prompts = PromptBatch(
        prompts=[
            Prompt(text="I love ice cream and I live in Berlin.", role=MessageRole.USER),
            Prompt(text="I broke up with my girlfriend last week.", role=MessageRole.USER),
            Prompt(text="He hates Mr. X.", role=MessageRole.AI),
        ]
    )

    fact_ids = engine.process(prompts)
    print("Stored fact IDs:", fact_ids)
    print("Stats:", engine.get_stats())


if __name__ == "__main__":
    main()
