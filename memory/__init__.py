from memory import embedder, store
from memory.extractor import queue_turn, flush


def load():
    embedder.load()
    store.load()


def retrieve(text: str) -> str:
    return store.retrieve(text)


__all__ = ["load", "retrieve", "queue_turn", "flush"]
