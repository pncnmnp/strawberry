from lm import chat

_SYSTEM = "You summarize voice-assistant conversations for memory compaction. Terse, factual, no preamble."


def compact(history: list) -> str:
    lines = [
        f"{'Sir' if m['role'] == 'user' else 'Strawberry'}: {m['content']}"
        for m in history
        if m["role"] in ("user", "assistant") and isinstance(m.get("content"), str)
    ]
    prompt = (
        "\n".join(lines)
        + "\n\nIn 3-5 short bullets, capture: facts Sir shared about himself, "
        "ongoing state (music playing, notes written, etc.), and anything Strawberry "
        "should remember going forward. Skip small talk and pleasantries."
    )
    return chat(_SYSTEM, prompt).strip()
