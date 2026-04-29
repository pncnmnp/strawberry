from lm import chat
from log import log

_SYSTEM = "You extract relevant facts from web pages. Include specific details, names, dates, and figures where present. Do not add anything not in the source."


def summarize(query: str, sources: list[tuple[str, str]]) -> str:
    """Synthesize multiple sources in a single LLM call. sources: [(title, text), ...]."""
    blocks = "\n\n".join(
        f"[{i+1}] {title}\n{text}"
        for i, (title, text) in enumerate(sources)
    )
    prompt = (
        f"Sources:\n{blocks}\n\n"
        f"In 6-8 sentences, synthesize what these sources say about: {query}. "
        f"Cite source numbers like [1], [2] when stating facts."
    )
    return chat(_SYSTEM, prompt).strip()
