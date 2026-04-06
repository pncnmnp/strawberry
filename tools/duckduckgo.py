import re
from ddgs import DDGS
from log import log


def _clean(text: str) -> str:
    # DDG strips HTML tags, leaving words jammed together — insert spaces at boundaries
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def search_duckduckgo(query: str) -> str:
    results = DDGS().text(query, max_results=4)
    if not results:
        return f"No results found for '{query}'."

    for r in results:
        log("duckduckgo", r["href"])

    return "\n\n".join(
        f"{r['title']}\n{_clean(r['body'])}"
        for r in results
    )
