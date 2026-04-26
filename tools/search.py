from concurrent.futures import ThreadPoolExecutor, Future
from log import log
from pixies import search_route
from tools.wikipedia import _fetch, search_and_fetch
from tools.duckduckgo import search_duckduckgo

def search(query: str, backend: str = "auto") -> str:
    """Search the web for information. Very slow (30-60s) — only use when necessary.
    Good reasons to search: current events, recent news, real-time data, or when you are genuinely uncertain about a fact and accuracy matters.
    Bad reasons to search: you can answer confidently, the user is just chatting, or the question is straightforward reasoning or math.

    Args:
        query: The topic or question to search for.
        backend: Which search backend to use. Use 'duckduckgo' if the user asks to avoid Wikipedia or wants web results. Defaults to 'auto' (router decides).
    """
    if backend == "duckduckgo":
        log("search", f"[duckduckgo] {query}")
        return search_duckduckgo(query)

    if backend != "auto":
        # explicit wikipedia
        log("search", f"[wikipedia] {query}")
        return _wiki_path(query, query) or search_duckduckgo(query)

    # Speculative parallelism: fire duckduckgo (network) alongside the fused
    # router+normalizer (LLM) so the losing branch is already in-flight.
    with ThreadPoolExecutor(max_workers=1) as pool:
        ddg_future: Future = pool.submit(search_duckduckgo, query)

        backend, title = search_route(query)
        log("search", f"[{backend}] {query}")

        if backend == "wikipedia" and title:
            log("wikipedia", title)
            if (result := _wiki_path(query, title)) is not None:
                return result

        return ddg_future.result()


def _wiki_path(query: str, title: str) -> str | None:
    """Try Wikipedia fetch + search. Returns content or None on miss."""
    result = _fetch(title)
    if result.startswith("No Wikipedia page"):
        log("search", "normalized title miss, trying wikipedia search")
        result = search_and_fetch(query)
    if not result.startswith("No Wikipedia page"):
        return result
    log("search", "wikipedia miss, falling back to duckduckgo")
    return None
