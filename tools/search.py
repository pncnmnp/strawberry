from log import log
from pixies import wiki_normalize, search_route
from tools.wikipedia import _fetch, search_and_fetch
from tools.duckduckgo import search_duckduckgo

def search(query: str, backend: str = "auto") -> str:
    """Search for information on any topic — facts, people, places, current events, products, and more.

    Args:
        query: The topic or question to search for.
        backend: Which search backend to use. Use 'duckduckgo' if the user asks to avoid Wikipedia or wants web results. Defaults to 'auto' (router decides).
    """
    if backend == "auto":
        backend = search_route(query)
    log("search", f"[{backend}] {query}")

    if backend == "wikipedia":
        title = wiki_normalize(query)
        log("wikipedia", title)
        result = _fetch(title)
        if result.startswith("No Wikipedia page"):
            log("search", "normalized title miss, trying wikipedia search")
            result = search_and_fetch(query)
        if not result.startswith("No Wikipedia page"):
            return result
        log("search", "wikipedia miss, falling back to duckduckgo")

    return search_duckduckgo(query)
