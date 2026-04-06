from log import log
from pixies import wiki_normalize, search_route
from tools.wikipedia import _fetch
from tools.duckduckgo import search_duckduckgo

DEFINITION = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search for information on any topic — facts, people, places, current events, products, and more.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic or question to search for.",
                },
            },
            "required": ["query"],
        },
    },
}


def search(query: str) -> str:
    backend = search_route(query)
    log("search", f"[{backend}] {query}")

    if backend == "wikipedia":
        title = wiki_normalize(query)
        log("wikipedia", title)
        result = _fetch(title)
        if not result.startswith("No Wikipedia page"):
            return result
        log("search", "wikipedia miss, falling back to duckduckgo")

    return search_duckduckgo(query)
