import wikipedia


def _fetch(title: str) -> str:
    try:
        page = wikipedia.page(title, auto_suggest=False)
        return page.summary
    except wikipedia.DisambiguationError as e:
        return _fetch(e.options[0])
    except wikipedia.PageError:
        return f"No Wikipedia page found for '{title}'."


def search_and_fetch(query: str) -> str:
    """Use Wikipedia's own search to find the best article for a query."""
    results = wikipedia.search(query, results=10)
    if not results:
        return f"No Wikipedia page found for '{query}'."
    for title in results:
        result = _fetch(title)
        if not result.startswith("No Wikipedia page"):
            return result
    return f"No Wikipedia page found for '{query}'."
