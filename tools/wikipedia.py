from concurrent.futures import ThreadPoolExecutor
import wikipedia


def _fetch(title: str) -> str | None:
    try:
        page = wikipedia.page(title, auto_suggest=False)
        return page.summary
    except wikipedia.DisambiguationError as e:
        return _fetch(e.options[0])
    except wikipedia.PageError:
        return None


def fetch_source(query: str) -> tuple[str, str] | None:
    """Search Wikipedia and return (title, summary) of the highest-ranked matching article, or None."""
    results = wikipedia.search(query, results=10)
    if not results:
        return None
    # Fan out fetches in parallel; walk in rank order so we still return the highest-ranked hit.
    with ThreadPoolExecutor(max_workers=len(results)) as pool:
        future_titles = [(title, pool.submit(_fetch, title)) for title in results]
        for title, future in future_titles:
            if (text := future.result()) is not None:
                return (title, text)
    return None
