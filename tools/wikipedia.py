import wikipedia


def _fetch(title: str) -> str:
    try:
        page = wikipedia.page(title, auto_suggest=False)
        return page.summary
    except wikipedia.DisambiguationError as e:
        return _fetch(e.options[0])
    except wikipedia.PageError:
        return f"No Wikipedia page found for '{title}'."
