import wikipedia
from log import log
from pixies import wiki_normalize

DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_wikipedia",
        "description": "Search Wikipedia for a summary of a topic when the user asks about a person, place, concept, or event.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic to search for on Wikipedia.",
                },
            },
            "required": ["query"],
        },
    },
}


def _fetch(title: str) -> str:
    try:
        page = wikipedia.page(title, auto_suggest=False)
        return page.summary
    except wikipedia.DisambiguationError as e:
        return _fetch(e.options[0])
    except wikipedia.PageError:
        return f"No Wikipedia page found for '{title}'."


def search_wikipedia(query: str) -> str:
    title = wiki_normalize(query)
    log("wiki", f"normalized: {title}")
    result = _fetch(title)
    log("wiki", result)
    return result
