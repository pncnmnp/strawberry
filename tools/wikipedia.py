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


def search_wikipedia(query: str) -> str:
    query = wiki_normalize(query)
    log("wiki", f"normalized query: {query}")
    try:
        result = wikipedia.summary(query, sentences=3, auto_suggest=True)
    except wikipedia.DisambiguationError as e:
        return search_wikipedia(e.options[0])
    except wikipedia.PageError:
        return f"No Wikipedia page found for '{query}'."
    log("wiki", result)
    return result
