import ollama

_SYSTEM = """You decide whether a search query should go to Wikipedia or DuckDuckGo.

Reply with ONLY one word: "wikipedia" or "duckduckgo"

Use wikipedia when the query is a straightforward encyclopedic lookup — the kind answered by a single well-defined article.
Examples: "what is a black hole", "who was Napoleon", "what is the Y combinator", "history of the Roman Empire"

Use duckduckgo when the query is complex, comparative, opinionated, or requires synthesising across many sources — the kind of thing you'd type into Google.
Examples: "competitors of American Express", "best noise cancelling headphones 2024", "how does Stripe compare to Braintree", "latest iPhone specs", "what is Discover card known for", "why is inflation rising" """


def route(query: str) -> str:
    response = ollama.chat(
        model="gemma4:e2b",
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": query},
        ],
    )
    decision = response["message"]["content"].strip().lower()
    return "wikipedia" if "wikipedia" in decision else "duckduckgo"
