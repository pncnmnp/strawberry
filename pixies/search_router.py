from lm import chat

_SYSTEM = """You decide whether a search query should go to Wikipedia or DuckDuckGo, and if Wikipedia, provide the article title.

If Wikipedia: reply with ONLY the Wikipedia article title — no quotes, no explanation.
If DuckDuckGo: reply with ONLY the word "DUCKDUCKGO".

Use wikipedia when the query is a straightforward encyclopedic lookup — the kind answered by a single well-defined article.
Examples:
- "what is a black hole"          → Black hole
- "who was Napoleon"              → Napoleon
- "what is the Y combinator"      → Fixed-point combinator
- "history of the Roman Empire"   → Roman Empire
- "jaguar car"                    → Jaguar Cars
- "python language"               → Python (programming language)

Use duckduckgo when the query is complex, comparative, opinionated, or requires synthesising across many sources.
Examples:
- "competitors of American Express"           → DUCKDUCKGO
- "best noise cancelling headphones 2024"     → DUCKDUCKGO
- "how does Stripe compare to Braintree"      → DUCKDUCKGO
- "latest iPhone specs"                       → DUCKDUCKGO"""


def route(query: str) -> tuple[str, str | None]:
    """Returns (backend, wiki_title | None)."""
    decision = chat(_SYSTEM, query).strip()
    if decision.upper() == "DUCKDUCKGO":
        return "duckduckgo", None
    return "wikipedia", decision
