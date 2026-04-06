import json
import ollama

_SYSTEM = """You expand a search query with synonyms and closely related terms to improve search recall.

Return ONLY a JSON array of strings. Include the original term. 5-7 terms max. Be specific — no generic filler.

Examples:
- "gym"      → ["gym", "fitness", "workout", "exercise", "training", "weights"]
- "passport" → ["passport", "visa", "travel document", "immigration"]
- "meeting"  → ["meeting", "call", "standup", "sync", "appointment", "discussion"]
- "money"    → ["money", "payment", "budget", "expense", "cost", "invoice", "bill"]"""


def expand(query: str) -> list[str]:
    response = ollama.chat(
        model="gemma4:e2b",
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": query},
        ],
    )
    try:
        terms = json.loads(response["message"]["content"].strip())
        if isinstance(terms, list):
            return [str(t) for t in terms]
    except (json.JSONDecodeError, ValueError):
        pass
    return [query]
