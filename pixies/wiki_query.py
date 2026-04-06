import ollama

_SYSTEM = """You are a Wikipedia search assistant. Given a topic or phrase, return the most likely Wikipedia article title for it.

Rules:
- Return ONLY the article title, nothing else
- No punctuation, no explanation, no quotes
- Prefer the canonical Wikipedia title (e.g. "Y combinator" → "Fixed-point combinator", "WW2" → "World War II")
- If already a clean title, return it as-is"""


def normalize(query: str) -> str:
    response = ollama.chat(
        model="gemma4:e2b",
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": query},
        ],
    )
    return response["message"]["content"].strip()
