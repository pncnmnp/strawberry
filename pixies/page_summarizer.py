from lm import chat

_SYSTEM = "You extract relevant facts from web pages. Include specific details, names, dates, and figures where present. Do not add anything not in the source."


def summarize(query: str, title: str, text: str) -> str:
    prompt = (
        f"Title: {title}\n\n"
        f"Content:\n{text[:4000]}\n\n"
        f"In 4-6 sentences, summarize the key facts this page contains about: {query}"
    )
    return chat(_SYSTEM, prompt).strip()
