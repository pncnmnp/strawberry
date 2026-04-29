from concurrent.futures import ThreadPoolExecutor

from log import log
from pixies import page_summarize
from tools.duckduckgo import fetch_sources as ddg_fetch_sources
from tools.wikipedia import fetch_source as wiki_fetch_source


def search(query: str, backend: str = "auto") -> str:
    """Search the web for information. Slow (~30s) — only use when necessary.
    Good reasons to search: current events, recent news, real-time data, or when you are genuinely uncertain about a fact and accuracy matters.
    Bad reasons to search: you can answer confidently, the user is just chatting, or the question is straightforward reasoning or math.

    Args:
        query: The topic or question to search for.
        backend: 'auto' (default) combines Wikipedia and web results. Use 'duckduckgo' to skip Wikipedia.
    """
    log("search", f"[{backend}] {query}")
    sources: list[tuple[str, str]] = []

    if backend == "duckduckgo":
        sources.extend(ddg_fetch_sources(query))
    elif backend == "wikipedia":
        if wiki := wiki_fetch_source(query):
            sources.append(wiki)
        if not sources:
            sources.extend(ddg_fetch_sources(query))
    else:
        # auto: pull wiki and ddg in parallel and let the summarizer blend them
        # NOTE: Previously, we used to have a pixie call that would decide which backend to 
        # use based on the query, however, that added significant latency. For local llms + 
        # realtime conversations, this is a better approximation.
        with ThreadPoolExecutor(max_workers=2) as pool:
            wiki_future = pool.submit(wiki_fetch_source, query)
            ddg_future = pool.submit(ddg_fetch_sources, query)
            if wiki := wiki_future.result():
                sources.append(wiki)
            sources.extend(ddg_future.result())

    sources = sources[:3]
    if not sources:
        return f"No results found for '{query}'."

    # NOTE: High latency call, needs further optimization.
    synthesis = page_summarize(query, sources)
    header = "Sources:\n" + "\n".join(f"[{i+1}] {title}" for i, (title, _) in enumerate(sources))
    return f"{header}\n\n{synthesis}"
