import re
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from ddgs import DDGS
from log import log
from pixies import page_summarize


def _clean(text: str) -> str:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Tags whose content we skip entirely (scripts, styles, etc.)
_SKIP_TAGS = {"script", "style", "noscript", "head", "nav", "footer", "aside"}


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._skip_depth = 0  # how many "junk" tags deep we currently are
        self._parts: list[str] = []  # collected text fragments

    def handle_starttag(self, tag, attrs):  # noqa: ARG002
        # entering a junk tag (script, nav, etc.) — start ignoring content
        if tag in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        # leaving a junk tag — stop ignoring if we're back at the surface
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        # only collect text when we're not inside a junk tag
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._parts.append(text)

    def get_text(self) -> str:
        # join all fragments and collapse any leftover whitespace
        raw = " ".join(self._parts)
        raw = re.sub(r"\s+", " ", raw)
        return raw.strip()


def _clean_url(url: str) -> str:
    """Unwrap DuckDuckGo redirect URLs to get the real destination."""
    if "duckduckgo.com/l/?uddg=" in url:
        url = urllib.parse.unquote(url.split("uddg=")[1].split("&")[0])
    return url


def _fetch_page(url: str, timeout: int = 6) -> str:
    url = _clean_url(url)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                return ""
            html = resp.read(50_000).decode("utf-8", errors="ignore")
    except Exception:
        return ""

    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_text()[:5000]


def search_duckduckgo(query: str) -> str:
    results = DDGS().text(query + " -site:youtube.com -site:youtu.be -site:vimeo.com -site:twitch.tv", max_results=6)
    if not results:
        return f"No results found for '{query}'."

    for r in results:
        log("duckduckgo", r["href"])

    # Fetch pages in parallel; summarize each as soon as it lands (pipeline).
    # If scraping fails, fall back to the DDG snippet — short but real content.
    summaries = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_fetch_page, r["href"]): r for r in results}
        for future in as_completed(futures):
            r = futures[future]
            text = future.result()
            if text:
                summary = page_summarize(query, r["title"], text)
            else:
                snippet = _clean(r.get("body", ""))
                log("duckduckgo", f"scrape failed, using snippet: {r['href']}")
                summary = snippet
            if summary:
                summaries.append(f"Source: {r['title']}\n{summary}")
            if len(summaries) == 4:
                break

    if not summaries:
        return f"No usable results found for '{query}'."

    return "\n\n".join(summaries)
