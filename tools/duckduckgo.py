import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
import httpx
from ddgs import DDGS
from log import log


_CLIENT = httpx.Client(
    http2=True,
    follow_redirects=True,
    timeout=3.0,
    headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    },
)


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


def _fetch_page(url: str) -> str:
    url = _clean_url(url)
    try:
        with _CLIENT.stream("GET", url) as resp:
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return ""
            buf = bytearray()
            for chunk in resp.iter_bytes():
                buf.extend(chunk)
                if len(buf) >= 500_000:
                    break
            html = bytes(buf[:500_000]).decode("utf-8", errors="ignore")
    except Exception as e:
        log("duckduckgo", f"fetch failed {url}: {type(e).__name__}: {e}")
        return ""

    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_text()[:4000]


def fetch_sources(query: str) -> list[tuple[str, str]]:
    """Run a DDG search, scrape pages in parallel, return up to 4 (title, text) sources.
    Falls back to the DDG snippet for any page that fails to scrape (commercial sites often block bots)."""
    results = DDGS().text(query + " -site:youtube.com -site:youtu.be -site:vimeo.com -site:twitch.tv", max_results=6)
    if not results:
        return []

    for r in results:
        log("duckduckgo", r["href"])

    sources: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_fetch_page, r["href"]): r for r in results}
        for future in as_completed(futures):
            r = futures[future]
            if text := future.result():
                sources.append((r["title"], text))
            elif snippet := _clean(r.get("body", "")):
                log("duckduckgo", f"scrape failed, using snippet: {r['href']}")
                sources.append((r["title"], snippet))

    return sources[:3]
