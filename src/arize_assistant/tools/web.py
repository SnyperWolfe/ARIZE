"""Web tools: search (DuckDuckGo, no API key needed) and reading a page as plain text."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from .base import Tool, ToolContext, ToolError

USER_AGENT = "Mozilla/5.0 (compatible; ARIZE-assistant/0.1)"
SEARCH_URL = "https://html.duckduckgo.com/html/"

_RESULT = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
    r'(?:.*?class="result__snippet"[^>]*>(.*?)</a>)?',
    re.S,
)
_TAGS = re.compile(r"<[^>]+>")


def _clean(fragment: str) -> str:
    return " ".join(html.unescape(_TAGS.sub("", fragment or "")).split())


def _real_url(href: str) -> str:
    # DuckDuckGo wraps links as //duckduckgo.com/l/?uddg=<real url>
    if "uddg=" in href:
        return unquote(parse_qs(urlparse(href).query).get("uddg", [href])[0])
    return "https:" + href if href.startswith("//") else href


def parse_search_results(page: str, limit: int) -> list[dict]:
    results = []
    for href, title, snippet in _RESULT.findall(page):
        url = _real_url(href)
        if "duckduckgo.com/y.js" in url:  # ads
            continue
        results.append({"title": _clean(title), "url": url, "snippet": _clean(snippet)})
        if len(results) >= limit:
            break
    return results


def web_search(ctx: ToolContext, query: str, max_results: int = 5) -> str:
    try:
        resp = httpx.post(
            SEARCH_URL,
            data={"q": query},
            headers={"User-Agent": USER_AGENT},
            timeout=20,
            follow_redirects=True,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise ToolError(f"Search failed: {exc}") from exc
    results = parse_search_results(resp.text, max(1, min(int(max_results), 10)))
    if not results:
        return "No results found."
    return "\n\n".join(
        f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}" for i, r in enumerate(results, 1)
    )


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer", "form"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def html_to_text(page: str) -> tuple[str, str]:
    parser = _TextExtractor()
    parser.feed(page)
    lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
    text = "\n".join(line for line in lines if line)
    return " ".join(parser.title.split()), text


def read_webpage(ctx: ToolContext, url: str) -> str:
    if urlparse(url).scheme not in {"http", "https"}:
        raise ToolError("Only http and https links can be read")
    try:
        resp = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise ToolError(f"Could not load {url}: {exc}") from exc
    if "html" not in resp.headers.get("content-type", "html"):
        return resp.text
    title, text = html_to_text(resp.text)
    return f"# {title}\n\n{text}" if title else text


TOOLS = [
    Tool(
        "web_search",
        "Search the web. Returns titles, links and short snippets.",
        {
            "query": {"type": "string", "description": "What to search for"},
            "max_results": {"type": "integer", "description": "1-10, default 5"},
        },
        web_search,
        ["query"],
    ),
    Tool(
        "read_webpage",
        "Download a web page and return its readable text.",
        {"url": {"type": "string", "description": "Full http(s) link"}},
        read_webpage,
        ["url"],
    ),
]
