"""
ingest.py
---------
Stage: PAGES_FETCHED -> CONTENT_EXTRACTED

Fetches pages listed in pages.json, extracts translatable text segments
while preserving HTML structure, links, placeholders, and markup.
Saves results to extracted_segments.json.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
PAGES_JSON = ROOT / "pages.json"
EXTRACTED_SEGMENTS_JSON = ROOT / "extracted_segments.json"
FETCH_FAILURES_JSON = ROOT / "fetch_failures.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Tags whose text content we want to extract
EXTRACTABLE_TAGS = {
    "h1", "h2", "h3", "h4", "h5", "h6",
    "p", "li", "td", "th", "caption",
    "label", "button", "a",
    "span", "div",
    "blockquote", "figcaption",
}

# Tags we always skip entirely (no extraction from inside)
SKIP_TAGS = {
    "script", "style", "noscript", "meta", "link",
    "head", "svg", "path", "iframe", "code", "pre",
}

# Minimum character length for a segment to be worth translating
MIN_SEGMENT_LENGTH = 3

# Placeholder pattern: {{...}}, {%...%}, <%...%>, etc.
PLACEHOLDER_PATTERN = re.compile(
    r"(\{\{[^}]+\}\}|\{%[^%]+%\}|<%[^%]+%>|\$\{[^}]+\}|\[[A-Z_]+\])"
)

# URL-like strings we want to skip translating
URL_PATTERN = re.compile(r"https?://\S+")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_segment_id(page_url: str, html_path: str, source_text: str) -> str:
    """Stable, short ID derived from content + location."""
    raw = f"{page_url}|{html_path}|{source_text}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _build_html_path(element: Tag) -> str:
    """Build a CSS-selector-style path from the root to *element*."""
    parts: list[str] = []
    node = element
    while node and node.name:
        tag = node.name
        siblings = [s for s in node.parent.children if isinstance(s, Tag) and s.name == tag] if node.parent else []
        if len(siblings) > 1:
            idx = siblings.index(node) + 1
            tag = f"{tag}[{idx}]"
        parts.append(tag)
        node = node.parent  # type: ignore[assignment]
    return " > ".join(reversed(parts))


def _extract_placeholders(text: str) -> list[str]:
    return PLACEHOLDER_PATTERN.findall(text)


def _extract_links(element: Tag) -> list[str]:
    links: list[str] = []
    for a in element.find_all("a", href=True):
        href = a["href"].strip()
        if href and not href.startswith("#"):
            links.append(href)
    if element.name == "a" and element.get("href"):
        href = element["href"].strip()
        if href and not href.startswith("#") and href not in links:
            links.append(href)
    return links


def _contains_html(text: str) -> bool:
    """True if the raw inner-HTML contains any tags."""
    return bool(re.search(r"<[a-zA-Z]", text))


def _is_translatable(text: str) -> bool:
    """Return True when the text is worth translating."""
    stripped = text.strip()
    if len(stripped) < MIN_SEGMENT_LENGTH:
        return False
    if re.fullmatch(r"[\d\s\W]+", stripped):
        return False
    if URL_PATTERN.fullmatch(stripped):
        return False
    return True


def _inner_html(element: Tag) -> str:
    return "".join(str(c) for c in element.children)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def _extract_from_element(
    element: Tag,
    page_url: str,
    seen_paths: set[str],
) -> list[dict[str, Any]]:
    """
    Recursively walk *element*. Yield segment dicts for leaf-ish nodes
    whose direct text content (including inline markup) is translatable.
    """
    segments: list[dict[str, Any]] = []

    if element.name in SKIP_TAGS:
        return segments

    has_direct_text = any(
        isinstance(c, NavigableString) and c.strip()
        for c in element.children
    )

    if element.name in EXTRACTABLE_TAGS and has_direct_text:
        raw_inner = _inner_html(element).strip()
        plain_text = element.get_text(" ", strip=True)

        if _is_translatable(plain_text):
            html_path = _build_html_path(element)
            path_key = f"{page_url}|{html_path}"
            if path_key not in seen_paths:
                seen_paths.add(path_key)

                segment_id = _make_segment_id(page_url, html_path, plain_text)
                links = _extract_links(element)
                placeholders = _extract_placeholders(raw_inner)

                segments.append(
                    {
                        "segment_id": segment_id,
                        "page_url": page_url,
                        "html_path": html_path,
                        "source_text": raw_inner,
                        "plain_text": plain_text,
                        "contains_html": _contains_html(raw_inner),
                        "placeholders": placeholders,
                        "links": links,
                    }
                )

    for child in element.children:
        if isinstance(child, Tag):
            segments.extend(_extract_from_element(child, page_url, seen_paths))

    return segments


def _extract_alt_segments(
    root_element: Tag,
    page_url: str,
    seen_paths: set[str],
) -> list[dict[str, Any]]:
    """Extract translatable `alt` text from image tags."""
    segments: list[dict[str, Any]] = []
    for img in root_element.find_all("img"):
        alt_text = (img.get("alt") or "").strip()
        if not _is_translatable(alt_text):
            continue

        html_path = _build_html_path(img) + "[@alt]"
        path_key = f"{page_url}|{html_path}"
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)

        segment_id = _make_segment_id(page_url, html_path, alt_text)
        segments.append(
            {
                "segment_id": segment_id,
                "page_url": page_url,
                "html_path": html_path,
                "source_text": alt_text,
                "plain_text": alt_text,
                "contains_html": False,
                "placeholders": _extract_placeholders(alt_text),
                "links": [],
                "attribute_name": "alt",
            }
        )
    return segments


def fetch_page(url: str, timeout: int = 20) -> str | None:
    """Fetch *url* and return HTML text, or None on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as exc:
        print(f"  [WARN] Failed to fetch {url}: {exc}")
        return None


def extract_segments_from_html(html: str, page_url: str) -> list[dict[str, Any]]:
    """Parse *html* and return a list of translatable segment dicts."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(SKIP_TAGS):
        tag.decompose()

    body = soup.find("body") or soup
    seen_paths: set[str] = set()
    segments = _extract_from_element(body, page_url, seen_paths)  # type: ignore[arg-type]
    segments.extend(_extract_alt_segments(body, page_url, seen_paths))  # type: ignore[arg-type]
    return segments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_ingest(pages: list[str], delay: float = 1.5) -> list[dict[str, Any]]:
    """
    Fetch each page in *pages*, extract segments, and return the combined list.
    *delay* seconds are added between fetches to be polite.
    """
    all_segments: list[dict[str, Any]] = []
    fetch_failures: list[dict[str, str]] = []

    for i, url in enumerate(pages):
        print(f"  Fetching [{i+1}/{len(pages)}]: {url}")
        html = fetch_page(url)

        if html is None:
            print(f"  [WARN] No content retrieved for {url}; recording auditable fetch failure")
            fetch_failures.append(
                {
                    "page_url": url,
                    "error": "fetch_failed_or_empty_response",
                }
            )

            # Keep fetch failures auditable in extracted_segments.json.
            all_segments.append(
                {
                    "segment_id": _make_segment_id(url, "__FETCH_ERROR__", "fetch_failed"),
                    "page_url": url,
                    "html_path": "__FETCH_ERROR__",
                    "source_text": "[FETCH_ERROR] Source page could not be fetched.",
                    "plain_text": "[FETCH_ERROR] Source page could not be fetched.",
                    "contains_html": False,
                    "placeholders": [],
                    "links": [],
                    "extraction_error": True,
                }
            )
            continue

        segs = extract_segments_from_html(html, url)
        print(f"    -> {len(segs)} segments extracted")
        all_segments.extend(segs)

        if i < len(pages) - 1:
            time.sleep(delay)

    FETCH_FAILURES_JSON.write_text(
        json.dumps(fetch_failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if fetch_failures:
        print(f"  Fetch failures recorded -> {FETCH_FAILURES_JSON}")

    return all_segments


def save_segments(segments: list[dict[str, Any]], path: Path = EXTRACTED_SEGMENTS_JSON) -> None:
    path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved {len(segments)} segments -> {path}")


def load_pages(path: Path = PAGES_JSON) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["pages"]


if __name__ == "__main__":
    pages = load_pages()
    segments = run_ingest(pages)
    save_segments(segments)
    print("Done.")
