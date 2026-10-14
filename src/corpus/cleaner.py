"""
cleaner.py
==========
Cleans raw HTML and Markdown text into plain, normalized prose
suitable for chunking and embedding.

Pipeline:
  HTML input  ->  strip boilerplate  ->  convert to Markdown  ->  normalize  ->  clean text
  MD input    ->  strip front-matter  ->  normalize  ->  clean text
"""

import re
import unicodedata
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup
from markdownify import markdownify as md2text


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def clean_document(raw: str, source_type: str = "auto", url: str = "") -> str:
    """
    Clean a raw document string (HTML or Markdown) into plain text.

    Args:
        raw:         Raw file content.
        source_type: 'html' | 'markdown' | 'auto'  (auto detects from content).
        url:         Source URL, used only for logging.

    Returns:
        Cleaned plain-text string, or empty string if the document is unusable.
    """
    if not raw or not raw.strip():
        return ""

    # Auto-detect format
    if source_type == "auto":
        source_type = "html" if raw.lstrip().startswith("<") else "markdown"

    if source_type == "html":
        text = _clean_html(raw)
    else:
        text = _clean_markdown(raw)

    text = _normalize_whitespace(text)
    text = _remove_boilerplate_lines(text)
    return text.strip()


# ---------------------------------------------------------------------------
# HTML cleaning
# ---------------------------------------------------------------------------

# Tags that contain non-content boilerplate
_BOILERPLATE_TAGS = [
    "nav", "header", "footer", "aside", "script",
    "style", "noscript", "svg", "form", "button",
    "cookie", "banner", "advertisement",
]

# CSS classes / IDs that signal boilerplate
_BOILERPLATE_PATTERNS = re.compile(
    r"(nav|sidebar|footer|header|cookie|banner|menu|toc|breadcrumb|"
    r"advertisement|social|share|comment|pagination|related)",
    re.IGNORECASE,
)


def _clean_html(html: str) -> str:
    """Strip boilerplate tags then convert remaining HTML -> plain text."""
    soup = BeautifulSoup(html, "lxml")

    # Remove boilerplate structural tags
    for tag in soup.find_all(_BOILERPLATE_TAGS):
        tag.decompose()

    # Remove elements whose class/id looks like navigation or ads
    for tag in soup.find_all(True):
        if not tag or not getattr(tag, "attrs", None):
            continue
        classes = tag.attrs.get("class", [])
        if isinstance(classes, str):
            classes = [classes]
        elif not isinstance(classes, list):
            classes = []
        elem_id = tag.attrs.get("id", "")
        if isinstance(elem_id, list):
            elem_id = " ".join(str(x) for x in elem_id)
        elif not isinstance(elem_id, str):
            elem_id = str(elem_id) if elem_id else ""
        attrs = " ".join(str(v) for v in (classes + [elem_id]))
        if _BOILERPLATE_PATTERNS.search(attrs):
            try:
                tag.decompose()
            except Exception:
                pass

    # Convert surviving HTML to markdown (preserves code blocks, headers)
    cleaned_md = md2text(str(soup), heading_style="ATX", code_language="")
    return cleaned_md


# ---------------------------------------------------------------------------
# Markdown cleaning
# ---------------------------------------------------------------------------

_YAML_FRONT_MATTER = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_HUGO_SHORTCODES   = re.compile(r"\{\{[^}]*\}\}")          # Hugo {{< ... >}}
_HTML_TAGS         = re.compile(r"<[^>]+>")                 # Residual HTML
_MD_LINK_IMAGES    = re.compile(r"!\[([^\]]*)\]\([^)]*\)")  # Images -> alt text


def _clean_markdown(text: str) -> str:
    """Strip Markdown front-matter and non-content elements."""
    text = _YAML_FRONT_MATTER.sub("", text)   # Remove YAML front-matter
    text = _HUGO_SHORTCODES.sub("", text)      # Remove Hugo shortcodes
    text = _MD_LINK_IMAGES.sub(r"\1", text)    # Replace images with alt text
    text = _HTML_TAGS.sub(" ", text)           # Remove any inline HTML
    return text


# ---------------------------------------------------------------------------
# Shared normalization
# ---------------------------------------------------------------------------

_MULTIPLE_BLANK_LINES = re.compile(r"\n{3,}")
_MULTIPLE_SPACES      = re.compile(r"[ \t]{2,}")

# Lines that are purely decorative / navigation
_BOILERPLATE_LINE_PATTERNS = [
    re.compile(r"^\s*[•\-\*]\s*$"),            # Lone bullet points
    re.compile(r"^\s*\|[-|]+\|\s*$"),           # Table separator rows
    re.compile(r"^(Edit this page|Was this|Feedback|Previous|Next|Contents|Table of Contents)\b", re.IGNORECASE),
    re.compile(r"^(Copyright|All rights reserved|Privacy Policy)", re.IGNORECASE),
]


def _normalize_whitespace(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)          # Normalize unicode
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTIPLE_SPACES.sub(" ", text)
    text = _MULTIPLE_BLANK_LINES.sub("\n\n", text)
    return text


def _remove_boilerplate_lines(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        if any(p.match(line) for p in _BOILERPLATE_LINE_PATTERNS):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)
