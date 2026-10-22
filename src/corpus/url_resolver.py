"""
src/corpus/url_resolver.py
==========================
Generic URL and citation resolver for enterprise RAG documents.
"""

from typing import Optional, Dict

CHANNEL_MAP = {
    "confluence": "https://www.atlassian.com/software/confluence",
    "jira": "https://www.atlassian.com/software/jira",
    "slack": "https://slack.com/",
    "linear": "https://linear.app/",
    "google_drive": "https://drive.google.com/",
    "googledrive": "https://drive.google.com/",
    "github": "https://github.com/",
    "hubspot": "https://www.hubspot.com/",
    "gmail": "https://workspace.google.com/products/gmail/",
    "fireflies": "https://fireflies.ai/",
}

# Optional mapping registry for canonical documentation overrides
REAL_DOCUMENTATION_MAP: Dict[str, str] = dict(CHANNEL_MAP)


def resolve_url(url: Optional[str], default: str = "https://www.atlassian.com/software/confluence") -> str:
    """Resolve any document URL or relative channel path to its canonical link."""
    if not url:
        return default
    clean = url.strip()
    if clean in REAL_DOCUMENTATION_MAP:
        return REAL_DOCUMENTATION_MAP[clean]

    # Handle internal enterprise-rag-dataset paths like "enterprise-rag-dataset/confluence/..."
    if clean.startswith("enterprise-rag-dataset/"):
        parts = clean.split("/")
        if len(parts) > 1 and parts[1].lower() in CHANNEL_MAP:
            return CHANNEL_MAP[parts[1].lower()]

    # If it's a channel name directly
    lower_clean = clean.lower()
    for ch, link in CHANNEL_MAP.items():
        if ch in lower_clean and not clean.startswith("http"):
            return link

    # If already a valid web URL, return as is
    if clean.startswith("http://") or clean.startswith("https://"):
        return clean

    return default
