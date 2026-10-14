"""
chunker.py
==========
Splits cleaned text into overlapping chunks with metadata.

Strategy:
  - Recursive character splitting: tries to split on paragraph boundaries
    first, then sentences, then words.  This keeps semantic units together.
  - Code blocks are kept intact as their own chunk (never split mid-command).
  - Each chunk receives a unique ID and metadata dict.

Output schema per chunk:
  {
      "chunk_id":    "docker_networking_042",
      "source":      "docker",
      "topic":       "networking",
      "version":     "26.0",
      "url":         "https://docs.docker.com/network/",
      "date_scraped":"2026-10-01",
      "type":        "text" | "code",
      "token_count": 387,
      "text":        "..."
  }
"""

import re
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Generator, List, Optional

import tiktoken

# ---------------------------------------------------------------------------
# Tokenizer (using cl100k_base — compatible with most modern LLMs)
# ---------------------------------------------------------------------------

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text, disallowed_special=()))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id:        str
    source:          str
    topic:           str
    version:         str = "latest"
    url:             str = ""
    date_scraped:    str = ""
    type:            str = "text"          # "text" | "code"
    token_count:     int = 0
    text:            str = ""
    subtopic:        Optional[str] = None
    document_id:     Optional[str] = None
    section_id:      Optional[str] = None
    title:           Optional[str] = None
    source_url:      Optional[str] = None
    source_type:     Optional[str] = None
    source_version:  Optional[str] = None
    snapshot_date:   Optional[str] = None
    language:        Optional[str] = "en"
    security_class:  Optional[str] = "clean"
    keywords:        Optional[list] = None
    parent_section:  Optional[str] = None
    license_note:    Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


# ---------------------------------------------------------------------------
# Code-block extractor
# ---------------------------------------------------------------------------

# Matches fenced code blocks:  ```lang\n...\n```  or  ~~~\n...\n~~~
_CODE_BLOCK_RE = re.compile(r"(```[^\n]*\n[\s\S]*?```|~~~[^\n]*\n[\s\S]*?~~~)", re.MULTILINE)


def _split_code_and_text(text: str):
    """
    Yield (segment_text, is_code) tuples by splitting on fenced code blocks.
    Code blocks are never split further; text segments are chunked normally.
    """
    last_end = 0
    for match in _CODE_BLOCK_RE.finditer(text):
        # Text before this code block
        before = text[last_end:match.start()]
        if before.strip():
            yield before, False
        # The code block itself
        yield match.group(0), True
        last_end = match.end()
    # Trailing text
    remainder = text[last_end:]
    if remainder.strip():
        yield remainder, False


# ---------------------------------------------------------------------------
# Recursive text splitter
# ---------------------------------------------------------------------------

# Ordered separators: try largest semantic unit first
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _recursive_split(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    """
    Split `text` into segments each ≤ max_tokens, with overlap_tokens
    repeated at the start of the next segment.
    Uses recursive separator strategy.
    """
    if count_tokens(text) <= max_tokens:
        return [text]

    chunks: List[str] = []
    _split_with_separator(text, max_tokens, overlap_tokens, _SEPARATORS, chunks)
    return chunks


def _split_with_separator(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
    separators: List[str],
    result: List[str],
):
    sep = separators[0] if separators else ""
    next_seps = separators[1:] if len(separators) > 1 else []

    parts = text.split(sep) if sep else list(text)
    current: List[str] = []
    current_tokens = 0

    for part in parts:
        part_tokens = count_tokens(part + sep)

        if current_tokens + part_tokens > max_tokens:
            if current:
                merged = sep.join(current)
                if count_tokens(merged) <= max_tokens:
                    result.append(merged)
                else:
                    # Still too big — recurse with next separator
                    if next_seps:
                        _split_with_separator(merged, max_tokens, overlap_tokens, next_seps, result)
                    else:
                        result.append(merged)  # Accept oversized as last resort

                # Keep overlap: walk back from end until we have overlap_tokens worth
                overlap_parts: List[str] = []
                overlap_tok = 0
                for p in reversed(current):
                    p_tok = count_tokens(p + sep)
                    if overlap_tok + p_tok > overlap_tokens:
                        break
                    overlap_parts.insert(0, p)
                    overlap_tok += p_tok
                current = overlap_parts
                current_tokens = overlap_tok

            current.append(part)
            current_tokens += part_tokens
        else:
            current.append(part)
            current_tokens += part_tokens

    if current:
        merged = sep.join(current)
        if count_tokens(merged) <= max_tokens:
            result.append(merged)
        elif next_seps:
            _split_with_separator(merged, max_tokens, overlap_tokens, next_seps, result)
        else:
            result.append(merged)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_document(
    text: str,
    source: str,
    topic: str,
    version: str,
    url: str,
    date_scraped: str,
    max_tokens: int = 400,
    overlap_tokens: int = 50,
    min_tokens: int = 80,
) -> List[Chunk]:
    """
    Chunk a cleaned document into Chunk objects.

    Args:
        text:          Cleaned plain text.
        source:        Short source label (e.g. 'docker', 'kubernetes').
        topic:         Sub-topic label (e.g. 'networking', 'storage').
        version:       Documentation version string (e.g. '26.0').
        url:           Canonical URL of the source page.
        date_scraped:  ISO date string.
        max_tokens:    Maximum tokens per chunk.
        overlap_tokens: Overlap between consecutive chunks.
        min_tokens:    Discard chunks shorter than this.

    Returns:
        List of Chunk objects.
    """
    chunks: List[Chunk] = []
    chunk_index = 0

    for segment, is_code in _split_code_and_text(text):
        if is_code:
            # Code blocks: keep intact; only discard if trivially short
            tok = count_tokens(segment)
            if tok >= min_tokens:
                cid = _make_id(source, topic, chunk_index)
                chunks.append(Chunk(
                    chunk_id=cid, source=source, topic=topic,
                    version=version, url=url, date_scraped=date_scraped,
                    type="code", token_count=tok, text=segment.strip(),
                ))
                chunk_index += 1
        else:
            # Text segments: recursive split
            sub_chunks = _recursive_split(segment, max_tokens, overlap_tokens)
            for sub in sub_chunks:
                sub = sub.strip()
                tok = count_tokens(sub)
                if tok < min_tokens:
                    continue
                cid = _make_id(source, topic, chunk_index)
                chunks.append(Chunk(
                    chunk_id=cid, source=source, topic=topic,
                    version=version, url=url, date_scraped=date_scraped,
                    type="text", token_count=tok, text=sub,
                ))
                chunk_index += 1

    return chunks


def _make_id(source: str, topic: str, index: int) -> str:
    """Generate a deterministic, human-readable chunk ID."""
    safe_source = re.sub(r"[^a-z0-9]", "_", source.lower())
    safe_topic  = re.sub(r"[^a-z0-9]", "_", topic.lower())
    return f"{safe_source}_{safe_topic}_{index:05d}"
