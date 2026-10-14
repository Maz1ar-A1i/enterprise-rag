"""
src/advanced_rag/advanced_prompt_builder.py
============================================
Advanced Prompt Assembly & Structured Partial-Grounding Rules:
- Supports multi-topic structured answers with inline citations.
- Replaces rigid all-or-nothing refusal with nuanced coverage breakdown.
- Retains strict zero-hallucination guardrails for ungrounded content.
"""

import re
import tiktoken
from typing import List, Tuple, Dict, Any

from src.corpus.chunker import Chunk
from src.retrieval.hybrid_retriever import RetrievalResult

_TOKENIZER = tiktoken.get_encoding("cl100k_base")

def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text, disallowed_special=()))

ADVANCED_SYSTEM_PROMPT = """You are an expert enterprise technical assistant specialized in DevOps, Containers, Cloud Infrastructure, and Databases.

Follow these operational guidelines:
1. TRUTHFULNESS & GROUNDING:
   - Base your technical answers on the provided DOCUMENTATION PASSAGES below.
   - User queries may contain typos, misspellings, or abbreviations (e.g. 'postgress' -> PostgreSQL, 'kubernetees' -> Kubernetes, 'tocker' -> Docker, 'fiass' -> FAISS, 'fastgpi' -> FastAPI).
   - Accurately map misspelled terms to their corresponding technologies.

2. COMPREHENSIVE MULTI-TOPIC & COMPARISON SYNTHESIS:
   - When the user asks to explain multiple technologies or compare/contrast them (with pros, cons, and tables):
     a) For EACH requested technology (e.g., Docker, Kubernetes, PostgreSQL), provide a concise technical section (`## Technology Name`) covering its core purpose, key components, and operational role.
     b) Provide a comprehensive Markdown Comparison Table comparing all requested technologies side-by-side with columns:
        `| Technology | Category / Role | Core Architecture | Storage / State | Key Pros | Key Cons / Trade-offs | Source Documentation |`
     c) CITATIONS: Every claim and table row MUST cite its documentation source using the format `[Source: <URL>]` (e.g., `[Source: https://docs.docker.com/]`, `[Source: https://kubernetes.io/docs/home/]`, `[Source: https://www.postgresql.org/docs/current/]`). Always complete and close all citation brackets.

3. DEVOPS & INFRASTRUCTURE SYNTHESIS:
   - If the user asks general or overarching questions about DevOps, its definition, or its operational scope:
     a) Synthesize how the core technologies in the passages (such as Docker containerization, Kubernetes container orchestration, and CI/CD pipelines) define and power the operational scope of modern DevOps engineering.
     b) Break down the lifecycle phases (Build & Package with Docker, Deploy & Orchestrate with Kubernetes, CI/CD automation) using the technical concepts and citations from the passages.

4. PARTIAL GROUNDING & OUT-OF-CORPUS CLARIFICATION:
   - If a user question combines technical concepts with external market/commercial requests (e.g., salary packages, hiring companies, job postings):
     a) Answer the technical and operational scope using the documentation passages with inline citations.
     b) For commercial or job-market topics not in the documentation (like specific salary compensation or hiring company names), explicitly clarify that live salary compensation figures and recruitment job postings are outside the scope of this technical documentation index.
   - ONLY return the complete refusal phrase ("I cannot find sufficient information in the provided documentation to answer this question.") if the query has zero technical concepts and none of the retrieved passages are relevant.

5. FORMATTING:
   - Use clean Markdown with headers (`##`), bold terms, concise bullet points, and complete formatted tables. Ensure all sections are fully written out."""


def build_advanced_context_block(chunks: List[Chunk], token_budget: int = 5000) -> str:
    """Concatenate chunk texts with rich metadata headers."""
    from src.corpus.url_resolver import resolve_url
    parts = []
    total_tokens = 0

    for i, chunk in enumerate(chunks, start=1):
        clean_url = resolve_url(chunk.url)
        header = (
            f"[PASSAGE {i} | Source: {chunk.source.upper()} | Topic: {chunk.topic} | "
            f"Version: {chunk.version} | URL: {clean_url}]"
        )
        block = f"{header}\n{chunk.text}\n---"
        block_tokens = _count_tokens(block)

        if total_tokens + block_tokens > token_budget:
            break

        parts.append(block)
        total_tokens += block_tokens

    return "\n\n".join(parts)


def build_advanced_prompt(
    query: str,
    retrieval_result: RetrievalResult,
    sub_queries: List[str] = None,
    token_budget: int = 5000,
) -> Tuple[str, str]:
    """
    Build the advanced system prompt and formatted user message.
    """
    context = build_advanced_context_block(retrieval_result.chunks, token_budget)
    
    user_message = f"""DOCUMENTATION PASSAGES:
{context}

USER QUESTION:
{query}

TECHNICAL ANSWER:"""

    return ADVANCED_SYSTEM_PROMPT, user_message


# Canonical topic documentation links
CANONICAL_DOC_LINKS = {
    "docker": "https://docs.docker.com/",
    "kubernetes": "https://kubernetes.io/docs/home/",
    "postgresql": "https://www.postgresql.org/docs/current/",
    "postgres": "https://www.postgresql.org/docs/current/",
    "fastapi": "https://fastapi.tiangolo.com/",
    "nginx": "https://nginx.org/en/docs/",
    "prometheus": "https://prometheus.io/docs/introduction/overview/",
    "github_actions": "https://docs.github.com/en/actions",
    "github": "https://docs.github.com/en/actions",
    "helm": "https://helm.sh/docs/",
    "linux_commands": "https://man7.org/linux/man-pages/",
    "linux": "https://man7.org/linux/man-pages/",
    "faiss": "https://github.com/facebookresearch/faiss/wiki",
    "security": "https://kubernetes.io/docs/concepts/security/",
    "kyverno": "https://kubernetes.io/docs/concepts/security/",
    "cross_topic": "https://kubernetes.io/docs/home/",
}

# Regex for citations and refusal
_CITATION_ANY_RE = re.compile(r"\[Source:\s*([^\]\n]+)\]", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s\]\)\<\>\"']+", re.IGNORECASE)

_REFUSAL_PATTERNS = [
    r"i cannot find sufficient information",
    r"i don['\u2019]t have enough information",
    r"insufficient information.*in.*documentation",
    r"not covered in the provided",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def extract_advanced_citations(answer_text: str, chunks: List[Chunk] = None) -> List[Dict[str, str]]:
    """
    Extract citations from answer text, resolving raw URLs, passage IDs, or topic names
    to valid canonical documentation links.
    """
    citations = []
    seen_urls = set()

    # 1. Match any [Source: ...] pattern
    for match in _CITATION_ANY_RE.finditer(answer_text):
        raw_target = match.group(1).strip()
        
        # Check if direct URL is present
        url_match = _URL_RE.search(raw_target)
        if url_match:
            from src.corpus.url_resolver import resolve_url
            found_url = resolve_url(url_match.group(0).rstrip(".,;"))
            source_topic = "doc"
            for k in CANONICAL_DOC_LINKS:
                if k in found_url.lower():
                    source_topic = k
                    break
            if found_url not in seen_urls:
                seen_urls.add(found_url)
                citations.append({"source": source_topic, "url": found_url})
            continue

        # Check for PASSAGE <N> pattern
        pass_m = re.search(r"passage\s*(\d+)", raw_target, re.IGNORECASE)
        if pass_m and chunks:
            from src.corpus.url_resolver import resolve_url
            idx = int(pass_m.group(1)) - 1
            if 0 <= idx < len(chunks):
                ch = chunks[idx]
                target_url = resolve_url(ch.url, CANONICAL_DOC_LINKS.get(ch.topic, "https://docs.docker.com/"))
                if target_url not in seen_urls:
                    seen_urls.add(target_url)
                    citations.append({"source": ch.topic, "url": target_url})
                continue

        # Check for topic keywords in source string (e.g. 'Docker – Networking basics')
        lower_raw = raw_target.lower()
        matched_topic = None
        for top_key, doc_url in CANONICAL_DOC_LINKS.items():
            if top_key in lower_raw:
                matched_topic = (top_key, doc_url)
                break

        if matched_topic:
            top_k, doc_url = matched_topic
            if doc_url not in seen_urls:
                seen_urls.add(doc_url)
                citations.append({"source": top_k, "url": doc_url})

    # If no citations were found via tags but chunks exist and answer is technical,
    # pull canonical links from the retrieved chunks that appear in the answer
    if not citations and chunks:
        lower_ans = answer_text.lower()
        for ch in chunks:
            if ch.topic in lower_ans or ch.source in lower_ans:
                target_url = ch.url or CANONICAL_DOC_LINKS.get(ch.topic, "https://docs.docker.com/")
                if target_url not in seen_urls:
                    seen_urls.add(target_url)
                    citations.append({"source": ch.topic, "url": target_url})

    return citations


def format_inline_citations(answer_text: str, citations: List[Dict[str, str]]) -> str:
    """
    Replaces unlinked [Source: ...] annotations with clean Markdown clickable links.
    Also trims any trailing unclosed `[Source:` tags if generation was stopped.
    """
    # Clean dangling unclosed `[Source:` at the very end
    cleaned = re.sub(r"\|\s*\[Source:\s*$", "| — |", answer_text.rstrip())
    cleaned = re.sub(r"\[Source:\s*$", "", cleaned.rstrip())

    # Map topics to URLs
    url_map = {c["source"].lower(): c["url"] for c in citations}

    def _replace_cite(match):
        raw_content = match.group(1).strip()
        # If it's already markdown link, keep it
        if "http" in raw_content:
            return f"[[Source Documentation]({raw_content})]"
        
        # Check topic
        lower = raw_content.lower()
        for k, url in CANONICAL_DOC_LINKS.items():
            if k in lower:
                return f"[[Source: {k.capitalize()} Docs]({url})]"
        
        # Default to first citation URL if available
        if citations:
            return f"[[Source: Docs]({citations[0]['url']})]"
        return f"[Source: {raw_content}]"

    return _CITATION_ANY_RE.sub(_replace_cite, cleaned)


def is_advanced_refusal(answer_text: str) -> bool:
    trimmed = answer_text.strip().lower()
    if trimmed.startswith("i cannot find sufficient information in the provided documentation"):
        return True
    return False

