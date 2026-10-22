"""
src/advanced_rag/query_transformer.py
======================================
Pre-Retrieval Query Transformation & Optimization:
1. Technical Spell & Synonym Canonicalization (resolves typos, abbreviations, and word forms).
2. Deterministic Multi-Query Decomposition (splits compound/multi-part questions into distinct parallel queries).
3. Keyword Expansion & LLM Normalization fallback.
"""

import re
import json
import difflib
import logging
from typing import List, Dict, Optional

log = logging.getLogger(__name__)

# Common enterprise workplace terms, typos, and abbreviations mapping
CANONICAL_SYNONYMS = {
    # Slack
    "slac": "slack",
    "slak": "slack",

    # Jira
    "jria": "jira",
    "jra": "jira",

    # Confluence
    "confluece": "confluence",
    "conflunce": "confluence",
    "confluance": "confluence",

    # GitHub
    "githb": "github",
    "git hub": "github",
    "github pr": "github pull request",
    "pr": "pull request",

    # Linear
    "lineer": "linear",
    "liner": "linear",

    # HubSpot
    "hubspt": "hubspot",
    "hub spot": "hubspot",
    "crm": "hubspot crm",

    # Google Drive
    "gdrive": "google drive",
    "g drive": "google drive",
    "gdocs": "google docs",

    # Gmail
    "gmal": "gmail",
    "gmial": "gmail",

    # Meetings / Fireflies
    "firelies": "fireflies",
    "meeting trasncript": "meeting transcript",
    "trasncript": "transcript",

    # Enterprise common terms
    "sso": "single sign-on SSO",
    "rbac": "role-based access control RBAC",
    "sla": "service level agreement SLA",
    "rpo": "recovery point objective RPO",
    "rto": "recovery time objective RTO",
    "mfa": "multi-factor authentication MFA",
    "okta": "okta SSO identity",
    "saml": "SAML SSO authentication",
    "oidc": "OIDC OAuth identity",
    "idp": "identity provider IdP",
    "api key": "API key credential",
    "apikey": "API key credential",
    "oncall": "on-call escalation",
    "on-call": "on-call escalation",
    "runbook": "runbook incident response",
    "postmortem": "post-mortem incident review",
    "k8s": "kubernetes",
    "kms": "key management service KMS",

    # Query intent typos
    "failuer": "failure",
    "failuers": "failures",
    "diffrence": "difference",
    "diffrences": "differences",
    "betweeen": "between",
    "betwen": "between",
}

# Regex to find word boundaries
_WORD_BOUNDARY_RE = re.compile(r"\b[a-zA-Z0-9_\-]+\b")

# Words that must NEVER be fuzzy matched to technical terms or hijack domain search
COMMON_ENGLISH_WORDS = {
    "what", "when", "where", "which", "who", "whom", "this", "that", "these", "those",
    "them", "then", "there", "their", "they", "have", "has", "had", "does", "done",
    "will", "with", "from", "into", "some", "more", "such", "than", "very", "also",
    "look", "looks", "used", "uses", "using", "like", "make", "made", "good", "best",
    "core", "main", "each", "both", "show", "tell", "give", "help", "need", "know",
    "role", "type", "part", "work", "runs", "cost", "time", "date", "high", "over",
    "fail", "failure", "failures", "error", "errors", "issue", "issues",
    # General non-technical words that should not pull in arbitrary library chunks
    "scope", "scopes", "salary", "salaries", "package", "packages", "company", "companies",
    "position", "positions", "offering", "job", "jobs", "market", "demand",
}

# Core vocabulary for fuzzy spell matching — enterprise workplace domain
KNOWN_VOCABULARY = [
    # Enterprise channels
    "slack", "jira", "confluence", "gmail", "google", "drive", "github",
    "hubspot", "linear", "fireflies", "transcript", "meeting",
    # Enterprise concepts
    "sso", "rbac", "sla", "rpo", "rto", "mfa", "okta", "saml", "oidc",
    "runbook", "postmortem", "incident", "escalation", "oncall",
    "authentication", "authorization", "entitlement", "credential",
    "kubernetes", "faiss", "embedding", "retrieval", "vector",
    "policy", "contract", "onboarding", "offboarding", "contractor",
    "deployment", "rollback", "latency", "throughput", "failover",
    "ticket", "sprint", "milestone", "epic", "backlog",
    "pipeline", "workflow", "integration", "endpoint",
]


class QueryTransformer:
    """
    Transforms user queries to overcome lexical mismatches, typos, and vector dilution.
    """

    def __init__(self, groq_client=None):
        self.groq = groq_client

    def canonicalize(self, query: str) -> str:
        """
        Fast dictionary + fuzzy spell matching typo canonicalization.
        """
        tokens = query.split()
        normalized = []
        for tok in tokens:
            clean = tok.strip("?,.:;!'\"()[]{}").lower()
            if clean in CANONICAL_SYNONYMS:
                replaced = CANONICAL_SYNONYMS[clean]
                normalized.append(replaced)
            elif clean not in COMMON_ENGLISH_WORDS and len(clean) >= 5:
                # Try fuzzy matching against known technical vocabulary
                matches = difflib.get_close_matches(clean, KNOWN_VOCABULARY, n=1, cutoff=0.80)
                if matches:
                    normalized.append(matches[0])
                else:
                    normalized.append(tok)
            else:
                normalized.append(tok)
        
        return " ".join(normalized)

    def decompose(self, query: str) -> List[str]:
        """
        Pre-Retrieval Normalization & Decomposition:
        1. Canonicalizes typos & misspellings within the DevOps domain (e.g. postgre -> postgresql, tocker -> docker).
        2. Prioritizes deterministic multi-technology detection to guarantee accurate topic extraction.
        3. Uses LLM Domain Query Normalizer for complex natural language questions.
        """
        # Step 1: Fast local canonicalization
        canonical_q = self.canonicalize(query)

        # Step 2: Deterministic check for compound multi-technology queries
        if self._is_compound_query(canonical_q):
            sub_queries = self._rule_based_split(canonical_q)
            if len(sub_queries) > 1:
                log.info(f"[QueryTransformer] Deterministic decomposed {query!r} -> {sub_queries}")
                return sub_queries

        # Step 3: Use LLM Domain Query Normalizer if available
        if self.groq:
            try:
                llm_queries = self._llm_domain_normalize(canonical_q)
                if llm_queries and len(llm_queries) > 1:
                    log.info(f"[QueryTransformer] LLM decomposed {query!r} -> {llm_queries}")
                    return llm_queries
                elif llm_queries and len(llm_queries) == 1 and llm_queries[0] != canonical_q:
                    log.info(f"[QueryTransformer] LLM normalized {query!r} -> {llm_queries}")
                    canonical_q = llm_queries[0]
            except Exception as e:
                log.warning(f"[QueryTransformer] LLM query normalization fallback: {e}")

        # Fallback to single canonical query
        return [canonical_q]

    def _is_compound_query(self, query: str) -> bool:
        q_lower = query.lower()
        has_multiple_ands = q_lower.count(" and ") >= 1
        has_question_marks = q_lower.count("?") >= 2
        has_comparison = any(w in q_lower for w in [" vs ", " versus ", " difference ", " differences ", " compare ", " comparison "])
        has_multiple_sentences = len([c for c in re.split(r"[.?;]", query) if len(c.strip()) > 5]) >= 2
        return has_multiple_ands or has_question_marks or has_comparison or has_multiple_sentences

    def _rule_based_split(self, query: str) -> List[str]:
        """Split compound queries and multi-topic comparison requests."""
        q_clean = query.strip()
        q_lower = query.lower()
        
        # Detect all mentioned core enterprise channels / topics
        core_techs = [
            "slack", "jira", "confluence", "gmail", "google drive", "github",
            "hubspot", "linear", "fireflies", "meeting",
            "sso", "rbac", "sla", "runbook", "incident", "failover",
        ]
        
        found_techs = []
        for tech in core_techs:
            # Check with word boundaries
            if re.search(rf"\b{re.escape(tech)}\b", q_lower):
                found_techs.append(tech)
        
        # If multiple distinct technologies are identified in a compound query
        if len(found_techs) >= 2:
            # Check if failures/troubleshooting is asked
            has_failures = any(w in q_lower for w in ["fail", "failure", "failures", "troubleshoot", "errors", "issues"])
            
            sub_queries = []
            for tech in found_techs:
                if has_failures:
                    sub_queries.append(f"{tech} failure troubleshooting incident resolution")
                else:
                    sub_queries.append(f"{tech} content information details")
            
            # Add comparison query if user asked for comparison, differences or pros/cons
            if any(w in q_lower for w in ["compare", "contrast", "difference", "differences", "vs", "versus", "pros", "cons", "table"]):
                sub_queries.append(f"compare and contrast {' vs '.join(found_techs)} pros and cons")
            return sub_queries

        # Split on sentence/clause boundaries (periods, question marks, semicolons)
        clauses = [c.strip() for c in re.split(r"[.?;]", q_clean) if len(c.strip()) > 5]
        if len(clauses) >= 2:
            return clauses

        # Pattern: "what is/are A and B and C..."
        prefix_match = re.match(r"^(what\s+is|what\s+are|explain|how\s+to\s+use|tell\s+me\s+about)\s+(.+)$", q_clean, re.IGNORECASE)
        if prefix_match:
            prefix = prefix_match.group(1)
            rest = prefix_match.group(2)
            parts = [p.strip() for p in re.split(r",|\sand\s", rest) if p.strip()]
            if len(parts) >= 2:
                return [f"{prefix} {part}" for part in parts if len(part) > 1]

        return [query]

    def _llm_domain_normalize(self, query: str) -> List[str]:
        """
        Stage 1 Domain Query Normalizer:
        Analyzes user input within the DevOps / Cloud / Containers / CI/CD domain.
        Corrects keyboard typos, expands abbreviations, and splits compound questions.
        """
        system_prompt = (
            "You are an expert Enterprise Workplace Knowledge Query Normalizer. "
            "Your objective is to prepare user queries for high-precision retrieval from enterprise internal knowledge bases (Slack, Jira, Confluence, Gmail, Google Drive, GitHub, HubSpot, Linear, meeting transcripts).\n"
            "Rules:\n"
            "1. TYPO RECOVERY: Detect and correct all keyboard typos, misspellings, and abbreviations in enterprise terms "
            "(e.g., 'slac' -> 'slack', 'jria' -> 'jira', 'confluece' -> 'confluence', 'sso' -> 'single sign-on SSO', 'rbac' -> 'role-based access control').\n"
            "2. NEVER SUBSTITUTE: NEVER replace a requested topic (e.g., 'SLA') with an unrelated one. Preserve all user-specified topics.\n"
            "3. DECOMPOSITION: If the user query asks about multiple distinct topics, "
            "decompose it into separate focused search queries for EACH topic.\n"
            "4. SINGLE TOPIC: If the query asks about a single topic, return a 1-element array with the corrected query.\n"
            "5. OUTPUT FORMAT: Return ONLY a valid JSON array of strings: [\"query 1\", \"query 2\"]"
        )
        user_msg = f"User Query: {query}"
        
        result = self.groq.call_json(
            system_prompt=system_prompt,
            user_message=user_msg,
            model=None,
            temperature=0.0,
            max_tokens=256,
        )
        if isinstance(result, list) and all(isinstance(x, str) for x in result) and len(result) > 0:
            return result
        elif isinstance(result, dict) and "queries" in result and isinstance(result["queries"], list):
            return result["queries"]
        return [query]
