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

# Common DevOps / Cloud terms, typos, and abbreviations mapping
CANONICAL_SYNONYMS = {
    # Kubernetes
    "kubernete": "kubernetes",
    "kubernet": "kubernetes",
    "kubernot": "kubernetes",
    "kubernetees": "kubernetes",
    "kuberneties": "kubernetes",
    "kubernetis": "kubernetes",
    "kuber": "kubernetes",
    "k8s": "kubernetes",
    "k8": "kubernetes",
    "kubectl": "kubernetes kubectl",
    "kubelet": "kubernetes kubelet",
    "kube-proxy": "kubernetes kube-proxy",
    "apiserver": "kubernetes api-server",
    "ingress": "kubernetes ingress",
    "configmap": "kubernetes configmap",
    "daemonset": "kubernetes daemonset",
    "statefulset": "kubernetes statefulset",
    
    # Docker
    "tocker": "docker",
    "dockr": "docker",
    "docke": "docker",
    "dokcer": "docker",
    "dockers": "docker",
    "docker": "docker",
    "dockerfile": "docker dockerfile",
    "docker-compose": "docker compose",
    
    # Vector DB / Search
    "fiass": "faiss",
    "fais": "faiss",
    "bm-25": "bm25",
    "vectordb": "vector database faiss",
    
    # Databases & Frameworks
    "postgre": "postgresql",
    "postgres": "postgresql",
    "postgress": "postgresql",
    "postgresq": "postgresql",
    "postgresql": "postgresql",
    "psql": "postgresql",
    "pg": "postgresql",
    "fastgpi": "fastapi",
    "fastapi": "fastapi",
    "fast-api": "fastapi",
    "gh actions": "github actions",
    "gha": "github actions workflow",
    "promql": "prometheus promql",
    "promethus": "prometheus",
    "prometheos": "prometheus",
    "helm chart": "helm chart package",
    "nginx": "nginx",
    "linux": "linux",

    # DevOps, CI/CD & Containerization Domain Expansion
    "devops": "devops docker kubernetes github actions ci cd",
    "dev-ops": "devops docker kubernetes github actions ci cd",
    "cicd": "ci cd continuous integration github actions",
    "ci/cd": "ci cd continuous integration github actions",
    "containerization": "docker container containerization kubernetes",
    "containers": "docker container containers",
    "container": "docker container",
    
    # Query intent words & typos
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

# Core vocabulary for fuzzy spell matching
KNOWN_VOCABULARY = [
    "kubernetes", "kubectl", "kubelet", "kube-proxy", "ingress", "configmap",
    "daemonset", "statefulset", "deployment", "service", "pod", "namespace",
    "docker", "dockerfile", "container", "image", "registry", "compose", "volume",
    "postgresql", "postgres", "psql", "mvcc", "vacuum", "pgpool",
    "fastapi", "uvicorn", "pydantic", "endpoint", "dependency",
    "faiss", "embedding", "vector", "dense", "similarity", "retrieval",
    "prometheus", "grafana", "metrics", "promql", "alertmanager",
    "nginx", "reverse-proxy", "upstream", "ssl", "tls",
    "github", "actions", "workflow", "ci", "cd", "pipeline",
    "helm", "chart", "release", "values",
    "linux", "systemd", "journalctl", "cgroups",
    "devops", "cicd", "containerization"
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
        
        # Detect all mentioned core technologies
        core_techs = [
            "docker", "kubernetes", "postgresql", "fastapi", "faiss", 
            "prometheus", "nginx", "github actions", "helm", "linux", "bm25",
            "devops", "cicd", "ci/cd"
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
                if tech in ("devops", "cicd", "ci/cd"):
                    sub_queries.append("docker kubernetes containerization ci cd github actions devops architecture")
                elif has_failures:
                    sub_queries.append(f"{tech} architecture features purpose components and failure troubleshooting")
                else:
                    sub_queries.append(f"{tech} architecture features components and operations")
            
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
            "You are an expert DevOps, Containers, Cloud Infrastructure, and Databases Query Normalizer. "
            "Your objective is to prepare user queries for high-precision technical documentation retrieval.\n"
            "Rules:\n"
            "1. TYPO RECOVERY: Detect and correct all keyboard typos, misspellings, and phonetic slips in technical terms "
            "(e.g., 'tocker' -> 'docker', 'postgre' -> 'postgresql', 'kubernete' -> 'kubernetes', 'fastgpi' -> 'fastapi', 'fiass' -> 'faiss', 'psql' -> 'postgresql', 'gha' -> 'github actions').\n"
            "2. NEVER SUBSTITUTE TECHNOLOGIES: NEVER replace a requested technology (e.g., PostgreSQL / postgres) with an unrequested one (e.g., Helm, Nginx). Always preserve all user-specified technologies.\n"
            "3. DECOMPOSITION: If the user query asks about multiple distinct technologies or topics (e.g. 'what is X and Y'), "
            "decompose it into separate focused search queries for EACH requested technology.\n"
            "4. SINGLE TOPIC: If the query asks about a single topic (e.g. 'what is tocker'), return a 1-element array with the corrected query (e.g. ['what is docker']).\n"
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
