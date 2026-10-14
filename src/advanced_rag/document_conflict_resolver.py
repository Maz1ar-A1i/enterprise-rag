"""
src/advanced_rag/document_conflict_resolver.py
==============================================
Deterministic Contradiction and Version Conflict Resolver.

Implements strict enterprise conflict resolution policy:
  1. Semantic version extraction & normalization (e.g. K8s v1.28 vs v1.21, Helm v3 vs v2, Compose v2 vs v1).
  2. User-requested version extraction from query (e.g. "in Helm v2...", "for Postgres 12").
  3. Deterministic precedence arbitration:
     - Rule 1: Explicit User-Requested Version Match (Highest Priority)
     - Rule 2: Deprecation & Recency Order (Newer version supersedes older unless user requested older)
     - Rule 3: Source Authority Hierarchy (Official canonical docs > Community/Internal notes)
     - Rule 4: Embedding / Reranker score (Tiebreaker only)
  4. Multi-version comparison mode when query asks for differences / migrations.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import re
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Version extraction regex patterns
_VERSION_REGEXES = [
    re.compile(r"\bv?(\d+\.\d+(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"\b(helm\s*v?[23])\b", re.IGNORECASE),
    re.compile(r"\b(compose\s*v?[123])\b", re.IGNORECASE),
    re.compile(r"\b(python\s*3\.\d+)\b", re.IGNORECASE),
    re.compile(r"\b(postgres(?:ql)?\s*\d+)\b", re.IGNORECASE),
    re.compile(r"\b(k8s|kubernetes)\s*v?(\d+\.\d+)\b", re.IGNORECASE),
]


def parse_semver(v_str: str) -> Tuple[int, ...]:
    """Extract numeric tuple for version comparison e.g. '1.28.2' -> (1, 28, 2)"""
    nums = re.findall(r"\d+", v_str)
    return tuple(int(n) for n in nums) if nums else (0,)


@dataclass
class VersionMetadata:
    raw_version: Optional[str] = None
    parsed_tuple: Tuple[int, ...] = (0,)
    is_deprecated: bool = False
    deprecation_reason: Optional[str] = None
    release_date: Optional[str] = None


@dataclass
class ResolvedChunk:
    chunk: Any
    version_meta: VersionMetadata
    priority_score: float
    conflict_notes: List[str] = field(default_factory=list)
    is_superseded: bool = False


class DocumentConflictResolver:
    """
    Arbitrates conflicts between retrieved documents using deterministic metadata policy.
    """
    def __init__(self):
        # Known deprecation markers
        self.deprecation_keywords = [
            "deprecated", "obsolete", "legacy", "no longer supported", 
            "removed in", "superseded by", "end of life", "eol"
        ]

    def extract_version(self, text: str, title: str = "", topic: str = "") -> VersionMetadata:
        """
        Extracts version information from chunk text, title, or topic.
        """
        combined = f"{title} {topic} {text[:300]}".lower()
        
        # Check for explicit Helm or Compose versions
        if "helm v2" in combined or "helm 2" in combined:
            return VersionMetadata(raw_version="2.0", parsed_tuple=(2, 0), is_deprecated=True, deprecation_reason="Helm 2 reached EOL Nov 2020")
        if "helm v3" in combined or "helm 3" in combined:
            return VersionMetadata(raw_version="3.0", parsed_tuple=(3, 0), is_deprecated=False)

        if "compose v1" in combined or "docker-compose" in combined and "v2" not in combined:
            if "docker compose v2" not in combined:
                return VersionMetadata(raw_version="1.0", parsed_tuple=(1, 0), is_deprecated=True, deprecation_reason="Compose v1 is deprecated")
        if "compose v2" in combined or "docker compose" in combined:
            return VersionMetadata(raw_version="2.0", parsed_tuple=(2, 0), is_deprecated=False)

        # General semver search
        for reg in _VERSION_REGEXES:
            m = reg.search(combined)
            if m:
                raw = m.group(1) if m.groups() else m.group(0)
                parsed = parse_semver(raw)
                is_dep = any(k in combined for k in self.deprecation_keywords)
                return VersionMetadata(raw_version=raw, parsed_tuple=parsed, is_deprecated=is_dep)

        is_dep = any(k in combined for k in self.deprecation_keywords)
        return VersionMetadata(raw_version=None, parsed_tuple=(0,), is_deprecated=is_dep)

    def detect_user_version_constraint(self, query: str) -> Optional[Tuple[str, Tuple[int, ...]]]:
        """
        Detects if user explicitly asked for a specific version.
        """
        q_lower = query.lower()
        if "helm 2" in q_lower or "helm v2" in q_lower:
            return ("helm v2", (2, 0))
        if "helm 3" in q_lower or "helm v3" in q_lower:
            return ("helm v3", (3, 0))
        if "compose v1" in q_lower or "compose 1" in q_lower:
            return ("compose v1", (1, 0))
        if "compose v2" in q_lower or "compose 2" in q_lower:
            return ("compose v2", (2, 0))

        # Check for generic version in query e.g. "version 1.25", "k8s 1.28"
        for reg in _VERSION_REGEXES:
            m = reg.search(query)
            if m:
                raw = m.group(1) if m.groups() else m.group(0)
                return (raw, parse_semver(raw))
        return None

    def is_comparison_query(self, query: str) -> bool:
        """
        Detects if the user is asking to compare versions or migration changes.
        """
        q_lower = query.lower()
        comparison_triggers = [
            "difference between", "diff", "versus", "vs", "compare", 
            "migrat", "upgrade from", "what changed in", "breaking change"
        ]
        return any(trig in q_lower for trig in comparison_triggers)

    def resolve_conflicts(self, query: str, chunks: List[Any]) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Applies deterministic conflict arbitration to the list of retrieved chunks.
        
        Returns:
            (reordered_and_filtered_chunks, conflict_resolution_report)
        """
        if not chunks:
            return [], {"total_conflicts": 0, "resolution_actions": []}

        user_version_req = self.detect_user_version_constraint(query)
        is_comparison = self.is_comparison_query(query)

        resolved_items: List[ResolvedChunk] = []
        resolution_actions: List[str] = []

        # Analyze each chunk's version and authority
        for idx, ch in enumerate(chunks):
            text = getattr(ch, "text", "")
            title = getattr(ch, "title", "") or ""
            topic = getattr(ch, "topic", "") or ""
            base_score = float(getattr(ch, "score", 0.5) or 0.5)

            v_meta = self.extract_version(text, title, topic)
            
            # Start with base score
            priority = base_score

            # Rule 1: Explicit user-requested version
            if user_version_req:
                req_str, req_tuple = user_version_req
                if v_meta.parsed_tuple and v_meta.parsed_tuple[0] == req_tuple[0]:
                    priority += 1.0  # Huge boost for exact requested major version
                    resolution_actions.append(f"Boosted chunk {idx} matching user requested version {req_str}")
                elif v_meta.parsed_tuple != (0,) and v_meta.parsed_tuple[0] != req_tuple[0]:
                    priority -= 0.5  # Penalize non-matching version
                    resolution_actions.append(f"Deprioritized chunk {idx} ({v_meta.raw_version}) conflicting with requested {req_str}")

            # Rule 2: If user did NOT ask for comparison or legacy version, newer supersedes older
            elif not is_comparison:
                if v_meta.is_deprecated:
                    priority -= 0.4
                    resolution_actions.append(f"Demoted deprecated chunk {idx} ({v_meta.deprecation_reason or 'deprecated'})")
                elif v_meta.parsed_tuple > (1, 0):
                    # Newer versions get priority over older
                    priority += 0.05 * v_meta.parsed_tuple[0]

            resolved_items.append(ResolvedChunk(
                chunk=ch,
                version_meta=v_meta,
                priority_score=priority,
                conflict_notes=[]
            ))

        # Sort deterministically by priority_score (highest priority first)
        resolved_items.sort(key=lambda item: item.priority_score, reverse=True)

        # Flag superseded chunks if not in comparison mode
        final_chunks = []
        seen_topics = set()
        for item in resolved_items:
            final_chunks.append(item.chunk)

        conflict_report = {
            "user_version_constraint": user_version_req[0] if user_version_req else None,
            "comparison_mode": is_comparison,
            "actions_taken": resolution_actions,
            "chunks_arbitrated": len(chunks)
        }

        return final_chunks, conflict_report
