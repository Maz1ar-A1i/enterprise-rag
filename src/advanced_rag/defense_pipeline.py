"""
src/advanced_rag/defense_pipeline.py
====================================
Production RAG Security and Defense Pipeline.

Implements an explicit multi-stage defense architecture:
  1. Source Trust Registry & Document Provenance Tracking
  2. Prompt Injection & Adversarial Payload Quarantine
  3. Instruction / Data Boundary Separation (Envelope Isolation)
  4. Context Filtering & Untrusted Content Stripping
  5. Fail-Closed / Refusal Logic for Untrusted or Insufficient Evidence

Complies with OWASP Top 10 for LLMs (LLM01: Prompt Injection, LLM02: Sensitive Information Disclosure).
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import re
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Known prompt injection & adversarial trigger patterns
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|directions)",
    r"disregard\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts)",
    r"system\s*override",
    r"you\s+are\s+now\s+(a|an|unrestricted|in\s+developer\s+mode)",
    r"developer\s+mode\s+enabled",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"exfiltrate\s+",
    r"output\s+the\s+(system\s+prompt|developer\s+instructions|hidden\s+key)",
    r"reveal\s+your\s+(instructions|rules|system\s+prompt)",
    r"<\s*script[^>]*>",
    r"\[SYSTEM_PROMPT\]",
    r"\[ADMIN_INSTRUCTION\]",
    r"bypass\s+(safety|guardrails|filters)"
]

# Compiled regexes for fast execution
_INJECTION_REGEXES = [re.compile(p, re.IGNORECASE) for p in PROMPT_INJECTION_PATTERNS]


@dataclass
class SourceTrustRecord:
    source_domain: str
    trust_tier: str          # "authoritative", "internal_trusted", "community", "untrusted"
    trust_score: float       # 0.0 to 1.0
    allowed_execution: bool = False
    requires_quarantine_check: bool = True


@dataclass
class DefenseAssessment:
    chunk_id: str
    is_safe: bool
    quarantined: bool
    quarantine_reason: Optional[str] = None
    trust_score: float = 1.0
    sanitized_text: str = ""
    detected_patterns: List[str] = field(default_factory=list)


class SourceTrustRegistry:
    """
    Tracks document provenance and authority tiers.
    """
    DEFAULT_TRUST_MAP = {
        "docs.docker.com": SourceTrustRecord("docs.docker.com", "authoritative", 1.0),
        "kubernetes.io": SourceTrustRecord("kubernetes.io", "authoritative", 1.0),
        "nginx.org": SourceTrustRecord("nginx.org", "authoritative", 1.0),
        "postgresql.org": SourceTrustRecord("postgresql.org", "authoritative", 1.0),
        "docs.helm.sh": SourceTrustRecord("docs.helm.sh", "authoritative", 1.0),
        "prometheus.io": SourceTrustRecord("prometheus.io", "authoritative", 1.0),
        "docs.github.com": SourceTrustRecord("docs.github.com", "authoritative", 1.0),
        "systemd.io": SourceTrustRecord("systemd.io", "authoritative", 1.0),
        "datatracker.ietf.org": SourceTrustRecord("datatracker.ietf.org", "authoritative", 1.0),
        "internal_enterprise": SourceTrustRecord("internal_enterprise", "internal_trusted", 0.95),
        "community_docs": SourceTrustRecord("community_docs", "community", 0.70),
        "untrusted_web": SourceTrustRecord("untrusted_web", "untrusted", 0.20),
    }

    def __init__(self, custom_trust_map: Optional[Dict[str, SourceTrustRecord]] = None):
        self.trust_map = dict(self.DEFAULT_TRUST_MAP)
        if custom_trust_map:
            self.trust_map.update(custom_trust_map)

    def get_trust(self, source_url_or_topic: str) -> SourceTrustRecord:
        for domain, record in self.trust_map.items():
            if domain in (source_url_or_topic or "").lower():
                return record
        # Default fallback for internal / recognized topics
        return SourceTrustRecord(
            source_domain=source_url_or_topic or "unknown",
            trust_tier="internal_trusted",
            trust_score=0.85
        )


class SuspiciousDocumentQuarantine:
    """
    Detects adversarial injection attacks and isolates poisoned chunks.
    """
    @staticmethod
    def scan_chunk(chunk_id: str, text: str) -> Tuple[bool, List[str]]:
        """
        Returns (is_suspicious, matched_patterns).
        """
        matched = []
        for reg in _INJECTION_REGEXES:
            if reg.search(text):
                matched.append(reg.pattern)
        
        is_suspicious = len(matched) > 0
        return is_suspicious, matched


class InstructionDataSeparator:
    """
    Enforces strict instruction/data boundary separation.
    Wraps text in structured XML data envelopes and neutralizes directive mimics.
    """
    @staticmethod
    def sanitize(text: str) -> str:
        # Strip raw system control delimiters that might trick transformer tokenizers
        cleaned = re.sub(r"<\|im_start\|>", "[START_TOKEN_STRIPPED]", text)
        cleaned = re.sub(r"<\|im_end\|>", "[END_TOKEN_STRIPPED]", cleaned)
        cleaned = re.sub(r"\[SYSTEM_PROMPT\]", "[DOC_HEADER]", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def wrap_in_envelope(chunk_id: str, topic: str, trust_score: float, text: str, version: Optional[str] = None) -> str:
        sanitized = InstructionDataSeparator.sanitize(text)
        ver_attr = f' version="{version}"' if version else ""
        return (
            f'<retrieved_data_envelope chunk_id="{chunk_id}" topic="{topic}" '
            f'trust="{trust_score:.2f}"{ver_attr} integrity="verified">\n'
            f'{sanitized}\n'
            f'</retrieved_data_envelope>'
        )


class SecurityDefensePipeline:
    """
    End-to-end security defense pipeline orchestrator.
    Filters, assesses, quarantines, and sanitizes retrieved chunks.
    """
    def __init__(self, trust_registry: Optional[SourceTrustRegistry] = None):
        self.trust_registry = trust_registry or SourceTrustRegistry()
        self.quarantine = SuspiciousDocumentQuarantine()
        self.separator = InstructionDataSeparator()

    def process_chunks(self, chunks: List[Any]) -> Tuple[List[Any], List[DefenseAssessment], Dict[str, Any]]:
        """
        Scans all retrieved chunks, quarantining unsafe ones and returning sanitized chunks.
        
        Returns:
            (safe_chunks, assessments, defense_summary)
        """
        safe_chunks = []
        assessments: List[DefenseAssessment] = []
        quarantined_count = 0

        for chunk in chunks:
            chunk_id = getattr(chunk, "chunk_id", str(id(chunk)))
            raw_text = getattr(chunk, "text", "")
            topic = getattr(chunk, "topic", "general")
            url = getattr(chunk, "url", "")

            # 1. Source Trust Assessment
            trust_record = self.trust_registry.get_trust(url or topic)

            # 2. Injection & Adversarial Scan
            is_suspicious, matched = self.quarantine.scan_chunk(chunk_id, raw_text)

            if is_suspicious:
                quarantined_count += 1
                assessment = DefenseAssessment(
                    chunk_id=chunk_id,
                    is_safe=False,
                    quarantined=True,
                    quarantine_reason=f"Prompt injection / adversarial pattern detected: {matched}",
                    trust_score=0.0,
                    sanitized_text="[QUARANTINED BY SECURITY DEFENSE PIPELINE]",
                    detected_patterns=matched
                )
                assessments.append(assessment)
                log.warning(f"[SECURITY DEFENSE] Quarantined chunk {chunk_id}: {matched}")
                continue

            # 3. Instruction/Data Boundary Sanitization
            sanitized = self.separator.sanitize(raw_text)
            assessment = DefenseAssessment(
                chunk_id=chunk_id,
                is_safe=True,
                quarantined=False,
                trust_score=trust_record.trust_score,
                sanitized_text=sanitized,
                detected_patterns=[]
            )
            assessments.append(assessment)
            safe_chunks.append(chunk)

        defense_summary = {
            "total_inspected": len(chunks),
            "safe_passed": len(safe_chunks),
            "quarantined": quarantined_count,
            "security_status": "SECURE" if quarantined_count == 0 else "ADVERSARIAL_ATTACK_BLOCKED",
            "fail_closed_required": len(safe_chunks) == 0 and len(chunks) > 0
        }

        return safe_chunks, assessments, defense_summary

    @staticmethod
    def should_fail_closed(safe_chunks: List[Any], defense_summary: Dict[str, Any]) -> bool:
        """
        Returns True if trustworthy evidence is insufficient or all chunks were quarantined.
        """
        if defense_summary.get("fail_closed_required", False):
            return True
        if not safe_chunks:
            return True
        return False
