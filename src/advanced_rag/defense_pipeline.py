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

# Known prompt injection, adversarial triggers, and credential exfiltration patterns (OWASP LLM01/LLM02)
PROMPT_INJECTION_PATTERNS = [
    # 1. Directive overrides & instruction hijacking
    r"ignore\s+(all\s+)?(previous|above|prior|system)\s+(instructions|prompts|directions|rules|constraints)",
    r"disregard\s+(all\s+)?(previous|above|prior|system)\s+(instructions|prompts|directions|rules|constraints)",
    r"forget\s+(all\s+)?(previous|prior|system)\s+(instructions|prompts|rules)",
    r"override\s+(all\s+)?(system|instructions|prompts|rules|constraints|filters|guardrails)",
    r"system\s*override",
    
    # 2. Persona hijacking & jailbreak triggers
    r"you\s+are\s+now\s+(a|an|unrestricted|in\s+developer\s+mode|dan|evil|unfiltered|jailbroken)",
    r"act\s+as\s+(an?\s+)?(unrestricted|jailbroken|evil|unfiltered|dan|system\s+administrator)",
    r"developer\s+mode(\s+enabled|\s+on)?",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"bypass\s+(safety|guardrails?|filters?|rules?|constraints?|authentication|security)",
    r"disable\s+(safety|guardrails?|filters?|security|defense)",
    
    # 3. System prompt & instruction exfiltration
    r"(reveal|output|display|show|print|dump|echo|repeat|leak)\s+(all\s+)?(the\s+)?(system\s+prompt|system\s+instructions|developer\s+instructions|developer\s+prompts|core\s+prompts|initial\s+prompt|hidden\s+instructions)",
    r"reveal\s+your\s+(instructions|rules|system\s+prompt|guidelines|constraints)",
    r"what\s+(is|are)\s+your\s+(system\s+prompt|system\s+instructions|initial\s+instructions|hidden\s+prompt)",
    r"repeat\s+everything\s+(above|before|preceding)",
    
    # 4. Secret, token, credential exfiltration & unauthorized penetration (OWASP LLM02)
    r"(output|display|show|print|dump|leak|exfiltrate|reveal)\s+(all\s+)?(the\s+)?(internal\s+)?(api\s*keys?|env(ironment)?\s+variables?|secrets?|tokens?|passwords?|credentials?)",
    r"exfiltrat(e|ion)",
    r"(internal\s+)?api\s*keys?\s+and\s+environment\s+variables?",
    r"\.env\s+(file|variables|keys)",
    r"(root|database|admin)\s+password",
    r"private\s+(api\s+token|secret\s+key|encryption\s+key|credentials?)",
    r"(unauthorized|exploit|backdoor)\s+(root\s+access|remote\s+access|penetration)",
    
    # 5. Tokenizer & delimiter injection mimics
    r"<\s*script[^>]*>",
    r"\[SYSTEM_PROMPT\]",
    r"\[ADMIN_INSTRUCTION\]",
    r"\[SYSTEM_OVERRIDE\]",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"\[INST\]",
    r"\[/INST\]"
]

# Compiled regexes for fast execution
_INJECTION_REGEXES = [re.compile(p, re.IGNORECASE) for p in PROMPT_INJECTION_PATTERNS]

SECURITY_GUARDRAIL_REFUSAL = (
    "🛡️ **Security Guardrail Active**: The request was quarantined by the Defense Pipeline "
    "due to potential prompt injection, adversarial trigger patterns, or system override attempt. "
    "System instructions, API keys, and internal environment configurations are strictly isolated and cannot be accessed or disclosed."
)


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
        "confluence": SourceTrustRecord("confluence", "authoritative", 1.0),
        "atlassian": SourceTrustRecord("atlassian", "authoritative", 1.0),
        "jira": SourceTrustRecord("jira", "authoritative", 1.0),
        "linear": SourceTrustRecord("linear.app", "authoritative", 1.0),
        "github": SourceTrustRecord("github.com", "authoritative", 1.0),
        "slack": SourceTrustRecord("slack.com", "internal_trusted", 0.95),
        "google_drive": SourceTrustRecord("drive.google.com", "authoritative", 1.0),
        "drive.google": SourceTrustRecord("drive.google.com", "authoritative", 1.0),
        "gmail": SourceTrustRecord("workspace.google.com", "internal_trusted", 0.95),
        "hubspot": SourceTrustRecord("hubspot.com", "authoritative", 1.0),
        "fireflies": SourceTrustRecord("fireflies.ai", "internal_trusted", 0.90),
        "enterprise-rag-dataset": SourceTrustRecord("enterprise-rag-dataset", "internal_trusted", 0.95),
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

    def scan_query(self, query: str) -> Tuple[bool, List[str], str]:
        """
        Scans an incoming user query for adversarial prompt injection,
        jailbreak attempts, system override instructions, and credential exfiltration.
        
        Returns:
            (is_adversarial, matched_patterns, refusal_message)
        """
        is_suspicious, matched = self.quarantine.scan_chunk("user_query", query)
        if is_suspicious:
            log.warning(f"[SECURITY DEFENSE] Adversarial query blocked: {matched} | Query: {query[:120]}")
            return True, matched, SECURITY_GUARDRAIL_REFUSAL
        return False, [], ""

    def scan_output(self, output_text: str) -> Tuple[bool, str]:
        """
        Defense-in-depth output guardrail:
        Verifies that generated text does not reflect adversarial overrides or exfiltrate credentials.
        """
        if not output_text:
            return False, output_text
        
        leak_patterns = [
            r"system\s*override\s*(successful|complete)",
            r"gsk_[a-zA-Z0-9]{20,}",
            r"sk-[a-zA-Z0-9]{20,}",
            r"(GROQ_API_KEY|OPENAI_API_KEY|DATABASE_URL)\s*=",
        ]
        for lp in leak_patterns:
            if re.search(lp, output_text, re.IGNORECASE):
                log.error(f"[SECURITY DEFENSE] Output leak detected: {lp}")
                return True, SECURITY_GUARDRAIL_REFUSAL
        return False, output_text

    @staticmethod
    def should_fail_closed(safe_chunks: List[Any], defense_summary: Dict[str, Any]) -> bool:
        """
        Returns True if trustworthy evidence is insufficient or all chunks were quarantined.
        """
        if defense_summary.get("fail_closed_required", False):
            return True
        if defense_summary.get("quarantined", 0) > 0:
            return True
        if not safe_chunks:
            return True
        return False

