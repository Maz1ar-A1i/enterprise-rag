"""
metrics.py
==========
All evaluation metrics for the RAG pipeline.

Organized by group:
  GROUP 1 — Retrieval: Recall@K, MRR, nDCG@K
  GROUP 2 — Answer quality: Correctness, Faithfulness, Citation Precision,
             Refusal Precision/Recall, Hallucination Rate
  GROUP 3 — System: Latency percentiles, RAM usage

Each function is pure (no side effects) and thoroughly documented.
The LLM-as-judge functions call Groq to score answer quality.
"""

import re
import math
import logging
from typing import List, Dict, Optional, Any

import numpy as np

log = logging.getLogger(__name__)

# ============================================================================
# GROUP 1 — RETRIEVAL METRICS
# ============================================================================

def _matches_relevant(retrieved_id: str, relevant: set) -> bool:
    if retrieved_id in relevant:
        return True
    if "_" in retrieved_id:
        base_id = retrieved_id.rsplit("_", 1)[0]
        if base_id in relevant:
            return True
    return False


def recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """
    Fraction of relevant chunks that appear in the top-K retrieved results.

    Formula:  |{retrieved[:k]} ∩ {relevant}| / |{relevant}|

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs (most relevant first).
        relevant_ids:  Ground-truth relevant chunk IDs for this question.
        k:             Cutoff rank.

    Returns:
        Float in [0, 1].  Returns 0.0 if relevant_ids is empty.
    """
    if not relevant_ids:
        return 0.0
    relevant = set(relevant_ids)
    matched = set()
    for doc_id in retrieved_ids[:k]:
        if doc_id in relevant:
            matched.add(doc_id)
        elif "_" in doc_id:
            base_id = doc_id.rsplit("_", 1)[0]
            if base_id in relevant:
                matched.add(base_id)
    return len(matched) / len(relevant)


def mean_reciprocal_rank(
    retrieved_ids: List[str],
    relevant_ids: List[str],
) -> float:
    """
    Reciprocal rank of the first relevant document in the retrieved list.

    Formula:  1 / rank_of_first_relevant  (0 if none found in list)

    Returns:
        Float in [0, 1].
    """
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if _matches_relevant(doc_id, relevant):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int = 10,
) -> float:
    """
    Normalized Discounted Cumulative Gain at cutoff K.

    Uses binary relevance (1 = relevant, 0 = not relevant).
    Rewards finding relevant documents earlier in the ranking.

    Formula:
        DCG@K  = Σ_{i=1}^{K}  rel_i / log2(i+1)
        IDCG@K = DCG of the ideal ranking (all relevant first)
        nDCG@K = DCG / IDCG

    Returns:
        Float in [0, 1].  Returns 0.0 if no relevant documents.
    """
    if not relevant_ids:
        return 0.0

    relevant = set(relevant_ids)
    top_k    = retrieved_ids[:k]

    # Actual DCG
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(top_k, start=1)
        if _matches_relevant(doc_id, relevant)
    )

    # Ideal DCG: all relevant documents at the top positions
    ideal_k = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_k + 1))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_retrieval_metrics(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k_values: List[int] = None,
    ndcg_k: int = 10,
) -> Dict[str, float]:
    """
    Compute all retrieval metrics for a single question.

    Args:
        retrieved_ids: Ordered retrieved chunk IDs.
        relevant_ids:  Ground-truth relevant chunk IDs.
        k_values:      List of K cutoffs for Recall@K (default [1, 5, 10]).
        ndcg_k:        K cutoff for nDCG.

    Returns:
        Dict with keys: recall_1, recall_5, recall_10, mrr, ndcg_10.
    """
    if k_values is None:
        k_values = [1, 5, 10]

    result = {}
    for k in k_values:
        result[f"recall_{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)

    result["mrr"]     = mean_reciprocal_rank(retrieved_ids, relevant_ids)
    result[f"ndcg_{ndcg_k}"] = ndcg_at_k(retrieved_ids, relevant_ids, ndcg_k)
    return result


# ============================================================================
# GROUP 2 — ANSWER QUALITY METRICS (LLM-as-Judge)
# ============================================================================

# ── Correctness ──────────────────────────────────────────────────────────────

CORRECTNESS_SYSTEM_PROMPT = """You are an expert technical evaluator assessing the correctness of answers
to DevOps and cloud infrastructure questions.

Evaluate the System Answer against the Reference Answer.
Score from 0.0 to 1.0:
  1.0 — Fully correct, covers all key facts, no errors
  0.8 — Mostly correct, minor omission or slight imprecision
  0.6 — Partially correct, main concept right but key details missing
  0.4 — Some relevant content but significant errors or gaps
  0.2 — Mostly incorrect, only surface-level overlap
  0.0 — Completely wrong, irrelevant, or fabricated

Return ONLY a JSON object: {"score": <float 0.0-1.0>, "reason": "<one sentence>"}"""


def judge_correctness(
    groq_client,
    question: str,
    reference_answer: str,
    system_answer: str,
) -> Optional[Dict]:
    """
    Use the Groq LLM as a judge to score answer correctness.

    Returns:
        {"score": float, "reason": str}  or None on parse failure.
    """
    user_msg = f"""Question: {question}

Reference Answer: {reference_answer}

System Answer: {system_answer}

Score the System Answer for factual correctness."""

    return groq_client.call_json(
        system_prompt=CORRECTNESS_SYSTEM_PROMPT,
        user_message=user_msg,
    )


# ── Faithfulness ─────────────────────────────────────────────────────────────

FAITHFULNESS_SYSTEM_PROMPT = """You are evaluating whether an answer is fully grounded in the provided context.

Task:
1. Extract all atomic factual claims from the System Answer.
2. For each claim, check if it is explicitly supported by the Context Passages.
3. Return the count of supported and total claims.

A claim is SUPPORTED if it directly follows from the context text.
A claim is UNSUPPORTED if it adds information not present in the context (hallucination).

Return ONLY a JSON object:
{"supported": <int>, "total": <int>, "faithfulness_score": <float 0.0-1.0>, "unsupported_claims": ["claim1", "claim2"]}"""


def judge_faithfulness(
    groq_client,
    system_answer: str,
    context_texts: List[str],
) -> Optional[Dict]:
    """
    Score how well the answer is grounded in the retrieved context.

    Returns:
        {"supported": int, "total": int, "faithfulness_score": float, "unsupported_claims": list}
    """
    context_block = "\n\n---\n\n".join(context_texts)
    user_msg = f"""Context Passages:
{context_block}

System Answer:
{system_answer}

Evaluate faithfulness."""

    return groq_client.call_json(
        system_prompt=FAITHFULNESS_SYSTEM_PROMPT,
        user_message=user_msg,
    )


# ── Citation precision ────────────────────────────────────────────────────────

_CITATION_RE = re.compile(r"\[Source:\s*(https?://[^\]]+)\]", re.IGNORECASE)


def citation_precision(
    answer_text: str,
    retrieved_urls: List[str],
) -> Dict[str, Any]:
    """
    Compute citation precision: of all citations in the answer,
    what fraction point to URLs that were actually retrieved?

    Args:
        answer_text:    Raw answer string from the LLM.
        retrieved_urls: URLs of the chunks that were retrieved + shown.

    Returns:
        {"precision": float, "cited_urls": list, "valid_citations": int, "total_citations": int}
    """
    cited    = _CITATION_RE.findall(answer_text)
    valid_url_set = set(retrieved_urls)
    valid    = sum(1 for url in cited if url in valid_url_set)

    precision = valid / len(cited) if cited else 0.0
    return {
        "precision":        precision,
        "cited_urls":       cited,
        "valid_citations":  valid,
        "total_citations":  len(cited),
    }


# ── Refusal metrics ───────────────────────────────────────────────────────────

def refusal_metrics(
    predicted_refusals: List[bool],
    expected_refusals: List[bool],
) -> Dict[str, float]:
    """
    Compute precision and recall for refusal behavior.

    A True Positive = system refused AND question was truly unanswerable.
    A False Positive = system refused BUT question had an answer (too cautious).
    A False Negative = system answered BUT question was unanswerable (hallucinated).

    Args:
        predicted_refusals: List of bool — did system refuse each question?
        expected_refusals:  List of bool — is each question truly unanswerable?

    Returns:
        {"precision": float, "recall": float, "f1": float}
    """
    tp = fp = fn = tn = 0
    for pred, exp in zip(predicted_refusals, expected_refusals):
        if pred and exp:     tp += 1   # Correct refusal
        elif pred and not exp: fp += 1  # Wrong refusal (over-cautious)
        elif not pred and exp: fn += 1  # Missed refusal (hallucination)
        else:                 tn += 1   # Correct answer provided

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ── Hallucination rate ────────────────────────────────────────────────────────

def hallucination_rate(
    faithfulness_scores: List[float],
    threshold: float = 0.9,
) -> float:
    """
    Fraction of answers where faithfulness < threshold.
    A faithfulness score below threshold indicates at least one
    hallucinated claim in the answer.

    Args:
        faithfulness_scores: Per-question faithfulness scores.
        threshold:           Below this = counted as hallucination.

    Returns:
        Float in [0, 1]. Lower is better.
    """
    if not faithfulness_scores:
        return 0.0
    hallucinated = sum(1 for s in faithfulness_scores if s < threshold)
    return hallucinated / len(faithfulness_scores)


# ============================================================================
# GROUP 4 — SYSTEM PERFORMANCE METRICS
# ============================================================================

def latency_percentiles(
    latencies_ms: List[float],
) -> Dict[str, float]:
    """
    Compute p50 and p95 latency from a list of measurements.

    Args:
        latencies_ms: List of end-to-end latency values in milliseconds.

    Returns:
        {"p50_ms": float, "p95_ms": float, "min_ms": float, "max_ms": float, "mean_ms": float}
    """
    arr = np.array(latencies_ms)
    return {
        "p50_ms":  float(np.percentile(arr, 50)),
        "p95_ms":  float(np.percentile(arr, 95)),
        "min_ms":  float(arr.min()),
        "max_ms":  float(arr.max()),
        "mean_ms": float(arr.mean()),
    }
