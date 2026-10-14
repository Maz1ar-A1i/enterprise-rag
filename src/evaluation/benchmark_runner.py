"""
benchmark_runner.py
===================
Runs the full evaluation benchmark and produces frozen result files.

Usage:
    runner = BenchmarkRunner(pipeline, groq_client, config)
    report = runner.run(benchmark_path="datasets/benchmark/questions_visible.jsonl",
                        output_path="results/baseline_naive_rag.json")

Benchmark JSONL format (one JSON object per line):
{
    "id":                 "q_001",
    "question":           "How do I configure Nginx to use HTTPS?",
    "answer":             "To enable HTTPS in Nginx...",
    "relevant_chunk_ids": ["nginx_http_00042", "nginx_http_00198"],
    "category":           "direct",
    "expected_refusal":   false
}

Output JSON format:
{
    "meta": { "benchmark_file": ..., "timestamp": ..., "total_questions": ... },
    "aggregate": { "recall_1": ..., "recall_5": ..., ... },
    "by_category": { "direct": {...}, "multi_hop": {...}, ... },
    "per_question": [ {...}, {...}, ... ],
    "latency": { "p50_ms": ..., "p95_ms": ... },
    "system": { "ram_mb": ... }
}
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from tqdm import tqdm

from src.pipeline.naive_rag import NaiveRAGPipeline, RAGResult
from src.generation.groq_client import GroqClient
from src.evaluation.metrics import (
    compute_retrieval_metrics,
    judge_correctness,
    judge_faithfulness,
    citation_precision,
    refusal_metrics,
    hallucination_rate,
    latency_percentiles,
)

log = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Runs all benchmark questions through the pipeline and computes all metrics.

    Args:
        pipeline:    Initialized NaiveRAGPipeline.
        groq_client: GroqClient for LLM-as-judge calls.
        k_values:    Recall@K cutoffs.
        ndcg_k:      nDCG cutoff.
        use_llm_judge: Whether to call Groq for correctness/faithfulness scoring.
                       Set False for a fast retrieval-only eval run.
        save_per_question: Whether to include per-question details in output.
    """

    def __init__(
        self,
        pipeline: NaiveRAGPipeline,
        groq_client: GroqClient,
        k_values: List[int] = None,
        ndcg_k: int = 10,
        use_llm_judge: bool = True,
        save_per_question: bool = True,
    ):
        self.pipeline          = pipeline
        self.groq              = groq_client
        self.k_values          = k_values or [1, 5, 10]
        self.ndcg_k            = ndcg_k
        self.use_llm_judge     = use_llm_judge
        self.save_per_question = save_per_question

    # ── Main evaluation run ────────────────────────────────────────────────

    def run(
        self,
        benchmark_path: str,
        output_path: str,
    ) -> dict:
        """
        Run the full benchmark and write results to disk.

        Args:
            benchmark_path: Path to JSONL benchmark file.
            output_path:    Where to write the JSON result file.

        Returns:
            The full results dict (also written to output_path).
        """
        questions = self._load_benchmark(benchmark_path)
        total = len(questions)
        print(f"\n[Eval] Starting evaluation: {total} questions")
        print(f"[Eval] LLM judge enabled: {self.use_llm_judge}")

        per_question_results = []
        all_retrieval: List[dict] = []
        all_correctness: List[float] = []
        all_faithfulness: List[float] = []
        all_citation_precisions: List[float] = []
        all_latencies: List[float] = []
        all_ram: List[float] = []

        predicted_refusals: List[bool] = []
        expected_refusals:  List[bool] = []

        for item in tqdm(questions, desc="[Eval] Questions"):
            q_result = self._evaluate_one(item)
            per_question_results.append(q_result)

            # Collect for aggregation
            all_retrieval.append(q_result["retrieval_metrics"])
            all_latencies.append(q_result["total_latency_ms"])
            all_ram.append(q_result["ram_mb"])
            predicted_refusals.append(q_result["is_refusal"])
            expected_refusals.append(item.get("expected_refusal", False))

            if q_result.get("correctness_score") is not None:
                all_correctness.append(q_result["correctness_score"])
            if q_result.get("faithfulness_score") is not None:
                all_faithfulness.append(q_result["faithfulness_score"])
            if q_result.get("citation_precision") is not None:
                all_citation_precisions.append(q_result["citation_precision"])

        # ── Aggregate metrics ──────────────────────────────────────────────
        aggregate = self._aggregate_retrieval(all_retrieval)

        if all_correctness:
            aggregate["answer_correctness"] = float(np.mean(all_correctness))
        if all_faithfulness:
            aggregate["faithfulness"]       = float(np.mean(all_faithfulness))
            aggregate["hallucination_rate"] = hallucination_rate(all_faithfulness)
        if all_citation_precisions:
            aggregate["citation_precision"] = float(np.mean(all_citation_precisions))

        # Refusal metrics
        refusal_q_indices = [
            i for i, item in enumerate(questions)
            if item.get("expected_refusal", False)
        ]
        if refusal_q_indices:
            ref_metrics = refusal_metrics(predicted_refusals, expected_refusals)
            aggregate["refusal_precision"] = ref_metrics["precision"]
            aggregate["refusal_recall"]    = ref_metrics["recall"]
            aggregate["refusal_f1"]        = ref_metrics["f1"]

        # ── By-category breakdown ──────────────────────────────────────────
        by_category = self._aggregate_by_category(per_question_results, questions)

        # ── System metrics ─────────────────────────────────────────────────
        latency_stats = latency_percentiles(all_latencies)
        system_stats  = {
            "ram_mb_mean": float(np.mean(all_ram)),
            "ram_mb_max":  float(np.max(all_ram)),
        }

        # ── Build final report ─────────────────────────────────────────────
        report = {
            "meta": {
                "benchmark_file": benchmark_path,
                "timestamp":      datetime.now().isoformat(),
                "total_questions": total,
                "llm_judge_used": self.use_llm_judge,
            },
            "aggregate": aggregate,
            "by_category": by_category,
            "latency": latency_stats,
            "system": system_stats,
        }
        if self.save_per_question:
            report["per_question"] = per_question_results

        # ── Save to disk ───────────────────────────────────────────────────
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self._print_summary(aggregate, latency_stats, total)
        print(f"\n[Eval] Results saved -> {output_path}")
        return report

    # ── Evaluate a single question ─────────────────────────────────────────

    def _evaluate_one(self, item: dict) -> dict:
        """Run pipeline on one benchmark question and compute all metrics."""
        question        = item["question"]
        reference       = item.get("answer", "")
        relevant_ids    = item.get("relevant_chunk_ids", [])
        category        = item.get("category", "unknown")

        # Run pipeline
        rag_result: RAGResult = self.pipeline.run(question)

        # Retrieval metrics (using all retrieved IDs including pre-rerank)
        # We evaluate on retrieved_chunk_ids (hybrid output) for recall
        retrieval_metrics_dict = compute_retrieval_metrics(
            retrieved_ids=rag_result.retrieved_chunk_ids,
            relevant_ids=relevant_ids,
            k_values=self.k_values,
            ndcg_k=self.ndcg_k,
        )

        result = {
            "id":                  item.get("id", ""),
            "category":            category,
            "question":            question,
            "answer":              rag_result.answer,
            "retrieved_chunk_ids": rag_result.retrieved_chunk_ids,
            "reranked_chunk_ids":  rag_result.reranked_chunk_ids,
            "citations":           rag_result.citations,
            "is_refusal":          rag_result.is_refusal,
            "expected_refusal":    item.get("expected_refusal", False),
            "retrieval_metrics":   retrieval_metrics_dict,
            "total_latency_ms":    rag_result.total_latency_ms,
            "retrieval_latency_ms":rag_result.retrieval_latency_ms,
            "rerank_latency_ms":   rag_result.rerank_latency_ms,
            "generation_latency_ms":rag_result.generation_latency_ms,
            "ram_mb":              rag_result.ram_mb_at_query,
        }

        # Citation precision
        cit = citation_precision(rag_result.answer, rag_result.retrieved_urls)
        result["citation_precision"] = cit["precision"]
        result["cited_urls"]         = cit["cited_urls"]

        # LLM-as-judge (slower — can be disabled for quick retrieval-only runs)
        if self.use_llm_judge and reference:
            corr = judge_correctness(self.groq, question, reference, rag_result.answer)
            if corr:
                result["correctness_score"] = corr.get("score")
                result["correctness_reason"]= corr.get("reason")

            faith = judge_faithfulness(self.groq, rag_result.answer, rag_result.retrieved_texts)
            if faith:
                result["faithfulness_score"] = faith.get("faithfulness_score")
                result["unsupported_claims"] = faith.get("unsupported_claims", [])

        return result

    # ── Aggregation helpers ────────────────────────────────────────────────

    def _aggregate_retrieval(self, retrieval_list: List[dict]) -> dict:
        """Average all retrieval metrics across questions."""
        if not retrieval_list:
            return {}
        keys = retrieval_list[0].keys()
        return {
            k: float(np.mean([r[k] for r in retrieval_list if k in r]))
            for k in keys
        }

    def _aggregate_by_category(
        self, per_q: List[dict], questions: List[dict]
    ) -> Dict[str, dict]:
        """Compute retrieval metrics grouped by question category."""
        from collections import defaultdict
        category_map = defaultdict(list)

        for q_result, item in zip(per_q, questions):
            cat = item.get("category", "unknown")
            category_map[cat].append(q_result["retrieval_metrics"])

        return {
            cat: self._aggregate_retrieval(metrics_list)
            for cat, metrics_list in category_map.items()
        }

    def _load_benchmark(self, path: str) -> List[dict]:
        """Load a JSONL benchmark file."""
        questions = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))
        print(f"[Eval] Loaded {len(questions)} questions from {path}")
        return questions

    def _print_summary(self, aggregate: dict, latency: dict, total: int) -> None:
        """Print a formatted summary table to the terminal."""
        print("\n" + "=" * 60)
        print("  EVALUATION SUMMARY")
        print("=" * 60)
        print(f"  Total questions evaluated: {total}")
        print(f"\n  RETRIEVAL METRICS:")
        for k in [1, 5, 10]:
            key = f"recall_{k}"
            if key in aggregate:
                print(f"    Recall@{k:2d} : {aggregate[key]:.4f}")
        if "mrr" in aggregate:
            print(f"    MRR      : {aggregate['mrr']:.4f}")
        if "ndcg_10" in aggregate:
            print(f"    nDCG@10  : {aggregate['ndcg_10']:.4f}")
        print(f"\n  ANSWER QUALITY:")
        for key, label in [
            ("answer_correctness", "Correctness"),
            ("faithfulness",       "Faithfulness"),
            ("hallucination_rate", "Hallucination↓"),
            ("citation_precision", "Citation Prec"),
            ("refusal_precision",  "Refusal Prec"),
            ("refusal_recall",     "Refusal Recall"),
        ]:
            if key in aggregate:
                print(f"    {label:20s}: {aggregate[key]:.4f}")
        print(f"\n  LATENCY:")
        print(f"    p50: {latency['p50_ms']:.1f} ms")
        print(f"    p95: {latency['p95_ms']:.1f} ms")
        print("=" * 60)
