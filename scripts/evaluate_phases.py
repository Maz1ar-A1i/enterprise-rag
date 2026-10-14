"""
scripts/evaluate_phases.py
==========================
Run benchmark evaluation across progressive checkpoints (Phase 1 vs Phase 2)
side-by-side on the same queries and compute exact delta improvements.

Usage:
    # Quick retrieval-only comparison on 20 benchmark questions:
    python scripts/evaluate_phases.py --limit 20

    # Full benchmark comparison on visible dataset:
    python scripts/evaluate_phases.py --no-judge

    # Single query comparison test:
    python scripts/evaluate_phases.py --query "How do I configure PostgreSQL connection pooling in FastAPI?"
"""

import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.phase_manager import PhasePipelineFactory
from src.generation.groq_client import GroqClient
from src.evaluation.metrics import (
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
)

def evaluate_retrieval_for_phases(
    benchmark_file: str,
    phases: List[int] = [1, 2],
    limit: int = None,
) -> Dict[int, Dict[str, float]]:
    """
    Run the same benchmark questions through each phase pipeline and compute metrics.
    """
    questions = []
    with open(benchmark_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    if limit:
        questions = questions[:limit]

    print(f"\n[Evaluator] Running comparative evaluation over {len(questions)} questions...")
    phase_metrics = {}

    for phase in phases:
        print(f"\n--- Initializing Phase {phase} Pipeline ---")
        pipeline = PhasePipelineFactory.get_pipeline(phase=phase)
        
        mrr_scores = []
        r1_scores = []
        r5_scores = []
        r10_scores = []
        ndcg10_scores = []
        latencies = []

        print(f"[Evaluator] Processing questions through Phase {phase}...")
        t0_total = time.time()
        for idx, q in enumerate(questions):
            query = q["question"]
            gold_cids = q.get("relevant_chunk_ids", [])
            
            t0 = time.time()
            rag_res = pipeline.run(query)
            elapsed_ms = (time.time() - t0) * 1000
            latencies.append(elapsed_ms)

            # Reranked chunk IDs are the final retrieved context for generation
            ranked_ids = rag_res.reranked_chunk_ids

            mrr_scores.append(mean_reciprocal_rank(ranked_ids, gold_cids))
            r1_scores.append(recall_at_k(ranked_ids, gold_cids, k=1))
            r5_scores.append(recall_at_k(ranked_ids, gold_cids, k=5))
            r10_scores.append(recall_at_k(ranked_ids, gold_cids, k=10))
            ndcg10_scores.append(ndcg_at_k(ranked_ids, gold_cids, k=10))

            if (idx + 1) % 10 == 0 or (idx + 1) == len(questions):
                print(f"  Processed {idx + 1}/{len(questions)} questions...")

        import numpy as np
        phase_metrics[phase] = {
            "mrr": float(np.mean(mrr_scores)),
            "recall_at_1": float(np.mean(r1_scores)),
            "recall_at_5": float(np.mean(r5_scores)),
            "recall_at_10": float(np.mean(r10_scores)),
            "ndcg_at_10": float(np.mean(ndcg10_scores)),
            "avg_latency_ms": float(np.mean(latencies)),
        }

    return phase_metrics

def main():
    parser = argparse.ArgumentParser(description="Side-by-side evaluation of RAG phases")
    parser.add_argument("--benchmark", default="datasets/benchmark/questions_visible.jsonl",
                        help="Benchmark questions file")
    parser.add_argument("--limit", type=int, default=15,
                        help="Limit number of questions for quick run")
    parser.add_argument("--phases", type=int, nargs="+", default=[1, 2],
                        help="Phase numbers to evaluate (e.g. 1 2 or 1 2 3)")
    parser.add_argument("--query", type=str, default=None,
                        help="Test a single query interactively across all phases")
    parser.add_argument("--output", default="results/phase_comparison_results.json",
                        help="Output comparison results file")
    args = parser.parse_args()

    phase_names = {
        1: "Phase 1 (Baseline)",
        2: "Phase 2 (+ Reranker FT)",
        3: "Phase 3 (+ Embedding FT)",
        4: "Phase 4 (+ LLM FT)",
    }

    if args.query:
        print(f"\n======================================================================")
        print(f"SINGLE QUERY SIDE-BY-SIDE EVALUATION ACROSS PHASES {args.phases}")
        print(f"Query: '{args.query}'")
        print(f"======================================================================")
        results = PhasePipelineFactory.run_all_available_phases(args.query, phases=args.phases)
        for p, res in results.items():
            label = phase_names.get(p, f"Phase {p}")
            print(f"\n[{label}]")
            print(f"Top 3 Chunks: {res.reranked_chunk_ids[:3]}")
            print(f"Scores:       {[round(s, 4) for s in res.retrieval_scores[:3]]}")
            print(f"Answer:       {res.answer[:250]}...")
            print(f"Total Latency:{res.total_latency_ms:.2f} ms")
        return

    metrics = evaluate_retrieval_for_phases(
        benchmark_file=args.benchmark,
        phases=args.phases,
        limit=args.limit,
    )

    print("\n" + "=" * 95)
    print(f"PROGRESSIVE RAG EVALUATION MATRIX ({' vs '.join(f'Phase {p}' for p in args.phases)})")
    print("=" * 95)
    
    header = f"{'Metric':<18}"
    for p in args.phases:
        header += f" | {phase_names.get(p, f'Phase {p}'):<22}"
    print(header)
    print("-" * 95)

    for k in ["mrr", "recall_at_1", "recall_at_5", "recall_at_10", "ndcg_at_10", "avg_latency_ms"]:
        row = f"{k:<18}"
        for p in args.phases:
            val = metrics[p][k]
            if k == "avg_latency_ms":
                row += f" | {val:<22.2f}"
            else:
                row += f" | {val:<22.4f}"
        print(row)
    print("=" * 95)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "evaluated_questions": args.limit,
            "evaluated_phases": args.phases,
            "metrics": metrics,
        }, f, indent=2)
    print(f"\nDetailed comparison saved to: {args.output}")

if __name__ == "__main__":
    main()
