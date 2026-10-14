"""
scripts/run_evaluation.py
=========================
Step 4: Run the full evaluation benchmark and produce the frozen results file.

This is the ONE COMMAND that produces the official benchmark results.
Run it only when you want to lock in results — don't run it repeatedly
on partial implementations or you'll pollute your results history.

Usage:
    # Full evaluation with LLM judge:
    python scripts/run_evaluation.py --config configs/naive_rag_config.yaml

    # Fast retrieval-only eval (no Groq judge calls):
    python scripts/run_evaluation.py --config configs/naive_rag_config.yaml --no-judge

    # Evaluate only specific categories:
    python scripts/run_evaluation.py --categories direct troubleshooting

    # Dry run with 10 questions:
    python scripts/run_evaluation.py --limit 10

Output:
    results/baseline_naive_rag.json   — frozen evaluation results
    results/charts/                   — visualizations
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.naive_rag       import NaiveRAGPipeline
from src.generation.groq_client   import GroqClient
from src.evaluation.benchmark_runner import BenchmarkRunner
from src.evaluation.report_generator import ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run the RAG evaluation benchmark")
    parser.add_argument("--config",   default="configs/naive_rag_config.yaml")
    parser.add_argument("--benchmark", default=None,
                        help="Override benchmark file path")
    parser.add_argument("--output",   default=None,
                        help="Override output JSON file path")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM-as-judge (retrieval metrics only, much faster)")
    parser.add_argument("--limit",    type=int, default=None,
                        help="Evaluate only the first N questions (for testing)")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Only evaluate questions from these categories")
    parser.add_argument("--phase", type=int, default=None, choices=[1, 2],
                        help="Phase checkpoint to evaluate (1: Baseline, 2: + Reranker FT)")
    parser.add_argument("--pipeline", default="naive", choices=["naive", "advanced"],
                        help="Pipeline type to evaluate ('naive' or 'advanced')")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip chart/CSV generation")
    args = parser.parse_args()

    # Determine config by phase if specified
    if args.phase == 1:
        args.config = "configs/naive_rag_config.yaml"
    elif args.phase == 2:
        args.config = "configs/phase2_rag_config.yaml"

    # ── Load config ────────────────────────────────────────────────────────
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    eval_cfg   = cfg["evaluation"]
    bench_cfg  = cfg["benchmark"]

    benchmark_path = args.benchmark or bench_cfg["visible_file"]
    timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.phase:
        output_prefix = f"phase_{args.phase}_evaluation"
    else:
        output_prefix = "advanced_rag" if args.pipeline == "advanced" else "baseline_naive_rag"
    output_path    = args.output or f"results/{output_prefix}_{timestamp}.json"

    # ── Validate benchmark file ────────────────────────────────────────────
    if not Path(benchmark_path).exists():
        log.error(f"Benchmark file not found: {benchmark_path}")
        log.error("Run scripts/generate_benchmark.py first.")
        sys.exit(1)

    # ── Filter benchmark if needed ─────────────────────────────────────────
    if args.limit or args.categories:
        temp_path = _filter_benchmark(benchmark_path, args.limit, args.categories)
        benchmark_path = temp_path

    # ── Load pipeline ──────────────────────────────────────────────────────
    print(f"\n[Runner] Initializing {args.pipeline.upper()} pipeline...")
    print("[Runner] (This loads models into memory — may take 30–60 seconds)")

    try:
        if args.pipeline == "advanced":
            from src.pipeline.advanced_rag import AdvancedRAGPipeline
            pipeline = AdvancedRAGPipeline.from_config(args.config)
        else:
            pipeline = NaiveRAGPipeline.from_config(args.config)
    except FileNotFoundError as e:
        log.error(str(e))
        log.error("Run scripts/build_index.py first.")
        sys.exit(1)

    groq = GroqClient()

    # ── Run evaluation ─────────────────────────────────────────────────────
    print("\n[Runner] Starting evaluation run...")
    runner = BenchmarkRunner(
        pipeline=pipeline,
        groq_client=groq,
        k_values=eval_cfg.get("recall_k_values", [1, 5, 10]),
        ndcg_k=eval_cfg.get("ndcg_k", 10),
        use_llm_judge=not args.no_judge,
        save_per_question=eval_cfg.get("save_per_question", True),
    )

    report = runner.run(
        benchmark_path=benchmark_path,
        output_path=output_path,
    )

    # ── Generate charts ────────────────────────────────────────────────────
    if not args.no_report:
        print("\n[Runner] Generating reports...")
        reporter = ReportGenerator(
            results_path=output_path,
            charts_dir=str(Path("results") / "charts"),
        )
        reporter.generate_all()

    print(f"\n{'='*60}")
    print("  EVALUATION COMPLETE")
    print(f"  Results: {output_path}")
    print(f"  Charts:  results/charts/")
    print(f"{'='*60}")


def _filter_benchmark(path: str, limit: int, categories: list) -> str:
    """Write a filtered temporary benchmark file."""
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            if categories and q.get("category") not in categories:
                continue
            questions.append(q)
            if limit and len(questions) >= limit:
                break

    temp_path = "results/temp_benchmark_filtered.jsonl"
    Path("results").mkdir(exist_ok=True)
    with open(temp_path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"[Runner] Filtered benchmark: {len(questions)} questions -> {temp_path}")
    return temp_path


if __name__ == "__main__":
    main()
