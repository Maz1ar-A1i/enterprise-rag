"""
report_generator.py
===================
Reads a saved evaluation JSON and produces:
  1. A formatted text table (terminal / log output)
  2. A CSV file (one row per question)
  3. Charts:
     - Radar chart: all retrieval metrics
     - Bar chart: metrics by category
     - Latency histogram
     - Faithfulness vs. Correctness scatter

All outputs go to results/charts/
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (works without display)
from tabulate import tabulate

log = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates visual and tabular reports from a benchmark results JSON.

    Args:
        results_path: Path to the JSON written by BenchmarkRunner.
        charts_dir:   Directory where chart PNGs are saved.
    """

    def __init__(self, results_path: str, charts_dir: str = "results/charts"):
        self.results_path = Path(results_path)
        self.charts_dir   = Path(charts_dir)
        self.charts_dir.mkdir(parents=True, exist_ok=True)

        with open(results_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    # ── Master generate method ─────────────────────────────────────────────

    def generate_all(self) -> None:
        """Run all report types."""
        print("[Report] Generating all reports...")
        self.print_summary_table()
        self.export_csv()
        self.plot_aggregate_bar()
        self.plot_by_category_bar()
        self.plot_latency_histogram()
        print(f"[Report] All outputs written to {self.charts_dir}")

    # ── 1. Terminal table ──────────────────────────────────────────────────

    def print_summary_table(self) -> None:
        agg = self.data.get("aggregate", {})
        lat = self.data.get("latency", {})
        meta = self.data.get("meta", {})

        rows = [
            ["Benchmark file",   meta.get("benchmark_file", "?")],
            ["Total questions",  meta.get("total_questions", "?")],
            ["Timestamp",        meta.get("timestamp", "?")],
            ["---", "---"],
            ["Recall@1",         f"{agg.get('recall_1', 0):.4f}"],
            ["Recall@5",         f"{agg.get('recall_5', 0):.4f}"],
            ["Recall@10",        f"{agg.get('recall_10', 0):.4f}"],
            ["MRR",              f"{agg.get('mrr', 0):.4f}"],
            ["nDCG@10",          f"{agg.get('ndcg_10', 0):.4f}"],
            ["---", "---"],
            ["Answer Correctness",f"{agg.get('answer_correctness', 0):.4f}"],
            ["Faithfulness",     f"{agg.get('faithfulness', 0):.4f}"],
            ["Hallucination Rate",f"{agg.get('hallucination_rate', 0):.4f}"],
            ["Citation Precision",f"{agg.get('citation_precision', 0):.4f}"],
            ["Refusal Precision", f"{agg.get('refusal_precision', 0):.4f}"],
            ["Refusal Recall",   f"{agg.get('refusal_recall', 0):.4f}"],
            ["---", "---"],
            ["p50 Latency",      f"{lat.get('p50_ms', 0):.1f} ms"],
            ["p95 Latency",      f"{lat.get('p95_ms', 0):.1f} ms"],
        ]

        print("\n" + tabulate(rows, headers=["Metric", "Value"], tablefmt="grid"))

    # ── 2. CSV export ──────────────────────────────────────────────────────

    def export_csv(self) -> Path:
        """Write one CSV row per question."""
        per_q = self.data.get("per_question", [])
        if not per_q:
            print("[Report] No per-question data to export.")
            return

        csv_path = self.charts_dir / "per_question_results.csv"
        fieldnames = [
            "id", "category", "question",
            "recall_1", "recall_5", "recall_10", "mrr", "ndcg_10",
            "correctness_score", "faithfulness_score",
            "citation_precision", "is_refusal", "expected_refusal",
            "total_latency_ms", "retrieval_latency_ms",
            "rerank_latency_ms", "generation_latency_ms",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for q in per_q:
                row = {k: q.get(k, "") for k in fieldnames}
                # Flatten nested retrieval_metrics
                rm = q.get("retrieval_metrics", {})
                row["recall_1"]  = rm.get("recall_1", "")
                row["recall_5"]  = rm.get("recall_5", "")
                row["recall_10"] = rm.get("recall_10", "")
                row["mrr"]       = rm.get("mrr", "")
                row["ndcg_10"]   = rm.get("ndcg_10", "")
                writer.writerow(row)

        print(f"[Report] CSV saved -> {csv_path}")
        return csv_path

    # ── 3. Aggregate bar chart ─────────────────────────────────────────────

    def plot_aggregate_bar(self) -> None:
        agg = self.data.get("aggregate", {})
        metrics = {
            "Recall@1": agg.get("recall_1", 0),
            "Recall@5": agg.get("recall_5", 0),
            "Recall@10":agg.get("recall_10", 0),
            "MRR":      agg.get("mrr", 0),
            "nDCG@10":  agg.get("ndcg_10", 0),
            "Correctness": agg.get("answer_correctness", 0),
            "Faithfulness": agg.get("faithfulness", 0),
            "Citation Prec": agg.get("citation_precision", 0),
        }

        fig, ax = plt.subplots(figsize=(12, 5))
        colors = ["#2196F3"] * 5 + ["#4CAF50"] * 3
        bars = ax.bar(list(metrics.keys()), list(metrics.values()), color=colors, edgecolor="white")

        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score (0–1)", fontsize=12)
        ax.set_title("Naive RAG Baseline — Aggregate Metrics", fontsize=14, fontweight="bold")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        for bar, val in zip(bars, metrics.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        # Legend
        from matplotlib.patches import Patch
        legend_elems = [
            Patch(facecolor="#2196F3", label="Retrieval"),
            Patch(facecolor="#4CAF50", label="Answer Quality"),
        ]
        ax.legend(handles=legend_elems)
        ax.tick_params(axis="x", labelrotation=15)
        plt.tight_layout()
        out = self.charts_dir / "aggregate_metrics.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Report] Chart saved -> {out}")

    # ── 4. By-category bar chart ───────────────────────────────────────────

    def plot_by_category_bar(self) -> None:
        by_cat = self.data.get("by_category", {})
        if not by_cat:
            return

        categories = list(by_cat.keys())
        recall5    = [by_cat[c].get("recall_5", 0) for c in categories]
        ndcg       = [by_cat[c].get("ndcg_10", 0) for c in categories]
        mrr        = [by_cat[c].get("mrr", 0) for c in categories]

        x = np.arange(len(categories))
        width = 0.28

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - width, recall5, width, label="Recall@5", color="#2196F3")
        ax.bar(x,         ndcg,    width, label="nDCG@10",  color="#FF9800")
        ax.bar(x + width, mrr,     width, label="MRR",      color="#9C27B0")

        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Retrieval Performance by Question Category", fontsize=14, fontweight="bold")
        ax.legend()
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        plt.tight_layout()

        out = self.charts_dir / "by_category.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Report] Chart saved -> {out}")

    # ── 5. Latency histogram ───────────────────────────────────────────────

    def plot_latency_histogram(self) -> None:
        per_q = self.data.get("per_question", [])
        if not per_q:
            return

        latencies = [q.get("total_latency_ms", 0) for q in per_q]
        lat_stats = self.data.get("latency", {})

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(latencies, bins=40, color="#2196F3", edgecolor="white", alpha=0.85)
        ax.axvline(lat_stats.get("p50_ms", 0), color="orange", linestyle="--",
                   linewidth=2, label=f"p50 = {lat_stats.get('p50_ms', 0):.0f}ms")
        ax.axvline(lat_stats.get("p95_ms", 0), color="red", linestyle="--",
                   linewidth=2, label=f"p95 = {lat_stats.get('p95_ms', 0):.0f}ms")
        ax.set_xlabel("End-to-end latency (ms)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Query Latency Distribution", fontsize=14, fontweight="bold")
        ax.legend()
        plt.tight_layout()

        out = self.charts_dir / "latency_histogram.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Report] Chart saved -> {out}")
