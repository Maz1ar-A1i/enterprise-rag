"""
src/pipeline/phase_manager.py
=============================
Unified Phase Architecture for Progressive RAG Evaluation.

Maintains ONE integrated RAG product with progressive evaluation checkpoints:
  - Phase 1: Baseline (Baseline Embeddings + Baseline Reranker + Baseline LLM)
  - Phase 2: + Reranker Fine-Tuning (Baseline Embeddings + Fine-Tuned Reranker + Baseline LLM)
  - Phase 3: + Embedding Fine-Tuning (Fine-Tuned Embeddings + Fine-Tuned Reranker + Baseline LLM) [Upcoming]
  - Phase 4: + LLM Fine-Tuning (Fine-Tuned Embeddings + Fine-Tuned Reranker + Fine-Tuned LLM) [Final Product]
"""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import yaml
import logging

from src.pipeline.naive_rag import NaiveRAGPipeline, RAGResult
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.reranker import Reranker
from src.generation.groq_client import GroqClient, DEFAULT_GENERATE_MODEL

log = logging.getLogger(__name__)

PHASE_CONFIGS = {
    1: "configs/naive_rag_config.yaml",
    2: "configs/phase2_rag_config.yaml",
    3: "configs/phase3_rag_config.yaml",
}

class PhasePipelineFactory:
    """
    Factory to instantiate or retrieve specific Phase pipelines
    without duplicating system components.
    """

    @staticmethod
    def get_pipeline(phase: int = 1, config_path: Optional[str] = None) -> NaiveRAGPipeline:
        """
        Build a pipeline for a specific evaluation phase.

        Args:
            phase: 1 (Baseline), 2 (Reranker FT), 3 (Embedding FT), 4 (LLM FT)
            config_path: Optional custom config YAML path.
        """
        if config_path is None:
            config_path = PHASE_CONFIGS.get(phase, f"configs/phase{phase}_rag_config.yaml")

        if not Path(config_path).exists():
            # Fallback to Phase 1 config if specific phase config not found
            log.warning(f"Config '{config_path}' not found. Falling back to configs/naive_rag_config.yaml")
            config_path = "configs/naive_rag_config.yaml"

        print(f"[PhaseManager] Initializing Pipeline for Phase {phase} using config: {config_path}")
        return NaiveRAGPipeline.from_config(config_path)

    @classmethod
    def run_all_available_phases(
        cls,
        question: str,
        phases: List[int] = [1, 2],
    ) -> Dict[int, RAGResult]:
        """
        Execute the exact same query across all available phases
        for side-by-side comparative evaluation.

        Returns:
            Dict mapping phase number -> RAGResult
        """
        results = {}
        for p in phases:
            pipeline = cls.get_pipeline(phase=p)
            results[p] = pipeline.run(question)
        return results
