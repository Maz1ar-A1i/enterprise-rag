"""
naive_rag.py
============
End-to-end Naive RAG pipeline.

This is the main class that wires all components together:
  Query -> Hybrid Retrieval -> Reranking -> Groq Generation -> Answer

It is the single entry-point used by:
  - scripts/run_evaluation.py  (batch evaluation)
  - Any interactive demo you build

Design principle:
  Components are injected (not instantiated here) so they can be
  swapped for fine-tuned versions in later phases without changing
  this file.

Usage:
    pipeline = NaiveRAGPipeline.from_config("configs/naive_rag_config.yaml")
    result   = pipeline.run("How do I configure Nginx upstream keepalive?")
    print(result.answer)
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List

from pathlib import Path
import psutil
import os
import yaml

from src.retrieval.bm25_retriever  import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult
from src.retrieval.reranker         import Reranker
try:
    from src.retrieval.dense_retriever import DenseRetriever
except Exception:
    DenseRetriever = object
from src.generation.groq_client     import GroqClient, DEFAULT_GENERATE_MODEL
from src.generation.prompt_builder  import (
    build_prompt, extract_citations, is_refusal
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RAGResult:
    """
    Complete output from one pipeline run.
    Stored for every benchmark question to enable post-hoc analysis.
    """
    question:               str
    answer:                 str
    retrieved_chunk_ids:    List[str]       # After hybrid retrieval (top-50)
    reranked_chunk_ids:     List[str]       # After reranking (top-5)
    retrieved_texts:        List[str]       # Reranked chunk texts
    retrieved_urls:         List[str]       # For citation verification
    retrieval_scores:       List[float]     # Reranker scores
    citations:              List[str]       # URLs cited in the answer
    is_refusal:             bool            # Did the model refuse to answer?
    retrieval_latency_ms:   float
    rerank_latency_ms:      float
    generation_latency_ms:  float
    total_latency_ms:       float
    ram_mb_at_query:        float

    def to_dict(self) -> dict:
        return {
            "question":             self.question,
            "answer":               self.answer,
            "retrieved_chunk_ids":  self.retrieved_chunk_ids,
            "reranked_chunk_ids":   self.reranked_chunk_ids,
            "citations":            self.citations,
            "is_refusal":           self.is_refusal,
            "retrieval_latency_ms": round(self.retrieval_latency_ms, 2),
            "rerank_latency_ms":    round(self.rerank_latency_ms, 2),
            "generation_latency_ms":round(self.generation_latency_ms, 2),
            "total_latency_ms":     round(self.total_latency_ms, 2),
            "ram_mb":               round(self.ram_mb_at_query, 1),
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class NaiveRAGPipeline:
    """
    The complete naive RAG pipeline.

    Args:
        bm25:        Loaded BM25Retriever.
        dense:       Loaded DenseRetriever.
        reranker:    Initialized Reranker.
        groq:        Initialized GroqClient.
        rrf_k:       RRF smoothing constant.
        bm25_top_k:  BM25 candidates.
        dense_top_k: Dense candidates.
        rerank_top_k: Final chunks kept after reranking.
        context_budget: Max context tokens for LLM.
        generation_model: Groq model name.
        generation_temperature: Sampling temperature.
        generation_max_tokens: Max output tokens.
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        reranker: Reranker,
        groq: GroqClient,
        rrf_k: int = 60,
        bm25_top_k: int = 50,
        dense_top_k: int = 50,
        rerank_top_k: int = 5,
        context_budget: int = 5000,
        generation_model: str = DEFAULT_GENERATE_MODEL,
        generation_temperature: float = 0.1,
        generation_max_tokens: int = 1024,
    ):
        self.hybrid = HybridRetriever(
            bm25=bm25,
            dense=dense,
            rrf_k=rrf_k,
            bm25_top_k=bm25_top_k,
            dense_top_k=dense_top_k,
            output_top_k=max(bm25_top_k, dense_top_k),
        )
        self.reranker              = reranker
        self.groq                  = groq
        self.rerank_top_k          = rerank_top_k
        self.context_budget        = context_budget
        self.generation_model      = generation_model
        self.generation_temperature= generation_temperature
        self.generation_max_tokens = generation_max_tokens

    # ── Main run method ────────────────────────────────────────────────────

    def run(self, question: str) -> RAGResult:
        """
        Process one question through the full pipeline.

        Args:
            question: The user's question string.

        Returns:
            RAGResult with answer, retrieved chunks, latency metrics, etc.
        """
        process = psutil.Process(os.getpid())
        t_total_start = time.perf_counter()

        # ── Step 1: Hybrid retrieval ───────────────────────────────────────
        t0 = time.perf_counter()
        hybrid_result = self.hybrid.retrieve(question)
        retrieval_ms  = (time.perf_counter() - t0) * 1000

        retrieved_ids = hybrid_result.chunk_ids  # All 50 candidates

        # ── Step 2: Reranking ─────────────────────────────────────────────
        t0 = time.perf_counter()
        reranked_result = self.reranker.rerank(
            query=question,
            retrieval_result=hybrid_result,
            top_k=self.rerank_top_k,
        )
        rerank_ms = (time.perf_counter() - t0) * 1000

        reranked_ids = reranked_result.chunk_ids   # Top-5

        # ── Step 3: Build prompt and generate ─────────────────────────────
        system_prompt, user_message = build_prompt(
            query=question,
            reranked_result=reranked_result,
            token_budget=self.context_budget,
        )

        t0 = time.perf_counter()
        answer = self.groq.call(
            system_prompt=system_prompt,
            user_message=user_message,
            model=self.generation_model,
            temperature=self.generation_temperature,
            max_tokens=self.generation_max_tokens,
        )
        generation_ms = (time.perf_counter() - t0) * 1000

        total_ms = (time.perf_counter() - t_total_start) * 1000
        ram_mb   = process.memory_info().rss / (1024 * 1024)

        # ── Step 4: Post-processing ────────────────────────────────────────
        citations  = extract_citations(answer)
        refused    = is_refusal(answer)

        return RAGResult(
            question=question,
            answer=answer,
            retrieved_chunk_ids=retrieved_ids,
            reranked_chunk_ids=reranked_ids,
            retrieved_texts=reranked_result.texts,
            retrieved_urls=reranked_result.urls,
            retrieval_scores=reranked_result.scores,
            citations=citations,
            is_refusal=refused,
            retrieval_latency_ms=retrieval_ms,
            rerank_latency_ms=rerank_ms,
            generation_latency_ms=generation_ms,
            total_latency_ms=total_ms,
            ram_mb_at_query=ram_mb,
        )

    # ── Factory: build from YAML config ───────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str) -> "NaiveRAGPipeline":
        """
        Load a fully initialized pipeline from a YAML config file.
        This is the standard entry-point for scripts.
        """
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        index_dir = cfg["paths"]["indexes"]
        emb_cfg   = cfg["embedding"]
        rer_cfg   = cfg["reranker"]
        gen_cfg   = cfg["generation"]
        hyb_cfg   = cfg["hybrid"]
        bm25_cfg  = cfg["bm25"]
        dense_cfg = cfg["dense"]

        print("[Pipeline] Loading BM25 retriever...")
        bm25 = BM25Retriever.load(index_dir)

        dense = None
        dense_index_file = Path(index_dir) / "faiss_index.bin"
        if dense_index_file.exists():
            try:
                print("[Pipeline] Loading Dense retriever...")
                from src.retrieval.dense_retriever import DenseRetriever
                dense = DenseRetriever.load(
                    index_dir,
                    model_name=emb_cfg["model_name"],
                    adapter_path=emb_cfg.get("adapter_path", None),
                    device=emb_cfg.get("device", "cpu"),
                )
            except Exception as e:
                log.warning(f"[Pipeline] Dense retriever load error: {e}. Using BM25 sparse retrieval.")
        else:
            print("[Pipeline] FAISS index not found -> Using BM25 sparse retrieval mode.")

        reranker = Reranker(
            model_name=rer_cfg["model_name"],
            adapter_path=rer_cfg.get("adapter_path", None),
            device=rer_cfg.get("device", "cpu"),
            batch_size=rer_cfg.get("batch_size", 16),
        )

        groq = GroqClient()

        return cls(
            bm25=bm25,
            dense=dense,
            reranker=reranker,
            groq=groq,
            rrf_k=hyb_cfg.get("rrf_k", 60),
            bm25_top_k=bm25_cfg.get("top_k", 50),
            dense_top_k=dense_cfg.get("top_k", 50),
            rerank_top_k=rer_cfg.get("top_k_output", 5),
            context_budget=gen_cfg.get("context_token_budget", 5000),
            generation_model=gen_cfg.get("model", DEFAULT_GENERATE_MODEL),
            generation_temperature=gen_cfg.get("temperature", 0.1),
            generation_max_tokens=gen_cfg.get("max_tokens", 1024),
        )
