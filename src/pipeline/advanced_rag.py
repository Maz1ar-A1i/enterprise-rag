"""
src/pipeline/advanced_rag.py
============================
End-to-end Advanced RAG pipeline.

Integrates:
  Query -> Query Transformation (Canonicalization & Decomposition)
        -> Parallel Multi-Query Hybrid Retrieval
        -> Fair-Share Round-Robin Balanced Topic Allocation
        -> Cross-Encoder Reranking
        -> Multi-Topic Context Formulation & Grounding Prompt
        -> Groq Generation & Precision Citations

Usage:
    pipeline = AdvancedRAGPipeline.from_config("configs/naive_rag_config.yaml")
    result   = pipeline.run("what is kubernet and docker and fastapi and fiass")
    print(result.answer)
"""

import os
import time
import psutil
import logging
from dataclasses import dataclass
from typing import Optional, List

import yaml

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult
from src.retrieval.reranker import Reranker
from src.generation.groq_client import GroqClient, DEFAULT_GENERATE_MODEL
from src.advanced_rag.query_transformer import QueryTransformer
from src.advanced_rag.advanced_retriever import AdvancedRetriever
from src.advanced_rag.advanced_prompt_builder import (
    build_advanced_prompt,
    extract_advanced_citations,
    is_advanced_refusal,
)
from src.pipeline.naive_rag import RAGResult

from src.advanced_rag.defense_pipeline import SecurityDefensePipeline, InstructionDataSeparator
from src.advanced_rag.document_conflict_resolver import DocumentConflictResolver
from src.advanced_rag.claim_verifier import ClaimVerifier

log = logging.getLogger(__name__)


class AdvancedRAGPipeline:
    """
    Production Advanced Multi-Stage RAG Pipeline.
    Implements 8-stage enterprise runtime architecture:
      1. Query Ingestion
      2. Query Transformation (Canonicalization & Decomposition)
      3. Parallel Multi-Query Hybrid Retrieval (BM25 + Dense)
      4. Cross-Encoder Reranking
      5. Security Defense & Prompt Injection Quarantine
      6. Deterministic Document Conflict & Version Resolution
      7. Context Assembly & Grounded LLM Generation
      8. Independent Claim & Citation Verification -> Answer / Refusal
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        reranker: Reranker,
        groq: GroqClient,
        rrf_k: int = 60,
        candidates_per_subquery: int = 15,
        rerank_top_k: int = 6,
        context_budget: int = 5000,
        generation_model: str = DEFAULT_GENERATE_MODEL,
        generation_temperature: float = 0.1,
        generation_max_tokens: int = 1024,
    ):
        self.groq = groq
        self.transformer = QueryTransformer(groq_client=self.groq)
        self.retriever = AdvancedRetriever(
            bm25=bm25,
            dense=dense,
            reranker=reranker,
            query_transformer=self.transformer,
            rrf_k=rrf_k,
            candidates_per_subquery=candidates_per_subquery,
            final_top_k=rerank_top_k,
        )
        self.defense_pipeline = SecurityDefensePipeline()
        self.conflict_resolver = DocumentConflictResolver()
        self.claim_verifier = ClaimVerifier(entailment_threshold=0.60)
        self.rerank_top_k = rerank_top_k
        self.context_budget = context_budget
        self.generation_model = generation_model
        self.generation_temperature = generation_temperature
        self.generation_max_tokens = generation_max_tokens

    def run(self, question: str) -> RAGResult:
        """
        Process one question through the 8-stage defended runtime pipeline.
        """
        process = psutil.Process(os.getpid())
        t_total_start = time.perf_counter()

        # Stages 1-4: Advanced Hybrid Retrieval & Reranking
        t0 = time.perf_counter()
        retrieval_res, sub_queries = self.retriever.retrieve(
            question,
            top_k=self.rerank_top_k,
            use_reranker=True,
        )
        retrieval_ms = (time.perf_counter() - t0) * 1000

        # Stage 5: Security Defense Pipeline (Quarantine & Envelope Isolation)
        raw_chunks = retrieval_res.chunks or []
        safe_chunks, defense_assessments, defense_summary = self.defense_pipeline.process_chunks(raw_chunks)

        if self.defense_pipeline.should_fail_closed(safe_chunks, defense_summary):
            total_ms = (time.perf_counter() - t_total_start) * 1000
            ram_mb = process.memory_info().rss / (1024 * 1024)
            refusal_reason = "I cannot find sufficient information in the provided documentation to answer this question."
            if defense_summary.get("quarantined", 0) > 0:
                refusal_reason = "The request or retrieved evidence was quarantined due to potential adversarial injection content."
            return RAGResult(
                question=question,
                answer=refusal_reason,
                retrieved_chunk_ids=[c.chunk_id for c in safe_chunks],
                reranked_chunk_ids=[c.chunk_id for c in safe_chunks],
                retrieved_texts=[c.text for c in safe_chunks],
                retrieved_urls=[getattr(c, "url", "") for c in safe_chunks],
                retrieval_scores=[getattr(c, "score", 0.0) for c in safe_chunks],
                citations=[],
                is_refusal=True,
                retrieval_latency_ms=retrieval_ms * 0.4,
                rerank_latency_ms=retrieval_ms * 0.6,
                generation_latency_ms=0.0,
                total_latency_ms=total_ms,
                ram_mb_at_query=ram_mb,
            )

        # Stage 6: Deterministic Document Conflict & Version Resolution
        arbitrated_chunks, conflict_report = self.conflict_resolver.resolve_conflicts(question, safe_chunks)
        retrieval_res.chunks = arbitrated_chunks

        # Stage 7: Context Assembly & Grounded Generation
        system_prompt, user_message = build_advanced_prompt(
            query=question,
            retrieval_result=retrieval_res,
            sub_queries=sub_queries,
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
        ram_mb = process.memory_info().rss / (1024 * 1024)

        # Stage 8: Independent Claim-Level Citation & Entailment Verification
        verification_report = self.claim_verifier.verify_answer(answer, arbitrated_chunks)
        citations = extract_advanced_citations(answer, chunks=arbitrated_chunks)
        refused = is_advanced_refusal(answer) or verification_report.verified_refusal

        return RAGResult(
            question=question,
            answer=answer,
            retrieved_chunk_ids=[c.chunk_id for c in arbitrated_chunks],
            reranked_chunk_ids=[c.chunk_id for c in arbitrated_chunks],
            retrieved_texts=[c.text for c in arbitrated_chunks],
            retrieved_urls=[getattr(c, "url", "") for c in arbitrated_chunks],
            retrieval_scores=[getattr(c, "score", 0.0) for c in arbitrated_chunks],
            citations=citations,
            is_refusal=refused,
            retrieval_latency_ms=retrieval_ms * 0.4,
            rerank_latency_ms=retrieval_ms * 0.6,
            generation_latency_ms=generation_ms,
            total_latency_ms=total_ms,
            ram_mb_at_query=ram_mb,
        )

    @classmethod
    def from_config(cls, config_path: str) -> "AdvancedRAGPipeline":
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        index_dir = cfg["paths"]["indexes"]
        emb_cfg = cfg["embedding"]
        rer_cfg = cfg["reranker"]
        gen_cfg = cfg["generation"]
        hyb_cfg = cfg["hybrid"]

        print("[Advanced Pipeline] Loading BM25 retriever...")
        bm25 = BM25Retriever.load(index_dir)

        print("[Advanced Pipeline] Loading Dense retriever...")
        dense = DenseRetriever.load(
            index_dir,
            model_name=emb_cfg["model_name"],
            device=emb_cfg.get("device", "cpu"),
        )

        reranker = Reranker(
            model_name=rer_cfg["model_name"],
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
            candidates_per_subquery=15,
            rerank_top_k=rer_cfg.get("top_k_output", 6),
            context_budget=gen_cfg.get("context_token_budget", 5000),
            generation_model=gen_cfg.get("model", DEFAULT_GENERATE_MODEL),
            generation_temperature=gen_cfg.get("temperature", 0.1),
            generation_max_tokens=gen_cfg.get("max_tokens", 1024),
        )
