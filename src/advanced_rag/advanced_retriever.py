"""
src/advanced_rag/advanced_retriever.py
=======================================
Multi-Query Balanced Hybrid Retriever + Cross-Encoder Reranking:
1. Executes parallel retrieval for each decomposed sub-query.
2. Performs balanced round-robin topic allocation so no single topic monopolizes context.
3. Applies Cross-Encoder Reranker across the fused candidate pool for precision scoring.
"""

import logging
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

from src.corpus.chunker import Chunk
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult
from src.retrieval.reranker import Reranker
from src.advanced_rag.query_transformer import QueryTransformer

log = logging.getLogger(__name__)


class AdvancedRetriever:
    """
    Advanced multi-stage retriever with query decomposition, balanced candidate
    allocation, and cross-encoder reranking.
    """

    def __init__(
        self,
        bm25: Optional[BM25Retriever],
        dense: Optional[DenseRetriever],
        reranker: Optional[Reranker] = None,
        query_transformer: Optional[QueryTransformer] = None,
        rrf_k: int = 60,
        candidates_per_subquery: int = 15,
        final_top_k: int = 5,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.transformer = query_transformer or QueryTransformer()
        self.rrf_k = rrf_k
        self.candidates_per_subquery = candidates_per_subquery
        self.final_top_k = final_top_k
        
        self.hybrid = None
        if self.bm25 and self.dense:
            self.hybrid = HybridRetriever(
                bm25=self.bm25,
                dense=self.dense,
                rrf_k=self.rrf_k,
                output_top_k=50,
            )

    def retrieve(
        self,
        raw_query: str,
        top_k: Optional[int] = None,
        use_reranker: bool = True,
    ) -> Tuple[RetrievalResult, List[str]]:
        """
        Execute advanced retrieval pipeline:
        1. Decompose raw query into sub-queries.
        2. Retrieve hybrid candidates for each sub-query.
        3. Fair-share merge across sub-queries.
        4. Cross-encoder rerank if enabled.

        Returns:
            Tuple of (RetrievalResult, sub_queries_list)
        """
        limit = top_k or self.final_top_k

        # Step 1: Decompose query
        sub_queries = self.transformer.decompose(raw_query)
        log.info(f"[AdvancedRetriever] Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")

        # Step 2: Retrieve per sub-query
        sub_results: List[List[Tuple[Chunk, float]]] = []
        chunk_registry: Dict[str, Chunk] = {}

        for sq in sub_queries:
            if self.hybrid:
                res = self.hybrid.retrieve(sq, top_k=self.candidates_per_subquery)
                pairs = list(zip(res.chunks, res.scores))
            elif self.dense:
                pairs = self.dense.retrieve(sq, top_k=self.candidates_per_subquery)
            elif self.bm25:
                pairs = self.bm25.retrieve(sq, top_k=self.candidates_per_subquery)
            else:
                pairs = []

            for c, _ in pairs:
                chunk_registry[c.chunk_id] = c
            sub_results.append(pairs)

        # Step 3: Balanced Round-Robin Merge across sub-queries
        # Guarantees that if 4 sub-queries were asked, each gets top slots!
        merged_chunks: List[Chunk] = []
        seen_ids = set()
        
        max_depth = max((len(r) for r in sub_results), default=0)
        for rank_idx in range(max_depth):
            for sq_idx, res_list in enumerate(sub_results):
                if rank_idx < len(res_list):
                    chunk, _ = res_list[rank_idx]
                    if chunk.chunk_id not in seen_ids:
                        seen_ids.add(chunk.chunk_id)
                        merged_chunks.append(chunk)

        if not merged_chunks:
            return RetrievalResult([], []), sub_queries

        # Step 4: Cross-Encoder Reranking (if enabled)
        if use_reranker and self.reranker:
            try:
                candidate_pool = RetrievalResult(
                    chunks=merged_chunks[: max(limit * 3, 20)],
                    scores=[1.0 / (i + 1) for i in range(min(len(merged_chunks), max(limit * 3, 20)))]
                )
                rerank_query = " ".join(sub_queries) if sub_queries else raw_query
                final_res = self.reranker.rerank(rerank_query, candidate_pool, top_k=limit)
                return final_res, sub_queries
            except Exception as e:
                log.warning(f"Reranking failed, using balanced merged pool: {e}")

        # If reranker is disabled or errored, return balanced merged pool
        selected = merged_chunks[:limit]
        scores = [1.0 / (i + 1) for i in range(len(selected))]
        return RetrievalResult(selected, scores), sub_queries
