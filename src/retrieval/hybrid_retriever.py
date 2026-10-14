"""
hybrid_retriever.py
===================
Fuses BM25 and dense retrieval results using Reciprocal Rank Fusion (RRF).

Why RRF?
  BM25 is great at keyword matching ("kube-apiserver --authorization-mode").
  Dense retrieval is great at semantic similarity ("how to restrict API access").
  Neither is universally better.  RRF combines both without needing to tune
  weights — it just uses the rank positions, not raw scores.

RRF formula:
  For each document d from combined pool:
    rrf_score(d) = Σ_r  1 / (k + rank_r(d))

  where:
    k   = smoothing constant (default 60, empirically best across BEIR)
    r   = each retrieval system (BM25 and Dense)
    rank_r(d) = position of d in retrieval system r's ranked list (1-indexed)
    If d doesn't appear in a system's list: rank = ∞ -> contribution = 0

Why k=60?
  Originally proposed in Cormack et al. (2009).  Empirically validated on
  BEIR and MS MARCO as the best single constant value.
"""

from collections import defaultdict
from typing import List, Tuple, Dict

from src.corpus.chunker import Chunk
from src.retrieval.bm25_retriever import BM25Retriever
try:
    from src.retrieval.dense_retriever import DenseRetriever
except Exception:
    DenseRetriever = object


# ---------------------------------------------------------------------------
# RetrievalResult — clean wrapper for downstream components
# ---------------------------------------------------------------------------

class RetrievalResult:
    """Holds the retrieved chunks after fusion, with scores and ranks."""

    def __init__(self, chunks: List[Chunk], scores: List[float]):
        assert len(chunks) == len(scores)
        self.chunks = chunks
        self.scores = scores

    @property
    def chunk_ids(self) -> List[str]:
        return [c.chunk_id for c in self.chunks]

    @property
    def texts(self) -> List[str]:
        return [c.text for c in self.chunks]

    @property
    def urls(self) -> List[str]:
        return [c.url for c in self.chunks]

    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self):
        return iter(zip(self.chunks, self.scores))


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Combines BM25 and dense retrievers with RRF fusion.

    Args:
        bm25:   Initialized BM25Retriever.
        dense:  Initialized DenseRetriever.
        rrf_k:  RRF smoothing constant (default 60).
        bm25_top_k:   Candidates from BM25 before fusion.
        dense_top_k:  Candidates from dense before fusion.
        output_top_k: Final top-K after fusion (fed to reranker).
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        rrf_k: int = 60,
        bm25_top_k: int = 50,
        dense_top_k: int = 50,
        output_top_k: int = 50,
    ):
        self.bm25         = bm25
        self.dense        = dense
        self.rrf_k        = rrf_k
        self.bm25_top_k   = bm25_top_k
        self.dense_top_k  = dense_top_k
        self.output_top_k = output_top_k

    def retrieve(self, query: str, top_k: int = None) -> RetrievalResult:
        """
        Retrieve hybrid top-K chunks for a query.

        Flow:
          1. BM25 retrieves top-50  (keyword matching)
          2. Dense retrieves top-50 (semantic matching)
          3. RRF fuses both lists
          4. Return top-K by RRF score
        """
        limit = top_k if top_k is not None else self.output_top_k

        # Step 1: Retrieve from each system
        bm25_results = self.bm25.retrieve(query, top_k=max(self.bm25_top_k, limit))
        if self.dense is not None:
            dense_results = self.dense.retrieve(query, top_k=max(self.dense_top_k, limit))
            # Step 2: RRF fusion
            fused = self._rrf_fuse(bm25_results, dense_results)
        else:
            fused = bm25_results

        # Step 3: Return top-K
        top = fused[: limit]
        chunks = [item[0] for item in top]
        scores = [item[1] for item in top]
        return RetrievalResult(chunks, scores)

    def _rrf_fuse(
        self,
        bm25_results:  List[Tuple[Chunk, float]],
        dense_results: List[Tuple[Chunk, float]],
    ) -> List[Tuple[Chunk, float]]:
        """
        Apply Reciprocal Rank Fusion to two ranked lists.

        Returns sorted list of (Chunk, rrf_score) descending.
        """
        # Map chunk_id -> Chunk object (need it for output)
        chunk_registry: Dict[str, Chunk] = {}
        rrf_scores: Dict[str, float] = defaultdict(float)

        # Process BM25 rankings
        for rank, (chunk, _raw_score) in enumerate(bm25_results, start=1):
            chunk_registry[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] += 1.0 / (self.rrf_k + rank)

        # Process Dense rankings
        for rank, (chunk, _raw_score) in enumerate(dense_results, start=1):
            chunk_registry[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] += 1.0 / (self.rrf_k + rank)

        # Sort by fused score descending
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
        return [(chunk_registry[cid], rrf_scores[cid]) for cid in sorted_ids]
