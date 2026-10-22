"""
bm25_retriever.py
=================
Sparse lexical retrieval using BM25 (Best Match 25).

BM25 is the gold-standard sparse baseline for text retrieval.
It works purely on word overlap + term frequency statistics —
no neural models required.  It's fast, interpretable, and
surprisingly competitive for technical documentation where
exact terminology (e.g., 'kube-apiserver', 'nginx upstream')
is highly diagnostic.

Saved state:
  indexes/bm25_index.pkl   — serialized BM25Okapi object
  indexes/chunk_lookup.pkl — chunk_id -> Chunk dict
"""

import pickle
import re
import logging
from pathlib import Path
from typing import List, Tuple

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.corpus.chunker import Chunk

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer (simple word tokenizer — BM25 does not use subword tokens)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """
    Lowercase + split on non-alphanumeric characters.
    Keeps technical tokens like 'kube-proxy', 'pgpool-II' intact by
    splitting on spaces and punctuation but preserving hyphens in words.
    """
    text = text.lower()
    # Split on whitespace and common punctuation, keep hyphens within words
    tokens = re.findall(r"\b[\w-]+\b", text)
    return tokens


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    Wraps rank_bm25.BM25Okapi for indexed retrieval over a corpus of Chunks.

    Usage:
        retriever = BM25Retriever()
        retriever.build(chunks)
        retriever.save("indexes/")
        # --- later ---
        retriever = BM25Retriever.load("indexes/")
        results = retriever.retrieve("how to configure nginx upstream", top_k=50)
    """

    def __init__(self):
        self.bm25: BM25Okapi = None
        self.chunks: List[Chunk] = []
        self._chunk_id_to_idx: dict = {}

    # ── Building the index ─────────────────────────────────────────────────

    def build(self, chunks: List[Chunk]) -> None:
        """
        Build BM25 index from a list of Chunk objects.

        Each chunk's text is tokenized and added to the corpus.
        The chunk list order is preserved — BM25 ranks by integer index,
        which we map back to chunk IDs.
        """
        print(f"[BM25] Building index over {len(chunks):,} chunks...")
        self.chunks = chunks
        self._chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(chunks)}

        tokenized = [_tokenize(c.text) for c in tqdm(chunks, desc="[BM25] Tokenizing")]
        self.bm25 = BM25Okapi(tokenized)
        self._tokenized_corpus = tokenized
        print("[BM25] Index built.")

    def add_chunks(self, new_chunks: List[Chunk]) -> int:
        """
        Incrementally append new chunks to the existing BM25 index.

        Args:
            new_chunks: List of new Chunk objects to add.

        Returns:
            Number of chunks successfully added.
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built or loaded. Call build() or load() first.")
        if not new_chunks:
            return 0

        # Tokenize only new chunks
        new_tokens = [_tokenize(c.text) for c in new_chunks]

        # Reconstruct corpus list if not cached in memory
        if not hasattr(self, "_tokenized_corpus") or self._tokenized_corpus is None:
            self._tokenized_corpus = [_tokenize(c.text) for c in self.chunks]

        self._tokenized_corpus.extend(new_tokens)
        self.chunks.extend(new_chunks)
        self._chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        self.bm25 = BM25Okapi(self._tokenized_corpus)
        print(f"[BM25] Incrementally added {len(new_chunks):,} chunks. Total BM25 docs: {len(self.chunks):,}")
        return len(new_chunks)

    # ── Retrieval ──────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 50) -> List[Tuple[Chunk, float]]:
        """
        Retrieve top_k chunks most relevant to query.

        Returns:
            List of (Chunk, bm25_score) tuples, sorted descending by score.
        """
        if self.bm25 is None:
            raise RuntimeError("Index not built. Call build() or load() first.")

        query_tokens = _tokenize(query)
        stopwords = {
            "what", "is", "are", "the", "a", "an", "and", "or", "in", "of", "to", "for",
            "with", "on", "at", "by", "from", "up", "about", "into", "over", "after",
            "how", "do", "does", "did", "can", "could", "should", "would", "will",
            "tell", "me", "give", "explain", "describe", "show", "please", "compare",
            "contrast", "versus", "vs", "table", "pros", "cons", "their"
        }
        effective_tokens = [t for t in query_tokens if t not in stopwords]
        if not effective_tokens:
            effective_tokens = query_tokens

        scores = self.bm25.get_scores(effective_tokens)  # Array of float scores

        # Pair each chunk with its score, sort descending
        scored = sorted(
            zip(self.chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return scored[:top_k]

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, index_dir: str) -> None:
        """Serialize the BM25 index and chunk list to disk."""
        dir_path = Path(index_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        bm25_path  = dir_path / "bm25_index.pkl"
        chunks_path = dir_path / "chunks.pkl"

        with open(bm25_path, "wb") as f:
            pickle.dump(self.bm25, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[BM25] Saved index -> {bm25_path}")
        print(f"[BM25] Saved chunks -> {chunks_path}")

    @classmethod
    def load(cls, index_dir: str) -> "BM25Retriever":
        """Load a previously saved BM25 index."""
        dir_path = Path(index_dir)
        instance = cls()

        bm25_path   = dir_path / "bm25_index.pkl"
        chunks_path = dir_path / "chunks.pkl"

        if not bm25_path.exists():
            raise FileNotFoundError(f"BM25 index not found at {bm25_path}. Run build_index.py first.")

        print(f"[BM25] Loading index from {bm25_path}...")
        with open(bm25_path, "rb") as f:
            instance.bm25 = pickle.load(f)

        with open(chunks_path, "rb") as f:
            instance.chunks = pickle.load(f)

        instance._chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(instance.chunks)}
        print(f"[BM25] Loaded {len(instance.chunks):,} chunks.")
        return instance
