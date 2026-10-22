"""
dense_retriever.py
==================
Dense semantic retrieval using FAISS + BAAI/bge-base-en-v1.5 embeddings.

Why FAISS IndexFlatIP with L2-normalized vectors?
  - Inner Product (IP) of two L2-normalized vectors = cosine similarity.
  - IndexFlatIP does an exact (brute-force) search — no approximation.
  - For ≤100k chunks this is fast enough (< 100ms per query on CPU).
  - If corpus grows to millions, swap to IndexIVFFlat (approximate) later.

Why bge-base-en-v1.5?
  - Top MTEB retrieval score in the free/small class (63.4 avg)
  - 768-dim rich representations — better than MiniLM's 384
  - Designed for retrieval; responds well to the BGE query prefix
  - Same BAAI family as our reranker -> shared training distribution
"""

import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Any

from src.corpus.chunker import Chunk

log = logging.getLogger(__name__)

# BGE requires a specific prefix on queries (not on documents)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ---------------------------------------------------------------------------
# DenseRetriever
# ---------------------------------------------------------------------------

class DenseRetriever:
    """
    Encodes chunks with BGE-base, indexes them in FAISS, retrieves by cosine sim.

    Usage:
        retriever = DenseRetriever(model_name="BAAI/bge-base-en-v1.5")
        retriever.build(chunks)
        retriever.save("indexes/")
        # --- later ---
        retriever = DenseRetriever.load("indexes/")
        results = retriever.retrieve("how to configure nginx upstream", top_k=50)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        adapter_path: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 64,
    ):
        self.model_name   = model_name
        self.adapter_path = adapter_path
        self.device       = device
        self.batch_size   = batch_size
        self.model        = None
        self.tokenizer    = None
        self.index: Optional[Any] = None
        self.chunks: List[Chunk] = []
        self._is_peft     = False

    def _load_model(self) -> None:
        if self.model is not None:
            return

        import os
        import torch

        if self.adapter_path and Path(self.adapter_path).exists():
            adapter_dir = Path(self.adapter_path)
            if (adapter_dir / "adapter_config.json").exists():
                print(f"[Dense] Loading Phase 3 Fine-Tuned Embedding LoRA Adapter: {self.adapter_path}")
                from transformers import AutoTokenizer, AutoModel
                from peft import PeftModel

                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                base_model = AutoModel.from_pretrained(self.model_name)
                self.model = PeftModel.from_pretrained(base_model, str(self.adapter_path))
                self.model.to(self.device)
                self.model.eval()
                self._is_peft = True
                print("[Dense] Phase 3 Embedding LoRA Adapter attached.")
            else:
                print(f"[Dense] Loading Phase 3 Fine-Tuned SentenceTransformer: {self.adapter_path}")
                from sentence_transformers import SentenceTransformer
                num_threads = min(8, os.cpu_count() or 4)
                torch.set_num_threads(num_threads)
                self.model = SentenceTransformer(str(self.adapter_path), device=self.device)
                self.model.max_seq_length = 384
                self._is_peft = False
                print("[Dense] Phase 3 Fine-Tuned SentenceTransformer loaded successfully.")
        else:
            print(f"[Dense] Loading Baseline Model: {self.model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                num_threads = min(8, os.cpu_count() or 4)
                torch.set_num_threads(num_threads)
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.model.max_seq_length = 384
                self._is_peft = False
            except Exception as e:
                log.info(f"[Dense] Loading via HuggingFace transformers ({e})")
                from transformers import AutoTokenizer, AutoModel
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                self._is_peft = True

    # ── Building the index ─────────────────────────────────────────────────

    def build(self, chunks: List[Chunk]) -> None:
        """
        Encode all chunks and build a FAISS inner-product index.

        Steps:
          1. Encode chunk texts in batches -> float32 embeddings
          2. L2-normalize (so IP = cosine similarity)
          3. Add to FAISS IndexFlatIP
        """
        self._load_model()
        self.chunks = chunks

        print(f"[Dense] Encoding {len(chunks):,} chunks (batch_size={self.batch_size})...")
        texts = [c.text for c in chunks]

        # Encode without query prefix (prefix is for queries only)
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalize in place
        )
        # Shape: (N, 768)
        embeddings = embeddings.astype(np.float32)

        # Build FAISS index
        import faiss
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)   # Inner product (= cosine after L2-norm)
        self.index.add(embeddings)

        print(f"[Dense] FAISS index built. Vectors: {self.index.ntotal:,}, dim={dim}")

    def add_chunks(self, new_chunks: List[Chunk]) -> int:
        """
        Incrementally embed and append new chunks to the existing FAISS index.
        Does NOT re-embed old chunks — takes 1-2 seconds.

        Args:
            new_chunks: List of new Chunk objects to add.

        Returns:
            Number of chunks successfully added.
        """
        if self.index is None:
            raise RuntimeError("FAISS index not loaded. Load or build index first.")
        if not new_chunks:
            return 0

        self._load_model()
        texts = [c.text for c in new_chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        self.index.add(embeddings)
        self.chunks.extend(new_chunks)
        print(f"[Dense] Incrementally added {len(new_chunks):,} chunks. Total FAISS vectors: {self.index.ntotal:,}")
        return len(new_chunks)

    # ── Retrieval ──────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 50) -> List[Tuple[Chunk, float]]:
        """
        Retrieve top_k most similar chunks.

        Args:
            query:  Raw query string (BGE prefix added automatically).
            top_k:  Number of results to return.

        Returns:
            List of (Chunk, cosine_similarity_score) tuples, sorted descending.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() or load() first.")

        self._load_model()

        # Add BGE query prefix
        prefixed_query = BGE_QUERY_PREFIX + query

        if self._is_peft:
            import torch
            import torch.nn.functional as F
            inputs = self.tokenizer([prefixed_query], padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inputs)
                cls_rep = out.last_hidden_state[:, 0]
                norm_emb = F.normalize(cls_rep, p=2, dim=1)
                query_emb = norm_emb.cpu().numpy().astype(np.float32)
        else:
            query_emb = self.model.encode(
                [prefixed_query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)

        scores, indices = self.index.search(query_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:       # FAISS returns -1 for padded results
                continue
            results.append((self.chunks[idx], float(score)))

        return results   # Already sorted descending by FAISS

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, index_dir: str) -> None:
        """Write FAISS index and chunk list to disk."""
        dir_path = Path(index_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        import faiss
        faiss_path  = dir_path / "faiss_index.bin"
        chunks_path = dir_path / "chunks.pkl"

        faiss.write_index(self.index, str(faiss_path))

        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[Dense] Saved FAISS index -> {faiss_path}")
        print(f"[Dense] Saved chunks -> {chunks_path}")

    @classmethod
    def load(
        cls,
        index_dir: str,
        model_name: str = "BAAI/bge-base-en-v1.5",
        adapter_path: Optional[str] = None,
        device: str = "cpu",
    ) -> "DenseRetriever":
        """Load a saved DenseRetriever."""
        dir_path = Path(index_dir)
        instance = cls(model_name=model_name, adapter_path=adapter_path, device=device)

        faiss_path  = dir_path / "faiss_index.bin"
        chunks_path = dir_path / "chunks.pkl"

        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {faiss_path}.")

        import faiss
        print(f"[Dense] Loading FAISS index from {faiss_path}...")
        instance.index = faiss.read_index(str(faiss_path))

        with open(chunks_path, "rb") as f:
            instance.chunks = pickle.load(f)

        instance._load_model()
        print(f"[Dense] Loaded {len(instance.chunks):,} chunks.")
        return instance

    def reload_adapter(self, new_adapter_path: Optional[str] = None) -> bool:
        """
        Hot-swap the active embedding model weights in memory without rebuilding FAISS.
        Zero downtime: subsequent queries immediately use the newly fine-tuned weights.
        """
        import os
        import torch
        from sentence_transformers import SentenceTransformer

        path = new_adapter_path or self.adapter_path
        if not path or not Path(path).exists():
            log.warning(f"[Dense] Cannot reload adapter: path does not exist ({path})")
            return False

        try:
            print(f"[Dense] Hot-swapping embedding adapter to: {path}")
            num_threads = min(8, os.cpu_count() or 4)
            torch.set_num_threads(num_threads)
            new_model = SentenceTransformer(str(path), device=self.device)
            new_model.max_seq_length = 384

            # Atomic in-memory swap
            self.model = new_model
            self.adapter_path = str(path)
            self._is_peft = False
            print("[Dense] [OK] Model hot-swapped successfully. Live queries now use updated weights.")
            return True
        except Exception as e:
            log.error(f"[Dense] Hot-swap failed: {e}", exc_info=True)
            return False

