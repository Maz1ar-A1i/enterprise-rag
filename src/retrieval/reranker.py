"""
reranker.py
===========
Cross-encoder reranking using BAAI/bge-reranker-base.

Supports:
  - Phase 1 Baseline: Pretrained BAAI/bge-reranker-base
  - Phase 2 Fine-Tuned: QLoRA PEFT Adapter (e.g. reranker_phase2/adapter)
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

from src.corpus.chunker import Chunk
from src.retrieval.hybrid_retriever import RetrievalResult

log = logging.getLogger(__name__)


class Reranker:
    """
    Reranks a candidate set from the hybrid retriever.

    The cross-encoder scores each (query, chunk_text) pair independently.
    Higher score = more relevant.

    Usage:
        # Baseline (Phase 1):
        reranker = Reranker(model_name="BAAI/bge-reranker-base")

        # Fine-Tuned (Phase 2):
        reranker = Reranker(
            model_name="BAAI/bge-reranker-base",
            adapter_path="reranker_phase2/adapter"
        )
        final_result = reranker.rerank(query, retrieval_result, top_k=5)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        adapter_path: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 16,
    ):
        self.model_name   = model_name
        self.adapter_path = adapter_path
        self.device       = device
        self.batch_size   = batch_size
        self._model       = None
        self._tokenizer   = None
        self._is_peft     = False

    def _load_model(self) -> None:
        if self._model is not None:
            return

        if self.adapter_path and Path(self.adapter_path).exists():
            print(f"[Reranker] Loading Phase 2 Fine-Tuned LoRA adapter: {self.adapter_path}")
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from peft import PeftModel
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=1,
            )
            self._model = PeftModel.from_pretrained(base_model, str(self.adapter_path))
            self._model.to(self.device)
            self._model.eval()
            self._is_peft = True
            print("[Reranker] Phase 2 LoRA adapter attached and ready.")
        else:
            print(f"[Reranker] Loading Phase 1 Baseline model: {self.model_name}")
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                    max_length=512,
                )
                self._is_peft = False
            except Exception as e:
                log.info(f"[Reranker] Loading via HuggingFace transformers ({e})")
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=1,
                )
                self._model.to(self.device)
                self._model.eval()
                self._is_peft = True

    def rerank(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        top_k: int = 5,
    ) -> RetrievalResult:
        """
        Score and re-sort candidate chunks, return top-K.

        Args:
            query:            Raw query string.
            retrieval_result: Output from HybridRetriever.retrieve().
            top_k:            How many chunks to keep after reranking.

        Returns:
            RetrievalResult containing the top_k most relevant chunks.
        """
        try:
            self._load_model()
        except Exception as e:
            log.warning(f"[Reranker] Model load error: {e}. Using retrieval order fallback.")
            return RetrievalResult(retrieval_result.chunks[:top_k], retrieval_result.scores[:top_k])

        if len(retrieval_result) == 0:
            return RetrievalResult([], [])

        candidates = retrieval_result.chunks
        pairs = [(query, chunk.text) for chunk in candidates]

        if self._is_peft:
            import torch
            all_scores = []
            with torch.no_grad():
                for i in range(0, len(pairs), self.batch_size):
                    batch = pairs[i:i + self.batch_size]
                    inputs = self._tokenizer(
                        [p[0] for p in batch],
                        [p[1] for p in batch],
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    ).to(self.device)
                    logits = self._model(**inputs).logits.view(-1)
                    batch_scores = torch.sigmoid(logits).cpu().tolist()
                    if isinstance(batch_scores, float):
                        batch_scores = [batch_scores]
                    all_scores.extend(batch_scores)
            scores = all_scores
        else:
            raw_scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            scores = raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)

        # Sort by score descending
        scored = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        top = scored[:top_k]
        chunks = [item[0] for item in top]
        rerank_scores = [item[1] for item in top]

        return RetrievalResult(chunks, rerank_scores)

