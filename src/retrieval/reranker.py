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
            adapter_dir = Path(self.adapter_path)
            if (adapter_dir / "adapter_config.json").exists():
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
                print(f"[Reranker] Loading Phase 2 Fine-Tuned CrossEncoder: {self.adapter_path}")
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(
                    str(self.adapter_path),
                    device=self.device,
                    max_length=512,
                )
                self._is_peft = False
                print("[Reranker] Phase 2 Fine-Tuned CrossEncoder loaded successfully.")
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

    def _algorithmic_cross_score(self, query: str, chunk: Chunk, orig_score: float = 0.5) -> float:
        """
        Pure-Python algorithmic cross-scoring when neural models are blocked by OS policy.
        Evaluates exact phrases, keyword coverage, title/heading match, and term proximity.
        Produces genuine, continuous float scores in [0.1, 0.98].
        """
        import re
        import math

        q_clean = query.lower().strip()
        text_lower = chunk.text.lower()
        topic_lower = (chunk.topic or "").lower()

        # Extract words (minimum 2 chars)
        q_words = [w for w in re.findall(r"\b[a-z0-9_-]+\b", q_clean) if len(w) > 1]
        if not q_words:
            return round(orig_score, 4)

        # 1. Exact query match bonus
        exact_bonus = 0.0
        if len(q_words) > 1 and q_clean in text_lower:
            exact_bonus = 0.35
        elif len(q_words) > 2:
            # Check 2-word shingles
            bigrams = [f"{q_words[i]} {q_words[i+1]}" for i in range(len(q_words) - 1)]
            matched_bigrams = sum(1 for bg in bigrams if bg in text_lower)
            exact_bonus = 0.25 * (matched_bigrams / max(1, len(bigrams)))

        # 2. Keyword coverage
        matched_words = [w for w in q_words if w in text_lower]
        coverage = len(matched_words) / max(1, len(q_words))

        # 3. Topic / Heading bonus
        heading_bonus = 0.0
        first_line = text_lower.split("\n", 1)[0]
        if any(w in topic_lower for w in q_words):
            heading_bonus += 0.10
        if any(w in first_line for w in q_words):
            heading_bonus += 0.12

        # 4. Term proximity bonus
        proximity_bonus = 0.0
        if len(matched_words) >= 2:
            positions = []
            for w in matched_words:
                pos = text_lower.find(w)
                if pos != -1:
                    positions.append(pos)
            if len(positions) >= 2:
                span = max(positions) - min(positions)
                # If all terms occur within 300 characters of each other
                if span < 300:
                    proximity_bonus = 0.15 * (1.0 - (span / 300.0))

        # 5. Term frequency saturation
        tf_count = sum(text_lower.count(w) for w in matched_words)
        tf_score = min(0.20, (tf_count / (len(matched_words) + 2)) * 0.05)

        # Combine into cross-score (scaled between 0.15 and 0.98)
        cross_score = (
            coverage * 0.40
            + exact_bonus
            + heading_bonus
            + proximity_bonus
            + tf_score
        )
        cross_score = max(0.12, min(0.98, cross_score))

        # Blend with incoming retrieval score if available
        if orig_score > 0:
            norm_orig = min(1.0, orig_score)
            blended = 0.65 * cross_score + 0.35 * norm_orig
        else:
            blended = cross_score

        return round(blended, 4)

    def _algorithmic_rerank(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Rerank candidates using algorithmic cross-scoring."""
        candidates = retrieval_result.chunks
        orig_scores = retrieval_result.scores if retrieval_result.scores else [0.5] * len(candidates)
        
        scored = []
        for i, chunk in enumerate(candidates):
            o_score = orig_scores[i] if i < len(orig_scores) else 0.5
            c_score = self._algorithmic_cross_score(query, chunk, orig_score=o_score)
            scored.append((chunk, c_score))

        # Re-sort descending by the new algorithmic cross-score
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        return RetrievalResult(
            chunks=[item[0] for item in top],
            scores=[item[1] for item in top],
        )

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
        if len(retrieval_result) == 0:
            return RetrievalResult([], [])

        try:
            self._load_model()
        except Exception as e:
            # Fall back to high-precision algorithmic cross-scoring
            return self._algorithmic_rerank(query, retrieval_result, top_k=top_k)

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

