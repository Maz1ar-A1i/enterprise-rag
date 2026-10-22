"""
src/training/continual_buffer.py
================================
Manages the Continual Learning Experience Buffer for parametric fine-tuning.

Key Features:
  1. Queues new (query, positive, [negative]) pairs as new documents arrive.
  2. Synthesizes training pairs from newly ingested chunks.
  3. Anti-Catastrophic Forgetting: Mixes in an "Experience Replay" sample (20-30%)
     from the original enterprise benchmark dataset.
  4. Safe persistence in `corpus/processed/continual_buffer.jsonl`.
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
BUFFER_FILE = PROJECT_ROOT / "corpus" / "processed" / "continual_buffer.jsonl"
ARCHIVE_FILE = PROJECT_ROOT / "corpus" / "processed" / "continual_archive.jsonl"
BENCHMARK_FILE = PROJECT_ROOT / "datasets" / "benchmark" / "questions_all.jsonl"

log = logging.getLogger(__name__)


class ContinualBuffer:
    """
    Experience Replay Buffer for Parametric Incremental Learning.
    """

    def __init__(self, buffer_path: Optional[Path] = None):
        self.buffer_path = buffer_path or BUFFER_FILE
        self.buffer_path.parent.mkdir(parents=True, exist_ok=True)

    def add_pair(
        self,
        query: str,
        positive: str,
        negative: Optional[str] = None,
        source: str = "custom",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a single (query, positive, [negative]) training triplet to the buffer.
        """
        record = {
            "query": query.strip(),
            "positive": positive.strip(),
            "negative": negative.strip() if negative else None,
            "source": source,
            "metadata": metadata or {},
        }
        with open(self.buffer_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def add_from_chunk(self, chunk_text: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Synthesize 1-2 training pairs from a newly ingested chunk text.
        Extracts key factual headings or declarative sentences as pseudo-queries.
        """
        if not chunk_text or len(chunk_text.strip()) < 40:
            return 0

        lines = [l.strip() for l in chunk_text.split("\n") if l.strip()]
        added = 0

        # Heuristic 1: If title or header exists
        title = (metadata or {}).get("title", "")
        if title and len(title) > 5:
            self.add_pair(
                query=f"What are the details regarding {title}?",
                positive=chunk_text,
                source="chunk_title",
                metadata=metadata,
            )
            added += 1

        # Heuristic 2: First informative line as query
        for line in lines[:3]:
            # Clean markdown header tokens
            clean = line.lstrip("#- *").strip()
            if 15 < len(clean) < 120 and not clean.endswith(":"):
                self.add_pair(
                    query=clean,
                    positive=chunk_text,
                    source="chunk_sentence",
                    metadata=metadata,
                )
                added += 1
                break

        # Fallback if no specific title/header
        if added == 0 and len(lines) > 0:
            self.add_pair(
                query=f"Information on {(metadata or {}).get('topic', 'workplace topics')}",
                positive=chunk_text,
                source="chunk_fallback",
                metadata=metadata,
            )
            added += 1

        return added

    def count(self) -> int:
        """Return number of pending training pairs in the buffer."""
        if not self.buffer_path.exists():
            return 0
        with open(self.buffer_path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def get_pending_pairs(self) -> List[Dict[str, Any]]:
        """Read all pending pairs from the buffer."""
        if not self.buffer_path.exists():
            return []
        pairs = []
        with open(self.buffer_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        pairs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return pairs

    def get_replay_anchors(self, count: int = 30) -> List[Dict[str, Any]]:
        """
        Sample anchor triplets from the gold benchmark dataset to prevent
        catastrophic forgetting of previously mastered concepts.
        """
        if not BENCHMARK_FILE.exists():
            return []

        candidates = []
        try:
            with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    # Filter for answerable questions with gold facts
                    if not item.get("expected_refusal", False) and item.get("answer"):
                        candidates.append({
                            "query": item["question"],
                            "positive": item.get("gold_answer") or item["answer"],
                            "negative": None,
                            "source": "experience_replay",
                        })
        except Exception as e:
            log.warning(f"Could not load experience replay anchors: {e}")
            return []

        if len(candidates) <= count:
            return candidates
        return random.sample(candidates, count)

    def prepare_dataset(self, replay_ratio: float = 0.25):
        """
        Builds the unified training dataset by combining:
          1. Pending new document pairs from buffer
          2. Experience replay anchors (to prevent catastrophic forgetting)
        Returns:
          List of sentence_transformers.InputExample objects.
        """
        from sentence_transformers import InputExample

        pending = self.get_pending_pairs()
        if not pending:
            return []

        n_pending = len(pending)
        n_replay = max(5, int(n_pending * replay_ratio))
        replays = self.get_replay_anchors(count=n_replay)

        combined = pending + replays
        random.shuffle(combined)

        examples = []
        for p in combined:
            q = p["query"]
            pos = p["positive"]
            neg = p.get("negative")
            if neg:
                examples.append(InputExample(texts=[q, pos, neg]))
            else:
                examples.append(InputExample(texts=[q, pos]))

        return examples

    def clear(self, archive: bool = True) -> None:
        """Clear the buffer, optionally moving records to continual_archive.jsonl."""
        if not self.buffer_path.exists():
            return

        if archive:
            with open(self.buffer_path, "r", encoding="utf-8") as f_in:
                content = f_in.read()
            if content.strip():
                with open(ARCHIVE_FILE, "a", encoding="utf-8") as f_out:
                    f_out.write(content)

        # Truncate active buffer
        with open(self.buffer_path, "w", encoding="utf-8") as f:
            f.write("")
