"""
src/corpus/incremental_ingestor.py
====================================
Production Incremental Ingestion Engine for Enterprise RAG.

Key Capabilities:
  - Takes new documents (files, folders, or raw text) and chunks them.
  - Automatically embeds ONLY new chunks using the fine-tuned embedding model.
  - Incrementally appends vectors into FAISS (faiss_index.add) and BM25 index.
  - Persists updates to indexes/ and corpus/processed/chunks.jsonl without full rebuilding.
  - Can update the live running server memory in real-time.
"""

import os
import re
import json
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from src.corpus.chunker import Chunk, chunk_document, count_tokens

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_DIR = PROJECT_ROOT / "indexes"
CORPUS_JSONL = PROJECT_ROOT / "corpus" / "processed" / "chunks.jsonl"
CORPUS_STATS = PROJECT_ROOT / "corpus" / "processed" / "corpus_stats.json"


def extract_text_from_file(file_path: Path) -> str:
    """Extract plain text from supported document types (.txt, .md, .json, .csv, .pdf)."""
    suffix = file_path.suffix.lower()

    if suffix in (".txt", ".md", ".markdown", ".log", ".yaml", ".yml"):
        return file_path.read_text(encoding="utf-8", errors="replace")

    elif suffix == ".json":
        data = json.loads(file_path.read_text(encoding="utf-8", errors="replace"))
        if isinstance(data, str):
            return data
        elif isinstance(data, list):
            return "\n\n".join(str(item) for item in data)
        elif isinstance(data, dict):
            # If standard document dict with 'text' or 'content' key
            for key in ("text", "content", "body", "article"):
                if key in data and isinstance(data[key], str):
                    return data[key]
            return json.dumps(data, indent=2)

    elif suffix == ".csv":
        import csv
        lines = []
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            for row in reader:
                lines.append(" | ".join(row))
        return "\n".join(lines)

    elif suffix == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages)
        except ImportError:
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(str(file_path))
                pages = [page.get_text() for page in doc]
                return "\n\n".join(pages)
            except ImportError:
                raise RuntimeError("PDF parsing requires 'pypdf' or 'PyMuPDF'. Run: pip install pypdf")

    # Fallback to plain text read
    return file_path.read_text(encoding="utf-8", errors="replace")


class IncrementalIngestor:
    """
    Handles end-to-end incremental ingestion into the Enterprise RAG System.
    """

    def __init__(
        self,
        index_dir: Optional[Path] = None,
        corpus_jsonl: Optional[Path] = None,
        corpus_stats: Optional[Path] = None,
        dense_retriever = None,
        bm25_retriever = None,
    ):
        self.index_dir = index_dir or INDEX_DIR
        self.corpus_jsonl = corpus_jsonl or CORPUS_JSONL
        self.corpus_stats = corpus_stats or CORPUS_STATS
        self.dense = dense_retriever
        self.bm25 = bm25_retriever

    def _ensure_retrievers_loaded(self):
        """Lazy-load retrievers if not provided."""
        if self.dense is None:
            from src.retrieval.dense_retriever import DenseRetriever
            adapter_path = PROJECT_ROOT / "embedding_adapter"
            self.dense = DenseRetriever.load(
                str(self.index_dir),
                model_name=str(adapter_path) if adapter_path.exists() else "BAAI/bge-base-en-v1.5",
                adapter_path=str(adapter_path) if adapter_path.exists() else None,
            )

        if self.bm25 is None:
            from src.retrieval.bm25_retriever import BM25Retriever
            self.bm25 = BM25Retriever.load(str(self.index_dir))

    def ingest_text(
        self,
        raw_text: str,
        source: str = "custom",
        topic: str = "general",
        version: str = "1.0",
        url: str = "",
        document_id: Optional[str] = None,
        title: Optional[str] = None,
        min_tokens: int = 20,
        max_tokens: int = 400,
        overlap_tokens: int = 50,
    ) -> Dict[str, Any]:
        """
        Chunk and incrementally ingest raw text directly into the live indexes.
        """
        clean_text = raw_text.strip()
        if not clean_text:
            return {"status": "error", "message": "Input text is empty."}

        self._ensure_retrievers_loaded()

        # Generate unique document ID if not provided
        if not document_id:
            h = hashlib.md5(clean_text[:500].encode("utf-8")).hexdigest()[:12]
            document_id = f"inc_{source}_{h}"

        date_str = datetime.now().strftime("%Y-%m-%d")
        effective_url = url or f"internal://{source}/{document_id}"

        # 1. Chunk document
        raw_chunks = chunk_document(
            text=clean_text,
            source=source,
            topic=topic,
            version=version,
            url=effective_url,
            date_scraped=date_str,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            min_tokens=min_tokens,
        )

        if not raw_chunks:
            # If text is too short for min_tokens, create a single chunk
            tok_count = count_tokens(clean_text)
            raw_chunks = [
                Chunk(
                    chunk_id=f"{source}_{topic}_00000",
                    source=source,
                    topic=topic,
                    version=version,
                    url=effective_url,
                    date_scraped=date_str,
                    type="text",
                    token_count=tok_count,
                    text=clean_text,
                )
            ]

        # 2. Assign globally unique chunk IDs based on current corpus size
        current_total = len(self.dense.chunks)
        prepared_chunks: List[Chunk] = []

        for idx, c in enumerate(raw_chunks):
            unique_id = f"{c.source}_{c.topic}_{current_total + idx:06d}"
            c.chunk_id = unique_id
            c.document_id = document_id
            c.title = title or f"{source.upper()} {topic.title()} Document"
            prepared_chunks.append(c)

        # 3. Incrementally add to Dense FAISS index (Only encodes the new chunks!)
        t_dense_start = time.perf_counter()
        added_dense = self.dense.add_chunks(prepared_chunks)
        self.dense.save(str(self.index_dir))
        dense_time_ms = round((time.perf_counter() - t_dense_start) * 1000, 1)

        # 4. Incrementally add to BM25 index
        t_bm25_start = time.perf_counter()
        added_bm25 = self.bm25.add_chunks(prepared_chunks)
        self.bm25.save(str(self.index_dir))
        bm25_time_ms = round((time.perf_counter() - t_bm25_start) * 1000, 1)

        # 5. Append to chunks.jsonl on disk
        if self.corpus_jsonl.exists():
            with open(self.corpus_jsonl, "a", encoding="utf-8") as f:
                for c in prepared_chunks:
                    f.write(json.dumps(c.to_dict()) + "\n")

        # 6. Update corpus_stats.json
        stats = self._update_corpus_stats(prepared_chunks)

        # 7. Enqueue new chunks into ContinualBuffer for future parametric fine-tuning
        buffer_count = 0
        try:
            from src.training.continual_buffer import ContinualBuffer
            buf = ContinualBuffer()
            for c in prepared_chunks:
                chunk_meta = {"title": getattr(c, "title", title or ""), "source": c.source, "topic": c.topic, "chunk_id": c.chunk_id}
                buffer_count += buf.add_from_chunk(c.text, metadata=chunk_meta)
        except Exception as e:
            log.warning(f"Could not enqueue chunks into continual buffer: {e}")

        return {
            "status": "success",
            "document_id": document_id,
            "chunks_added": len(prepared_chunks),
            "buffer_pairs_added": buffer_count,
            "new_total_chunks": len(self.dense.chunks),
            "source": source,
            "topic": topic,
            "dense_time_ms": dense_time_ms,
            "bm25_time_ms": bm25_time_ms,
            "chunk_ids": [c.chunk_id for c in prepared_chunks],
            "sample_snippet": prepared_chunks[0].text[:180] + "..." if prepared_chunks else "",
            "corpus_stats": stats,
        }

    def ingest_file(
        self,
        file_path: Path,
        source: Optional[str] = None,
        topic: Optional[str] = None,
        url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ingest a single file from the filesystem."""
        path = Path(file_path)
        if not path.exists():
            return {"status": "error", "message": f"File not found: {path}"}

        text = extract_text_from_file(path)
        doc_source = source or path.parent.name or "custom"
        doc_topic = topic or path.stem.replace("_", " ").title()

        return self.ingest_text(
            raw_text=text,
            source=doc_source,
            topic=doc_topic,
            url=url or f"file:///{path.resolve()}",
            document_id=f"doc_{path.stem}",
            title=path.stem.replace("-", " ").replace("_", " ").title(),
        )

    def _update_corpus_stats(self, new_chunks: List[Chunk]) -> Dict[str, Any]:
        """Update corpus_stats.json with newly ingested chunks."""
        stats = {}
        if self.corpus_stats.exists():
            try:
                stats = json.loads(self.corpus_stats.read_text(encoding="utf-8"))
            except Exception:
                stats = {}

        stats["total_chunks"] = len(self.dense.chunks)
        stats["completed_at"] = datetime.now().isoformat()

        sources = stats.get("sources", {})
        for c in new_chunks:
            sources[c.source] = sources.get(c.source, 0) + 1
        stats["sources"] = sources

        self.corpus_stats.parent.mkdir(parents=True, exist_ok=True)
        self.corpus_stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        return stats
