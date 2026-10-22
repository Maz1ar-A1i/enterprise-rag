"""
scripts/build_index.py
======================
Step 2: Build the BM25 and FAISS indexes from the processed corpus.

Must run AFTER build_corpus.py.

Usage:
    python scripts/build_index.py --config configs/naive_rag_config.yaml

Output:
    indexes/bm25_index.pkl    — BM25 index
    indexes/faiss_index.bin   — FAISS index
    indexes/chunks.pkl        — Chunk objects (shared by both)
"""

import sys
import json
import argparse
import logging
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corpus.chunker       import Chunk
from src.retrieval.bm25_retriever  import BM25Retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_chunks(chunks_path: str) -> list:
    """Load chunks from JSONL file."""
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            chunks.append(Chunk(**d))
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Build BM25 and FAISS indexes")
    parser.add_argument("--config", default="configs/naive_rag_config.yaml")
    parser.add_argument("--bm25-only",  action="store_true", help="Only build BM25 sparse index")
    parser.add_argument("--dense-only", action="store_true", help="Only build FAISS dense index")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Limit number of chunks to index (e.g. 3000 for quick test)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override dense embedding batch size (e.g. 64 or 128)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    chunks_path = Path(cfg["paths"]["corpus_processed"]) / "chunks.jsonl"
    index_dir   = Path(cfg["paths"]["indexes"])
    index_dir.mkdir(parents=True, exist_ok=True)

    if not chunks_path.exists():
        log.error(f"Chunks file not found: {chunks_path}")
        log.error("Run scripts/build_corpus.py first.")
        sys.exit(1)

    print("\n" + "="*65)
    print("  STEP 2: BUILDING BM25 & FAISS RETRIEVAL INDEXES")
    print("="*65)
    print(f"  Corpus File : {chunks_path}")
    print(f"  Output Dir  : {index_dir.resolve()}")
    
    chunks = load_chunks(str(chunks_path))
    if args.max_chunks and args.max_chunks < len(chunks):
        print(f"  Subsetting  : Using first {args.max_chunks:,} of {len(chunks):,} chunks")
        chunks = chunks[:args.max_chunks]
    else:
        print(f"  Total Chunks: {len(chunks):,}")

    # ── BM25 Index ─────────────────────────────────────────────────────────
    if not args.dense_only:
        print("\n" + "-"*65)
        print("  [1/2] BUILDING BM25 SPARSE INDEX (Term Frequency / Lexical)")
        print("-"*65)
        bm25 = BM25Retriever()
        bm25.build(chunks)
        bm25.save(str(index_dir))
        print("  [OK] BM25 Index successfully built and saved.")

    # ── Dense (FAISS) Index ────────────────────────────────────────────────
    if not args.bm25_only:
        print("\n" + "-"*65)
        print("  [2/2] BUILDING DENSE VECTOR INDEX (FAISS + BGE-base-en-v1.5)")
        print("-"*65)
        from src.retrieval.dense_retriever import DenseRetriever
        emb_cfg = cfg["embedding"]
        batch_size = args.batch_size or emb_cfg.get("batch_size", 64)
        print(f"  Model      : {emb_cfg['model_name']}")
        print(f"  Dimensions : {emb_cfg.get('dimensions', 768)}")
        print(f"  Batch Size : {batch_size}")
        print(f"  Device     : {emb_cfg.get('device', 'cpu')}")
        print("  Starting embedding computation...\n")

        dense = DenseRetriever(
            model_name=emb_cfg["model_name"],
            adapter_path=emb_cfg.get("adapter_path"),
            device=emb_cfg.get("device", "cpu"),
            batch_size=batch_size,
        )
        dense.build(chunks)
        dense.save(str(index_dir))
        print("\n  [OK] FAISS Dense Index successfully built and saved.")

    print("\n" + "="*65)
    print("  ALL INDEXES READY FOR RETRIEVAL & EVALUATION!")
    print("="*65)
    print(f"  Artifacts saved to: {index_dir.resolve()}")
    print("  Next step -> Run evaluation:")
    print("    python scripts/run_evaluation.py --config configs/naive_rag_config.yaml\n")


if __name__ == "__main__":
    main()
