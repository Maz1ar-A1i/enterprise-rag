"""
scripts/build_phase3_index.py
==============================
Builds the Phase 3 FAISS dense index using the fine-tuned BGE embedding adapter.

Usage:
    python scripts/build_phase3_index.py
    python scripts/build_phase3_index.py --batch-size 32
    python scripts/build_phase3_index.py --dry-run
"""

import sys
import time
import argparse
import pickle
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_phase3_index")

ADAPTER_PATH     = PROJECT_ROOT / "phase3_bge_fixed_final" / "adapter"
CHUNKS_PKL       = PROJECT_ROOT / "indexes" / "chunks.pkl"
PHASE3_INDEX_DIR = PROJECT_ROOT / "indexes" / "phase3"
MODEL_NAME       = "BAAI/bge-base-en-v1.5"


def main():
    parser = argparse.ArgumentParser(description="Build Phase 3 FAISS index with fine-tuned embeddings")
    parser.add_argument("--batch-size", type=int, default=64, help="Encoding batch size (default: 64)")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup only, do not build index")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  Phase 3 FAISS Index Builder")
    print("  Model: BGE-base-en-v1.5 + LoRA Adapter (Phase 3)")
    print("=" * 70)

    if not ADAPTER_PATH.exists():
        log.error(f"Phase 3 adapter not found at: {ADAPTER_PATH}")
        sys.exit(1)
    log.info(f"V Phase 3 adapter found: {ADAPTER_PATH}")

    if not CHUNKS_PKL.exists():
        log.error(f"Chunks pickle not found at: {CHUNKS_PKL}. Run corpus ingestion first.")
        sys.exit(1)
    log.info(f"V Chunks pickle found: {CHUNKS_PKL}")

    if args.dry_run:
        print("\nV Dry-run complete. All files present. Run without --dry-run to build index.")
        return

    log.info("Loading chunks from pickle...")
    t0 = time.time()
    with open(CHUNKS_PKL, "rb") as f:
        chunks = pickle.load(f)
    log.info(f"Loaded {len(chunks):,} chunks in {time.time() - t0:.1f}s")

    from src.retrieval.dense_retriever import DenseRetriever

    log.info(f"Building Phase 3 FAISS index (batch_size={args.batch_size})...")
    retriever = DenseRetriever(
        model_name=MODEL_NAME,
        adapter_path=str(ADAPTER_PATH),
        device="cpu",
        batch_size=args.batch_size,
    )
    t_build = time.time()
    retriever.build(chunks)
    build_time = time.time() - t_build
    log.info(f"Index built in {build_time / 60:.1f} minutes")

    PHASE3_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    retriever.save(str(PHASE3_INDEX_DIR))

    print("\n" + "=" * 70)
    print(f"  Phase 3 FAISS index saved to: {PHASE3_INDEX_DIR}")
    print(f"  Chunks indexed: {len(chunks):,}")
    print(f"  Build time: {build_time / 60:.1f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()
