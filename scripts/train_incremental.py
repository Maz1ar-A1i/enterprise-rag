"""
scripts/train_incremental.py
============================
Command-line interface for CPU Parametric Continual Fine-Tuning.

Usage:
    # Train 1 epoch on accumulated buffer + experience replay:
    python scripts/train_incremental.py

    # Preview buffer size without training:
    python scripts/train_incremental.py --dry-run

    # Train 2 epochs with custom replay ratio:
    python scripts/train_incremental.py --epochs 2 --replay-ratio 0.30
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.continual_buffer import ContinualBuffer
from src.training.incremental_trainer import IncrementalTrainer, get_status


def main():
    parser = argparse.ArgumentParser(
        description="CPU-Optimized Continual Fine-Tuning for BGE Embedding Adapter."
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs (default: 1)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for CPU training (default: 4)")
    parser.add_argument("--replay-ratio", type=float, default=0.25, help="Ratio of anchor replay samples (default: 0.25)")
    parser.add_argument("--dry-run", action="store_true", help="Display buffer status without launching training")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear buffer upon completion")
    args = parser.parse_args()

    buffer = ContinualBuffer()
    pending_count = buffer.count()

    print("=" * 65)
    print("  ENTERPRISE RAG: PARAMETRIC CONTINUAL FINE-TUNING (CPU)")
    print("=" * 65)
    print(f"  Buffer File       : {buffer.buffer_path}")
    print(f"  Pending Samples   : {pending_count}")
    print(f"  Replay Anchors    : ~{max(5, int(pending_count * args.replay_ratio))}")
    print(f"  Training Epochs   : {args.epochs}")
    print(f"  CPU Batch Size    : {args.batch_size}")
    print("=" * 65)

    if args.dry_run:
        print("\n[Dry Run] Buffer inspection complete. Exiting without training.")
        pairs = buffer.get_pending_pairs()
        for idx, p in enumerate(pairs[:5], 1):
            print(f"  {idx}. Q: {p['query'][:70]}...")
            print(f"     A: {p['positive'][:70]}...")
        if len(pairs) > 5:
            print(f"     ... and {len(pairs) - 5} more.")
        return

    if pending_count == 0:
        print("\n[Notice] No training pairs found in buffer.")
        print("Ingest new documents first via 'python scripts/ingest_document.py' or Web UI.")
        return

    print("\nStarting CPU continual fine-tuning (Process isolated)...")
    trainer = IncrementalTrainer(batch_size=args.batch_size)
    result = trainer.train_epoch(
        epochs=args.epochs,
        replay_ratio=args.replay_ratio,
        clear_buffer_on_finish=not args.no_clear,
    )

    print("\n" + "=" * 65)
    if result.get("status") == "success":
        print("  [SUCCESS] CONTINUAL FINE-TUNING COMPLETED SUCCESSFULLY!")
        print(f"  Samples Trained : {result.get('samples_trained')}")
        print(f"  Time Elapsed    : {result.get('elapsed_seconds')}s")
        print(f"  Updated Adapter : {result.get('adapter_path')}")
        print("\n  Tip: If your Web UI server is running, hot-reload the weights live via:")
        print("  curl -X POST http://127.0.0.1:8000/api/continual/hot-reload")
    else:
        print(f"  [FAILED] TRAINING FAILED: {result.get('message')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
