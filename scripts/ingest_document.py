"""
scripts/ingest_document.py
==========================
CLI tool for Incremental Document Ingestion in Enterprise RAG.

Usage Examples:
    # 1. Ingest a single file:
    python scripts/ingest_document.py --file docs/new_policy.md --source confluence --topic hr

    # 2. Ingest an entire folder of documents:
    python scripts/ingest_document.py --dir docs/updates/ --source github --topic backend

    # 3. Ingest raw text snippet directly:
    python scripts/ingest_document.py --text "The incident response escalation time is now 15 minutes." --source slack --topic oncall
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.corpus.incremental_ingestor import IncrementalIngestor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Incremental Document Ingestion for Enterprise RAG")
    parser.add_argument("--file", type=str, help="Path to a single file to ingest (.txt, .md, .json, .csv, .pdf)")
    parser.add_argument("--dir", type=str, help="Path to a directory of files to ingest")
    parser.add_argument("--text", type=str, help="Raw text string to ingest directly")
    parser.add_argument("--source", type=str, default="custom", help="Knowledge source tag (confluence, github, jira, slack, etc.)")
    parser.add_argument("--topic", type=str, default="general", help="Topic or category name")
    parser.add_argument("--title", type=str, default=None, help="Document title")
    parser.add_argument("--url", type=str, default="", help="Canonical document URL")

    args = parser.parse_args()

    if not args.file and not args.dir and not args.text:
        parser.error("You must specify either --file, --dir, or --text.")

    print("\n" + "="*65)
    print("  INCREMENTAL DOCUMENT INGESTION")
    print("="*65)

    ingestor = IncrementalIngestor()

    if args.text:
        print(f"  Ingesting raw text (source={args.source}, topic={args.topic})...")
        res = ingestor.ingest_text(
            raw_text=args.text,
            source=args.source,
            topic=args.topic,
            title=args.title,
            url=args.url,
        )
        print("\n  [OK] Ingestion Result:")
        print(f"    - Chunks added      : {res['chunks_added']}")
        print(f"    - New total chunks  : {res['new_total_chunks']:,}")
        print(f"    - Dense FAISS time  : {res['dense_time_ms']} ms")
        print(f"    - BM25 index time   : {res['bm25_time_ms']} ms")
        print(f"    - Chunk IDs         : {res['chunk_ids']}")

    elif args.file:
        file_path = Path(args.file)
        print(f"  Ingesting file: {file_path}...")
        res = ingestor.ingest_file(
            file_path=file_path,
            source=args.source,
            topic=args.topic,
            url=args.url,
        )
        if res.get("status") == "error":
            print(f"  [ERROR] {res.get('message')}")
            sys.exit(1)
        print("\n  [OK] Ingestion Result:")
        print(f"    - Chunks added      : {res['chunks_added']}")
        print(f"    - New total chunks  : {res['new_total_chunks']:,}")
        print(f"    - Dense FAISS time  : {res['dense_time_ms']} ms")
        print(f"    - BM25 index time   : {res['bm25_time_ms']} ms")
        print(f"    - Document ID       : {res['document_id']}")

    elif args.dir:
        dir_path = Path(args.dir)
        if not dir_path.exists():
            print(f"  [ERROR] Directory not found: {dir_path}")
            sys.exit(1)

        files = [p for p in dir_path.rglob("*") if p.is_file() and p.suffix.lower() in (".txt", ".md", ".json", ".csv", ".pdf")]
        print(f"  Found {len(files)} files in {dir_path}. Ingesting...\n")

        total_added = 0
        for f in files:
            print(f"    -> Ingesting {f.name}...")
            r = ingestor.ingest_file(f, source=args.source, topic=args.topic)
            if r.get("status") == "success":
                total_added += r.get("chunks_added", 0)

        print("\n" + "="*65)
        print(f"  [COMPLETED] Ingested {len(files)} files.")
        print(f"  Total new chunks added: {total_added:,}")
        print(f"  Current index total   : {len(ingestor.dense.chunks):,} chunks")
        print("="*65 + "\n")


if __name__ == "__main__":
    main()
