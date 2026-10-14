"""
scripts/start_ui.py
===================
Launches the Enterprise RAG Studio Light-Theme Web UI on http://localhost:8000.

Usage:
    python scripts/start_ui.py
    python scripts/start_ui.py --port 8080
"""

import sys
import argparse
import webbrowser
import uvicorn
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser(description="Start Enterprise RAG Web Studio")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    args = parser.parse_args()

    # Load stats if available
    stats_file = PROJECT_ROOT / "corpus" / "processed" / "corpus_stats.json"
    chunk_count_str = "10,548 Chunks"
    if stats_file.exists():
        try:
            import json
            with open(stats_file, "r", encoding="utf-8") as f:
                c_stats = json.load(f)
                total = c_stats.get("total_chunks", 10548)
                chunk_count_str = f"{total:,} Chunks"
        except Exception:
            pass

    url = f"http://{args.host}:{args.port}"
    print("\n" + "="*65)
    print("  STARTING ENTERPRISE RAG STUDIO (WHITE THEME)")
    print("="*65)
    print(f"  Web UI URL  : {url}")
    print(f"  Corpus View : {chunk_count_str}")
    print("  Benchmark   : Active Evaluation Dashboard")
    print("  Press Ctrl+C to stop the server.\n")

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    uvicorn.run("src.server.app:app", host=args.host, port=args.port, reload=False, log_level="info")

if __name__ == "__main__":
    main()
