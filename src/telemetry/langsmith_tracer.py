"""
src/telemetry/langsmith_tracer.py
===================================
Production-Grade Observability & Debugging with LangSmith:
- Automatic full-trace visualization of RAG stages in LangSmith dashboard.
- Safe fallbacks: If LangSmith API key is absent, writes local JSONL traces to logs/rag_traces.jsonl.
- Traces: Query Transformation, BM25, Dense, Hybrid Fusion, Reranking, Prompt Construction, LLM Generation.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from functools import wraps

log = logging.getLogger(__name__)

# Check if LangSmith is enabled via environment variables
LANGSMITH_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in ("true", "1") and bool(os.getenv("LANGCHAIN_API_KEY"))

try:
    from langsmith import traceable, Client
    _LANGSMITH_AVAILABLE = True
    if LANGSMITH_ENABLED:
        log.info(f"[LangSmith] Active project: {os.getenv('LANGCHAIN_PROJECT', 'enterprise-rag-studio')}")
except ImportError:
    _LANGSMITH_AVAILABLE = False
    traceable = lambda *args, **kwargs: (lambda f: f)  # No-op decorator fallback


LOCAL_TRACE_DIR = Path("logs")
LOCAL_TRACE_FILE = LOCAL_TRACE_DIR / "rag_traces.jsonl"


def log_local_trace(stage: str, inputs: Any, outputs: Any, latency_ms: float, metadata: Optional[Dict[str, Any]] = None):
    """Fallback local JSONL logger for offline inspection and debugging."""
    try:
        LOCAL_TRACE_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage": stage,
            "latency_ms": round(latency_ms, 2),
            "inputs": inputs,
            "outputs": outputs,
            "metadata": metadata or {},
        }
        with open(LOCAL_TRACE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning(f"Failed to write local trace: {e}")


def trace_stage(name: str, run_type: str = "chain"):
    """
    Decorator that instruments any RAG function with LangSmith tracing
    and automatic local JSONL logging.
    """
    def decorator(fn):
        # Apply LangSmith @traceable if available
        if _LANGSMITH_AVAILABLE:
            instrumented = traceable(name=name, run_type=run_type)(fn)
        else:
            instrumented = fn

        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                result = instrumented(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                log_local_trace(
                    stage=name,
                    inputs={"args": [str(a)[:300] for a in args], "kwargs": {k: str(v)[:300] for k, v in kwargs.items()}},
                    outputs={"result": str(result)[:500]},
                    latency_ms=elapsed_ms,
                    metadata={"run_type": run_type},
                )
                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                log_local_trace(
                    stage=name,
                    inputs={"args": [str(a)[:300] for a in args], "kwargs": {k: str(v)[:300] for k, v in kwargs.items()}},
                    outputs={"error": str(e)},
                    latency_ms=elapsed_ms,
                    metadata={"run_type": run_type, "status": "failed"},
                )
                raise

        return wrapper
    return decorator
