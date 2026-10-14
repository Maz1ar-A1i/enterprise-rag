# Domain-Trained Enterprise RAG — Naive Baseline

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up Groq API key
```bash
copy .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Build corpus (downloads ~10,000+ chunks)
```bash
python scripts/build_corpus.py --config configs/naive_rag_config.yaml
```

### 4. Generate benchmark questions (DO THIS BEFORE BUILDING INDEXES)
```bash
python scripts/generate_benchmark.py --config configs/naive_rag_config.yaml
# Then: manually review datasets/benchmark/questions_raw.jsonl
# Then: give datasets/benchmark/questions_hidden.jsonl to mentor — NEVER open it again
```

### 5. Build indexes
```bash
python scripts/build_index.py --config configs/naive_rag_config.yaml
```

### 6. Run evaluation (produces frozen baseline)
```bash
python scripts/run_evaluation.py --config configs/naive_rag_config.yaml
```

---

## Project Structure

```
project/
├── src/
│   ├── corpus/          # Scraper, cleaner, chunker, deduplicator
│   ├── retrieval/       # BM25, Dense (FAISS), Hybrid (RRF), Reranker
│   ├── generation/      # Groq client, prompt builder
│   ├── pipeline/        # End-to-end NaiveRAGPipeline
│   └── evaluation/      # Metrics, benchmark runner, report generator
├── scripts/             # One-command runnable entry points
├── configs/             # All hyperparameters (never hardcode values)
├── corpus/              # Raw + processed corpus (gitignored)
├── indexes/             # FAISS + BM25 indexes (gitignored)
├── datasets/benchmark/  # Questions (visible + hidden)
├── results/             # Evaluation outputs + charts
└── ai_use_log/          # Required AI usage documentation
```

## Technology Stack

| Component | Model/Tool | Why |
|-----------|-----------|-----|
| Embeddings | `BAAI/bge-base-en-v1.5` | #1 MTEB retrieval score, free, 768-dim |
| Reranker | `BAAI/bge-reranker-base` | Same BAAI family, best nDCG/speed ratio |
| Generator | `llama-3.3-70b-versatile` via Groq | 128k context, free tier, 300 tok/sec |
| Vector store | FAISS (IndexFlatIP) | Local, exact search, no server |
| Sparse search | BM25 (rank_bm25) | Keyword matching, complementary to dense |
| Fusion | Reciprocal Rank Fusion (k=60) | Proven, no hyperparameter tuning |

## Evaluation Command
```bash
# Full eval with LLM judge
python scripts/run_evaluation.py

# Fast retrieval-only (no Groq calls)
python scripts/run_evaluation.py --no-judge

# Dry run with 10 questions
python scripts/run_evaluation.py --limit 10
```
