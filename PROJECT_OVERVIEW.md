# 🚀 Domain-Trained Enterprise RAG System — Complete Project Overview

> **A production-grade, multi-phase Retrieval-Augmented Generation (RAG) system built for enterprise DevOps & Cloud Infrastructure Q&A.**

---

## 📌 Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [What Problem Does It Solve?](#2-what-problem-does-it-solve)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Project Phases](#4-project-phases)
5. [Data Flow — End to End](#5-data-flow--end-to-end)
6. [Component Deep Dive](#6-component-deep-dive)
7. [Technology Stack](#7-technology-stack)
8. [Evaluation Framework](#8-evaluation-framework)
9. [Directory Structure](#9-directory-structure)
10. [How to Run the Project](#10-how-to-run-the-project)
11. [Benchmark Question Categories](#11-benchmark-question-categories)
12. [Key Design Decisions](#12-key-design-decisions)

---

## 1. What Is This Project?

This is a **Domain-Trained Enterprise RAG (Retrieval-Augmented Generation)** system that answers complex technical questions about DevOps, Cloud Infrastructure, Containers, Databases, and CI/CD by retrieving and grounding its answers in **official documentation sources**.

Rather than relying on an LLM's training memory alone (which can be outdated or hallucinated), this system:

1. **Collects and indexes** thousands of chunks from authoritative documentation (Docker, Kubernetes, PostgreSQL, Nginx, Helm, Prometheus, GitHub Actions, Linux man pages, FastAPI, etc.)
2. **Retrieves** the most relevant chunks when a user asks a question
3. **Generates** a grounded, citation-backed answer using an LLM only from what was retrieved

The system is built in **4 escalating phases**, each improving accuracy, robustness, and safety — from a naive baseline to a fully hardened production pipeline with fine-tuned models, query transformation, and security defenses.

---

## 2. What Problem Does It Solve?

### The Problem
Enterprise DevOps teams constantly ask technical questions like:
- *"How do I configure Kubernetes Ingress with TLS termination?"*
- *"What's the difference between Docker volumes and bind mounts?"*
- *"How do I fix PostgreSQL MVCC transaction ID wraparound?"*

Existing LLMs:
- **Hallucinate** technical details, version numbers, and configuration syntax
- **Cannot cite** the exact documentation paragraph they learned from
- **Become stale** after their knowledge cutoff date
- **Mix up versions** (e.g., Kubernetes v1.21 vs v1.28 has different API objects)

### The Solution
This RAG system:
- **Grounds every answer** in fresh, scraped official documentation
- **Cites the exact source URL** for every claim
- **Refuses to answer** if no relevant documentation exists (instead of hallucinating)
- **Handles version conflicts** by deterministically resolving which doc version applies
- **Detects and blocks prompt injection attacks** from malicious user queries

---

## 3. High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        ENTERPRISE RAG SYSTEM                               │
│                                                                            │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐    │
│   │  CORPUS      │    │   INDEXES    │    │      QUERY PIPELINE      │    │
│   │  BUILDING    │───▶│   (Offline)  │    │      (Online/Runtime)    │    │
│   └──────────────┘    └──────────────┘    └──────────────────────────┘    │
│         │                   │                          │                  │
│   ┌─────▼──────┐    ┌───────▼──────┐    ┌─────────────▼─────────────┐    │
│   │  Scraper   │    │  FAISS Index │    │  1. Query Transformation   │    │
│   │  Cleaner   │    │  (Dense)     │    │  2. Hybrid Retrieval       │    │
│   │  Chunker   │    ├──────────────┤    │  3. Reranking              │    │
│   │ Deduplicat.│    │  BM25 Index  │    │  4. Security Defense       │    │
│   └────────────┘    │  (Sparse)    │    │  5. Conflict Resolution    │    │
│                     └──────────────┘    │  6. LLM Generation         │    │
│                                         │  7. Claim Verification     │    │
│                                         └───────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Project Phases

The project is structured into **4 cumulative phases**, each building on the last:

---

### Phase 1 — Naive RAG Baseline

**Goal:** Establish a frozen, reproducible baseline to measure all future improvements against.

| What it does | Details |
|---|---|
| Corpus Building | Scrapes ~10,000+ chunks from official DevOps documentation |
| Embedding | `BAAI/bge-base-en-v1.5` — 768-dimensional dense vectors |
| Retrieval | Hybrid: BM25 (sparse) + FAISS (dense) fused via Reciprocal Rank Fusion |
| Reranking | `BAAI/bge-reranker-base` — re-scores top-50 candidates, keeps top-5 |
| Generation | `llama-3.3-70b-versatile` via Groq API |
| Evaluation | Recall@K, MRR, nDCG@10, Correctness, Faithfulness, Citation Precision |

**Config:** `configs/naive_rag_config.yaml`
**Pipeline:** `src/pipeline/naive_rag.py` → `NaiveRAGPipeline`

---

### Phase 2 — Fine-Tuned Reranker (QLoRA)

**Goal:** Improve reranking quality by fine-tuning the reranker on domain-specific (query, positive chunk, negative chunk) triplets.

| What it does | Details |
|---|---|
| Training | QLoRA fine-tuning of `BAAI/bge-reranker-base` on DevOps triplets |
| Training Platform | Kaggle GPU notebooks (free T4/P100) |
| Adapter | Saved to `reranker_phase2/adapter/` |
| Integration | Loaded by `Reranker` class via `adapter_path` config |

The fine-tuned reranker scores `(query, chunk)` pairs with domain awareness — it learns that "K8s deployment YAML" is more relevant to a Kubernetes deployment question than a general Docker YAML snippet.

**Config:** `configs/phase2_rag_config.yaml`
**Training Script:** `scripts/train_reranker_qlora_kaggle.py`

---

### Phase 3 — Fine-Tuned Embedding Model (QLoRA) + Advanced RAG Pipeline

**Goal:** Improve retrieval quality by fine-tuning the embedding model AND introduce the full Advanced RAG pipeline with query transformation, security defenses, and claim verification.

| What it does | Details |
|---|---|
| Embedding Fine-Tuning | QLoRA on `BAAI/bge-base-en-v1.5` using contrastive loss on (query, positive, hard negatives) |
| Adapter | Saved to `phase3_bge_fixed_final/adapter/` |
| Query Transformation | Canonicalization → Decomposition → Multi-Query Retrieval |
| Security Defense | Prompt injection detection, source trust registry, data envelope isolation |
| Conflict Resolution | Version-aware deterministic document arbitration |
| Claim Verification | NLI-based per-claim entailment checking |

**Config:** `configs/phase3_rag_config.yaml`
**Pipeline:** `src/pipeline/advanced_rag.py` → `AdvancedRAGPipeline`

---

### Phase 4 — Fine-Tuned LLM Adapter (QLoRA)

**Goal:** Fine-tune the LLM itself on domain-specific (instruction, context, answer) triples to improve generation quality for DevOps Q&A.

| What it does | Details |
|---|---|
| Training | QLoRA fine-tuning of the generation LLM on curated DevOps instruction pairs |
| Training Script | `scripts/train_llm_qlora_kaggle.py` |
| Notebook | `scripts/phase4_llm_qlora_kaggle.ipynb` |
| Adapter | Stored in `phase4_llm_adapter/` |

---

## 5. Data Flow — End to End

### Offline: Corpus Building and Indexing

```
STEP 1: SCRAPING
────────────────
Official Docs (GitHub repos + HTML sitemaps)
  ├── Docker docs   (github.com/docker/docs)
  ├── Kubernetes    (github.com/kubernetes/website)
  ├── GitHub Actions (github.com/actions/runner)
  ├── PostgreSQL    (postgresql.org sitemap)
  ├── Nginx         (nginx.org sitemap)
  ├── Helm          (helm.sh docs)
  ├── Prometheus    (prometheus.io docs)
  ├── Linux man7    (man7.org)
  └── FastAPI       (fastapi.tiangolo.com)
         │
         ▼  scraper.py → RawPage objects → saved to corpus/raw/

STEP 2: CLEANING
─────────────────
  RawPage (HTML/Markdown)
  └── cleaner.py:
        ├── Strip HTML tags, nav bars, footers
        ├── Normalize unicode & whitespace
        └── Clean Markdown artifacts
         │
         ▼  Cleaned plain text

STEP 3: CHUNKING
─────────────────
  Cleaned text → chunker.py:
        ├── Split into 30–400 token chunks
        ├── 50-token overlap between chunks
        └── Attach metadata: source, topic, version, url, chunk_id
         │
         ▼  List of Chunks → corpus/processed/

STEP 4: DEDUPLICATION
──────────────────────
  Chunks → deduplicator.py:
        ├── MinHash + Jaccard similarity (threshold: 0.85)
        └── Remove near-duplicate chunks across sources
         │
         ▼  Unique Chunks

STEP 5: INDEXING
─────────────────
  Unique Chunks → build_index.py:
        ├── Dense: BAAI/bge-base-en-v1.5 embeds all chunks
        │         → FAISS IndexFlatIP (cosine similarity)
        │         → saved: indexes/faiss_index.bin
        │         → saved: indexes/chunk_metadata.pkl
        └── Sparse: BM25 (rank_bm25)
                  → saved: indexes/bm25_index.pkl
```

---

### Online: Query to Answer (Naive RAG)

```
USER QUESTION
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: HYBRID RETRIEVAL                                        │
│   BM25Retriever.retrieve(question)     → top-50 BM25 chunks    │
│   DenseRetriever.retrieve(question)    → top-50 FAISS chunks   │
│   HybridRetriever (RRF k=60)           → merged top-50 chunks  │
└─────────────────────────────────────────────────────────────────┘
     │  50 candidates
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: RERANKING                                               │
│   BAAI/bge-reranker-base scores each (query, chunk) pair       │
│   → Top-5 chunks selected by cross-encoder relevance score     │
└─────────────────────────────────────────────────────────────────┘
     │  5 best chunks
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: PROMPT BUILDING                                         │
│   prompt_builder.py:                                            │
│     ├── System prompt: "You are a DevOps expert. Answer only   │
│     │    from the context. Cite sources. Refuse if unsure."    │
│     └── User message: Question + numbered context passages     │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: GENERATION                                              │
│   Groq API → llama-3.3-70b-versatile                           │
│   temperature=0.1, max_tokens=3000                              │
│   → Answer with [Source: URL] citations                        │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: POST-PROCESSING                                         │
│   extract_citations(answer) → list of cited URLs               │
│   is_refusal(answer)        → did model refuse to answer?      │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
  RAGResult (answer, citations, latency metrics, refusal flag)
```

---

### Online: Query to Answer (Advanced RAG — Phase 3+, 8 Stages)

```
USER QUESTION
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: QUERY TRANSFORMATION                                   │
│   QueryTransformer.canonicalize(query)                          │
│     ├── Dictionary lookup: "k8s" → "kubernetes",               │
│     │   "postgre" → "postgresql", "tocker" → "docker"          │
│     └── Fuzzy spell-check against DevOps vocabulary            │
│                                                                 │
│   QueryTransformer.decompose(query)                             │
│     ├── Detects compound queries ("What is X and Y and Z?")    │
│     ├── Splits into per-technology sub-queries                  │
│     └── Falls back to LLM Domain Normalizer for complex cases  │
└─────────────────────────────────────────────────────────────────┘
     │  List of sub-queries
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: PARALLEL MULTI-QUERY HYBRID RETRIEVAL                 │
│   For each sub-query (in parallel):                             │
│     BM25 top-15 + Dense top-15 → RRF fusion → sub-result      │
│   Fair-Share Round-Robin allocation across sub-queries          │
│   → Balanced merged candidate pool                             │
└─────────────────────────────────────────────────────────────────┘
     │  ~50 diverse candidates covering all topics
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: CROSS-ENCODER RERANKING                               │
│   Fine-tuned BAAI/bge-reranker-base scores all candidates      │
│   → Top-6 highest scoring chunks selected                      │
└─────────────────────────────────────────────────────────────────┘
     │  6 best chunks
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4: SECURITY DEFENSE PIPELINE                             │
│   SourceTrustRegistry: assess doc provenance tier              │
│   SuspiciousDocumentQuarantine: scan for prompt injection      │
│     patterns (jailbreak, "ignore previous instructions", etc.) │
│   InstructionDataSeparator: wrap text in XML data envelopes    │
│   Fail-Closed: if all chunks quarantined → refuse to answer    │
└─────────────────────────────────────────────────────────────────┘
     │  Safe, sanitized chunks
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 5: DOCUMENT CONFLICT RESOLUTION                           │
│   DocumentConflictResolver:                                     │
│     ├── Extract version metadata from each chunk               │
│     ├── Check if user requested a specific version in query    │
│     ├── Arbitrate conflicts: user version > newer > older      │
│     └── Flag deprecated docs with superseded notices           │
└─────────────────────────────────────────────────────────────────┘
     │  Version-resolved chunks
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 6: LLM GENERATION                                         │
│   build_advanced_prompt() → grounding prompt with context      │
│   Groq API → LLM generates answer with citations               │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 7: CLAIM VERIFICATION                                     │
│   ClaimVerifier:                                                │
│     ├── Decompose answer into atomic claims                    │
│     ├── NLI entailment scoring per claim vs. source chunks     │
│     ├── Flag UNSUPPORTED or CONTRADICTED claims                │
│     └── Compute faithfulness score and hallucination rate      │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
  RAGResult (grounded answer + citations + verification report)
```

---

## 6. Component Deep Dive

### Corpus Layer (`src/corpus/`)

| File | Role |
|---|---|
| `scraper.py` | Downloads docs from GitHub repos (shallow clone, parse `.md`) and HTML sitemaps |
| `cleaner.py` | Strips HTML, normalizes whitespace, removes navigation boilerplate |
| `chunker.py` | Splits cleaned text into overlapping token-sized chunks with metadata |
| `deduplicator.py` | Uses MinHash + Jaccard to remove near-duplicate chunks (threshold: 0.85) |
| `url_resolver.py` | Resolves relative links to canonical absolute URLs for citations |

**Sources scraped:**
- `docs.docker.com` → Docker networking, storage, compose, engine, build, security
- `kubernetes.io` → Pods, Services, Deployments, Ingress, ConfigMaps, etc.
- `docs.github.com` → GitHub Actions workflows, runners, secrets
- `postgresql.org` → SQL, MVCC, replication, vacuuming, performance
- `nginx.org` → Reverse proxy, upstream, SSL/TLS, load balancing
- `docs.helm.sh` → Helm charts, releases, values, templating
- `prometheus.io` → Metrics, PromQL, Alertmanager, scraping
- `man7.org` → Linux systemd, journalctl, cgroups, namespaces
- `fastapi.tiangolo.com` → FastAPI endpoints, dependencies, middleware

---

### Retrieval Layer (`src/retrieval/`)

#### BM25 Retriever (`bm25_retriever.py`)
- **What:** Keyword-based sparse retrieval using TF-IDF weighting
- **Library:** `rank_bm25`
- **How:** Tokenizes query → scores all chunks by term overlap → returns top-K
- **Strength:** Exact keyword matching; great for specific command names, error codes

#### Dense Retriever (`dense_retriever.py`)
- **What:** Semantic similarity via dense vector search
- **Model:** `BAAI/bge-base-en-v1.5` (768-dim, #1 MTEB leaderboard)
- **Index:** FAISS `IndexFlatIP` (exact inner product / cosine similarity)
- **Query Prefix:** `"Represent this sentence for searching relevant passages: <query>"`
- **Strength:** Understands semantic intent; retrieves conceptually similar docs

#### Hybrid Retriever (`hybrid_retriever.py`)
- **What:** Fuses BM25 and Dense results using Reciprocal Rank Fusion
- **RRF Formula:** `score(d) = Σ 1 / (k + rank_in_list(d))` where `k=60`
- **Why RRF?** No extra hyperparameter tuning; proven to outperform individual retrievers
- **Output:** Top-50 merged, de-duplicated candidates

#### Reranker (`reranker.py`)
- **What:** Cross-encoder that scores `(query, chunk)` pairs jointly
- **Model:** `BAAI/bge-reranker-base` (fine-tuned in Phase 2 via QLoRA)
- **Input:** Top-50 hybrid candidates
- **Output:** Top-5 highest-scoring chunks
- **Why cross-encoder?** Reads query + chunk together → much more accurate than bi-encoder similarity

---

### Advanced RAG Layer (`src/advanced_rag/`)

#### Query Transformer (`query_transformer.py`)
- **Canonicalization:** Dictionary + fuzzy spell-check maps technical typos
  - `"k8s"` → `"kubernetes"`, `"postgre"` → `"postgresql"`, `"tocker"` → `"docker"`
- **Decomposition:** Splits compound queries into focused sub-queries
  - `"What is Docker and Kubernetes?"` → `["docker architecture features", "kubernetes architecture features", "compare docker vs kubernetes"]`
- **LLM Fallback:** Calls Groq with a specialized DevOps Domain Query Normalizer prompt for complex cases

#### Security Defense Pipeline (`defense_pipeline.py`)
Implements OWASP Top 10 for LLMs mitigations:

| Component | What it does |
|---|---|
| `SourceTrustRegistry` | Maps domains to trust tiers (authoritative / community / untrusted) |
| `SuspiciousDocumentQuarantine` | Scans chunk text for 13+ prompt injection regex patterns |
| `InstructionDataSeparator` | Wraps retrieved text in XML data envelopes; strips system token delimiters |
| Fail-Closed Logic | If ALL chunks are quarantined → refuses to answer rather than hallucinate |

**Example quarantine patterns detected:**
- `"ignore all previous instructions"`
- `"jailbreak"`, `"developer mode enabled"`
- `"<script>"`, `"[SYSTEM_PROMPT]"`, `"bypass safety guardrails"`

#### Document Conflict Resolver (`document_conflict_resolver.py`)
Handles version conflicts in retrieved docs:
1. Extract semantic versions from chunk text (`v1.28`, `Helm v3`, `Postgres 15`)
2. Extract user's requested version from the query
3. **Priority:** User-requested version > Newer version > Older version > Source authority
4. Flags deprecated docs with superseded notices
5. Multi-version comparison mode when query asks "difference between v1 and v2"

#### Claim Verifier (`claim_verifier.py`)
Post-generation verification:
1. Decompose LLM answer into atomic factual claims
2. Score each claim against the retrieved source chunks using NLI entailment
3. Categorize: `SUPPORTED` / `UNSUPPORTED` / `CONTRADICTED`
4. Produce faithfulness score = supported / total claims
5. Hallucination rate = claims not supported by any source

---

### Generation Layer (`src/generation/`)

| File | Role |
|---|---|
| `groq_client.py` | Wraps Groq API; handles rate limiting, retries, JSON mode calls |
| `prompt_builder.py` | Builds system + user prompts; enforces grounding constraints |
| `local_llm_client.py` | Alternative client for locally running LLMs |

**Grounding Constraint in System Prompt:**
> "Answer ONLY from the provided context. If the context does not contain enough information to answer the question, say 'I don't have enough information in my knowledge base to answer this.' Cite your sources using `[Source: URL]` notation."

---

### Evaluation Layer (`src/evaluation/`)

| File | Role |
|---|---|
| `metrics.py` | All metric computation functions (pure, no side effects) |
| `benchmark_runner.py` | Runs pipeline over all benchmark questions, collects results |
| `report_generator.py` | Generates HTML/markdown reports with charts and tables |

---

## 7. Technology Stack

| Component | Model / Tool | Why Chosen |
|---|---|---|
| **Embedding** | `BAAI/bge-base-en-v1.5` | #1 MTEB retrieval benchmark; 768-dim; free; fast on CPU |
| **Reranker** | `BAAI/bge-reranker-base` | Same BAAI family; best nDCG/speed ratio |
| **Generator** | `llama-3.3-70b-versatile` via Groq | 128k context; free tier; 300 tok/sec throughput |
| **Vector Store** | FAISS `IndexFlatIP` | Local; exact search; no server required |
| **Sparse Search** | BM25 (`rank_bm25`) | Keyword matching; complementary to dense |
| **Fusion** | Reciprocal Rank Fusion (k=60) | Proven; no extra hyperparameters |
| **Fine-Tuning** | QLoRA (quantized LoRA) | Fits GPU memory constraints; runs on free Kaggle T4 |
| **API** | Groq Cloud | Ultra-fast LLM inference; free tier |
| **Scraping** | `requests` + `BeautifulSoup` | Simple, robust HTML parsing |
| **Deduplication** | MinHash (`datasketch`) | Approximate nearest neighbor; handles 10k+ chunks |

---

## 8. Evaluation Framework

### Benchmark Question Categories

The system is evaluated on **8 question categories** designed to probe different failure modes:

| Category | What It Tests |
|---|---|
| `direct` | Simple single-document factual questions |
| `multi_hop` | Questions requiring synthesis across multiple documents |
| `troubleshooting` | Debugging scenarios (e.g., "my container keeps restarting, why?") |
| `cross_document` | Questions spanning multiple technology domains |
| `version_sensitive` | Questions where the correct answer depends on version (e.g., K8s v1.25 vs v1.28) |
| `ambiguous` | Underspecified queries that need clarification or scoped answers |
| `unanswerable` | Questions outside the corpus scope (system must refuse, not hallucinate) |
| `distractor_heavy` | Corpus contains misleading near-matches (tests reranker precision) |

---

### Metrics Computed

#### Group 1 — Retrieval Quality

| Metric | Formula | What it measures |
|---|---|---|
| **Recall@1** | Relevant in top-1 / total relevant | Did the best chunk appear first? |
| **Recall@5** | Relevant in top-5 / total relevant | Are relevant chunks in the reranked set? |
| **Recall@10** | Relevant in top-10 / total relevant | Coverage in the hybrid retrieval pass |
| **MRR** | 1 / rank of first relevant | How early does the first relevant appear? |
| **nDCG@10** | DCG / IDCG at cutoff 10 | Ranking quality (rewards early relevant results) |

#### Group 2 — Answer Quality (LLM-as-Judge)

| Metric | What it measures |
|---|---|
| **Correctness** | 0–1 score: does the answer match the reference answer? |
| **Faithfulness** | Fraction of answer claims supported by the retrieved context |
| **Citation Precision** | Fraction of cited URLs that were actually retrieved |
| **Refusal Precision** | Of all refusals, how many were truly unanswerable? |
| **Refusal Recall** | Of all unanswerable questions, how many did the system refuse? |
| **Hallucination Rate** | Fraction of answers with faithfulness below 0.9 threshold |

#### Group 3 — System Performance

| Metric | What it measures |
|---|---|
| **p50 Latency (ms)** | Median end-to-end response time |
| **p95 Latency (ms)** | 95th percentile latency |
| **RAM Usage (MB)** | Memory footprint at query time |

---

## 9. Directory Structure

```
Project 3/
│
├── src/                          # All source code
│   ├── corpus/                   # Data pipeline: scrape → clean → chunk → dedup
│   │   ├── scraper.py            # Multi-source documentation downloader
│   │   ├── cleaner.py            # HTML/Markdown text normalizer
│   │   ├── chunker.py            # Token-aware overlapping chunker
│   │   ├── deduplicator.py       # MinHash near-duplicate remover
│   │   └── url_resolver.py       # Canonical URL resolver for citations
│   │
│   ├── retrieval/                # Search components
│   │   ├── bm25_retriever.py     # BM25 keyword retriever
│   │   ├── dense_retriever.py    # FAISS dense retriever + embedding model
│   │   ├── hybrid_retriever.py   # RRF fusion of BM25 + Dense
│   │   └── reranker.py           # BGE cross-encoder reranker
│   │
│   ├── generation/               # LLM interface
│   │   ├── groq_client.py        # Groq API client with retry + rate limiting
│   │   ├── prompt_builder.py     # Grounding prompt templates
│   │   └── local_llm_client.py   # Local LLM alternative
│   │
│   ├── advanced_rag/             # Phase 3+ advanced components
│   │   ├── query_transformer.py  # Canonicalize + decompose queries
│   │   ├── advanced_retriever.py # Multi-query parallel retrieval
│   │   ├── advanced_prompt_builder.py  # Multi-topic context prompts
│   │   ├── defense_pipeline.py   # Security: injection scan + trust + envelopes
│   │   ├── document_conflict_resolver.py  # Version conflict arbitration
│   │   └── claim_verifier.py     # NLI-based claim entailment verification
│   │
│   ├── pipeline/                 # End-to-end pipeline orchestrators
│   │   ├── naive_rag.py          # Phase 1: simple hybrid RAG pipeline
│   │   ├── advanced_rag.py       # Phase 3: 8-stage advanced pipeline
│   │   └── phase_manager.py      # Selects correct pipeline by phase
│   │
│   ├── evaluation/               # Benchmarking & metrics
│   │   ├── metrics.py            # Recall@K, MRR, nDCG, Faithfulness, etc.
│   │   ├── benchmark_runner.py   # Batch evaluation loop
│   │   └── report_generator.py   # HTML/chart report generation
│   │
│   ├── server/                   # (Optional) API server
│   └── telemetry/                # Logging & observability
│
├── scripts/                      # Runnable entry points
│   ├── build_corpus.py           # Run full corpus pipeline
│   ├── build_index.py            # Build FAISS + BM25 indexes
│   ├── build_phase3_index.py     # Build Phase 3 indexes with fine-tuned embeddings
│   ├── generate_benchmark.py     # Generate Q&A benchmark from corpus
│   ├── run_evaluation.py         # Evaluate pipeline on benchmark
│   ├── run_ablation_experiments.py  # Compare pipeline variants
│   ├── evaluate_phases.py        # Side-by-side phase comparison
│   ├── start_ui.py               # Launch interactive demo UI
│   ├── train_reranker_qlora_kaggle.py   # Phase 2 reranker fine-tuning
│   ├── train_embedding_qlora_kaggle.py  # Phase 3 embedding fine-tuning
│   └── train_llm_qlora_kaggle.py        # Phase 4 LLM fine-tuning
│
├── configs/
│   ├── naive_rag_config.yaml     # Phase 1 configuration
│   ├── phase2_rag_config.yaml    # Phase 2 configuration (fine-tuned reranker)
│   └── phase3_rag_config.yaml    # Phase 3 configuration (fine-tuned embeddings)
│
├── corpus/                       # Scraped raw + processed corpus (gitignored)
│   ├── raw/                      # Raw downloaded pages per source
│   └── processed/                # Cleaned, chunked JSONL files
│
├── indexes/                      # Built search indexes (gitignored)
│   ├── faiss_index.bin           # Dense vector index
│   ├── bm25_index.pkl            # BM25 index
│   └── chunk_metadata.pkl        # Chunk ID to text/URL/topic mapping
│
├── datasets/
│   └── benchmark/
│       ├── questions_visible.jsonl  # Questions used during development
│       └── questions_hidden.jsonl  # Held-out test set (NEVER open!)
│
├── domain_trained_enterprise_corpus/  # Pre-built seed corpus
├── phase3_bge_fixed_final/           # Fine-tuned embedding adapter
├── reranker_phase2/                  # Fine-tuned reranker adapter
├── phase4_llm_adapter/               # Fine-tuned LLM adapter
│
├── results/                      # Evaluation outputs, charts, reports
├── logs/                         # Runtime logs
├── ai_use_log/                   # AI usage documentation
│
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
└── verify_imports.py             # Dependency verification script
```

---

## 10. How to Run the Project

### Prerequisites
```bash
pip install -r requirements.txt
```

### Environment Setup
```bash
copy .env.example .env
# Edit .env and add your GROQ_API_KEY
```
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_GENERATE_MODEL=llama-3.3-70b-versatile
GROQ_JUDGE_MODEL=llama-3.1-8b-instant
```

### Step-by-Step Execution

```bash
# 1. Build the corpus (scrapes all documentation sources)
python scripts/build_corpus.py --config configs/naive_rag_config.yaml

# 2. Generate benchmark questions (do BEFORE building indexes)
python scripts/generate_benchmark.py --config configs/naive_rag_config.yaml
#    Review: datasets/benchmark/questions_raw.jsonl
#    Hand hidden set to mentor: datasets/benchmark/questions_hidden.jsonl

# 3. Build FAISS + BM25 indexes
python scripts/build_index.py --config configs/naive_rag_config.yaml

# 4. Run evaluation (frozen naive baseline)
python scripts/run_evaluation.py --config configs/naive_rag_config.yaml

# 5. (Phase 3) Build indexes with fine-tuned embeddings
python scripts/build_phase3_index.py --config configs/phase3_rag_config.yaml

# 6. Compare all phases side by side
python scripts/evaluate_phases.py

# 7. Launch interactive demo UI
python scripts/start_ui.py
```

### Evaluation Modes

```bash
# Full evaluation with LLM judge (correctness + faithfulness scoring)
python scripts/run_evaluation.py

# Fast retrieval-only (no Groq calls, just Recall/MRR/nDCG)
python scripts/run_evaluation.py --no-judge

# Quick dry run on first 10 questions
python scripts/run_evaluation.py --limit 10

# Ablation: compare BM25-only vs Dense-only vs Hybrid
python scripts/run_ablation_experiments.py
```

---

## 11. Benchmark Question Categories

The benchmark has **two splits**:

| Split | File | Purpose |
|---|---|---|
| **Visible** | `questions_visible.jsonl` | Used during development for tuning |
| **Hidden** | `questions_hidden.jsonl` | Final held-out evaluation (never used during development) |

Each benchmark question includes:
- `question` — the natural language question
- `category` — one of the 8 categories above
- `relevant_chunk_ids` — ground truth chunk IDs for retrieval evaluation
- `reference_answer` — gold answer for correctness judging
- `is_unanswerable` — true if system should refuse to answer

---

## 12. Key Design Decisions

### Why Hybrid Retrieval (BM25 + Dense)?
BM25 excels at exact keyword matching (command names, error codes, specific flags). Dense retrieval excels at semantic understanding ("how do I restart a crashed service?" retrieves systemd restart docs). RRF fusion keeps the best of both with zero extra tuning.

### Why BAAI/bge Models?
- Top-ranked on the MTEB (Massive Text Embedding Benchmark) leaderboard
- Open-source, runs locally on CPU — no API cost
- Bi-encoder (bge-base) and cross-encoder (bge-reranker) from the same family → compatible representations

### Why QLoRA for Fine-Tuning?
- Runs on a free Kaggle T4/P100 GPU (≤16GB VRAM)
- 4-bit quantization + LoRA adapters — models fine-tune in hours, not days
- Adapters are small (MBs) vs full weights (GBs) — easy to version and swap

### Why Fail-Closed on Unanswerable Questions?
Hallucinating a wrong technical answer (e.g., wrong Kubernetes flag) causes real production outages. Better to say "I don't know" than to confidently give a wrong answer. Evaluated separately: refusal precision + recall measure how well the system knows its limits.

### Why a Security Defense Pipeline?
RAG systems are vulnerable to **prompt injection via poisoned documents** — a malicious chunk in the corpus could override LLM instructions. Enterprise production systems process docs from many sources; not all can be trusted equally. OWASP LLM01 (Prompt Injection) and LLM02 (Sensitive Info Disclosure) are mitigated by the quarantine + envelope isolation design.

### Why an Independent Claim Verifier?
The LLM generating the answer and the LLM judging its faithfulness are the same model — a potential conflict of interest. The `ClaimVerifier` uses a separate NLI (Natural Language Inference) model to independently verify each claim against source chunks, providing an unbiased faithfulness signal.

---

## Summary

This project builds a **complete, production-ready enterprise RAG system** from scratch:

```
Scrape Official Docs
    → Clean & Chunk
        → Build Hybrid Index (BM25 + FAISS)
            → Fine-Tune Reranker (Phase 2)
                → Fine-Tune Embeddings + Advanced Pipeline (Phase 3)
                    → Fine-Tune LLM (Phase 4)
                        → Evaluate with 8-category benchmark + LLM Judge
```

Each phase is **independently evaluatable** against the frozen benchmark, enabling rigorous measurement of what each improvement actually contributes to answer quality.

---

*Project 3 — Domain-Trained Enterprise RAG | October 2026*
