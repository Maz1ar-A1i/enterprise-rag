---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:582
- loss:MultipleNegativesRankingLoss
base_model: BAAI/bge-base-en-v1.5
widget:
- source_sentence: For the runtime 1.19 Dedicated canary, what TP watchdog mode and
    default timeouts should we start with, and what specific metrics or alert conditions
    should trigger an immediate rollback if we see false positives or drain storms?
  sentences:
  - 'Fix billing double-counting on Dedicated burst retries (idempotency + reconciliation)


    Problem: we are double-counting burst usage when the same customer request is
    retried (client retries, gateway retries after 5xx/timeout, and some 429 backoff
    paths). In beta, this showed up as duplicate burst usage line items for the same
    request window and was flagged in SUP-28488 and during reconciliation sampling.
    Root cause: the burst metering emitter currently treats each attempt as billable
    because we do not have a stable request-scoped idempotency key carried end-to-end
    into the metering aggregation stage; request_id/trace_id are not consistently
    preserved across retry code paths. Fix: introduce a request-scoped idempotency
    key that is generated at the gateway if missing, propagated to the data plane
    and metering emitter, and used to dedupe burst billable events before exporting
    to billing. Also add a reconciliation guard that detects duplicates that slip
    through (e.g., telemetry lag / partial outages) and blocks export with an alert.
    Details: 1) API/gateway: accept and forward X-Redwood-Idempotency-Key; if absent,
    generate a ULID per incoming request and reuse it across internal retry attempts.
    2) Metering: include idempotency_key on burst usage events; dedupe in the burst_usage_aggregator
    on (tenant_id, region, hw_class, pool_id, idempotency_key) with a bounded TTL
    and hashed storage to avoid high-cardinality raw keys. 3) Billing export: export
    a deduped_usage_count and dropped_duplicate_attempts metric; add a hard reconciliation
    check that fails the export batch if duplicate keys exceed a small threshold (defaults
    to 0.01%) and pages on-call for billing pipeline'
  - "Audit log export: role-aware filtered dump (2026)\n\n## Overview\nThis document\
    \ specifies the Console and API capability to export filtered slices of tenant\
    \ audit logs as a bulk dump while enforcing RBAC v2, retention limits, and SSO-backed\
    \ administrative permissions.\n\nThe primary use cases are compliance reporting,\
    \ e-discovery preparation, and offline analytics. The design explicitly prevents\
    \ disclosure of sensitive fields to users without the appropriate allow-list role.\n\
    \n## Scope\n### In scope\n- Role-aware bulk export for audit logs, initiated from\
    \ Console (Admin) and backed by a server-side export worker.\n- Query-scoped exports:\
    \ time window, actor, resource, action, outcome.\n- Field-level masking and omission\
    \ using RBAC v2 policy evaluation.\n- Export bundles include a manifest describing\
    \ query, filters, snapshot time, and schema version.\n- Support for multi-GB exports\
    \ using cursor-based pagination and resumable multipart downloads.\n- Export request\
    \ auditing to an immutable audit stream separate from the exported data.\n\n###\
    \ Out of scope\n- Real-time streaming feeds for customer SIEMs (handled by the\
    \ Audit Streaming project).\n- Interactive “preview rows” in Console (removed;\
    \ see decision ADR-EXPORT-004 below).\n\n## User experience (Console)\n1. Admin\
    \ navigates to: Console → Security → Audit Logs → Export.\n2. Admin configures\
    \ filters (time range required; other filters optional).\n3. Console shows:\n\
    \ - Estimated record count and approximate size (best-effort)\n - Retention cutoff\
    \ date for the tenant\n - Whether any sensitive fields may be included (based\
    \ on role)\n4. Admin submits export request.\n5. Console shows export status:\
    \ queued → running → completed/failed.\n6. On completion, Console provides a time-limited\
    \ download link and checksum."
  - 'eng-runtime


    Noah Patel: we need to settle default watchdog thresholds for TP before we merge
    the abort/drain path. what are sane numbers by model size/workload? main worry
    is false positives -> draining healthy replicas.


    Sofia Ivanova: +1. also we should be explicit if it’s a *stall* watchdog (no forward
    progress) vs *slow* watchdog (p99 latency). i think we want “no progress” only,
    but we’re using elapsed time as proxy.


    Hiro Tanaka: agree. in kernel land we’ve seen collectives legitimately take longer
    during long-context prefill (esp with big batches), but decode step should be
    tight. recommendation: separate thresholds for (prefill step, decode iter, collective
    op).


    Yuki Sato: quick data point from last week’s A100 80GB TP=8 runs (70B-ish):

    - steady-state decode iter ~35–80ms

    - worst spikes (not stuck) up to ~400–600ms when compaction + stream sync happens

    - prefill for 16k context can be 2–6s depending on batch

    so if decode watchdog is like 1s we’re probably safe, but 500ms might false-positive.


    Grace Kim: +1, and cancelled streaming / client disconnect path can pause things
    too (we’re fixing but still). for safety i’d rather start higher and tighten.


    Omar El-Sayed: another wrinkle: KV compaction / GC can block a stream for >1s
    under memory pressure. it’s not a deadlock but looks like “no progress”. we can
    tag stall reason once we have heartbeat + queue depth, but until then we’ll get
    ambiguous triggers.


    Julian Moore: can we do two-stage? (1) “soft” watchdog logs + increments metric
    when timeout exceeded, (2) “hard” abort only when exceeded N times or exceeded
    much higher threshold. that reduces false-positive drains.'
- source_sentence: What performance goal was set for reducing GPU out of memory incidents
    in the stress tests for the windowed KV cache proposal?
  sentences:
  - "Re: Critical: inference latency spike + 5xx affecting throughput (Crucible Health)\n\
    \nFrom: Julia Adams \nTo: Stephanie Nguyen \nDate: 2026-08-28T08:12:00-07:00\n\
    Subject: Critical: sudden latency spike and 500s on inference endpoints\n\nHi\
    \ Stephanie,\n\nWe started seeing a large increase in end-to-end latency and a\
    \ burst of 5xx errors on the /v1/generate endpoint at ~07:55 PDT. Throughput dropped\
    \ by ~40% for our primary production inference queue and several batches timed\
    \ out. This is causing timeouts in our user-facing flow and triggering throttles\
    \ on our side.\n\nSymptoms observed: \n- Median latency jumped from ~120ms to\
    \ ~900ms between 07:55–08:05\n- 500 responses: ~8% of requests in the window\n\
    - Throughput fall from 250 req/s to ~150 req/s for the Crucible production key\n\
    \nWe attached recent logs (request IDs included) and a CSV snapshot of KPIs. Please\
    \ escalate — this is blocking a critical demo at 10:00 PDT.\n\nSample request\
    \ IDs: 6a4f9b2c, 6a4f9b45, 6a4f9c01\n\nAttachments: request_logs_20260828.zip,\
    \ detailed_kpi_snapshot.csv\n\nThanks,\nJulia Adams\nHead of Integrations, Crucible\
    \ Health\n\nFrom: Stephanie Nguyen \nTo: Marcus Lin , Hannah Schmitt , Laura Bennett\
    \ , Sergio Alvarez \nDate: 2026-08-28T08:25:00-07:00\nSubject: Fwd: Critical:\
    \ sudden latency spike and 500s on inference endpoints\n\nTeam — forwarding urgent\
    \ customer report from Crucible Health (Julia)."
  - 'CI Eval Determinism and Flake Mitigation


    ## Summary

    Our evaluation harness previously produced intermittent metric drops (e.g., ~0.04
    BLEU on a single run) for the quantized-v2 onboarding suite that disappeared on
    rerun. This page documents the current (2026) standard for reducing nondeterminism,
    preventing CI flakes, and making regressions actionable.


    The root causes we have observed include:

    - Stochastic decoding (temperature/top_p sampling) without per-example RNG control.

    - RNG state not consistently propagated to all relevant libraries/ops.

    - GPU-level nondeterminism (kernel selection, atomic ordering), especially in
    quantized/int8 paths.

    - Cross-hardware differences (e.g., A100 int8 kernels vs legacy fallback paths).


    ## What changed since the initial investigation

    The original debugging thread identified sampling parameters and missing explicit
    generators as primary contributors. The current harness incorporates the same
    core ideas, with the following updates:

    - We now treat deterministic decoding as the default for per-commit CI gates for
    onboarding suites.

    - The determinism flag is implemented as `--deterministic` / `deterministic: true`
    in eval config, and it hard-enforces decoding parameters (see below).

    - The runner always constructs and passes an explicit `torch.Generator` for generation,
    derived from a root seed plus a stable per-example index.

    - CI gating moved from single-run metrics to median-of-repeats, and flakiness
    is measured explicitly.


    ## Definitions

    ### Deterministic (harness-level)

    A run is considered deterministic (for CI purposes) if repeated executions on
    the same hardware + container image + model revision produce identical decoded
    token sequences for the same prompts.


    Notes:

    - Hardware drift (A100 vs K80) is not expected to be bitwise identical.

    - Some quantized kernels may still be nondeterministic even when framework-level
    deterministic flags are enabled.'
  - "Bounded-window KV cache with adaptive backpressure and graceful GPU-OOM fallback\n\
    \nContext: We're seeing tail-latency spikes and intermittent GPU OOMs in multi-tenant\
    \ serving when long sequences lead to large KV residency. Current paged/blocked\
    \ KV designs reduce average memory but cause sudden allocation pressure and head-of-line\
    \ stalls when many long-context requests overlap.\n\nProblem statement: We need\
    \ a practical runtime-level strategy that: 1) bounds per-request KV working set\
    \ growth using a windowed retention model, 2) signals and smooths load to upstream\
    \ scheduling to avoid synchronous OOM cascades, and 3) provides a deterministic\
    \ fallback path that preserves latency SLOs for short requests while allowing\
    \ long requests to continue with degraded throughput.\n\nProposal summary: Implement\
    \ a \"bounded-window KV\" mode that splits KV residency into fixed-duration windows\
    \ (logical time slices of attention history) with the following features: \n-\
    \ Soft retention limits per-request (N windows) with an LRU-like priority inside\
    \ each window.\n- Adaptive backpressure signals exposed to the scheduler: soft_shed\
    \ and holdback tokens that throttle batching and admission for requests that would\
    \ push resident set above a dynamic high-water mark.\n- Graceful OOM fallback:\
    \ when hard limit is reached, use a prioritized partial-eviction strategy (drop\
    \ older windows for low-priority requests while preserving most-recent window\
    \ + KV prefix cache for high-priority/latency-sensitive traffic).\n\nWhy different\
    \ from existing approaches: Unlike slab-only or chunked allocators, windowed retention\
    \ encodes temporal locality at a coarse granularity that simplifies eviction math\
    \ and gives predictable worst-case resident set sizes. Combining this with explicit\
    \ backpressure tokens reduces synchronous eviction storms that previously caused\
    \ cascading tail latency."
- source_sentence: What is the phased rollout schedule for removing the legacy compatibility
    proxy and migrating old persisted flag configs to a single newer schema?
  sentences:
  - 'Strip compat-proxy and unify persisted flag schemas


    Summary: A legacy compatibility proxy and an ad-hoc legacy-flag-mapper were introduced
    in early rollout phases to accept old client configs and route traffic to fallback
    handlers. Over time these have become maintenance burden: duplicated telemetry,
    increased tail latency variance, and brittle conditional logic that blocks safe
    schema evolution. This ticket removes the compat-proxy and migrates persisted
    flag keys/configs to a single canonical config-schema v2. The work includes safe
    migration tooling, operator controls, and a staged rollout with clear observability.

    Legacy compat-proxy binary and all HTTP/gRPC routes removed from the runtime and
    service mesh

    Config-schema v2 is the sole accepted on-disk format; v1 reader deprecated and
    gated behind operator toggle

    Migration scripts can perform idempotent dry-runs and produce a migration report
    listing changed keys

    No customer-facing errors attributable to missing legacy keys during a 72h full-ramp
    window

    Telemetry shows no net increase in P99 latency or error-rate > 0.5% vs baseline
    after full rollout

    Create operator-only toggle ''allow_v1_configs'' default=false

    Run discovery job across S3/DB to enumerate v1 configs and generate migration
    plan

    Execute idempotent migration in dry-run mode and publish report to migration-bucket

    Perform controlled migration on a small set of non-critical Dedicated clusters

    Flip runtime to reject v1 payloads only after all persisted configs migrated

    Remove v1 reader code and compat-proxy from runtime in the final release

    Phase 0: Nightly dry-runs for 3 days and manual review of reports

    Phase 1 Canary: Migrate 1-2 canary clusters (approx 3% traffic) for 6 hours

    Phase 2 Staged: Expand to 20% traffic over 24h if canary stable

    Phase 3 Ramp: 60% for 24h with automated health checks and human-on-call'
  - 'Probe heatmap -> rollout decision protocol


    Summary

    -------

    This document defines a reproducible protocol to convert probe heatmaps (telemetry-driven
    failure density views) into rollout decisions for model promotions, quantization
    migrations, and configuration changes. The goal is to make decisions auditable,
    time-bounded, and aligned with SLOs so product and ops can act consistently across
    hosted, dedicated, and private deployments.


    Scope

    -----

    - Applies to any promotion or rollout that is gated by probe-driven quality signals:
    model promotions, quantized rollouts, prompt rewrites, and routing policy changes.

    - Focuses on probe heatmaps produced by the eval harness and live shadow probes
    (not human labelling pipelines).

    - Excludes legislated/regulatory decisions (finance/compliance) - see separate
    compliance approval flow.


    Key definitions

    ---------------

    - Probe heatmap: a time-windowed 2D histogram where rows are probes (slice identifiers)
    and columns are failure modes (fidelity, hallucination, latency, embeddings drift).
    Each cell contains count and a severity score.

    - Decision score (DS): a normalized scalar computed from heatmap aggregates that
    maps to GO / CAUTION / HOLD / ROLLBACK actions.

    - Tiered slice: pre-defined group of probes that share an SLO (e.g., CriticalTenantFlows,
    BillingIntents, LowRiskExplorations).


    Inputs required

    ---------------

    1) Heatmap snapshot covering the last N hours (default N=24) with per-probe metrics:
    total_calls, failure_count, avg_severity, median_latency_ms, embedding_drift_score.

    2) Active rollout plan metadata: target model id, variant (quantized/float16),
    percent_traffic, scheduled_start.

    3) Baseline reference heatmap from the control population (previous model or canary).

    4) SLO manifest for target slices (SLO window, allowed error budget, latency SLO).

    5) Operational constraints: maintenance windows, capacity headroom, and dependency
    map.'
  - 'refactor-thoughtstream-kiran


    Context:\nThis is my late-night thoughtstream after skimming the large PR that
    touched batching, KV cache boundaries, and a few public SDK changes. Not a finished
    doc; more of a sequence of heuristics, micro-checks, and follow-ups I want to
    remember and possibly turn into a tidy checklist for the team.\n\nHigh-level concerns
    (gut):\n- Backwards compatibility: SDK surface was changed for that optional `batch_hint`
    param. Need to confirm SDKs are tolerant for older clients.\n- Performance regression
    risk: the new batching warm-up path may alter latency p50/p95 during short-burst
    workloads.\n- Cache safety: cross-tenant KV-cache key normalization looked subtly
    different (strip-vs-preserve trailing slashes). That can lead to cache poisoning
    or silent correctness issues.\n\nQuick manual checks I did (and outcomes):\n-
    Read the new unit tests: they cover happy paths but not the case where sequence_length
    >> batch window. Missing.\n- Ran a small local benchmark (dev machine) with sample
    trace: p50 unchanged, p95 increased ~12%'
- source_sentence: For the ClearSight Health pilot discussions with Redwood, what
    specific types of patient-related data did their security lead say should be treated
    as PHI for this integration?
  sentences:
  - 'Unified Capacity Transition PRD: SLO-aligned Consumption and Dedicated Migration


    Executive summary:\n\nThis document defines the product requirements, design constraints,
    rollout roadmap, pricing impacts, and release notes for the "Unified Capacity
    Transition" initiative'
  - "Data handling & retention — post-call follow-up\n\nFrom: Aditya Rao \nTo: Bethany\
    \ Park \nCc: Laura Bennett \nDate: Wed, 20 May 2026 14:12:00 -0700\nSubject: Post-call\
    \ recap & next steps — data handling questions\n\nHi Bethany — thanks again for\
    \ taking time today. Quick recap from our call and the items we said we'd follow\
    \ up on:\n\n- ClearSight requirement: minimal retention of PII and PHI in any\
    \ hosted logging/analytics (you mentioned: prefer no storage at all; audit evidence\
    \ required).\n- Questions for us: can Redwood support a true 0-retention mode\
    \ for token logs and request traces, or do we need to offer a short retention\
    \ window?\n- Next steps I promised: summarize options (hosted pseudonymization\
    \ / private deployment) and send SOC2/attestations.\n\nAction items for us: I\
    \ will loop in our solutions lead (Laura) and sales lead (Vivek) to outline concrete\
    \ options and a proposal. Could you confirm the scope of data you consider \"\
    PHI\" for this integration (examples help)? Also confirm whether procurement will\
    \ want a signed MOA or simply SOW-level commitments for the pilot.\n\nAttachments:\
    \ CSH-security-reqs.pdf (your upload from today) — I attached for reference.\n\
    \nBest,\nAditya Rao\nAccount Executive, Redwood Inference\naditya.rao@redwood.com\n\
    (415) 555-0142\nFrom: Bethany Park \nTo: Aditya Rao \nDate: Thu, 21 May 2026 09:08:00\
    \ -0700\nSubject: Re: Post-call recap & next steps — data handling questions\n\
    \nThanks Aditya — great notes and thanks for moving quickly. A few clarifications\
    \ from our side:"
  - 'Risk-thread notes — SOC2 prep (scratch)


    Quick brain dump while mapping controls to actual product touchpoints — messy.
    Intended to be a running diary of open questions, small experiments, and what
    I need to hand off to ENG/SRE/Product later.


    Context and goal:

    - Prep for SOC2 readiness review (targeting initial readiness check in April).

    - Focus: evidence collection for A.1 (access control), A.3 (change management),
    and A.5 (monitoring/logging) — mapping existing artifacts to common control statements.

    - Keep notes about how Hosted/Dedicated/Private deployments differ for evidence
    (esp. on private where customer-managed KMS comes in).


    Things I observed (raw):

    - Access control: console uses Okta SSO; service-to-service auth uses mTLS and
    signed tokens. Need explicit docs for service token rotation frequency and how
    we revoke (is it in infra-runbook?). Ticket: INF-2345 references token rotation
    work but not runbook.

    - KMS: Hosted uses Redwood KMS (AWS KMS CMKs). Private offering points to customer
    KMS; evidence trail inconsistent across private installs. Ask PS: what artifacts
    do we get from on-prem install? (audit logs, KMS policy snapshots).

    - Logs: central ELK cluster for hosted, but Dedicated customers may have separate
    logging sinks; need a matrix of where logs live per deployment type.'
- source_sentence: What is the formula used by the new autoscaling approach that combines
    a high-percentile request rate with request variability to compute effective demand,
    including the default weights?
  sentences:
  - "Dispatch surface playback: edge vs core econ memo\n\nTL;DR:\nWe ran a small set\
    \ of controlled playback tests to compare dispatch semantics across edge-forward\
    \ dispatch (short hops to nearby edge cells with small prefetch), core-tiling\
    \ multiplexing (centralized GPU tiles with larger batching windows), and a hybrid\
    \ with KV-prefetch evictions. Results show: edge-dispatch-prefetch gives best\
    \ p50 and p95 for short responses ( =512 tokens) but suffers p99 regressions under\
    \ bursty arrival; hybrid-evict trades in slightly higher median latency for better\
    \ p99 and solid $/1k for mid-length responses.\n\nWhy we ran this (context):\n\
    - Product asks for a short, shareable comparison to inform routing policy defaults\
    \ for chat-lite vs doc-embed routes.\n- We need simple knobs that map to developer-facing\
    \ toggles: prefer-edge, prefer-core, hybrid-prefetch.\n- This is intentionally\
    \ lightweight — not exhaustive — but focused on practical, explainable deltas\
    \ to communicate with PM and Console teams.\n\nExperiment setup (short):\n- Date:\
    \ 2026-05-15 to 2026-05-17.\n- Hardware: G-core (A100x2) for core tiles; Edge\
    \ cells running T4-equivalent virtual nodes (quantized runtime).\n- Model: open-llm\
    \ v2-3B (float16 in core, int8 quantized at edge).\n- Workloads: 3 synthetic mixes:\
    \ \n * Short-burst: 80% =512 tok (document summarization)\n- Traffic shape: poisson\
    \ baseline + injected microbursts (10s, 100req/s spikes) to exercise tail behavior.\n\
    - Measurement: p50, p95, p99 latency on response-ready (end-of-stream), tokens/sec\
    \ observed, and computed $/1k tokens using current infra cost model (GPU-hour\
    \ amortized + infra egress)."
  - 'Introduce defaults-suggester: heuristic profiles for batching, cache TTLs and
    quantization presets


    Context: Teams repeatedly ask for sensible defaults for batching, cache TTLs and
    quantization when onboarding models to Redwood. Currently defaults live in multiple
    places and require manual tuning with evals and internal heuristics. Goal: ship
    a small, auditable suggestion system that observes recent workload signals and
    proposes conservative default profiles for batching.window_ms, batch_size caps,
    prefix/KV cache TTLs, and quantization presets. The suggester is opt-in by default
    and produces human-readable suggestions surfaced in the Console and via a new
    API endpoint.


    What this PR does: 1) Adds a lightweight offline heuristic engine (defaults-suggester)
    that consumes short-term aggregated telemetry (per-route QPS, median latency,
    50/95/99 latency quantiles, token length distribution) and maps to a ranked set
    of suggested profiles. 2) Introduces a new API: POST /optimize/v1/suggestions
    which returns JSON suggestions with confidence scores and an explanation string.
    3) Wire up Console: a Suggestions panel on per-route settings with an Apply &
    Preview flow that shows estimated unit-cost delta and latency impact. 4) Add a
    gated runtime flag so the runtime can accept a profile payload for immediate application
    (useful for Dedicated/Private workflows). 5) Provide a small set of canonical
    profiles (conservative, balanced, aggressive) for batching and quantization, plus
    TTL presets for prefix/KV caches. 6) Add end-to-end test coverage and benchmark
    harness runs comparing before/after for representative traffic shapes.'
  - 'variance-aware-autoscaler-and-intermittent-surge-fairness


    Add variance-aware autoscaler controller and per-tenant variance telemetry; improves
    stability under intermittent surge workloads and increases fairness of burst allocation.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy@1
- cosine_accuracy@3
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_precision@1
- cosine_precision@3
- cosine_precision@5
- cosine_precision@10
- cosine_recall@1
- cosine_recall@3
- cosine_recall@5
- cosine_recall@10
- cosine_ndcg@10
- cosine_mrr@5
- cosine_mrr@10
- cosine_map@100
model-index:
- name: SentenceTransformer based on BAAI/bge-base-en-v1.5
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: eval holdout
      type: eval_holdout
    metrics:
    - type: cosine_accuracy@1
      value: 0.9333333333333333
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.9777777777777777
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.9777777777777777
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.9777777777777777
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.9333333333333333
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.585185185185185
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.46222222222222226
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.2911111111111111
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.5295326278659612
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.7434126984126983
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.8297971781305115
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.9130599647266314
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.8999897768474252
      name: Cosine Ndcg@10
    - type: cosine_mrr@5
      value: 0.9518518518518517
      name: Cosine Mrr@5
    - type: cosine_mrr@10
      value: 0.9518518518518517
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.879045451009766
      name: Cosine Map@100
---

# SentenceTransformer based on BAAI/bge-base-en-v1.5

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) <!-- at revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 768, 'pooling_mode': 'cls', 'include_prompt': True})
  (2): Normalize({})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
queries = [
    'What is the formula used by the new autoscaling approach that combines a high-percentile request rate with request variability to compute effective demand, including the default weights?',
]
documents = [
    'variance-aware-autoscaler-and-intermittent-surge-fairness\n\nAdd variance-aware autoscaler controller and per-tenant variance telemetry; improves stability under intermittent surge workloads and increases fairness of burst allocation.',
    'Introduce defaults-suggester: heuristic profiles for batching, cache TTLs and quantization presets\n\nContext: Teams repeatedly ask for sensible defaults for batching, cache TTLs and quantization when onboarding models to Redwood. Currently defaults live in multiple places and require manual tuning with evals and internal heuristics. Goal: ship a small, auditable suggestion system that observes recent workload signals and proposes conservative default profiles for batching.window_ms, batch_size caps, prefix/KV cache TTLs, and quantization presets. The suggester is opt-in by default and produces human-readable suggestions surfaced in the Console and via a new API endpoint.\n\nWhat this PR does: 1) Adds a lightweight offline heuristic engine (defaults-suggester) that consumes short-term aggregated telemetry (per-route QPS, median latency, 50/95/99 latency quantiles, token length distribution) and maps to a ranked set of suggested profiles. 2) Introduces a new API: POST /optimize/v1/suggestions which returns JSON suggestions with confidence scores and an explanation string. 3) Wire up Console: a Suggestions panel on per-route settings with an Apply & Preview flow that shows estimated unit-cost delta and latency impact. 4) Add a gated runtime flag so the runtime can accept a profile payload for immediate application (useful for Dedicated/Private workflows). 5) Provide a small set of canonical profiles (conservative, balanced, aggressive) for batching and quantization, plus TTL presets for prefix/KV caches. 6) Add end-to-end test coverage and benchmark harness runs comparing before/after for representative traffic shapes.',
    'Dispatch surface playback: edge vs core econ memo\n\nTL;DR:\nWe ran a small set of controlled playback tests to compare dispatch semantics across edge-forward dispatch (short hops to nearby edge cells with small prefetch), core-tiling multiplexing (centralized GPU tiles with larger batching windows), and a hybrid with KV-prefetch evictions. Results show: edge-dispatch-prefetch gives best p50 and p95 for short responses ( =512 tokens) but suffers p99 regressions under bursty arrival; hybrid-evict trades in slightly higher median latency for better p99 and solid $/1k for mid-length responses.\n\nWhy we ran this (context):\n- Product asks for a short, shareable comparison to inform routing policy defaults for chat-lite vs doc-embed routes.\n- We need simple knobs that map to developer-facing toggles: prefer-edge, prefer-core, hybrid-prefetch.\n- This is intentionally lightweight — not exhaustive — but focused on practical, explainable deltas to communicate with PM and Console teams.\n\nExperiment setup (short):\n- Date: 2026-05-15 to 2026-05-17.\n- Hardware: G-core (A100x2) for core tiles; Edge cells running T4-equivalent virtual nodes (quantized runtime).\n- Model: open-llm v2-3B (float16 in core, int8 quantized at edge).\n- Workloads: 3 synthetic mixes: \n * Short-burst: 80% =512 tok (document summarization)\n- Traffic shape: poisson baseline + injected microbursts (10s, 100req/s spikes) to exercise tail behavior.\n- Measurement: p50, p95, p99 latency on response-ready (end-of-stream), tokens/sec observed, and computed $/1k tokens using current infra cost model (GPU-hour amortized + infra egress).',
]
query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)
print(query_embeddings.shape, document_embeddings.shape)
# [1, 768] [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)
# tensor([[0.5864, 0.2474, 0.2254]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Information Retrieval

* Dataset: `eval_holdout`
* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.sentence_transformer.evaluation.InformationRetrievalEvaluator)

| Metric              | Value   |
|:--------------------|:--------|
| cosine_accuracy@1   | 0.9333  |
| cosine_accuracy@3   | 0.9778  |
| cosine_accuracy@5   | 0.9778  |
| cosine_accuracy@10  | 0.9778  |
| cosine_precision@1  | 0.9333  |
| cosine_precision@3  | 0.5852  |
| cosine_precision@5  | 0.4622  |
| cosine_precision@10 | 0.2911  |
| cosine_recall@1     | 0.5295  |
| cosine_recall@3     | 0.7434  |
| cosine_recall@5     | 0.8298  |
| cosine_recall@10    | 0.9131  |
| **cosine_ndcg@10**  | **0.9** |
| cosine_mrr@5        | 0.9519  |
| cosine_mrr@10       | 0.9519  |
| cosine_map@100      | 0.879   |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 582 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                          | sentence_1                                                                           |
  |:---------|:------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type     | string                                                                              | string                                                                               |
  | modality | text                                                                                | text                                                                                 |
  | details  | <ul><li>min: 22 tokens</li><li>mean: 44.44 tokens</li><li>max: 100 tokens</li></ul> | <ul><li>min: 54 tokens</li><li>mean: 354.65 tokens</li><li>max: 495 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                       | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>In the email thread about renewing the feature flag vendor for Redwood, what peak burst capacity was proposed in the multi year renewal option SLA?</code> | <code>Feature flag licensing review & stakeholder sync — LaunchDarkly renewal<br><br>From: Sam Jenkins <br>To: Neha Kapoor <br>Date: Tue, 14 Apr 2026 10:12:00 -0700<br>Subject: LaunchDarkly renewal overview for Redwood<br><br>Hi Neha —<br><br>Great talking earlier. Per our conversation, the current enterprise subscription for LaunchDarkly at Redwood is up for renewal on 2026-05-31. I attached a short summary of current entitlements and a baseline pricing sheet. Key highlights: standard SLA, 24/7 support addon currently enabled, and audit logs are part of the platinum package.<br><br>I've included two renewal options in the attachment: a 12-month renewal with a modest 8% net discount and a 24-month commitment with a 14% net discount plus a capped per-seat increase. Happy to walk the team through trade-offs.<br><br>Open Qs for your team:<br>- priority items for this renewal (audit logs, retention, seats)<br>- appetite for a 24-month committed discount vs annual flexibility<br>- any compliance gating items we need to include in SOW<br><br>Att: con...</code> |
  | <code>In the sandbox onboarding demo with NimbusApps, what median and p95 latency did the 50-concurrency 1-minute performance run report?</code>                 | <code>ChatPlay onboarding sandbox demo — streaming, params, retries<br><br>Intro demo of Redwood Hosted Chat/text generation for NimbusApps. Covered streaming behavior, parameter tuning (temperature/top_p/max_tokens), and retry/backoff strategies. Discussed SSO/KMS concerns and criteria for a pilot. Agreed on sandbox access, security docs, and a perf benchmark next week.<br>Meeting header:<br>Date: 2026-09-03<br>Start: 15:00 UTC<br>Duration: 42 minutes<br>Attendees: Alex Mercer (Redwood AE), Priya Nair (Redwood SE), Carlos Vega (Redwood Product), Samir Patel (NimbusApps CTO), Jordan Lee (NimbusApps Lead Eng), Maya Chen (NimbusApps Security), Terry Huang (NimbusApps PM)<br><br>Auto-generated sections (may be imperfect):<br>Summary: quick hosted demo, show streaming, show params, answer latency/cost questions, clarify security and data flows.<br>Action items: see metadata.</code>                                                                                                                                                                                     |
  | <code>How do audit log exports access records older than the retention cutoff (super-admin override vs legal hold)?</code>                                       | <code>Audit log export: role-aware filtered dump (2026)<br><br>## Overview<br>This document specifies the Console and API capability to export filtered slices of tenant audit logs as a bulk dump while enforcing RBAC v2, retention limits, and SSO-backed administrative permissions.<br><br>The primary use cases are compliance reporting, e-discovery preparation, and offline analytics. The design explicitly prevents disclosure of sensitive fields to users without the appropriate allow-list role.<br><br>## Scope<br>### In scope<br>- Role-aware bulk export for audit logs, initiated from Console (Admin) and backed by a server-side export worker.<br>- Query-scoped exports: time window, actor, resource, action, outcome.<br>- Field-level masking and omission using RBAC v2 policy evaluation.<br>- Export bundles include a manifest describing query, filters, snapshot time, and schema version.<br>- Support for multi-GB exports using cursor-based pagination and resumable multipart downloads.<br>- Export request auditing to an immutable audit stream...</code>             |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 4
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 4
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 16
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `dataloader_multiprocessing_context`: None
- `dataloader_in_order`: True
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}
- `warmup_ratio`: None

</details>

### Training Logs
| Epoch | Step | eval_holdout_cosine_ndcg@10 |
|:-----:|:----:|:---------------------------:|
| 1.0   | 19   | 0.8820                      |
| 2.0   | 38   | 0.9068                      |
| 3.0   | 57   | 0.9080                      |
| 4.0   | 76   | 0.9000                      |


### Training Time
- **Training**: 2.8 minutes

### Framework Versions
- Python: 3.13.15
- Sentence Transformers: 5.7.0
- Transformers: 5.16.1
- PyTorch: 2.11.0+cu128
- Accelerate: 1.14.0
- Datasets: 4.8.5
- Tokenizers: 0.23.1

## Additional Resources

- [Training and Finetuning Embedding Models with Sentence Transformers](https://huggingface.co/blog/train-sentence-transformers): the end-to-end guide for training or finetuning Sentence Transformer models.
- [Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka): variable-size embeddings that can be truncated with minimal quality loss.
- [Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval](https://huggingface.co/blog/embedding-quantization): post-training compression of embedding vectors.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): train multimodal embedding models, with a Visual Document Retrieval walkthrough.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->