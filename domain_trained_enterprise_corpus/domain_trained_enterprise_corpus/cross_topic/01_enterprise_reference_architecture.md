# Cross-Topic Enterprise RAG - 01 Enterprise Reference Architecture

> Cross-topic knowledge is designed for multi-hop, troubleshooting and architecture questions.

## CROSS-001 | Reference request path

A representative enterprise path is client -> NGINX/TLS -> FastAPI -> query processing -> hybrid retrieval -> reranker -> context selection -> generator -> citation verifier -> response. PostgreSQL can store structured metadata and experiment state, while a vector index stores dense retrieval data. Prometheus monitors the runtime. Docker packages services; Kubernetes and Helm can deploy them; GitHub Actions can build, test and promote artifacts.

## CROSS-002 | Trust boundaries

Treat client queries, uploaded documents, retrieved passages and external model outputs as untrusted data. Deployment credentials, tool permissions, system instructions and monitoring administration belong to a higher-trust control plane. The RAG security layer must prevent retrieved instructions from crossing that boundary.

## CROSS-003 | Version axes

Formal evaluation requires separate versions for corpus snapshot, benchmark, chunking/index configuration, retriever, reranker, generator, prompt/configuration and serving image. Changing any one of these can change results.

## CROSS-004 | Resource boundaries

The serving system should isolate expensive generation and indexing operations from lightweight API and proxy processes. Resource requests and limits should be based on measurements. Formal latency results should record hardware and concurrency.

## CROSS-005 | Deployment boundaries

A container image is an immutable application artifact; Helm/Kubernetes define deployment state; NGINX exposes the HTTP edge; FastAPI defines the service contract; PostgreSQL holds structured state; Prometheus observes behavior. Clear boundaries make experiments and failures attributable.

## CROSS-006 | Corpus-to-answer trace

A robust answer trace should identify which corpus snapshot was indexed, which retriever and reranker selected passages, which citations supported claims and which model generated the answer. This is more useful than logging a prompt alone because it makes provenance auditable.

## CROSS-007 | Hidden-benchmark protection

The mentor-controlled hidden benchmark should not be baked into training artifacts, public CI outputs or production dashboards. Evaluation infrastructure should make it technically difficult to accidentally mix hidden questions with the development corpus.

## CROSS-008 | Change approval

Changes that can move the formal benchmark should be explicit: data changes, index changes, model changes and prompt/configuration changes. Record the hypothesis, change and evidence so the final report can explain why each improvement was kept.

## CROSS-009 | Failure containment

A parser failure should not take down the API; a broken exporter should not stop request handling; a model timeout should not exhaust every API worker; a failed database connection should produce a bounded error instead of an infinite retry loop. This is the operational meaning of component isolation.

## CROSS-010 | One command evaluation

The final project should have one documented command that identifies the versions, runs the frozen benchmark, measures retrieval and answer metrics, stores results as JSON/CSV and produces charts. CI and dashboards should consume these machine-readable results rather than reimplementing the experiment logic.

