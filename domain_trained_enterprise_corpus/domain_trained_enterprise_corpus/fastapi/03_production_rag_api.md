# Fastapi - 03 Production Rag Api

> Corpus role: enterprise infrastructure knowledge for the Domain-Trained Enterprise RAG capstone. Content is intentionally modular so it can be chunked into independent retrieval units.

## Topic scope

Topic: **fastapi**

Each section below is an atomic knowledge unit. Preserve the heading and topic metadata during ingestion so retrieval results retain their semantic context.

## FASTAPI-001 | RAG query endpoint design

A useful production endpoint can accept a query and optional retrieval controls, then return an answer object containing the answer, citations, refusal status and possibly retrieval trace identifiers. Keep internal model prompts and raw tool outputs out of the public response unless the contract explicitly requires them.

## FASTAPI-002 | Health and readiness endpoints

Separate liveness from readiness. A liveness endpoint answers whether the process is alive. A readiness check answers whether required dependencies are available enough to serve traffic. Do not make liveness depend on PostgreSQL if a temporary database outage would cause an otherwise healthy process to be restarted continuously.

## FASTAPI-003 | Timeouts and cancellation

Set explicit timeouts for upstream calls. Retrieval, reranking and generation can have different latency expectations. A single unbounded request can consume worker capacity and cause cascading failures. When possible, propagate cancellation and stop work after the caller has disconnected.

## FASTAPI-004 | Streaming responses

Streaming can improve perceived latency by returning generated tokens or incremental events before the full answer finishes. It also changes error handling: the server may have already sent part of a response when a downstream failure occurs. Citation completeness and refusal behavior should be defined explicitly for streamed responses.

## FASTAPI-005 | Observability hooks

Expose metrics for request count, latency, error rate, retrieval latency, reranker latency, generation latency and dependency failures. Include low-cardinality labels such as route and outcome; avoid putting user queries into metric labels because high-cardinality data can overload the metrics system.

## FASTAPI-006 | Graceful degradation

When a non-critical component fails, the API should have a deliberate policy. It may refuse the query, use a safe fallback retriever, or return a dependency-unavailable response. Silent fallback to an unvalidated model can invalidate the evaluation protocol.

## FASTAPI-007 | Evaluation endpoint isolation

Formal evaluation should use a separate command or restricted endpoint so normal users cannot modify benchmark prompts, scores or system selection. The visible benchmark should be frozen before major tuning to prevent overfitting.

## FASTAPI-008 | Reproducibility metadata

Record model version, retriever version, reranker version, corpus snapshot and configuration version with formal experiment results. Store these values in machine-readable metadata rather than relying on memory or notebook cells.

## FASTAPI-009 | Deployment with NGINX

A common deployment places NGINX in front of FastAPI. NGINX can terminate TLS and proxy traffic to the API service. The FastAPI process should be bound to its internal interface and not be directly exposed to the public network unless there is a deliberate reason.

## FASTAPI-010 | Deployment with Kubernetes

On Kubernetes, the API container should be managed by a Deployment, exposed internally through a Service, and configured with probes and resource requests/limits. Configuration that varies by environment should be separated from the image where possible.

## Source basis

The corpus is intentionally grounded in official project documentation. Use the URLs below for version-specific verification rather than copying volatile version numbers into model-training text.

- https://fastapi.tiangolo.com/tutorial/first-steps/
- https://fastapi.tiangolo.com/tutorial/dependencies/
- https://fastapi.tiangolo.com/tutorial/security/first-steps/
- https://fastapi.tiangolo.com/tutorial/background-tasks/

## Versioning note

This document should be treated as a versioned corpus artifact. Record the source snapshot date and any later updates in the dataset manifest. Version-sensitive benchmark questions should state the relevant software version explicitly.
