# Prometheus - 03 Observability For Rag

> Corpus role: enterprise infrastructure knowledge for the Domain-Trained Enterprise RAG capstone. Content is intentionally modular so it can be chunked into independent retrieval units.

## Topic scope

Topic: **prometheus**

Each section below is an atomic knowledge unit. Preserve the heading and topic metadata during ingestion so retrieval results retain their semantic context.

## PROMETHEUS-001 | Request latency decomposition

Record separate latency measurements for edge proxy, API processing, retrieval, reranking, generation and database access. End-to-end latency alone does not reveal which component is responsible for regressions.

## PROMETHEUS-002 | p50 and p95

The p50 represents the median experience, while p95 shows a slower tail. The capstone explicitly requires p50/p95 reporting, so both should be calculated from the same defined request population and time window.

## PROMETHEUS-003 | Throughput and saturation

Track requests per second, active requests and queue depth where available. High throughput with rising latency can indicate saturation. The system should record hardware and concurrency settings for each formal experiment.

## PROMETHEUS-004 | Resource metrics

CPU, RAM and GPU metrics help explain model-serving behavior. Record them alongside benchmark runs rather than assuming the same model has the same performance on every environment.

## PROMETHEUS-005 | Citation verification metrics

Expose counters for SUPPORTED, PARTIAL and UNSUPPORTED claims. A useful dashboard can compare citation support rate with answer correctness and refusal behavior instead of treating one metric as sufficient.

## PROMETHEUS-006 | Adversarial benchmark observability

Record attack-class outcomes such as instruction-bearing documents, contradictory documents and outdated documents. Do not expose private benchmark content in public dashboards. Track labels such as attack class and outcome, not the hidden question text itself.

## PROMETHEUS-007 | Monitoring model versions

Include a bounded model or pipeline version label in metrics so a deployment can be compared with a previous version. Do not use a Git commit hash as an unbounded label if the system generates a new series for every commit over long periods.

## PROMETHEUS-008 | SLO-style thinking

Define expected availability and latency targets only after measuring real workloads. A target should be paired with a service-level indicator and an explicit interpretation of what counts as an error or unavailable response.

## PROMETHEUS-009 | Failure correlation

When a latency spike occurs, correlate Prometheus data with NGINX access logs, FastAPI logs, Kubernetes events and PostgreSQL activity. A single metric rarely proves root cause.

## PROMETHEUS-010 | Evaluation dashboard

The project requires a dashboard comparing system variants. It should show retrieval metrics, answer metrics, security results, latency and resource usage, with a clear corpus/benchmark/version selector so results are not mixed across incompatible runs.

## Source basis

The corpus is intentionally grounded in official project documentation. Use the URLs below for version-specific verification rather than copying volatile version numbers into model-training text.

- https://prometheus.io/docs/concepts/
- https://prometheus.io/docs/prometheus/latest/configuration/configuration/
- https://prometheus.io/docs/prometheus/latest/configuration/recording_rules/
- https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/

## Versioning note

This document should be treated as a versioned corpus artifact. Record the source snapshot date and any later updates in the dataset manifest. Version-sensitive benchmark questions should state the relevant software version explicitly.
