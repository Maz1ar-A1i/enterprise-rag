# Prometheus - 01 Metrics Promql Model

> Corpus role: enterprise infrastructure knowledge for the Domain-Trained Enterprise RAG capstone. Content is intentionally modular so it can be chunked into independent retrieval units.

## Topic scope

Topic: **prometheus**

Each section below is an atomic knowledge unit. Preserve the heading and topic metadata during ingestion so retrieval results retain their semantic context.

## PROMETHEUS-001 | Prometheus data model

Prometheus stores data as time series: timestamped samples identified by a metric name and a set of label dimensions. A unique combination of metric name and label values identifies a series. This dimensional model makes filtering and aggregation central to PromQL.

## PROMETHEUS-002 | Metric names

Metric names should describe the measured feature. Counters commonly use a `_total` suffix. Names should stay predictable and avoid encoding unbounded identifiers that belong in labels.

## PROMETHEUS-003 | Labels and cardinality

Labels distinguish series such as HTTP method or service. Every unique label-value combination creates a distinct series. High-cardinality labels such as request IDs or raw user queries can create enormous numbers of series and should not be used casually.

## PROMETHEUS-004 | Counter, gauge, histogram and summary concepts

Counters represent quantities that generally increase and can reset on restart. Gauges represent values that can rise or fall. Histograms sample observations into buckets and can be used to approximate distributions and quantiles. Summaries calculate quantiles client-side with different aggregation trade-offs.

## PROMETHEUS-005 | PromQL selection

PromQL selects and transforms time series. Label matchers filter the series set, and functions/operators aggregate, transform or calculate rates. Learn the difference between an instant vector, range vector and scalar because many query errors are type-shape errors.

## PROMETHEUS-006 | Rate and increase

`rate()` is commonly used with counters to calculate the per-second average increase over a range. `increase()` returns the total increase over the selected range. Use counter-aware functions rather than dividing raw counter values by time manually.

## PROMETHEUS-007 | Aggregation

Functions such as `sum`, `avg`, `max`, `min` and `count` can aggregate series. `sum by (...)` keeps selected labels while reducing dimensions. Before aggregating, decide which labels preserve an operational question and which are accidental detail.

## PROMETHEUS-008 | Recording rules

Recording rules precompute frequently used or expensive expressions and store the result as a new time series. They can make dashboards and alerts faster and reduce repeated query cost.

## PROMETHEUS-009 | Time-series naming for RAG

Useful metrics for the capstone include request totals, request duration histograms, retrieval duration, reranker duration, generation duration, refusal totals and citation verification outcomes. Keep labels low-cardinality: service, route, outcome and model version can be reasonable dimensions if bounded.

## PROMETHEUS-010 | Prometheus versus logs

Metrics answer numerical questions over time; logs hold detailed event context. A mature system uses both and connects them through timestamps and correlation identifiers without copying entire user content into metric labels.

## Source basis

The corpus is intentionally grounded in official project documentation. Use the URLs below for version-specific verification rather than copying volatile version numbers into model-training text.

- https://prometheus.io/docs/concepts/
- https://prometheus.io/docs/prometheus/latest/configuration/configuration/
- https://prometheus.io/docs/prometheus/latest/configuration/recording_rules/
- https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/

## Versioning note

This document should be treated as a versioned corpus artifact. Record the source snapshot date and any later updates in the dataset manifest. Version-sensitive benchmark questions should state the relevant software version explicitly.
