# Prometheus - 02 Scraping Rules Alerts

> Corpus role: enterprise infrastructure knowledge for the Domain-Trained Enterprise RAG capstone. Content is intentionally modular so it can be chunked into independent retrieval units.

## Topic scope

Topic: **prometheus**

Each section below is an atomic knowledge unit. Preserve the heading and topic metadata during ingestion so retrieval results retain their semantic context.

## PROMETHEUS-001 | Scrape model

Prometheus normally pulls metrics from configured targets at intervals. The configuration describes scrape jobs and targets and can also load rule files. A target can be an application endpoint or an exporter exposing metrics on behalf of another system.

## PROMETHEUS-002 | Configuration reload

Prometheus can reload configuration without restarting when the relevant lifecycle mechanism is enabled and the new configuration is valid. Validate configuration before relying on an automated reload in production.

## PROMETHEUS-003 | Exporters

Exporters convert metrics from systems that do not expose Prometheus metrics directly. An exporter should produce stable, useful metrics and avoid exposing secrets or unbounded dimensions.

## PROMETHEUS-004 | Recording rules versus alerting rules

Recording rules save computed time series for later use. Alerting rules create alert state when a PromQL expression returns matching series. Both are defined in YAML rule files and evaluated periodically.

## PROMETHEUS-005 | Alert `for` duration

An alert rule can use a `for` duration so a condition must remain true for a period before the alert becomes firing. This helps reduce noise from transient spikes, but it also delays notification.

## PROMETHEUS-006 | Alert labels and annotations

Labels classify alerts for routing and grouping; annotations provide human-readable context. Avoid putting raw high-cardinality identifiers into alert labels unless the alerting system can safely handle them.

## PROMETHEUS-007 | Common infrastructure alerts

Useful alerts include API error-rate increases, sustained high latency, database connection saturation, disk-space exhaustion, missing scrape targets and repeated Pod restarts. Each alert should have a meaningful response action, not merely a scary threshold.

## PROMETHEUS-008 | Missing data versus zero

No sample is not automatically equivalent to a metric value of zero. Queries and alerts should distinguish a genuinely observed zero from a target that stopped exposing data. This is especially important for scrape failures.

## PROMETHEUS-009 | Promtool validation

The `promtool` utility can validate rule files. Automated CI should reject malformed rules before deployment to avoid losing observability at runtime.

## PROMETHEUS-010 | Alert fatigue control

Start with alerts tied to user impact or actionable resource exhaustion. Add low-level diagnostic alerts as annotations or dashboards rather than sending every metric anomaly to an on-call channel.

## Source basis

The corpus is intentionally grounded in official project documentation. Use the URLs below for version-specific verification rather than copying volatile version numbers into model-training text.

- https://prometheus.io/docs/concepts/
- https://prometheus.io/docs/prometheus/latest/configuration/configuration/
- https://prometheus.io/docs/prometheus/latest/configuration/recording_rules/
- https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/

## Versioning note

This document should be treated as a versioned corpus artifact. Record the source snapshot date and any later updates in the dataset manifest. Version-sensitive benchmark questions should state the relevant software version explicitly.
