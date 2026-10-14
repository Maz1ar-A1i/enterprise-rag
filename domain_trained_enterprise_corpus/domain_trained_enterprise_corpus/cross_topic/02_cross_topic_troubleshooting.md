# Cross-Topic Enterprise RAG - 02 Cross Topic Troubleshooting

> Cross-topic knowledge is designed for multi-hop, troubleshooting and architecture questions.

## CROSS-001 | 502 from NGINX to FastAPI

Check NGINX upstream address and port, then test the upstream from the NGINX runtime. Confirm FastAPI is listening on the container interface, inspect the API process exit state and review container logs. Only after connectivity is confirmed should you investigate request-time behavior.

## CROSS-002 | FastAPI times out during generation

Measure whether time is spent before retrieval, in retrieval, in reranking, in model generation or while waiting for a database call. Check concurrency and resource pressure. Apply bounded timeouts and consider background workers or queueing for workloads that are not request/response sized.

## CROSS-003 | Kubernetes Service has no endpoints

Check the Service selector against Pod labels, then inspect Pod readiness. A healthy-looking Pod can still be excluded from a Service if labels do not match or readiness is false.

## CROSS-004 | Helm upgrade produces unhealthy Pods

Inspect the rendered manifest, release history, changed values, Pod events and logs. Compare the new image/configuration with the last successful revision. Do not assume Helm itself is the root cause.

## CROSS-005 | PostgreSQL query becomes slow

Capture the actual SQL and parameters, run `EXPLAIN`/`EXPLAIN ANALYZE` safely, inspect row estimates, indexes, locks and I/O. Check whether a recent schema or data distribution change invalidated an earlier assumption.

## CROSS-006 | Prometheus dashboard goes blank

Check whether the target is still being scraped, whether the metric name or labels changed, and whether the query expects a series that no longer exists. Distinguish a zero value from missing data.

## CROSS-007 | Docker container keeps restarting

Inspect the exit code and logs, then check environment variables, mounted files, permissions and dependency availability. If the process exits cleanly, inspect the command/entrypoint and container orchestration restart policy.

## CROSS-008 | Linux host is out of disk

Use `df` to identify the full filesystem, `du` to find large paths, and inspect container images, writable layers, logs, model files and database storage. Free space only after determining what the deleted or growing data represents.

## CROSS-009 | TLS works locally but fails through NGINX

Compare certificate hostname, certificate chain, protocol versions and NGINX certificate/key configuration. Test from an actual client path rather than only from localhost. Confirm the private key is readable by NGINX and that configuration reload succeeded.

## CROSS-010 | RAG answer cites irrelevant documents

Check the retrieval top-k, hybrid merge behavior, reranker ordering and chunk metadata. Evaluate whether the error is retrieval recall, reranker precision, query formulation or an out-of-date/contradictory source. The correction should be linked to the component that actually caused the miss.

