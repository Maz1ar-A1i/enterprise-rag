# Nginx - 02 Tls Load Balancing

> Corpus role: enterprise infrastructure knowledge for the Domain-Trained Enterprise RAG capstone. Content is intentionally modular so it can be chunked into independent retrieval units.

## Topic scope

Topic: **nginx**

Each section below is an atomic knowledge unit. Preserve the heading and topic metadata during ingestion so retrieval results retain their semantic context.

## NGINX-001 | TLS termination

NGINX can terminate HTTPS by presenting a server certificate and using the corresponding private key. The private key must be protected. Modern configurations should use current TLS versions and strong configuration rather than legacy protocols and ciphers.

## NGINX-002 | TLS certificate chain

A certificate proves the server identity within a trust model, but clients also need the appropriate chain. Incorrect certificate chains often produce browser or API-client validation failures even when the leaf certificate itself looks valid.

## NGINX-003 | HTTP load balancing

NGINX can distribute HTTP/HTTPS traffic across an upstream group. Round-robin is a common basic algorithm; additional algorithms and capabilities depend on the NGINX edition and configuration. Load balancing can improve resource utilization and fault tolerance when application instances are compatible with horizontal scaling.

## NGINX-004 | Health checks

Open-source NGINX provides passive health-check behavior for upstream failures; NGINX Plus adds active health-check capabilities. Do not describe active health checks as an Open Source feature without qualification.

## NGINX-005 | Connection reuse

Keepalive connections can reduce repeated connection setup and TLS handshake overhead. Proper connection reuse can improve latency and reduce CPU use, but connection pool sizes should reflect upstream limits.

## NGINX-006 | Rate limiting

Request rate limiting can protect expensive endpoints from abuse. A RAG query endpoint may deserve different limits from lightweight health endpoints. Limits should be aligned with expected request cost and should return predictable errors to clients.

## NGINX-007 | Caching

NGINX can cache content, but caching dynamic answers requires careful consideration of identity, authorization, freshness and data sensitivity. A knowledge-base change can make a previously correct cached answer stale.

## NGINX-008 | Upstream failure isolation

When one backend is unhealthy, traffic should be steered according to the configured load-balancing and health behavior. Clients still need sensible retry policies because retries can amplify load during an outage.

## NGINX-009 | NGINX in Kubernetes

In Kubernetes, NGINX may run as an Ingress controller or as a standalone reverse proxy service. The deployment model changes how certificates, Service discovery and configuration are managed. Keep the conceptual boundary clear: Kubernetes routes traffic to the NGINX workload; NGINX applies HTTP proxy rules.

## NGINX-010 | Observability at the edge

Record request count, response status and upstream latency. Exporting these values to the monitoring system lets the team distinguish edge latency from application, retrieval and generation latency.

## Source basis

The corpus is intentionally grounded in official project documentation. Use the URLs below for version-specific verification rather than copying volatile version numbers into model-training text.

- https://docs.nginx.com/nginx/admin-guide/web-server/reverse-proxy
- https://docs.nginx.com/nginx/admin-guide/load-balancer/http-load-balancer/
- https://docs.nginx.com/nginx/admin-guide/security-controls/terminating-ssl-http/
- https://docs.nginx.com/nginx/admin-guide/load-balancer/http-health-check/

## Versioning note

This document should be treated as a versioned corpus artifact. Record the source snapshot date and any later updates in the dataset manifest. Version-sensitive benchmark questions should state the relevant software version explicitly.
