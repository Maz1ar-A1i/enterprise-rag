# Nginx - 01 Reverse Proxy Http

> Corpus role: enterprise infrastructure knowledge for the Domain-Trained Enterprise RAG capstone. Content is intentionally modular so it can be chunked into independent retrieval units.

## Topic scope

Topic: **nginx**

Each section below is an atomic knowledge unit. Preserve the heading and topic metadata during ingestion so retrieval results retain their semantic context.

## NGINX-001 | NGINX request processing

NGINX can act as a web server, reverse proxy and gateway. In reverse-proxy mode it receives a client request, selects an upstream server, forwards the request, receives the upstream response and returns it to the client. This makes NGINX a natural edge layer for a FastAPI enterprise service.

## NGINX-002 | server and location blocks

NGINX configuration is organized into contexts such as `http`, `server` and `location`. A `server` block selects a virtual host based on properties such as the requested hostname and port. A `location` block applies routing and proxy behavior to matching request paths.

## NGINX-003 | proxy_pass

`proxy_pass` defines where an upstream request is sent. Be precise about URI rewriting behavior because the placement of a trailing slash can change the resulting path forwarded to the backend. Test actual request paths rather than relying on memory.

## NGINX-004 | Forwarded headers

Reverse proxies commonly forward information about the original request, such as host and client address, through headers. Backend frameworks must be configured carefully to trust these headers only from known proxies. Blindly trusting client-supplied forwarding headers can create security problems.

## NGINX-005 | Buffering and timeouts

NGINX can buffer upstream responses and has configurable connect, send and read timeouts. Increasing timeouts may help with legitimately long operations, but it also holds connections longer and can reduce concurrency. Tune timeouts with measured workload data.

## NGINX-006 | Static content and API traffic

NGINX can serve static content directly while proxying API requests to FastAPI. This can reduce application work and make caching behavior explicit. Dynamic RAG responses usually should not be cached like immutable JavaScript assets unless the cache key and freshness semantics are carefully defined.

## NGINX-007 | Client body limits

The proxy can enforce request-size limits. This is useful for corpus upload APIs because a very large file can consume memory, disk or parser time. The limit should align with the backend's own validation and with expected operational requirements.

## NGINX-008 | Request logging

Access logs provide a high-level record of client requests. Log route, status, latency and upstream timing where useful, but do not put full confidential prompts or authorization tokens into access logs.

## NGINX-009 | HTTP protocol behavior

HTTP method, headers and status codes form a contract between client, proxy and application. When debugging, compare the request at both sides of the proxy so header or path rewriting does not create hidden differences.

## NGINX-010 | 502, 504 and 499-style failures

A 502 often indicates an invalid or failed upstream response; a timeout-related 504 usually means the proxy did not receive an upstream response in time. Client-aborted requests may appear differently depending on NGINX configuration. Correlate the status with upstream logs and timing fields before changing configuration.

## Source basis

The corpus is intentionally grounded in official project documentation. Use the URLs below for version-specific verification rather than copying volatile version numbers into model-training text.

- https://docs.nginx.com/nginx/admin-guide/web-server/reverse-proxy
- https://docs.nginx.com/nginx/admin-guide/load-balancer/http-load-balancer/
- https://docs.nginx.com/nginx/admin-guide/security-controls/terminating-ssl-http/
- https://docs.nginx.com/nginx/admin-guide/load-balancer/http-health-check/

## Versioning note

This document should be treated as a versioned corpus artifact. Record the source snapshot date and any later updates in the dataset manifest. Version-sensitive benchmark questions should state the relevant software version explicitly.
