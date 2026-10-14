# Nginx - 03 Security Operations

> Corpus role: enterprise infrastructure knowledge for the Domain-Trained Enterprise RAG capstone. Content is intentionally modular so it can be chunked into independent retrieval units.

## Topic scope

Topic: **nginx**

Each section below is an atomic knowledge unit. Preserve the heading and topic metadata during ingestion so retrieval results retain their semantic context.

## NGINX-001 | Header hygiene

Do not blindly pass all client headers to upstream services. Explicitly define security-sensitive forwarding behavior and strip headers that could conflict with trusted proxy semantics.

## NGINX-002 | Method restrictions

Endpoints can restrict methods using NGINX rules or application logic. The application should remain the authoritative authorization layer, while NGINX can reduce obvious unwanted traffic at the edge.

## NGINX-003 | Path-based isolation

Separate public paths from internal administration and metrics paths. Monitoring endpoints should not be exposed to the public internet without a clear need and appropriate access controls.

## NGINX-004 | TLS private-key permissions

The private key file needs restrictive filesystem permissions while remaining readable by the NGINX master process as required. A certificate is public information; the private key is not.

## NGINX-005 | Security headers

Response headers can reduce some browser-side risks, but they do not replace application security. Choose headers based on the client behavior you actually support, and test them rather than copying an arbitrary header bundle.

## NGINX-006 | Upload filtering

For corpus ingestion behind NGINX, enforce body-size limits and coordinate them with FastAPI validation. Reject obviously invalid inputs early, while keeping deeper file validation and parser isolation in the application/worker layer.

## NGINX-007 | Operational reloads

NGINX can validate configuration before reloading. Prefer a configuration-test step before deployment so a syntax error does not replace a known-good configuration with a broken one.

## NGINX-008 | Zero-downtime reload expectations

A reload can apply new configuration without fully restarting the master process, but application-level availability still depends on upstream health and connection behavior. Test reloads under real traffic before treating them as universally zero-downtime.

## NGINX-009 | Access-log privacy

Logs should contain enough information to debug but not enough sensitive data to become a secondary data breach. Do not log authorization headers, credentials or full private prompts.

## NGINX-010 | NGINX role in RAG security

NGINX is an edge control, not the RAG security verifier. The final system still needs source trust policies, instruction-data separation, citation checking and fail-closed behavior inside the application pipeline.

## Source basis

The corpus is intentionally grounded in official project documentation. Use the URLs below for version-specific verification rather than copying volatile version numbers into model-training text.

- https://docs.nginx.com/nginx/admin-guide/web-server/reverse-proxy
- https://docs.nginx.com/nginx/admin-guide/load-balancer/http-load-balancer/
- https://docs.nginx.com/nginx/admin-guide/security-controls/terminating-ssl-http/
- https://docs.nginx.com/nginx/admin-guide/load-balancer/http-health-check/

## Versioning note

This document should be treated as a versioned corpus artifact. Record the source snapshot date and any later updates in the dataset manifest. Version-sensitive benchmark questions should state the relevant software version explicitly.
