# Docker - 03 Security Troubleshooting

> Corpus role: enterprise infrastructure knowledge for the Domain-Trained Enterprise RAG capstone. Content is intentionally modular so it can be chunked into independent retrieval units.

## Topic scope

Topic: **docker**

Each section below is an atomic knowledge unit. Preserve the heading and topic metadata during ingestion so retrieval results retain their semantic context.

## DOCKER-001 | Least privilege for containers

Run services as a non-root user when the service does not require root. Minimize installed packages and remove build-only tools from runtime images. Do not assume container isolation eliminates the need for operating-system permissions, filesystem restrictions or network controls.

## DOCKER-002 | Secrets in Docker builds

Build arguments and image layers should not be used as a convenient place to hide secrets. A value written into an image layer can survive even after the source line is removed. Use build secrets where supported for build-time credentials and inject runtime secrets through the deployment platform.

## DOCKER-003 | Image provenance and scanning

An enterprise build should record the base image, package set and application version. Image vulnerability scanning is useful but does not prove an application is secure. It complements code review, dependency pinning, minimal images, runtime hardening and network restrictions.

## DOCKER-004 | Troubleshooting HTTP 502 from a containerized API

A 502 at NGINX often means the proxy could not obtain a valid upstream response. Check that the API process is listening on the expected interface and container port, verify NGINX upstream configuration, test the upstream from inside the NGINX container, and inspect API logs. Do not fix a 502 by blindly increasing proxy timeouts before understanding whether the backend is reachable.

## DOCKER-005 | Troubleshooting slow model containers

Separate model-load time from request-time latency. Record startup duration, first-request latency, warm-request p50/p95 latency and resource usage. Model initialization may be dominated by weight loading, tokenizer setup or CPU memory pressure and should not be confused with retrieval or generation latency.

## DOCKER-006 | Container exit diagnosis

Start with `docker ps -a` and inspect the exited container. Check logs, exit code, configuration and mounted paths. Common causes include missing environment variables, wrong executable paths, permission errors, incompatible architecture, failed database connections and application exceptions.

## DOCKER-007 | RAG security boundary in containers

Do not give a retrieved document the ability to alter container commands or system configuration. Retrieved text is data, not authority. Keep model prompts, tool permissions, filesystem permissions and deployment credentials separate from knowledge-base content. This directly supports the project requirement to resist instruction-bearing and poisoned documents.

## DOCKER-008 | Auditability

Record image identifiers, deployment manifests, configuration versions and significant security changes. When a hidden benchmark exposes an attack, the team should be able to identify which container image and configuration produced the result.

## DOCKER-009 | Operational anti-patterns

Avoid running databases with random ephemeral storage, exposing every internal port, copying production secrets into local `.env` files, using `latest` for formal experiments, or changing an image manually inside a running container. Each practice makes reproducibility and incident investigation harder.

## DOCKER-010 | Corpus ingestion worker isolation

If corpus ingestion executes document parsing, conversion or code extraction, isolate that workload from the serving API. Untrusted documents can contain malicious payloads, enormous files or parser edge cases. A separate worker process or container with restricted permissions is safer than parsing arbitrary files inside the public API process.

## Source basis

The corpus is intentionally grounded in official project documentation. Use the URLs below for version-specific verification rather than copying volatile version numbers into model-training text.

- https://docs.docker.com/get-started/docker-overview/
- https://docs.docker.com/compose/how-tos/networking/
- https://docs.docker.com/build/building/best-practices/
- https://docs.docker.com/engine/security/

## Versioning note

This document should be treated as a versioned corpus artifact. Record the source snapshot date and any later updates in the dataset manifest. Version-sensitive benchmark questions should state the relevant software version explicitly.
