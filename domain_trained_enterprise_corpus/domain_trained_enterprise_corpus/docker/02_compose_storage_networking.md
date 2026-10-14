# Docker - 02 Compose Storage Networking

> Corpus role: enterprise infrastructure knowledge for the Domain-Trained Enterprise RAG capstone. Content is intentionally modular so it can be chunked into independent retrieval units.

## Topic scope

Topic: **docker**

Each section below is an atomic knowledge unit. Preserve the heading and topic metadata during ingestion so retrieval results retain their semantic context.

## DOCKER-001 | Compose as a multi-service definition

Docker Compose lets a project describe multiple services, networks and volumes in one YAML model. An enterprise RAG stack can use Compose for local integration: FastAPI API, PostgreSQL metadata store, NGINX edge proxy, Prometheus metrics, and optional model/retrieval workers can be defined as separate services with explicit dependencies and health checks.

## DOCKER-002 | Service discovery in Compose

Compose creates a project network by default, and each service can be reached by its service name on that network. This avoids hard-coding container IP addresses. If a service is called `postgres`, an API container can generally use `postgres` as the database host. IP addresses can change when containers are recreated.

## DOCKER-003 | Dependency ordering and readiness

A dependency declaration can control startup order, but startup order is not the same as application readiness. A database process can be listening before initialization is complete. Prefer explicit health checks and application-side retry logic for dependencies that may take time to become ready.

## DOCKER-004 | Persistent PostgreSQL storage

A PostgreSQL container should use a persistent volume for its data directory. The application container should not share PostgreSQL's internal data files through a bind mount unless there is a deliberate reason and the host filesystem semantics are understood. Backups must be tested separately; a volume is not a backup.

## DOCKER-005 | Compose configuration separation

Keep environment-specific values in separate Compose override files or environment inputs. Avoid duplicating full service definitions across files. Separate safe defaults from secrets and document required variables. Never commit production credentials to the repository.

## DOCKER-006 | Internal versus external networks

Create separate networks when you want to reduce unnecessary connectivity. For example, an edge-facing NGINX service can be on an ingress network and an internal API/database network. Only the components that must communicate should join both. Network segmentation reduces accidental exposure and makes trust boundaries explicit.

## DOCKER-007 | Logging and troubleshooting Compose

Use `docker compose ps`, `docker compose logs`, `docker compose exec`, and `docker compose config` when debugging. `docker compose config` is particularly useful for checking the fully resolved configuration. When a request fails, trace the path: client -> NGINX -> API -> retrieval/model service -> PostgreSQL/vector store, and inspect logs at each boundary.

## DOCKER-008 | Resource limits and enterprise behavior

Local Compose is useful for reproducibility, but resource constraints should be made explicit when testing latency or memory. CPU, RAM and GPU availability can differ across environments. Never claim a latency result from a constrained development container is representative of production without recording the runtime environment.

## DOCKER-009 | Common networking failure modes

A service can fail because it uses `localhost` when it should use another service name, because the two containers are not attached to the same network, because the target port is confused with the published host port, or because an application is binding only to `127.0.0.1` inside the container. Debug from inside the calling container rather than assuming host-level connectivity is equivalent.

## DOCKER-010 | Compose and the RAG project

A good local stack keeps retrieval, reranking and generation components independently replaceable. That makes ablations easier: the same API and monitoring layer can call a generic retriever, a fine-tuned retriever, or the final trained pipeline while infrastructure remains constant.

## Source basis

The corpus is intentionally grounded in official project documentation. Use the URLs below for version-specific verification rather than copying volatile version numbers into model-training text.

- https://docs.docker.com/get-started/docker-overview/
- https://docs.docker.com/compose/how-tos/networking/
- https://docs.docker.com/build/building/best-practices/
- https://docs.docker.com/engine/security/

## Versioning note

This document should be treated as a versioned corpus artifact. Record the source snapshot date and any later updates in the dataset manifest. Version-sensitive benchmark questions should state the relevant software version explicitly.
