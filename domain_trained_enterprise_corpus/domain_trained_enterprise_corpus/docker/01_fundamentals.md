# Docker - 01 Fundamentals

> Corpus role: enterprise infrastructure knowledge for the Domain-Trained Enterprise RAG capstone. Content is intentionally modular so it can be chunked into independent retrieval units.

## Topic scope

Topic: **docker**

Each section below is an atomic knowledge unit. Preserve the heading and topic metadata during ingestion so retrieval results retain their semantic context.

## DOCKER-001 | Docker images, containers, registries and layers

Docker separates the application artifact from the running process. An image is a read-only template containing application files, runtime dependencies and metadata. A container is a runnable instance of an image with its own writable layer and process lifecycle. Images are normally pulled from or pushed to registries. A registry stores image content; a repository is a named collection of related images; a tag is a human-readable reference that can move over time. A digest identifies image content immutably.

For enterprise RAG, this distinction matters because the service should be reproducible: the same image digest should run the same code even when a mutable tag such as `latest` changes.

## DOCKER-002 | Dockerfile fundamentals

A Dockerfile describes how an image is built. Typical instructions include `FROM`, `WORKDIR`, `COPY`, `RUN`, `ENV`, `EXPOSE`, `USER` and `CMD`. `RUN` executes during image construction; `CMD` defines the default command when a container starts; `ENTRYPOINT` can make the container behave like a dedicated executable. Keep build-time dependencies out of the final runtime image when possible.

## DOCKER-003 | Layering and build cache

Docker builds are layered. Instructions that remain unchanged can reuse cached layers, while a change to an earlier step can invalidate later steps. Put stable dependency installation steps before frequently changing application source when this improves cache reuse. Do not treat the cache as a security boundary: build inputs, package sources and secret-handling must still be controlled.

## DOCKER-004 | Container lifecycle

Common lifecycle operations are `docker create`, `docker start`, `docker stop`, `docker restart`, `docker kill`, `docker rm` and `docker inspect`. `docker run` is a convenience that creates and starts a container. A stopped container preserves its writable layer until removed. A robust service should be stateless when possible and store durable state in an explicit volume or external service.

## DOCKER-005 | Environment variables and configuration

Environment variables are useful for non-secret configuration and for changing runtime behavior between environments. They should not be treated as a secure secret store. For credentials and keys, use a proper secret mechanism appropriate to the deployment environment. Avoid baking environment-specific secrets into images because images are portable artifacts and may be copied to other registries or systems.

## DOCKER-006 | Volumes and bind mounts

A named volume is managed by Docker and is suited to durable container data. A bind mount maps a host path into the container and is useful for development or when the host deliberately owns the files. Containers should not rely on their writable layer for database durability. For PostgreSQL, persistent storage should be designed independently of the application container lifecycle.

## DOCKER-007 | Docker networking basics

A container can connect to networks and communicate with other containers according to the network configuration. In Compose, services on the default project network can discover one another by service name. This is important for an enterprise stack: an API container can reach PostgreSQL using its Compose service name instead of relying on a hard-coded container IP address.

## DOCKER-008 | Port publishing

A container port does not automatically become reachable from the host. `-p HOST:CONTAINER` publishes a container port to the host. Binding to `127.0.0.1` can limit host exposure, while binding to `0.0.0.0` exposes the port on host interfaces. In production, avoid publishing internal database and metrics ports unless there is a clear requirement.

## DOCKER-009 | Health checks versus process state

A running container only proves that its main process has not exited. A health check can test whether the service is actually ready to serve requests. Health checks should be cheap, deterministic and representative of the service dependency contract. They should not accidentally trigger expensive model loading or database migrations repeatedly.

## DOCKER-010 | Reproducible image identity

For controlled deployments, prefer immutable image digests or pinned release versions over floating tags. Tags are convenient human references but can move. Reproducibility requires recording the exact application version, base image version, dependency lock state and, for high-assurance deployments, the final image digest.

## Source basis

The corpus is intentionally grounded in official project documentation. Use the URLs below for version-specific verification rather than copying volatile version numbers into model-training text.

- https://docs.docker.com/get-started/docker-overview/
- https://docs.docker.com/compose/how-tos/networking/
- https://docs.docker.com/build/building/best-practices/
- https://docs.docker.com/engine/security/

## Versioning note

This document should be treated as a versioned corpus artifact. Record the source snapshot date and any later updates in the dataset manifest. Version-sensitive benchmark questions should state the relevant software version explicitly.
