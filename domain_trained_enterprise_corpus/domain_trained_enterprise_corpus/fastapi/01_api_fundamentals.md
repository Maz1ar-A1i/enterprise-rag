# Fastapi - 01 Api Fundamentals

> Corpus role: enterprise infrastructure knowledge for the Domain-Trained Enterprise RAG capstone. Content is intentionally modular so it can be chunked into independent retrieval units.

## Topic scope

Topic: **fastapi**

Each section below is an atomic knowledge unit. Preserve the heading and topic metadata during ingestion so retrieval results retain their semantic context.

## FASTAPI-001 | FastAPI application object

A FastAPI application is created from the `FastAPI` class. Route handlers are declared with decorators such as `@app.get()` and `@app.post()`. FastAPI uses Python type annotations and Pydantic models to validate and document API input and output. The framework exposes an OpenAPI schema and interactive documentation for the API.

## FASTAPI-002 | Path operations and HTTP methods

A path operation combines a URL path with an HTTP method. Common REST-style mappings use GET for retrieval, POST for creation or processing, PUT for replacement, PATCH for partial updates and DELETE for deletion. The project should choose semantics consistently and document exceptions when an endpoint represents an action such as `/evaluate` or `/search`.

## FASTAPI-003 | Request path, query and body parameters

Path parameters are part of the route, query parameters are supplied in the URL query string, and request bodies carry structured payloads. Use Pydantic models for JSON request bodies when structure and validation matter. Keep path and query parameters narrow enough that API behavior is predictable.

## FASTAPI-004 | Response models

Response models provide a contract for output. They can validate returned data and make OpenAPI documentation useful to consumers. In a RAG service, explicit response models can separate the answer text, citations, retrieval metadata, refusal state and evaluation fields instead of returning an unstructured dictionary.

## FASTAPI-005 | Automatic OpenAPI documentation

FastAPI automatically generates an OpenAPI specification and interactive documentation. The documentation can be used by developers to understand endpoints and by automated tools to inspect the contract. This is an engineering advantage because API behavior and schema stay close to code.

## FASTAPI-006 | Async versus normal path functions

FastAPI supports both `async def` and normal `def` path functions. Asynchronous functions are useful when the work is I/O-bound and the libraries involved support asynchronous use. CPU-heavy model inference does not become faster simply because the endpoint is declared `async`; CPU-bound work may need a worker process, queue or other execution strategy.

## FASTAPI-007 | Status codes and error responses

Use meaningful HTTP status codes. Validation errors should be distinguishable from authentication failures, missing resources, dependency failures and internal server errors. Do not expose stack traces or sensitive infrastructure details to external clients.

## FASTAPI-008 | Application lifecycle

Initialization can include loading configuration, opening shared clients, preparing telemetry and constructing service objects. Expensive model loading should happen once per worker process where appropriate rather than on every request. Be explicit about whether state is local to one process or shared through an external service.

## FASTAPI-009 | API versioning

For an evolving enterprise service, version the public contract deliberately, for example through a URL prefix or another stable compatibility strategy. Do not silently change the shape of the answer or citation objects while evaluation and clients depend on the existing schema.

## FASTAPI-010 | Request correlation

Attach a request identifier or correlation identifier to logs and response metadata when appropriate. The same identifier should help connect an NGINX access log, FastAPI request log, retrieval trace and Prometheus metrics without storing sensitive user content.

## Source basis

The corpus is intentionally grounded in official project documentation. Use the URLs below for version-specific verification rather than copying volatile version numbers into model-training text.

- https://fastapi.tiangolo.com/tutorial/first-steps/
- https://fastapi.tiangolo.com/tutorial/dependencies/
- https://fastapi.tiangolo.com/tutorial/security/first-steps/
- https://fastapi.tiangolo.com/tutorial/background-tasks/

## Versioning note

This document should be treated as a versioned corpus artifact. Record the source snapshot date and any later updates in the dataset manifest. Version-sensitive benchmark questions should state the relevant software version explicitly.
