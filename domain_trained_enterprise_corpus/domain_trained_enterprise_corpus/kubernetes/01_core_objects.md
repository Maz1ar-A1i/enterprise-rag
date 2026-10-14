# Kubernetes - 01 Core Objects

> Corpus role: enterprise infrastructure knowledge for the Domain-Trained Enterprise RAG capstone. Content is intentionally modular so it can be chunked into independent retrieval units.

## Topic scope

Topic: **kubernetes**

Each section below is an atomic knowledge unit. Preserve the heading and topic metadata during ingestion so retrieval results retain their semantic context.

## KUBERNETES-001 | Kubernetes control model

Kubernetes is a declarative platform for managing containerized workloads and services. Users describe a desired state, and controllers work continuously to reconcile the cluster toward that state. This differs from a script that simply starts processes once.

## KUBERNETES-002 | Pod

A Pod is the smallest deployable compute object in Kubernetes and represents one or more containers that share networking and selected storage namespaces. Containers in the same Pod communicate over `localhost`. Pods are generally replaceable rather than treated as permanent machines.

## KUBERNETES-003 | Deployment

A Deployment manages a set of replicated Pods and supports controlled updates. It specifies the desired number of replicas and the Pod template. Controllers create or replace Pods until the observed state matches the desired state.

## KUBERNETES-004 | Service

A Service provides a stable network abstraction for accessing a set of Pods selected by labels. Pods can change IP addresses as they are recreated, while the Service provides a stable endpoint for clients inside the cluster.

## KUBERNETES-005 | ConfigMap

ConfigMaps store non-sensitive configuration data. They can be exposed as environment variables or mounted as files. Changing a ConfigMap does not automatically mean an application reloads the new values; the application and deployment strategy must define how configuration changes take effect.

## KUBERNETES-006 | Secret

Secrets are intended for sensitive configuration such as credentials or tokens. They are not a substitute for an external secret management system, and base64 encoding is not equivalent to encryption. Access to Secrets should be restricted with RBAC.

## KUBERNETES-007 | Namespace

Namespaces partition names and can support policy and access-control boundaries. They are useful for separating environments or teams, but a namespace alone is not a complete security boundary.

## KUBERNETES-008 | Labels and selectors

Labels are key-value metadata used to identify and group resources. Selectors connect controllers and Services to the Pods they manage. A label mismatch can cause a Service to have no endpoints or a Deployment to fail to manage the intended Pods.

## KUBERNETES-009 | Jobs and CronJobs

Jobs run work to completion; CronJobs create Jobs on a schedule. They are suitable for one-off or periodic tasks such as re-indexing, backup verification or offline evaluation. Long-running API servers generally belong in Deployments instead.

## KUBERNETES-010 | Persistent volumes

Persistent storage separates application lifecycle from data lifecycle. PersistentVolumeClaims express storage requirements and can be bound to storage provided by the cluster. Stateful services require careful storage, backup and failover design; a Pod restart does not restore lost data by itself.

## Source basis

The corpus is intentionally grounded in official project documentation. Use the URLs below for version-specific verification rather than copying volatile version numbers into model-training text.

- https://kubernetes.io/docs/concepts/
- https://kubernetes.io/docs/concepts/workloads/
- https://kubernetes.io/docs/concepts/services-networking/

## Versioning note

This document should be treated as a versioned corpus artifact. Record the source snapshot date and any later updates in the dataset manifest. Version-sensitive benchmark questions should state the relevant software version explicitly.
