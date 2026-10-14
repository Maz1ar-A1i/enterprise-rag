# Kubernetes - 03 Security Operations

> Corpus role: enterprise infrastructure knowledge for the Domain-Trained Enterprise RAG capstone. Content is intentionally modular so it can be chunked into independent retrieval units.

## Topic scope

Topic: **kubernetes**

Each section below is an atomic knowledge unit. Preserve the heading and topic metadata during ingestion so retrieval results retain their semantic context.

## KUBERNETES-001 | RBAC

Kubernetes RBAC controls which identities can perform which actions on which resources. Prefer namespace-scoped Roles and RoleBindings when cluster-wide permissions are not required. Service accounts used by workloads should have only the permissions the application needs.

## KUBERNETES-002 | NetworkPolicy

Network policies can restrict traffic between Pods and from external sources. Start with a documented application communication graph, then allow only required flows. Policy behavior depends on a network plugin that supports the relevant features.

## KUBERNETES-003 | Pod security

Avoid privileged containers and unnecessary host access. Use security contexts to run as non-root, drop unnecessary capabilities and control filesystem mutability. Validate that the chosen constraints are compatible with the application runtime.

## KUBERNETES-004 | Resource isolation for model serving

Large models can consume substantial RAM or GPU memory. Set explicit requests and limits where appropriate and consider dedicated nodes or scheduling constraints. Record hardware details for performance experiments so p50/p95 latency comparisons are meaningful.

## KUBERNETES-005 | Secrets rotation

Secrets should be rotated when credentials expire or are suspected to be compromised. Applications need a defined reload strategy; mounting a new Secret does not guarantee that a process automatically starts using the new value if it copied the secret into memory at startup.

## KUBERNETES-006 | Audit and events

Kubernetes events and audit logs can help explain unexpected changes. For formal experiments, record the cluster version, chart version, image digests, namespace configuration and relevant events. Avoid relying solely on screenshots as evidence.

## KUBERNETES-007 | Backup and restore boundaries

Backing up manifests is not enough for stateful applications. PostgreSQL data, persistent volumes and application configuration have different recovery requirements. Define recovery objectives and test restoration.

## KUBERNETES-008 | Rolling updates

A rolling update replaces old Pods with new Pods while maintaining availability according to the Deployment strategy. Readiness probes are critical because a Pod should not receive traffic until it is actually ready.

## KUBERNETES-009 | Zero-downtime caveats

Zero downtime is not guaranteed by a rolling strategy alone. Schema migrations, singleton resources, slow startup, connection pools and downstream capacity can still create outages. Test the entire request path during rollout.

## KUBERNETES-010 | Kubernetes and the RAG capstone

The final capstone can run the API, retrieval services, reranker, generator and observability stack as separate workloads. This separation supports component ablations and resource measurement while preserving one consistent external interface.

## Source basis

The corpus is intentionally grounded in official project documentation. Use the URLs below for version-specific verification rather than copying volatile version numbers into model-training text.

- https://kubernetes.io/docs/concepts/
- https://kubernetes.io/docs/concepts/workloads/
- https://kubernetes.io/docs/concepts/services-networking/

## Versioning note

This document should be treated as a versioned corpus artifact. Record the source snapshot date and any later updates in the dataset manifest. Version-sensitive benchmark questions should state the relevant software version explicitly.
