# Helm - 03 Enterprise Chart Security

> Corpus role: enterprise infrastructure knowledge for the Domain-Trained Enterprise RAG capstone. Content is intentionally modular so it can be chunked into independent retrieval units.

## Topic scope

Topic: **helm**

Each section below is an atomic knowledge unit. Preserve the heading and topic metadata during ingestion so retrieval results retain their semantic context.

## HELM-001 | RBAC least privilege in charts

A chart should create only the Kubernetes permissions it needs. Avoid cluster-wide RBAC permissions when namespace-scoped access is enough. Review service accounts, Roles, ClusterRoles and bindings as part of chart security.

## HELM-002 | Pod security defaults

Charts should avoid privileged containers, host networking and unnecessary hostPath mounts. Run as a non-root user where supported, drop unnecessary Linux capabilities and use read-only filesystems when compatible with the workload.

## HELM-003 | Resource requests and limits

Define resource requests so the scheduler can place workloads predictably. Limits can protect a cluster from runaway resource usage, but setting arbitrary low limits can cause legitimate workloads to fail. RAG model services need realistic memory and CPU/GPU budgets derived from actual measurement.

## HELM-004 | Probes

Readiness probes control whether a pod should receive traffic; liveness probes detect a process that needs restart; startup probes can protect slow-starting applications from premature liveness failures. Probe endpoints should reflect actual application state.

## HELM-005 | Network policies

Where network policies are supported, restrict which pods can communicate. For example, only the API may need access to PostgreSQL, while an ingress controller should not need direct database access. Policy design should follow the service graph rather than broad allow-all rules.

## HELM-006 | Ingress and TLS boundaries

A chart can expose the API through an Ingress or gateway layer. TLS termination location should be explicit, and secrets containing certificates must be handled securely. Internal service traffic should not be assumed to be protected merely because edge traffic uses HTTPS.

## HELM-007 | ConfigMap versus Secret

Non-sensitive configuration belongs in ConfigMaps; confidential values belong in Secrets or an external secret store. Treat Kubernetes Secrets as sensitive data but do not confuse base64 encoding with encryption by itself.

## HELM-008 | Supply-chain controls

Pin image versions or digests, review chart dependencies and use trusted registries. The chart itself is executable configuration: a malicious template can create powerful resources. Review third-party charts before installing them into an enterprise cluster.

## HELM-009 | Pod disruption and availability

Production services may need disruption budgets and multiple replicas to tolerate planned maintenance. Availability mechanisms only help when the underlying application is stateless enough to support multiple instances.

## HELM-010 | Helm in CI/CD

CI should lint and render charts, validate manifests, and deploy a specific chart version with explicit values. Production credentials should be available only to the deployment stage that requires them.

## Source basis

The corpus is intentionally grounded in official project documentation. Use the URLs below for version-specific verification rather than copying volatile version numbers into model-training text.

- https://docs.helm.sh/docs/topics/charts/
- https://docs.helm.sh/docs/v3/chart_template_guide/values_files/
- https://docs.helm.sh/docs/helm/helm_upgrade/
- https://docs.helm.sh/docs/v3/helm/helm_template/

## Versioning note

This document should be treated as a versioned corpus artifact. Record the source snapshot date and any later updates in the dataset manifest. Version-sensitive benchmark questions should state the relevant software version explicitly.
