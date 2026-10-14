# Kubernetes - 02 Networking Scheduling Health

> Corpus role: enterprise infrastructure knowledge for the Domain-Trained Enterprise RAG capstone. Content is intentionally modular so it can be chunked into independent retrieval units.

## Topic scope

Topic: **kubernetes**

Each section below is an atomic knowledge unit. Preserve the heading and topic metadata during ingestion so retrieval results retain their semantic context.

## KUBERNETES-001 | Kubernetes network model

Kubernetes gives Pods cluster-wide IP addresses and supports Pod-to-Pod communication through the cluster network, subject to network policies or other controls. Containers in a Pod share the Pod network namespace and can communicate through `localhost`.

## KUBERNETES-002 | Service discovery

Services are discoverable through cluster DNS. Applications should usually call the Service name rather than storing Pod IPs. This is fundamental for reliable communication between a FastAPI API and internal retrieval services.

## KUBERNETES-003 | Service types

ClusterIP exposes a Service inside the cluster. NodePort and LoadBalancer provide other exposure models depending on the environment. Ingress or gateway layers can provide HTTP routing and TLS termination for multiple services.

## KUBERNETES-004 | Ingress routing

An Ingress defines HTTP(S) routing rules for external access through an Ingress controller. The controller is the component that actually implements the routing. Do not assume that creating an Ingress object by itself creates a load balancer.

## KUBERNETES-005 | Readiness, liveness and startup probes

Readiness indicates whether a Pod should receive traffic. Liveness indicates whether a container should be restarted. Startup probes can protect slow-starting applications from liveness checks that would otherwise restart them during initialization.

## KUBERNETES-006 | Resource requests and limits

Requests influence scheduling and communicate the expected resource reservation. Limits constrain usage according to the resource type. Poorly chosen values can either waste capacity or cause throttling/OOM failures. Model services should be tested under realistic memory pressure.

## KUBERNETES-007 | Horizontal scaling

A Deployment can run multiple replicas. Horizontal Pod Autoscaling can adjust replica counts based on metrics. CPU utilization is not always a useful signal for a RAG service; request rate, queue depth or latency may better represent demand if custom metrics are available.

## KUBERNETES-008 | Scheduling constraints

Node selectors, affinity and taints/tolerations can influence where Pods run. GPU workloads often require node labels, resource requests and device-plugin support. Scheduling rules should be justified by workload requirements rather than copied blindly.

## KUBERNETES-009 | Graceful termination

When a Pod is terminated, applications should close connections and finish safe in-flight work before the termination deadline. HTTP servers should stop accepting new work while allowing current requests to finish within a bounded time.

## KUBERNETES-010 | Failure debugging path

For a failing Service, check `kubectl get`, `kubectl describe`, endpoints, Pod status, events and logs. Start from the client request and trace through Service selection, Pod readiness, container port binding and application logs.

## Source basis

The corpus is intentionally grounded in official project documentation. Use the URLs below for version-specific verification rather than copying volatile version numbers into model-training text.

- https://kubernetes.io/docs/concepts/
- https://kubernetes.io/docs/concepts/workloads/
- https://kubernetes.io/docs/concepts/services-networking/

## Versioning note

This document should be treated as a versioned corpus artifact. Record the source snapshot date and any later updates in the dataset manifest. Version-sensitive benchmark questions should state the relevant software version explicitly.
