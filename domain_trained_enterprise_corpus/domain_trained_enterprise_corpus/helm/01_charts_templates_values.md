# Helm - 01 Charts Templates Values

> Corpus role: enterprise infrastructure knowledge for the Domain-Trained Enterprise RAG capstone. Content is intentionally modular so it can be chunked into independent retrieval units.

## Topic scope

Topic: **helm**

Each section below is an atomic knowledge unit. Preserve the heading and topic metadata during ingestion so retrieval results retain their semantic context.

## HELM-001 | Helm and Kubernetes

Helm is a package manager and templating tool for Kubernetes applications. A chart is a collection of files that describes a related set of Kubernetes resources. A release is an installed instance of a chart with a particular configuration.

## HELM-002 | Chart directory structure

A chart commonly contains `Chart.yaml`, `values.yaml`, a `templates/` directory and optional dependency or documentation files. `Chart.yaml` describes chart metadata and can declare dependencies. The `templates/` directory contains manifests that Helm renders into Kubernetes resources.

## HELM-003 | Chart.yaml versus values.yaml

`Chart.yaml` describes the chart itself: name, chart version, compatibility metadata and dependencies. `values.yaml` provides default configuration values for the templates. Users can override values during installation or upgrade.

## HELM-004 | Values precedence

Helm merges values from several sources. Chart defaults can be overridden by supplied values files and by command-line parameters such as `--set`; later or more specific inputs take precedence. This is useful for environment-specific settings but can also hide configuration if overused.

## HELM-005 | Template rendering

Helm templates are processed before Kubernetes receives the resulting manifests. Functions and conditional logic can choose which resources to emit. `helm template` renders locally and is useful for inspecting the generated YAML before installing it.

## HELM-006 | Named templates and helpers

Reusable template snippets should live in helper templates rather than being copied into every resource. This reduces duplication and makes naming conventions consistent across Deployments, Services, ConfigMaps and other resources.

## HELM-007 | Labels and annotations

Labels should be stable and semantically useful for selection, ownership and operations. Annotations hold non-selector metadata. Helm charts should use predictable labels for application name, instance, chart and managed-by information while allowing application-specific labels where appropriate.

## HELM-008 | Chart dependencies

A chart can declare other charts as dependencies. Dependencies should be versioned and reviewed like any other software dependency because they can change the resources and permissions installed in a cluster.

## HELM-009 | Secrets and values

Do not assume that putting a secret into `values.yaml` makes it secret. Helm values are configuration data and can appear in release metadata or CI logs depending on how they are supplied. Prefer external secret management when the platform provides it.

## HELM-010 | Compatibility metadata

Use chart metadata and Kubernetes version constraints where necessary to prevent installing a chart into an unsupported environment. Test the rendered manifests against the target cluster versions.

## Source basis

The corpus is intentionally grounded in official project documentation. Use the URLs below for version-specific verification rather than copying volatile version numbers into model-training text.

- https://docs.helm.sh/docs/topics/charts/
- https://docs.helm.sh/docs/v3/chart_template_guide/values_files/
- https://docs.helm.sh/docs/helm/helm_upgrade/
- https://docs.helm.sh/docs/v3/helm/helm_template/

## Versioning note

This document should be treated as a versioned corpus artifact. Record the source snapshot date and any later updates in the dataset manifest. Version-sensitive benchmark questions should state the relevant software version explicitly.
