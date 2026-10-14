# Helm - 02 Release Upgrade Rollback

> Corpus role: enterprise infrastructure knowledge for the Domain-Trained Enterprise RAG capstone. Content is intentionally modular so it can be chunked into independent retrieval units.

## Topic scope

Topic: **helm**

Each section below is an atomic knowledge unit. Preserve the heading and topic metadata during ingestion so retrieval results retain their semantic context.

## HELM-001 | Helm install

`helm install` creates a release from a chart and values. Use explicit release names and namespaces in automation. For production, validate the rendered manifests and values before installation.

## HELM-002 | Helm upgrade

`helm upgrade` changes an existing release to a new chart/configuration state. Values can be supplied with `-f/--values` or `--set` and related options. The order of multiple values files matters because later files can override earlier ones.

## HELM-003 | Rollback mindset

A rollback returns a release to a previous known revision. Rollbacks are safest when application images, configuration and schema changes are compatible with the older release. A database migration that cannot be reversed can make an otherwise successful Helm rollback unsafe.

## HELM-004 | Atomic deployment behavior

Helm supports options that make an upgrade/install wait for resources and roll back on failure. This is useful when the team wants the release to return to a previous state instead of remaining partially upgraded, but it does not make external side effects transactional.

## HELM-005 | Dry-run and validation

Use `helm template`, `helm install --dry-run` or equivalent validation paths to inspect generated manifests and catch obvious errors before changing the cluster. Local template rendering does not perform every server-side API validation, so cluster-level validation is still important.

## HELM-006 | Release history

Helm keeps release revisions that can be inspected and used for rollback. Treat release history as operational evidence: record which chart version and values produced a given production state.

## HELM-007 | Hooks

Helm hooks can run jobs at lifecycle points such as pre-install or post-upgrade. Hooks are powerful but easy to misuse. A hook that changes external state can make a rollback incomplete, and hook resources may require explicit cleanup or policies.

## HELM-008 | Upgrade failure diagnosis

When an upgrade fails, inspect Helm status/history and the Kubernetes resources that were changed. A failing readiness probe may be caused by application startup, configuration, image pull issues, permissions or an upstream dependency rather than Helm itself.

## HELM-009 | Chart testing

A chart should be tested with representative values files, rendered manifest validation and deployment smoke tests. Maintain at least one minimal configuration and one production-like configuration to catch assumptions that only appear with optional features enabled.

## HELM-010 | Release reproducibility

Store the exact chart version, Git commit, values files and image digest used for a formal deployment. This makes it possible to reproduce or audit a release months later.

## Source basis

The corpus is intentionally grounded in official project documentation. Use the URLs below for version-specific verification rather than copying volatile version numbers into model-training text.

- https://docs.helm.sh/docs/topics/charts/
- https://docs.helm.sh/docs/v3/chart_template_guide/values_files/
- https://docs.helm.sh/docs/helm/helm_upgrade/
- https://docs.helm.sh/docs/v3/helm/helm_template/

## Versioning note

This document should be treated as a versioned corpus artifact. Record the source snapshot date and any later updates in the dataset manifest. Version-sensitive benchmark questions should state the relevant software version explicitly.
