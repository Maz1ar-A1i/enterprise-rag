# Github Actions - 03 Ci Pipeline For Rag

> Corpus role: enterprise infrastructure knowledge for the Domain-Trained Enterprise RAG capstone. Content is intentionally modular so it can be chunked into independent retrieval units.

## Topic scope

Topic: **github_actions**

Each section below is an atomic knowledge unit. Preserve the heading and topic metadata during ingestion so retrieval results retain their semantic context.

## GITHUB_ACTIONS-001 | CI stages for the RAG project

A practical pipeline can validate formatting and static checks, run unit tests, run retrieval data integrity checks, build the container image, run integration tests and publish evaluation artifacts. Expensive fine-tuning jobs can be separated from normal CI and launched through an intentional training workflow.

## GITHUB_ACTIONS-002 | Corpus validation in CI

Treat the corpus as versioned data. CI can validate required metadata fields, duplicate rates, empty documents, malformed records, invalid source URLs, unexpected licenses and chunk-size distributions. This prevents bad data from silently entering a benchmark or training run.

## GITHUB_ACTIONS-003 | Benchmark leakage checks

Automated checks should ensure that benchmark questions or answers are not accidentally copied into training examples. Exact-match checks, normalized text comparisons and source provenance checks can catch obvious leakage. More sophisticated semantic leakage checks can be added later.

## GITHUB_ACTIONS-004 | Training workflow separation

Model fine-tuning can require GPU resources and longer run times. Keep the training workflow separate from the main test pipeline and make it accept explicit dataset/model/configuration versions. This supports reproducibility and makes it possible to compare experiments fairly.

## GITHUB_ACTIONS-005 | Evaluation artifact schema

Every formal evaluation should emit a machine-readable record with system variant, corpus version, benchmark version, metrics, runtime, hardware and configuration identifiers. Human-readable charts can be derived from this record.

## GITHUB_ACTIONS-006 | Container image promotion

Build an image once and promote the same immutable artifact between environments instead of rebuilding differently for each environment. The image digest should be the identity used when recording experiments or deployments.

## GITHUB_ACTIONS-007 | Deployment verification

After deployment, run smoke tests against health and representative API endpoints. Verify metrics are being scraped and logs are flowing before declaring the deployment successful.

## GITHUB_ACTIONS-008 | Rollback strategy

A deployment pipeline should support rollback to a known working image or chart version. Rollback is safer when releases are immutable and configuration changes are versioned.

## GITHUB_ACTIONS-009 | Hidden benchmark protection

The hidden benchmark must not be embedded in the repository or visible workflow artifacts. Production CI should validate the public benchmark; the private benchmark remains mentor-controlled until the final evaluation.

## GITHUB_ACTIONS-010 | One-command reproduction

The repository should expose a documented command that creates a clean environment, validates the selected corpus, runs evaluation and emits JSON/CSV results. GitHub Actions should call the same scripts rather than duplicating experiment logic in YAML.

## Source basis

The corpus is intentionally grounded in official project documentation. Use the URLs below for version-specific verification rather than copying volatile version numbers into model-training text.

- https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax
- https://docs.github.com/en/actions/reference/security/secure-use
- https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching

## Versioning note

This document should be treated as a versioned corpus artifact. Record the source snapshot date and any later updates in the dataset manifest. Version-sensitive benchmark questions should state the relevant software version explicitly.
