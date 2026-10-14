# Github Actions - 02 Security Reproducibility

> Corpus role: enterprise infrastructure knowledge for the Domain-Trained Enterprise RAG capstone. Content is intentionally modular so it can be chunked into independent retrieval units.

## Topic scope

Topic: **github_actions**

Each section below is an atomic knowledge unit. Preserve the heading and topic metadata during ingestion so retrieval results retain their semantic context.

## GITHUB_ACTIONS-001 | Pinning third-party actions

GitHub recommends pinning actions to a full-length commit SHA when immutable references are required. Tags are convenient but can move. Pinning reduces the risk that a later tag change silently executes different third-party code in a privileged workflow.

## GITHUB_ACTIONS-002 | Secrets handling

Secrets should be stored using GitHub's secret mechanisms and referenced only where necessary. Do not echo secrets, place them in artifacts, or write them into dependency caches. Be especially careful when workflows process forked pull requests or other untrusted contributions.

## GITHUB_ACTIONS-003 | Pull request trust boundary

Pull-request code can be attacker-controlled. A workflow that checks out untrusted code and exposes sensitive repository or deployment credentials can become a privilege-escalation path. Separate untrusted test execution from privileged deployment steps.

## GITHUB_ACTIONS-004 | Environment protection

Production deployment environments can require approvals or other protection rules. Use these controls to prevent automated workflows from changing production infrastructure merely because a build passed.

## GITHUB_ACTIONS-005 | Reproducible evaluation workflow

The evaluation workflow should check out a fixed commit, obtain a known corpus snapshot, run the same commands and output machine-readable results. Do not let benchmark questions be altered as part of the normal pull-request workflow.

## GITHUB_ACTIONS-006 | Dependency lock and cache keys

Use lock files where the package ecosystem supports them and make cache keys reflect the relevant lockfile and runtime version. A cache hit should mean a compatible dependency state, not merely a matching operating system name.

## GITHUB_ACTIONS-007 | Release evidence

For productionization, store the image digest, chart version, Git commit and test/evaluation summary together. This creates a trace from deployed behavior back to source code and benchmark results.

## GITHUB_ACTIONS-008 | AI coding assistant logging

This project requires an AI-use log for significant generated code or design decisions. The CI pipeline can validate that an AI-use log exists for major changes without assuming that generated code is correct. Metrics and experiment conclusions must always come from actual runs.

## GITHUB_ACTIONS-009 | Failure artifact retention

When a benchmark or deployment fails, upload enough evidence to reproduce the failure: test output, configuration summary, commit identifier and sanitized logs. Do not upload secrets, raw credentials or private benchmark questions.

## GITHUB_ACTIONS-010 | Workflow anti-patterns

Avoid `curl | bash` installation patterns from unknown sources, unpinned third-party actions in privileged jobs, broad write permissions, secrets in repository files and untrusted pull-request code running with deployment credentials.

## Source basis

The corpus is intentionally grounded in official project documentation. Use the URLs below for version-specific verification rather than copying volatile version numbers into model-training text.

- https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax
- https://docs.github.com/en/actions/reference/security/secure-use
- https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching

## Versioning note

This document should be treated as a versioned corpus artifact. Record the source snapshot date and any later updates in the dataset manifest. Version-sensitive benchmark questions should state the relevant software version explicitly.
