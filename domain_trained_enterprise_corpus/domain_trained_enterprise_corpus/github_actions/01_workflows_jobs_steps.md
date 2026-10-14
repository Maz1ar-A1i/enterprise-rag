# Github Actions - 01 Workflows Jobs Steps

> Corpus role: enterprise infrastructure knowledge for the Domain-Trained Enterprise RAG capstone. Content is intentionally modular so it can be chunked into independent retrieval units.

## Topic scope

Topic: **github_actions**

Each section below is an atomic knowledge unit. Preserve the heading and topic metadata during ingestion so retrieval results retain their semantic context.

## GITHUB_ACTIONS-001 | Workflow file location and structure

GitHub Actions workflow files use YAML and are stored under `.github/workflows`. A workflow contains one or more jobs. Jobs contain steps, and steps either run shell commands or invoke actions. The workflow should be treated as code: review changes, validate syntax and keep environment assumptions explicit.

## GITHUB_ACTIONS-002 | Events and triggers

The `on` key defines events that can start a workflow, such as pushes, pull requests, manual dispatches or scheduled events. Trigger scope matters: a workflow that runs on every branch and pull request can consume unnecessary resources, while a workflow that is too narrowly triggered may fail to validate important changes.

## GITHUB_ACTIONS-003 | Jobs and runners

A job executes on a runner environment. Jobs are independent by default and can be connected with `needs`. Use job boundaries to isolate build, test, security scan and deployment stages. Record the runner type when comparing performance results because CPU, memory and preinstalled tools affect timings.

## GITHUB_ACTIONS-004 | Steps and shell commands

Each job has a sequence of steps. A step can run a command or call an action. Keep shell commands explicit and fail fast where appropriate. Avoid constructing shell commands directly from untrusted pull-request content.

## GITHUB_ACTIONS-005 | Matrix testing

A matrix can run a job across multiple operating systems, Python versions or dependency configurations. Matrix tests are useful for a technical corpus project because parsing, indexing and serving code may behave differently across supported runtime versions.

## GITHUB_ACTIONS-006 | Artifacts

Workflow artifacts are files produced by a run and retained for later inspection or download according to repository settings. Evaluation jobs can upload JSON/CSV metrics, logs, benchmark summaries and packaged reports as artifacts.

## GITHUB_ACTIONS-007 | Caching dependencies

Dependency caching can reduce repeated installation time, but caches are not a trustworthy secret store. Cache content is not signed or verified, so workflows must avoid storing credentials or accepting untrusted cache writes that could alter files executed later.

## GITHUB_ACTIONS-008 | Reusable workflows

Reusable workflows allow an organization to centralize common CI or deployment logic. When reusing workflows from another repository, treat the referenced workflow as code with privileges and review its source, inputs and permissions.

## GITHUB_ACTIONS-009 | Job permissions

Use the minimum GitHub token permissions needed by a job. A read-only test job generally should not receive write access to repository contents. Separate release or deployment permissions from routine pull-request validation when possible.

## GITHUB_ACTIONS-010 | Concurrency control

Concurrency settings can cancel superseded runs or serialize deployment workflows. This prevents multiple commits from deploying simultaneously to the same environment and makes deployment state easier to reason about.

## Source basis

The corpus is intentionally grounded in official project documentation. Use the URLs below for version-specific verification rather than copying volatile version numbers into model-training text.

- https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax
- https://docs.github.com/en/actions/reference/security/secure-use
- https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching

## Versioning note

This document should be treated as a versioned corpus artifact. Record the source snapshot date and any later updates in the dataset manifest. Version-sensitive benchmark questions should state the relevant software version explicitly.
