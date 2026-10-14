# Linux Commands - 03 Enterprise Troubleshooting Playbooks

> Corpus role: enterprise infrastructure knowledge for the Domain-Trained Enterprise RAG capstone. Content is intentionally modular so it can be chunked into independent retrieval units.

## Topic scope

Topic: **linux_commands**

Each section below is an atomic knowledge unit. Preserve the heading and topic metadata during ingestion so retrieval results retain their semantic context.

## LINUX_COMMANDS-001 | API unreachable playbook

Check DNS, route, listening socket, local firewall, reverse proxy upstream and application logs in that order. A clear chain avoids changing multiple components simultaneously and losing the evidence about which boundary failed.

## LINUX_COMMANDS-002 | Database connection playbook

Confirm hostname resolution, port reachability, PostgreSQL listener configuration, authentication and database existence. Then check connection pool limits and PostgreSQL logs. Do not increase connection counts until the database capacity and workload are understood.

## LINUX_COMMANDS-003 | High memory playbook

Identify which process or container owns the memory, determine whether usage is steady or growing, and check for caches, model weights, large query results or leaks. For a RAG service, separate model memory, vector index memory and request concurrency effects.

## LINUX_COMMANDS-004 | High CPU playbook

Measure the process consuming CPU and correlate it with workload. CPU spikes may come from indexing, embedding generation, reranking, TLS, compression or pathological queries. Avoid treating a high CPU graph as a root cause by itself.

## LINUX_COMMANDS-005 | TLS failure playbook

Validate certificate chain, hostname, expiration, protocol versions and private-key permissions. Use `openssl s_client` and `curl -vk` carefully for diagnosis, but do not disable certificate validation as a permanent fix.

## LINUX_COMMANDS-006 | Permissions failure playbook

Inspect the effective user, groups, path permissions and mount ownership. Use `namei -l` to inspect each directory component and `stat` for file metadata. Prefer fixing the correct owner/group/ACL instead of granting world-write access.

## LINUX_COMMANDS-007 | Process leak playbook

Compare process counts over time, inspect child processes and open files, and correlate with request volume. A leak may be file descriptors, threads, subprocesses or memory rather than the number of top-level processes alone.

## LINUX_COMMANDS-008 | Incident evidence checklist

For reproducible incidents, collect UTC timestamps, host/container identifiers, deployment version, configuration hash, relevant logs, resource metrics and exact commands run. Avoid collecting secrets or unnecessary personal data.

## LINUX_COMMANDS-009 | Production change discipline

Make one meaningful change at a time during diagnosis, record the hypothesis, observe the effect and revert if it does not help. Ad hoc changes destroy the causal trail required for engineering learning.

## LINUX_COMMANDS-010 | Linux relevance to the capstone

The capstone ultimately runs on operating-system primitives whether through local services, containers or Kubernetes nodes. Understanding processes, sockets, permissions, logs and resource pressure helps explain failures that a Python traceback alone cannot.

## Source basis

The corpus is intentionally grounded in official project documentation. Use the URLs below for version-specific verification rather than copying volatile version numbers into model-training text.

- https://www.gnu.org/software/coreutils/manual/
- https://www.gnu.org/software/coreutils/manual/html_node/Mode-Structure.html
- https://kernel.org/doc/html/latest/

## Versioning note

This document should be treated as a versioned corpus artifact. Record the source snapshot date and any later updates in the dataset manifest. Version-sensitive benchmark questions should state the relevant software version explicitly.
