# Domain-Trained Enterprise RAG - Curated Infrastructure Corpus

## Purpose
This corpus is a high-quality seed collection for the capstone brief **Domain-Trained Enterprise RAG: Train the Retriever, Reranker & Generator**. The assignment requires a substantial DevOps/cloud/infra corpus covering Linux, NGINX, Docker, Kubernetes, PostgreSQL, networking, TLS, systemd, cloud and CI/CD, with a target of at least 10,000 meaningful cleaned sections/chunks and a benchmark of at least 1,000 categorized questions.

The uploaded brief explicitly requires: a substantial technical corpus; query-positive-hard-negative triples; fine-tuned retriever and reranker; several thousand validated generator training examples; advanced RAG; citation verification; and adversarial robustness against contradictory, outdated and instruction-bearing documents.

## Included topic folders
- docker
- fastapi
- github_actions
- helm
- kubernetes
- linux_commands
- nginx
- postgresql
- prometheus
- cross_topic

## Data design rules
1. Each `##` section is an atomic knowledge unit with a stable topic-local ID.
2. Sections are written for enterprise retrieval: definitions, configuration, failure modes, security, operational practices and architecture are separated instead of being mixed into giant pages.
3. Version-sensitive claims should be verified against the official source URLs listed at the end of each document and captured in the manifest with a snapshot date.
4. Do not manufacture repeated paragraphs just to reach a chunk count. Expand the corpus with additional official manuals, reference pages, release notes, security advisories and controlled internal runbooks, then deduplicate semantically.
5. Keep source provenance attached to each document during ingestion.

## Recommended expansion plan to meet the brief's >=10,000 meaningful chunks
The files in this package are the curated seed, not a synthetic 10,000-chunk filler set. To meet the formal target with quality, add versioned source material from official manuals for each topic, split it into semantic sections, remove navigation boilerplate, deduplicate near-duplicates, and retain source/version metadata.

A practical target mix is:
- 25% concepts and reference documentation
- 25% configuration and command reference
- 20% security and hardening
- 15% troubleshooting and failure modes
- 10% version-sensitive/release notes
- 5% cross-topic architecture and incident playbooks

## Retrieval metadata fields
Recommended fields per chunk:
`chunk_id`, `topic`, `subtopic`, `document_id`, `section_id`, `source_url`, `source_type`, `source_version`, `snapshot_date`, `security_class`, `text`, `parent_section`, `keywords`.

## Benchmark alignment
Use this corpus to create benchmark questions across the categories required by the brief: direct, multi-hop, troubleshooting, cross-document, version-sensitive, ambiguous, unanswerable and distractor-heavy. Keep the development benchmark separate from the mentor-held hidden benchmark.

## Security-specific corpus treatment
Documents containing instructions should be labeled as **knowledge data**, not executable instructions. Keep retrieved content isolated from system/tool instructions. Include deliberately poisoned, contradictory, outdated and instruction-bearing evaluation documents in a separate adversarial suite rather than silently mixing them into the clean training corpus.
