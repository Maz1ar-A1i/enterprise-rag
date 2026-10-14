# Dataset Build Plan for the Domain-Trained Enterprise RAG Capstone

## 1. What this package is
This package is the **curated seed corpus** for the project's infrastructure domain. It is deliberately organized into small, self-contained knowledge sections so the same material can support:

- dense-retriever fine-tuning;
- BM25/hybrid retrieval;
- reranker training;
- generator SFT/QLoRA examples;
- citation verification;
- troubleshooting and multi-hop benchmark questions;
- adversarial evaluation.

The capstone brief requires a substantially larger final corpus (>=10,000 cleaned meaningful sections/chunks) and >=1,000 benchmark questions. Do not reach that number by duplicating or lightly paraphrasing the same facts.

## 2. Required final corpus layers

### Layer A - authoritative clean corpus
Collect official documentation and high-quality technical references for each topic. Preserve source URL, source product, software version, publication/update date where available, and the original document title.

### Layer B - curated modular sections
Convert long pages into semantic sections. A chunk should answer a coherent question without requiring the entire page. Keep headings and parent headings in metadata.

### Layer C - deduplication
Remove exact duplicates, navigation boilerplate, cookie notices, repeated sidebars and near-duplicate versions. Keep version-specific material when the difference is meaningful and label the version explicitly.

### Layer D - conflict-aware corpus
Do not silently merge contradictory guidance. Keep conflicting versions or configurations as separate records with explicit version/source metadata so the RAG system can learn to reason about time and source authority.

### Layer E - adversarial corpus
Maintain a separate evaluation collection containing misleading, outdated, contradictory and instruction-bearing documents. These are for robustness testing and should not contaminate the clean training corpus unless intentionally labeled as adversarial examples.

## 3. Recommended metadata schema

```json
{
  "chunk_id": "kubernetes__networking__service_discovery__004",
  "topic": "kubernetes",
  "subtopic": "networking",
  "document_id": "kubernetes_networking_guide",
  "section_id": "Services, Service discovery",
  "title": "Service discovery",
  "text": "...",
  "source_url": "https://kubernetes.io/docs/...",
  "source_type": "official_documentation",
  "source_version": "1.34",
  "snapshot_date": "2026-10-05",
  "language": "en",
  "security_class": "clean",
  "keywords": ["service", "dns", "pod"],
  "parent_section": "Networking",
  "license_note": "verify source terms before redistribution"
}
```

## 4. Chunking guidance
Start with semantic sections, then apply a token-size guardrail. Avoid splitting code examples from the explanation that gives them meaning. Avoid creating one chunk that contains several unrelated commands or concepts.

A practical first experiment is to compare roughly 250-500 token chunks with 10-20% overlap against heading-aware chunks. Measure Recall@K, MRR and nDCG by question category rather than assuming the largest chunk is best.

## 5. Retriever training triples
For each query, create:

- one positive chunk that directly answers the question;
- one or more hard negatives from the same topic with overlapping vocabulary but a wrong answer;
- additional difficult negatives such as outdated versions, nearby configuration concepts and cross-product distractors.

Run at least two hard-negative mining iterations as required by the capstone. The second iteration should use errors found on the first trained retriever, not random negatives only.

## 6. Benchmark categories
The benchmark should include direct, multi-hop, troubleshooting, cross-document, version-sensitive, ambiguous, unanswerable and distractor-heavy questions. Keep a visible development set separate from the mentor-held hidden benchmark.

Examples:

- Direct: "What does a Kubernetes Service provide to Pods?"
- Multi-hop: "A FastAPI service is behind NGINX and deployed through Kubernetes; which component should own TLS termination in the described configuration?"
- Troubleshooting: "NGINX returns 502 but the FastAPI container is running. What should be checked first?"
- Version-sensitive: "Which Helm behavior/options are available in the specified Helm release?"
- Unanswerable: ask for a product feature not present in the corpus and require refusal rather than invention.
- Distractor-heavy: include 3-5 plausible passages from related components and require the correct one.

## 7. Generator training data
Build several thousand validated examples containing the retrieved context, answer, citations and explicit unsupported/refusal cases. Include examples where:

- the evidence fully supports the answer;
- evidence supports only part of the answer;
- evidence conflicts across versions;
- no evidence is sufficient;
- a retrieved document contains an instruction that the assistant must treat as data, not authority.

## 8. Citation verification target
Every claim in a generated answer should be mapped to one or more retrieved chunk IDs. The verifier should classify claim support as SUPPORTED, PARTIAL or UNSUPPORTED. Unsupported claims should trigger refusal or regeneration according to the final system policy.

## 9. Quality gates before training

- source provenance present;
- no accidental duplicate IDs;
- no empty text;
- no broken encoding;
- no leaked benchmark answers;
- version-sensitive facts have explicit version metadata;
- code examples are syntactically plausible;
- source links resolve;
- dataset snapshot is hashed/versioned;
- chunk statistics recorded;
- random manual review completed for every topic.

## 10. Important project alignment
The assignment requires the corpus to span Linux, NGINX, Docker, Kubernetes, PostgreSQL, networking, TLS, systemd, cloud and CI/CD, and it expects the dataset/evaluation protocol to be designed before tuning. This package covers the requested named topics plus cross-topic architecture; the final corpus should still add dedicated networking, TLS, systemd and cloud source material before the corpus-freeze gate.
