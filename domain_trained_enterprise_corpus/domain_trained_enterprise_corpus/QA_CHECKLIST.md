# Corpus QA Checklist

Before indexing this corpus:

- [ ] Validate UTF-8 and markdown parsing.
- [ ] Check for empty sections and accidental duplicate section IDs.
- [ ] Record source snapshot date and document version.
- [ ] Re-fetch official documentation for version-sensitive claims before the final benchmark freeze.
- [ ] Deduplicate near-identical sections across sources.
- [ ] Keep source URL and section provenance on every chunk.
- [ ] Run a chunk-length distribution report after the chosen chunker is configured.
- [ ] Create explicit positive passages for benchmark questions.
- [ ] Mine hard negatives from same-topic distractors and version-conflicting passages.
- [ ] Keep benchmark questions and answers outside the training corpus when evaluating leakage.
- [ ] Maintain a separate adversarial corpus for contradiction, stale documents and prompt injection.
- [ ] Validate that citations in generated answers resolve to the retrieved source chunk IDs.
