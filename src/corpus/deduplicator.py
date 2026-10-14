"""
deduplicator.py
===============
Removes near-duplicate chunks from the corpus using MinHash LSH.

Why MinHash?
  Exact hash dedup (MD5/SHA) only catches *identical* chunks.
  MinHash estimates Jaccard similarity between sets of word shingles,
  catching chunks that are 85%+ similar — e.g. the same paragraph
  repeated across Docker v24 and v25 docs with only the version number
  changed in one sentence.

Algorithm:
  1. Tokenize each chunk into 3-word shingles (trigrams).
  2. Hash each shingle set with N random hash functions -> MinHash signature.
  3. Use LSH (Locality Sensitive Hashing) to find candidate pairs efficiently.
  4. For candidates with estimated Jaccard ≥ threshold, keep only the first.
"""

import re
from typing import List, Tuple, Set
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH

from src.corpus.chunker import Chunk


# ---------------------------------------------------------------------------
# Shingling
# ---------------------------------------------------------------------------

def _shingles(text: str, k: int = 3) -> Set[str]:
    """
    Return a set of k-word shingles (word-level n-grams) from text.
    Word-level shingles are more robust than character n-grams for
    technical text where individual tokens carry high signal.
    """
    words = re.findall(r"\w+", text.lower())
    if len(words) < k:
        return {" ".join(words)}  # Short text: use whole text as one shingle
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def deduplicate(
    chunks: List[Chunk],
    threshold: float = 0.85,
    num_perm: int = 128,
) -> Tuple[List[Chunk], int]:
    """
    Remove near-duplicate chunks.

    Args:
        chunks:    All chunks from the corpus.
        threshold: Jaccard similarity threshold. Chunks above this are duplicates.
        num_perm:  Number of MinHash permutations. Higher = more accurate but slower.

    Returns:
        (deduplicated_chunks, num_removed)
    """
    # Exact dedup first (fastest, catches identical chunks)
    seen_hashes: Set[str] = set()
    exact_deduped: List[Chunk] = []
    for chunk in chunks:
        h = _fast_hash(chunk.text)
        if h not in seen_hashes:
            seen_hashes.add(h)
            exact_deduped.append(chunk)

    exact_removed = len(chunks) - len(exact_deduped)
    print(f"  [Dedup] Exact duplicates removed: {exact_removed}")

    # Near-duplicate dedup with MinHash LSH
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: List[MinHash] = []

    print("  [Dedup] Building MinHash signatures...")
    for i, chunk in enumerate(tqdm(exact_deduped, desc="MinHash")):
        mh = MinHash(num_perm=num_perm)
        for shingle in _shingles(chunk.text):
            mh.update(shingle.encode("utf-8"))
        minhashes.append(mh)

    # Insert into LSH; skip if a near-duplicate already inserted
    kept: List[Chunk] = []
    near_removed = 0

    print("  [Dedup] Detecting near-duplicates...")
    for i, (chunk, mh) in enumerate(
        tqdm(zip(exact_deduped, minhashes), total=len(exact_deduped), desc="LSH")
    ):
        key = f"chunk_{i}"
        try:
            result = lsh.query(mh)
        except Exception:
            result = []

        if result:
            # Near-duplicate found — skip this chunk
            near_removed += 1
        else:
            lsh.insert(key, mh)
            kept.append(chunk)

    print(f"  [Dedup] Near-duplicates removed: {near_removed}")
    print(f"  [Dedup] Final corpus size: {len(kept)} chunks")
    return kept, exact_removed + near_removed


def _fast_hash(text: str) -> str:
    """Quick MD5 hash for exact dedup (not cryptographic — speed is the goal)."""
    import hashlib
    return hashlib.md5(text.encode("utf-8")).hexdigest()
