"""Step 2a check: load the corpus and run lexical search end to end.

No model weights required -- this exercises corpus loading, the flattening step
and BM25 only. Run from the repository root:

    python prototype/scripts/check_bm25.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow running as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bm25 import BM25Index  # noqa: E402
from src.config import TOP_K  # noqa: E402
from src.corpus import load_corpus  # noqa: E402
from src.scoring import reciprocal_rank_fusion  # noqa: E402

SAMPLE_QUERIES = [
    "My maize leaves are turning yellow, what should I do?",
    "How do I control fall armyworm in my cassava farm?",
    "When is the best time to plant rice in northern Nigeria?",
    "How can I improve soil fertility without expensive fertiliser?",
]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("\n=== 1. Loading corpus ===")
    corpus = load_corpus()
    print(f"Documents loaded: {len(corpus)}")

    print("\n=== 2. Flattened search text (how the model sees a document) ===")
    sample = corpus.record(corpus.doc_ids[0])
    print(f"  {corpus.search_texts[0][:300]}...")

    print("\n=== 3. Building BM25 index ===")
    index = BM25Index(corpus).build()

    for query in SAMPLE_QUERIES:
        print(f"\n=== Query: {query} ===")
        hits = index.search(query, top_k=TOP_K)
        for rank, (doc_id, score) in enumerate(hits, start=1):
            record = corpus.record(doc_id)
            title = record["title"][:68]
            print(f"  {rank}. [{score:6.2f}] {title}")
            print(f"       crop={record['crop'] or '-'} | country={record['country'] or '-'} "
                  f"| source={record['source'] or '-'}")

    print("\n=== 4. Reciprocal rank fusion (BM25 vs a reversed list as a stand-in) ===")
    bm25_ids = [d for d, _ in index.search(SAMPLE_QUERIES[0], top_k=10)]
    fused = reciprocal_rank_fusion(bm25_ids, list(reversed(bm25_ids)))
    print(f"  {len(bm25_ids)} + {len(bm25_ids)} ranked lists -> {len(fused)} unique docs")
    print(f"  top of fused list: {fused[0][0]} (rrf={fused[0][1]:.5f})")

    print("\n=== 5. Sanity checks ===")
    problems = []
    if len(corpus) != 695:
        problems.append(f"expected 695 documents, got {len(corpus)}")
    if not index.is_built:
        problems.append("BM25 index did not build")
    if len(set(corpus.doc_ids)) != len(corpus):
        problems.append("duplicate document IDs found")

    if problems:
        print("  FAILED:")
        for problem in problems:
            print(f"    - {problem}")
        return 1

    print("  All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
