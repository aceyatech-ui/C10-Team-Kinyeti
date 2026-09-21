"""Exercise the query pipeline without loading any model weights.

The retriever is replaced by the seeded stub in ``stub_retriever.py``, which
makes the fusion, ordering and trace-assembly logic testable locally in seconds
instead of after a multi-gigabyte download.

This checks *structure*, not retrieval quality -- the stub's scores are
meaningless by construction. Ranking quality was verified separately against the
competition notebook (see ``verify_parity.ipynb``).

Run from the repository root:

    python prototype/scripts/check_pipeline.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for `src`
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `stub_retriever`

from src.bm25 import BM25Index  # noqa: E402
from src.config import MODES, TOP_K  # noqa: E402
from src.corpus import load_corpus  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402
from stub_retriever import StubRetriever, build_stub_pipeline  # noqa: E402

QUERY = "How do I control fall armyworm in my maize field?"


def check(label: str, condition: bool, detail: str = "") -> bool:
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    return condition


def main() -> int:
    logging.basicConfig(level=logging.WARNING)

    corpus = load_corpus()
    bm25 = BM25Index(corpus).build()
    pipeline = Pipeline(corpus, bm25, StubRetriever(corpus))

    print(f"Corpus: {len(corpus)} documents\n")

    all_ok = True
    for mode in MODES:
        print(f"Mode: {mode}")
        result = pipeline.run(QUERY, mode=mode)

        all_ok &= check("returns exactly TOP_K results", len(result.final) == TOP_K,
                        f"got {len(result.final)}")
        all_ok &= check(
            "final ranks are 1..TOP_K in order",
            [s.final_rank for s in result.final] == list(range(1, TOP_K + 1)),
            str([s.final_rank for s in result.final]),
        )
        all_ok &= check("no duplicate documents in top 5",
                        len({s.doc_id for s in result.final}) == TOP_K)
        all_ok &= check("every result resolves to a corpus record",
                        all(corpus.record(s.doc_id)["title"] for s in result.final))
        all_ok &= check("timings recorded", result.timings.total_ms > 0)

        if mode == "bm25_only":
            expected = [d for d, _ in bm25.search(QUERY, top_k=TOP_K)]
            all_ok &= check("matches BM25 order", [s.doc_id for s in result.final] == expected)
        elif mode == "dense_only":
            expected = [d for d, _ in pipeline.retriever.search_dense(QUERY, top_k=TOP_K)]
            all_ok &= check("matches dense order", [s.doc_id for s in result.final] == expected)
        elif mode == "hybrid":
            all_ok &= check("all results carry an RRF rank",
                            all(s.rrf_rank is not None for s in result.final))
            all_ok &= check("all results carry a reranker score",
                            all(s.reranker_score is not None for s in result.final))
            all_ok &= check(
                "hybrid reranks only the fused pool, not the whole corpus",
                all(s.doc_id in {p.doc_id for p in result.pool} for s in result.final),
            )
        elif mode == "pure_reranker":
            all_ok &= check("all results carry a reranker score",
                            all(s.reranker_score is not None for s in result.final))

        print()

    print("=== Trace completeness (what the UI renders) ===")
    result = pipeline.run(QUERY, mode="hybrid")
    top = result.final[0]

    # A fused result is guaranteed an RRF rank and a reranker score, but NOT
    # both first-stage ranks: RRF merges two pools, so a document may have been
    # found by BM25 alone or by dense alone. The UI must render that absence as
    # meaningful ("dense --" means lexical search was the only one to find it).
    all_ok &= check(
        "every hybrid result has an RRF rank and reranker score",
        all(s.rrf_rank is not None and s.reranker_score is not None for s in result.final),
    )
    all_ok &= check(
        "every hybrid result came from at least one retriever",
        all(s.bm25_rank is not None or s.dense_rank is not None for s in result.final),
    )

    both = sum(1 for s in result.final if s.bm25_rank and s.dense_rank)
    bm25_only = sum(1 for s in result.final if s.bm25_rank and not s.dense_rank)
    dense_only = sum(1 for s in result.final if s.dense_rank and not s.bm25_rank)
    print(f"  top-5 provenance: {both} found by both, {bm25_only} by BM25 only, "
          f"{dense_only} by dense only")
    print(f"  example: BM25 #{top.bm25_rank}, dense #{top.dense_rank}, "
          f"RRF #{top.rrf_rank}, reranker {top.reranker_score:.4f}")

    moved = [s for s in result.final if s.rrf_rank and s.rrf_rank > (s.final_rank or 0)]
    print(f"  {len(moved)} of {len(result.final)} results sit above their RRF rank "
          f"(reranker promoted them)")

    print("\n=== Lazy corpus encoding ===")
    lazy_pipeline, lazy_retriever = build_stub_pipeline(is_encoded=False)
    all_ok &= check("retriever starts un-encoded", not lazy_retriever.is_encoded)

    first = lazy_pipeline.run("When should I plant maize?", mode="hybrid")
    all_ok &= check("first query builds the index", lazy_retriever.encode_calls == 1,
                    f"got {lazy_retriever.encode_calls}")
    all_ok &= check(
        "first query explains the one-off delay to the visitor",
        any("just built" in n for n in first.notes),
    )

    second = lazy_pipeline.run("When should I plant maize?", mode="hybrid")
    all_ok &= check("later queries reuse the index", lazy_retriever.encode_calls == 1,
                    f"got {lazy_retriever.encode_calls}")
    all_ok &= check("later queries carry no delay note",
                    not any("just built" in n for n in second.notes))

    print("\n=== Empty query handling ===")
    try:
        pipeline.run("   ", mode="hybrid")
        all_ok &= check("blank query rejected", False, "no error raised")
    except ValueError:
        all_ok &= check("blank query rejected", True)

    print("\n=== Unknown mode handling ===")
    try:
        pipeline.run(QUERY, mode="nonsense")
        all_ok &= check("unknown mode rejected", False, "no error raised")
    except ValueError:
        all_ok &= check("unknown mode rejected", True)

    print()
    if all_ok:
        print("All structural checks passed.")
        return 0
    print("Some checks FAILED.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
