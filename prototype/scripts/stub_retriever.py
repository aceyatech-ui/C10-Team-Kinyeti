"""Deterministic stand-ins for the two fine-tuned models, for local checks.

``Pipeline`` only ever calls ``search_dense`` and ``rerank`` on its retriever, so
a seeded RNG can stand in for the real models. That makes the fusion, ordering,
trace-assembly and rendering logic testable in seconds instead of after a
multi-gigabyte download.

Scores produced here are meaningless by construction and must never be read as a
quality signal -- ranking quality was verified separately against the
competition notebook, in ``notebooks/verify_parity.ipynb``.
"""

from __future__ import annotations

import random


class StubRetriever:
    """Stand-in for the dense bi-encoder and the cross-encoder.

    Scores come from a seeded RNG keyed on the query, so a given query always
    produces the same ranking and assertions stay stable across runs.
    """

    def __init__(self, corpus, is_encoded: bool = True) -> None:
        self.corpus = corpus
        self.device = "cpu"
        self._encoded = is_encoded
        # Bumped by ensure_encoded(), so a check can prove the lazy-encode path
        # actually ran rather than silently short-circuiting.
        self.encode_calls = 0

    @property
    def is_encoded(self) -> bool:
        return self._encoded

    def ensure_encoded(self) -> None:
        if not self._encoded:
            self.encode_calls += 1
            self._encoded = True

    @staticmethod
    def _rng(seed_parts) -> random.Random:
        return random.Random(hash(tuple(str(p) for p in seed_parts)) & 0xFFFFFFFF)

    def search_dense(self, query: str, top_k: int = 50):
        rng = self._rng(("dense", query))
        scored = [(d, rng.random()) for d in self.corpus.doc_ids]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:top_k]

    def rerank(self, query: str, doc_ids: list[str], max_length: int):
        rng = self._rng(("rerank", query, max_length))
        return [(d, rng.uniform(-6.0, 6.0)) for d in doc_ids]


def build_stub_pipeline(corpus=None, is_encoded: bool = True):
    """Return a ``Pipeline`` wired to the real corpus and BM25 index, and a stub
    retriever. Returns ``(pipeline, retriever)``.
    """
    from src.bm25 import BM25Index
    from src.corpus import load_corpus
    from src.pipeline import Pipeline

    corpus = corpus if corpus is not None else load_corpus()
    bm25 = BM25Index(corpus).build()
    retriever = StubRetriever(corpus, is_encoded=is_encoded)
    return Pipeline(corpus, bm25, retriever), retriever
