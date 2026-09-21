"""BM25 lexical retrieval.

Kept separate from ``retrieve.py`` so the lexical half of the pipeline can be
built and tested without loading either neural model. On the Space this also
lets the BM25 index be ready immediately while the weights download.
"""

from __future__ import annotations

import logging

import numpy as np
from rank_bm25 import BM25Okapi

from .config import BM25_POOL_SIZE
from .corpus import Corpus

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """BM25 tokenisation, matching the notebooks' naive whitespace split.

    The competition notebooks do no stemming or stopword removal. That is
    reproduced exactly rather than improved, because the fine-tuned reranker was
    trained on candidates produced by this tokenisation.
    """
    return text.lower().split()


class BM25Index:
    """Okapi BM25 over the flattened corpus."""

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self._bm25: BM25Okapi | None = None

    def build(self) -> "BM25Index":
        """Build the index. Returns self so it can be chained."""
        logger.info("Building BM25 index over %d documents", len(self.corpus))
        self._bm25 = BM25Okapi([tokenize(t) for t in self.corpus.search_texts])
        return self

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None

    def search(self, query: str, top_k: int = BM25_POOL_SIZE) -> list[tuple[str, float]]:
        """Return ``(doc_id, score)`` ranked by descending BM25 score.

        ``np.argsort`` ascending then reversed reproduces the notebook, which
        also means ties resolve by corpus order rather than by score.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built; call build() first.")

        scores = self._bm25.get_scores(tokenize(query))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.corpus.doc_ids[int(i)], float(scores[int(i)])) for i in top_indices]
