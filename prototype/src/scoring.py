"""Lightweight ranking maths: fusion, blending, and result types.

Deliberately free of torch / sentence-transformers imports so the ranking logic
can be exercised without downloading any model weights. See ``bm25.py`` for the
lexical index and ``retrieve.py`` for the model-backed retrievers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import ALPHA, RRF_K


@dataclass
class Scored:
    """One document carrying the evidence of how it was retrieved.

    The per-stage fields are the point of the UI: showing BM25 rank, dense rank,
    RRF rank and the reranker's score side by side is what makes the fine-tuned
    reranker's contribution visible to a reader who has never heard of nDCG.
    """

    doc_id: str
    bm25_rank: int | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    rrf_rank: int | None = None
    rrf_score: float | None = None
    reranker_logit: float | None = None
    reranker_score: float | None = None
    final_rank: int | None = None

    @property
    def retrieved_by(self) -> list[str]:
        """Which first-stage retrievers surfaced this document."""
        found = []
        if self.bm25_rank is not None:
            found.append("BM25")
        if self.dense_rank is not None:
            found.append("dense")
        return found


@dataclass
class Timings:
    """Milliseconds spent per stage, surfaced in the UI."""

    bm25_ms: float = 0.0
    dense_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "BM25": round(self.bm25_ms, 1),
            "dense": round(self.dense_ms, 1),
            "rerank": round(self.rerank_ms, 1),
            "total": round(self.total_ms, 1),
        }


@dataclass
class RetrievalResult:
    """Everything one query produced, including the full audit trail."""

    query: str
    mode: str
    alpha: float = ALPHA
    final: list[Scored] = field(default_factory=list)
    #: Every candidate that entered the final stage, ranked. The UI uses this to
    #: show what the reranker demoted, not just what it promoted.
    pool: list[Scored] = field(default_factory=list)
    timings: Timings = field(default_factory=Timings)
    notes: list[str] = field(default_factory=list)


def reciprocal_rank_fusion(
    bm25_ids: list[str],
    dense_ids: list[str],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Fuse two ranked lists by reciprocal rank.

    Scores by rank *position* rather than raw score, which is what lets BM25's
    unbounded term-frequency scores and cosine similarities be combined without
    first putting them on a common scale.

    Ported from ``reciprocal_rank_fusion`` in the hybrid notebook.
    """
    rrf_scores: dict[str, float] = {}
    for ranked_list in (bm25_ids, dense_ids):
        for rank, doc_id in enumerate(ranked_list):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    return sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)


def sigmoid(x: float) -> float:
    """Logistic squashing, as used in the notebook's alpha-blending step.

    Guarded against overflow: reranker logits are unbounded and this is called
    per candidate in a loop.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def blend_scores(
    reranker_logits: list[float],
    dense_scores: list[float],
    alpha: float = ALPHA,
) -> list[float]:
    """Blend normalised reranker scores with dense similarities.

    At the competition's alpha of 0.0 this reduces to the pure reranker score,
    but the parameter is retained so the grid-searched alternatives
    (0.2 / 0.4 / 0.5 / 0.6 / 0.8 / 1.0) can be explored interactively.

    Documents that arrived via BM25 only have no dense similarity; the notebook
    defaults those to 0.0, faithfully reproduced here.
    """
    return [
        (alpha * dense) + ((1.0 - alpha) * sigmoid(logit))
        for logit, dense in zip(reranker_logits, dense_scores)
    ]
