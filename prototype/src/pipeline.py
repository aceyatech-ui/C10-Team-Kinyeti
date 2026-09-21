"""Query orchestration: runs a mode end to end and records the full trace.

Compared with the competition notebooks, which emitted a bare CSV of document
IDs, this returns every intermediate signal -- each candidate's BM25 rank, dense
rank, RRF rank and reranker score. The web UI renders that trace, because the
project's contribution is only legible if you can see the reranker promoting a
document that first-stage retrieval ranked poorly.
"""

from __future__ import annotations

import logging
from time import perf_counter

from .bm25 import BM25Index
from .config import (
    ALPHA,
    BM25_POOL_SIZE,
    DEFAULT_MODE,
    DENSE_POOL_SIZE,
    RERANKER_MAX_LENGTH,
    TOP_K,
)
from .corpus import Corpus
from .retrieve import Retriever
from .scoring import (
    RetrievalResult,
    Scored,
    Timings,
    blend_scores,
    reciprocal_rank_fusion,
    sigmoid,
)

logger = logging.getLogger(__name__)


class Pipeline:
    """Runs a query in one of the modes defined in ``config.MODES``."""

    def __init__(self, corpus: Corpus, bm25: BM25Index, retriever: Retriever) -> None:
        self.corpus = corpus
        self.bm25 = bm25
        self.retriever = retriever

    def run(
        self,
        query: str,
        mode: str = DEFAULT_MODE,
        alpha: float = ALPHA,
    ) -> RetrievalResult:
        """Retrieve for ``query`` and return the top 5 plus a full audit trail."""
        query = (query or "").strip()
        if not query:
            raise ValueError("Enter a question first.")

        if mode not in RERANKER_MAX_LENGTH and mode not in {"dense_only", "bm25_only"}:
            raise ValueError(f"Unknown mode: {mode}")

        result = RetrievalResult(query=query, mode=mode, alpha=alpha)
        started = perf_counter()

        # Deferred from startup: encoding the corpus requests a GPU, and doing
        # it at boot would claim one before any visitor arrives.
        first_query = not self.retriever.is_encoded
        self.retriever.ensure_encoded()
        if first_query:
            result.notes.append(
                "First request: the document index was just built, so this one "
                "is slower than subsequent questions."
            )

        # First-stage retrieval always runs, even in modes that do not rank by
        # it, so every card can show where BM25 and dense each placed a document.
        bm25_hits = self.bm25.search(query, top_k=BM25_POOL_SIZE)
        result.timings.bm25_ms = (perf_counter() - started) * 1000

        dense_started = perf_counter()
        dense_hits = self.retriever.search_dense(query, top_k=DENSE_POOL_SIZE)
        result.timings.dense_ms = (perf_counter() - dense_started) * 1000

        pool: dict[str, Scored] = {}
        for rank, (doc_id, score) in enumerate(bm25_hits, start=1):
            entry = pool.setdefault(doc_id, Scored(doc_id=doc_id))
            entry.bm25_rank = rank
            entry.bm25_score = score
        for rank, (doc_id, score) in enumerate(dense_hits, start=1):
            entry = pool.setdefault(doc_id, Scored(doc_id=doc_id))
            entry.dense_rank = rank
            entry.dense_score = score

        if mode == "bm25_only":
            ordered = [doc_id for doc_id, _ in bm25_hits]
        elif mode == "dense_only":
            ordered = [doc_id for doc_id, _ in dense_hits]
        elif mode == "hybrid":
            ordered = self._rank_hybrid(
                query,
                pool,
                result,
                bm25_ids=[d for d, _ in bm25_hits],
                dense_ids=[d for d, _ in dense_hits],
                alpha=alpha,
            )
        else:  # pure_reranker
            ordered = self._rank_pure(query, pool, result)

        for rank, doc_id in enumerate(ordered[:TOP_K], start=1):
            entry = pool.setdefault(doc_id, Scored(doc_id=doc_id))
            entry.final_rank = rank

        # The pool is returned whole so the UI can show what the reranker
        # demoted, not only what it promoted.
        result.final = [pool[d] for d in ordered[:TOP_K]]
        result.pool = sorted(
            pool.values(),
            key=lambda s: (s.final_rank is None, s.final_rank or 0),
        )
        result.timings.total_ms = (perf_counter() - started) * 1000

        if mode == "pure_reranker" and self.retriever.device == "cpu":
            result.notes.append(
                "Running on CPU, where scoring all "
                f"{len(self.corpus)} documents is slow. On the deployed Space "
                "this uses a GPU and takes a second or two."
            )

        return result

    def _rank_hybrid(
        self,
        query: str,
        pool: dict[str, Scored],
        result: RetrievalResult,
        bm25_ids: list[str],
        dense_ids: list[str],
        alpha: float = ALPHA,
    ) -> list[str]:
        """BM25 + dense -> reciprocal rank fusion -> cross-encoder rerank.

        Mirrors the hybrid notebook that produced the best private-leaderboard
        score. Reranking is limited to the fused candidate pool, which is what
        makes it affordable.
        """
        fused = reciprocal_rank_fusion(bm25_ids, dense_ids)[:BM25_POOL_SIZE]
        for rank, (doc_id, score) in enumerate(fused, start=1):
            entry = pool.setdefault(doc_id, Scored(doc_id=doc_id))
            entry.rrf_rank = rank
            entry.rrf_score = score

        candidate_ids = [doc_id for doc_id, _ in fused]

        rerank_started = perf_counter()
        logits = dict(
            self.retriever.rerank(
                query,
                candidate_ids,
                max_length=RERANKER_MAX_LENGTH["hybrid"],
            )
        )
        result.timings.rerank_ms = (perf_counter() - rerank_started) * 1000

        for doc_id, logit in logits.items():
            pool[doc_id].reranker_logit = logit

        blended = blend_scores(
            [logits[d] for d in candidate_ids],
            [pool[d].dense_score or 0.0 for d in candidate_ids],
            alpha=alpha,
        )
        for doc_id, score in zip(candidate_ids, blended):
            pool[doc_id].reranker_score = score

        order = sorted(
            range(len(candidate_ids)),
            key=lambda i: blended[i],
            reverse=True,
        )
        return [candidate_ids[i] for i in order]

    def _rank_pure(
        self,
        query: str,
        pool: dict[str, Scored],
        result: RetrievalResult,
    ) -> list[str]:
        """Score every document with the cross-encoder, skipping retrieval.

        This was the strongest public-leaderboard submission. Feasible only
        because the deployed Space has a GPU; see ``RetrievalResult.notes``.
        """
        all_ids = list(self.corpus.doc_ids)

        rerank_started = perf_counter()
        logits = dict(
            self.retriever.rerank(
                query,
                all_ids,
                max_length=RERANKER_MAX_LENGTH["pure_reranker"],
            )
        )
        result.timings.rerank_ms = (perf_counter() - rerank_started) * 1000

        for doc_id, logit in logits.items():
            entry = pool.setdefault(doc_id, Scored(doc_id=doc_id))
            entry.reranker_logit = logit
            entry.reranker_score = sigmoid(logit)

        return sorted(logits, key=lambda d: logits[d], reverse=True)
