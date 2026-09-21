"""Model-backed retrieval: dense bi-encoder search and cross-encoder reranking.

Ported from:

* ``scripts/running-on-fine-tune-of-bge-rera-bge-dense-bm25.ipynb``  (hybrid)
* ``scripts/running-inference-on-fine-tune-of-bge-reranker.ipynb``   (pure reranker)

Imports of ``torch`` and ``sentence_transformers`` are deferred into the methods
that need them. That keeps module import cheap, so the BM25-only paths and the UI
shell can start without pulling several gigabytes of wheels, and so the ranking
maths in ``scoring.py`` stays testable on its own.
"""

from __future__ import annotations

import logging

from .accelerator import gpu
from .config import DENSE_POOL_SIZE, DENSE_QUERY_PREFIX
from .corpus import Corpus

logger = logging.getLogger(__name__)

#: Seconds of GPU time per ``(query, document)`` pair at ``max_length=256``,
#: measured on the deployed Space (zero-a10g, 2026-09-20):
#: pure reranking of all 695 documents took ~74s -> ~0.106 s/pair.
#: At ``max_length=512`` each pair costs roughly twice as much, because the
#: cross-encoder attends over twice the tokens, so the estimate is scaled.
RERANK_SECONDS_PER_PAIR = 0.11

#: Multiplies the estimate. A declared duration is a *hard cap*: exceeding it
#: kills the call, so under-declaring is worse than over-declaring. But
#: over-declaring is not free either -- see :func:`rerank_duration`.
RERANK_SAFETY = 1.3


def rerank_duration(*args, **kwargs) -> int:
    """Declare the GPU time this rerank actually needs, rather than a flat cap.

    ZeroGPU checks the *declared* duration against a visitor's remaining daily
    quota **before the call runs**, and only charges the real time afterwards.
    So a flat ``duration=120`` does not consume 120s of quota -- but it does mean
    a visitor with under 120s left is refused outright, even when the work would
    have taken 15 seconds.

    That is not a corner case. Unauthenticated visitors get **2 minutes of GPU
    quota per day**, which is how newsletter readers arrive, and the durations
    this app originally declared summed to 180s for a single hybrid query --
    so every logged-out visitor was refused on their first click, with a quota
    error naming numbers that appear nowhere in the code.

    Accepts ``*args, **kwargs`` so it handles both standalone functions and
    instance methods where ``self`` is passed as the first argument.
    """
    doc_ids = kwargs.get("doc_ids")
    max_length = kwargs.get("max_length", 256)
    if doc_ids is None:
        for arg in args:
            if isinstance(arg, (list, tuple)):
                doc_ids = arg
                break
    count = len(doc_ids) if doc_ids is not None else 695
    per_pair = RERANK_SECONDS_PER_PAIR * (max_length / 256)
    return int(10 + count * per_pair * RERANK_SAFETY)


class Retriever:
    """Holds the two fine-tuned models and the encoded corpus.

    Models are loaded once and reused across queries. On ZeroGPU they are placed
    on ``cuda`` at load time as HF requires, which works because CUDA emulation
    is active outside ``@gpu`` calls.
    """

    def __init__(
        self,
        corpus: Corpus,
        dense_model_id: str,
        reranker_model_id: str,
        device: str | None = None,
    ) -> None:
        import torch
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.corpus = corpus
        self.dense_model_id = dense_model_id
        self.reranker_model_id = reranker_model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading dense bi-encoder %s on %s", dense_model_id, self.device)
        self.dense_model = SentenceTransformer(dense_model_id, device=self.device)

        logger.info("Loading cross-encoder %s on %s", reranker_model_id, self.device)
        # max_length is set per call, not here: the two source notebooks use
        # different values and baking one in would corrupt the other's rankings.
        self.reranker = CrossEncoder(reranker_model_id, device=self.device)

        self._doc_embeddings = None

    # -- indexing -------------------------------------------------------------

    @property
    def is_encoded(self) -> bool:
        return self._doc_embeddings is not None

    def ensure_encoded(self) -> None:
        """Encode the corpus if that has not happened yet.

        Called lazily on the first query rather than at import time. Encoding is
        decorated with ``@gpu``, and calling it during startup would ask ZeroGPU
        for an accelerator before any visitor has arrived.
        """
        if not self.is_encoded:
            self.encode_corpus()

    @gpu(duration=25)
    def _encode_documents(self):
        """Encode the corpus and *return* the embeddings.

        Declared at 25s: all 695 documents encode in under 4s on the deployed
        Space, so this is generous headroom rather than a realistic estimate.
        It is still far below the 120s it used to declare, and that matters
        because of the process-boundary note below -- the duration is checked
        against a visitor's quota *before* the call runs, and this one runs
        inside the first query a visitor makes.

        This RETURNS rather than assigning ``self._doc_embeddings``, and that is
        not a style choice. Under ZeroGPU a ``@gpu`` function runs in a forked
        worker process: the arguments are pickled in and only the return value is
        pickled back (see ``spaces/zero/wrappers.py``). An assignment to ``self``
        here writes to the child's copy of this object and is discarded when the
        call returns -- leaving the parent convinced nothing was encoded, and
        every query failing with "Corpus not encoded".

        ``.cpu()`` for the same reason: the parent has no real GPU, so the
        comparison must happen on tensors that survived the trip back.
        """
        return self.dense_model.encode(
            self.corpus.search_texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_tensor=True,
            device=self.device,
        ).cpu()

    def encode_corpus(self) -> None:
        """Pre-encode every document with the fine-tuned bi-encoder, once.

        ``normalize_embeddings=True`` so cosine similarity reduces to a dot
        product, exactly as in the notebook. The assignment happens *here*, in
        the parent process, not inside the ``@gpu`` call above.
        """
        if self._doc_embeddings is not None:
            return

        logger.info("Encoding %d passages", len(self.corpus))
        self._doc_embeddings = self._encode_documents()

    # -- dense retrieval ------------------------------------------------------

    @gpu(duration=12)
    def _encode_query(self, query_text: str):
        """Encode one query with the BGE instruction prefix.

        Declared at 12s against a measured sub-second cost. This one is charged
        on *every* query, so an inflated value would tax every visitor on every
        search rather than once per container.

        The prefix is not cosmetic: the dense model was fine-tuned with it, so
        dropping it degrades retrieval.
        """
        return self.dense_model.encode(
            f"{DENSE_QUERY_PREFIX}{query_text}",
            normalize_embeddings=True,
            convert_to_tensor=True,
            device=self.device,
        ).cpu()

    def search_dense(self, query_text: str, top_k: int = DENSE_POOL_SIZE) -> list[tuple[str, float]]:
        """Semantic retrieval by cosine similarity over the encoded corpus."""
        import torch

        if self._doc_embeddings is None:
            raise RuntimeError("Corpus not encoded; call encode_corpus() first.")

        query_embedding = self._encode_query(query_text)
        # Both sides are L2-normalised, so the dot product is the cosine.
        similarities = torch.matmul(self._doc_embeddings, query_embedding)
        top = torch.topk(similarities, k=min(top_k, len(self.corpus)))

        indices = top.indices.cpu().numpy()
        values = top.values.cpu().numpy()
        return [
            (self.corpus.doc_ids[int(i)], float(values[n]))
            for n, i in enumerate(indices)
        ]

    # -- reranking ------------------------------------------------------------

    @gpu(duration=rerank_duration)
    def rerank(
        self,
        query_text: str,
        doc_ids: list[str],
        max_length: int,
    ) -> list[tuple[str, float]]:
        """Score ``(query, document)`` pairs with the fine-tuned cross-encoder.

        Returns raw, unbounded logits; sigmoid normalisation happens in the
        blending step, matching the notebook.

        The duration is computed per call by :func:`rerank_duration`, because the
        two modes differ by an order of magnitude: hybrid reranks a 50-document
        pool at ``max_length=512``, pure reranker reranks all 695 at 256. A
        single flat cap cannot suit both -- it either refuses short calls or
        caps long ones.

        ``max_length`` is required rather than defaulted. The hybrid submission
        uses 512 and the pure-reranker submission uses 256, so a shared default
        would silently change one of them.
        """
        id_to_text = dict(zip(self.corpus.doc_ids, self.corpus.search_texts))

        # The reranker is constructed without max_length so it can vary per call.
        # It is set for the predict() immediately below and never read again, so
        # it does not need to survive back to the parent.
        # gpu-ok: same-call side effect, deliberately not persisted.
        self.reranker.max_length = max_length

        pairs = [[query_text, id_to_text[doc_id]] for doc_id in doc_ids]
        logits = self.reranker.predict(pairs, batch_size=32, convert_to_numpy=True)
        return [(doc_id, float(v)) for doc_id, v in zip(doc_ids, logits)]
