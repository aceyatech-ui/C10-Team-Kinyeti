"""Central configuration for the Team Kinyeti retrieval prototype.

Constants here are ported from the competition notebooks in ``scripts/``. Where
the notebooks disagree with each other, the disagreement is preserved per-mode
rather than averaged into a shared default -- see ``RERANKER_MAX_LENGTH``.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

# --- Models -----------------------------------------------------------------
# The Space cannot authenticate against Kaggle, so the fine-tuned weights must
# live on HF Hub. Override these on the Space via repository variables rather
# than editing this file, so the same commit runs against a different account.
DENSE_MODEL_ID = os.getenv("DENSE_MODEL_ID", "REPLACE-ME/bge-base-agri")
RERANKER_MODEL_ID = os.getenv("RERANKER_MODEL_ID", "REPLACE-ME/bge-reranker-large-agri")

# --- Document flattening ----------------------------------------------------
# Identical in all four notebooks. Used for both training and retrieval, which is
# why it must stay byte-for-byte stable: changing it silently invalidates the
# fine-tuned models' learned representation.
FLATTEN_TEMPLATE = "Crop {crop} | Country {country} | Title {title} | Text {text} | Source {source}"

# Prepended to *queries only* (never documents) for the dense bi-encoder. This is
# the instruction format BGE was trained with.
DENSE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --- First-stage retrieval --------------------------------------------------
BM25_POOL_SIZE = 50
DENSE_POOL_SIZE = 50

# Reciprocal rank fusion damping. 60 is the value used in the hybrid notebook.
RRF_K = 60

# Blend weight between dense cosine similarity (alpha) and sigmoid-normalised
# reranker logits (1 - alpha). The team grid-searched alpha and found 0.0 best,
# i.e. pure reranker scores. Kept explicit so the UI can expose it later.
ALPHA = 0.0

# --- Reranker max_length: DELIBERATELY PER-MODE -----------------------------
# The two inference notebooks disagree, and both are legitimate:
#
#   hybrid notebook        -> CrossEncoder(save_dir, max_length=512)
#   pure-reranker notebook -> InferencePairDataset(..., max_length=256)
#
# Training used 256. Unifying these into one shared constant would silently
# change one submission's rankings, so each mode carries its own value. Do not
# "clean this up".
RERANKER_MAX_LENGTH = {
    "hybrid": 512,
    "pure_reranker": 256,
}

# --- Retrieval modes exposed in the UI --------------------------------------
# `hybrid` and `pure_reranker` reproduce the two competition submissions.
# `dense_only` and `bm25_only` exist purely as baselines, so a visitor can see
# what the fine-tuned reranker actually adds.
MODES = {
    "hybrid": {
        "short": "Hybrid — BM25 + dense + reranker",
        "label": "Hybrid - BM25 + dense + reranker (best private LB, 0.93753)",
        "description": (
            "Retrieves 50 candidates with BM25 and 50 with the dense bi-encoder, "
            "fuses the two rankings with reciprocal rank fusion, then reranks the "
            "merged pool with the fine-tuned cross-encoder."
        ),
        "nDCG@5": "0.93753 (private)",
    },
    "pure_reranker": {
        "short": "Pure reranker — scores every document",
        "label": "Pure reranker - scores every document (best public LB, 0.96888)",
        "description": (
            "Skips first-stage retrieval entirely and scores all 695 documents "
            "with the fine-tuned cross-encoder. This was the strongest public "
            "leaderboard submission."
        ),
        "nDCG@5": "0.96888 (public)",
    },
    "dense_only": {
        "short": "Dense only — semantic baseline",
        "label": "Dense only - semantic similarity baseline",
        "description": "Fine-tuned bi-encoder cosine similarity. No lexical matching, no reranking.",
        "nDCG@5": "0.825 (held-out)",
    },
    "bm25_only": {
        "short": "BM25 only — keyword baseline",
        "label": "BM25 only - keyword baseline",
        "description": "Classic lexical search. No semantics, no reranking.",
        "nDCG@5": "not reported",
    },
}

DEFAULT_MODE = "hybrid"

# --- Corpus -----------------------------------------------------------------
DOCUMENTS_CSV = os.getenv("DOCUMENTS_CSV", "data/documents.csv")

# Result cards shown, matching the competition's submission format.
TOP_K = 5


def create_search_content(row: Mapping[str, Any]) -> str:
    """Flatten one corpus row into the single searchable string.

    Ported verbatim from ``create_search_content`` in all four notebooks. The
    odd-looking defaults (a single space rather than empty string) are preserved
    because they are part of what the models were fine-tuned against.
    """
    return FLATTEN_TEMPLATE.format(
        crop=str(row.get("crop", " ")).strip(),
        country=str(row.get("country", " ")).strip(),
        title=str(row.get("title", " ")).strip(),
        text=str(row.get("text", " ")).strip(),
        source=str(row.get("source", " ")).strip(),
    )


def is_placeholder_models() -> bool:
    """True when the model IDs have not been configured yet."""
    return DENSE_MODEL_ID.startswith("REPLACE-ME") or RERANKER_MODEL_ID.startswith("REPLACE-ME")
