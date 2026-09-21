"""Corpus loading and preprocessing.

Replaces the notebooks' hardcoded ``/kaggle/input/...`` reads with a local CSV
path, keeping the flattening step identical (see ``config.create_search_content``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import create_search_content

logger = logging.getLogger(__name__)

# Columns the retrieval pipeline depends on. Missing ones degrade the flattened
# string rather than raising, matching the notebooks' tolerant `.get(..., " ")`.
EXPECTED_COLUMNS = ("title", "text", "source", "crop", "country", "source_url")


@dataclass(frozen=True)
class Corpus:
    """An immutable, indexed corpus ready for retrieval.

    ``doc_ids`` and ``search_texts`` are parallel lists: index ``i`` in one
    corresponds to index ``i`` in the other. This ordering is what lets the
    numpy argsort steps map scored positions back to document IDs, so nothing
    here should ever be reordered or sorted in place.
    """

    doc_ids: list[str]
    search_texts: list[str]
    frame: pd.DataFrame

    def __len__(self) -> int:
        return len(self.doc_ids)

    def record(self, doc_id: str) -> dict:
        """Return the display fields for one document.

        ``origin`` and ``license`` are included because the UI has to be able to
        say when a document is synthetic. Most of the corpus is: those rows carry
        a ``source`` label naming the organisation whose material they were
        modelled on, not one that published them, and showing "ICRISAT" on such a
        card without qualification would misrepresent it.
        """
        row = self.frame.loc[doc_id]
        return {
            "document_id": doc_id,
            "title": _clean(row.get("title")),
            "text": _clean(row.get("text")),
            "source": _clean(row.get("source")),
            "crop": _clean(row.get("crop")),
            "country": _clean(row.get("country")),
            "source_url": _clean(row.get("source_url")),
            "origin": _clean(row.get("origin")),
            "license": _clean(row.get("license")),
        }


def _clean(value: object) -> str:
    """Normalise a cell to a stripped string, mapping NaN/None to empty."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


#: Where ``documents.csv`` may live, in priority order.
#: The repo-root location suits local development (the competition data sits in
#: the project's own ``data/``); the package-local location is what the Space
#: needs, since its repo root is ``prototype/``.
_CANDIDATE_PATHS = (
    Path("data/documents.csv"),
    Path(__file__).resolve().parent.parent / "data" / "documents.csv",
    Path(__file__).resolve().parent.parent.parent / "data" / "documents.csv",
)


def resolve_documents_csv(explicit: str | Path | None = None) -> Path:
    """Find ``documents.csv``, honouring an explicit path or ``DOCUMENTS_CSV``.

    Raises with an actionable message rather than letting pandas emit an opaque
    ``FileNotFoundError`` from deep inside ``read_csv``.
    """
    if explicit is not None:
        candidates = [Path(explicit)]
    elif os.getenv("DOCUMENTS_CSV"):
        candidates = [Path(os.environ["DOCUMENTS_CSV"])]
    else:
        candidates = list(_CANDIDATE_PATHS)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not find documents.csv. Searched:\n  "
        f"{searched}\n"
        "Download it from the competition and place it in the project's data/ "
        "folder, or set the DOCUMENTS_CSV environment variable to its path."
    )


def load_corpus(path: str | Path | None = None) -> Corpus:
    """Load ``documents.csv`` and flatten each row into its search string.

    The CSV is indexed by ``document_id``, matching the notebooks'
    ``read_csv(..., index_col="document_id")``.
    """
    csv_path = resolve_documents_csv(path)
    frame = pd.read_csv(csv_path, index_col="document_id")

    missing = [c for c in EXPECTED_COLUMNS if c not in frame.columns]
    if missing:
        logger.warning(
            "Corpus is missing expected column(s): %s. Retrieval will still run, "
            "but flattened documents will be weaker than the fine-tuned models expect.",
            ", ".join(missing),
        )

    frame["search_text"] = frame.apply(create_search_content, axis=1)

    # Normalise the index to strings *in place*, so the frame and `doc_ids`
    # always agree. The CSV stores integer IDs, but submission files and the
    # trace objects use string IDs; converting only one side made every lookup
    # by ID fail with a KeyError.
    frame.index = frame.index.map(str)

    doc_ids = frame.index.to_list()
    search_texts = frame["search_text"].tolist()

    logger.info("Loaded %d documents from %s", len(doc_ids), csv_path)
    return Corpus(doc_ids=doc_ids, search_texts=search_texts, frame=frame)
