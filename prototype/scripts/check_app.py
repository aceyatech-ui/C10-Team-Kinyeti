"""Render the Gradio app against the seeded stub and inspect the HTML it emits.

The models are never loaded: ``app.build_pipeline()`` bails out because the model
IDs are still placeholders, and the checks then install a stubbed pipeline in its
place. That keeps this runnable in seconds on any machine, and it is the check
that would have caught the missing-``accelerator``-module breakage before it
reached Kaggle.

Covers the things that silently rot in a UI: the number of cards rendered, that a
missing first-stage rank renders as meaningful rather than as a traceback, the
unconfigured and error states, and that document text cannot inject markup.

Run from the repository root:

    python prototype/scripts/check_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for `app`, `src`
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `stub_retriever`

import app  # noqa: E402
from src.config import MODES, TOP_K  # noqa: E402
from src.scoring import Scored  # noqa: E402
from stub_retriever import build_stub_pipeline  # noqa: E402

QUERY = "How do I control fall armyworm in my maize field?"


def check(label: str, condition: bool, detail: str = "") -> bool:
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    return condition


class HostileCorpus:
    """A corpus whose every display field tries to inject markup.

    Stands in for ``Corpus`` -- ``render`` only needs ``.record()`` -- to prove
    that document text is escaped on the way into the page.
    """

    PAYLOAD = '<img src=x onerror="alert(1)">'

    def record(self, doc_id: str) -> dict:
        return {
            "document_id": doc_id,
            "title": f"Title {self.PAYLOAD}",
            "text": f"Body {self.PAYLOAD}",
            "source": self.PAYLOAD,
            "crop": self.PAYLOAD,
            "country": self.PAYLOAD,
            "source_url": "javascript:alert(1)",
        }


class SafeCorpus(HostileCorpus):
    """Same, but with a legitimate source URL, to check the link still renders."""

    def record(self, doc_id: str) -> dict:
        record = super().record(doc_id)
        record["source_url"] = "https://example.org/a?b=1&c=2"
        return record


def main() -> int:
    all_ok = True

    # -- unconfigured state, before any stub is installed ---------------------
    print("=== Unconfigured state (no models wired up) ===")
    all_ok &= check("app imported without a pipeline",
                    app.PIPELINE is None, f"PIPELINE={app.PIPELINE!r}")
    all_ok &= check("import did not crash; a reason was recorded",
                    bool(app.STATE_ERROR), f"STATE_ERROR={app.STATE_ERROR!r}")
    unconfigured = app.search(QUERY, "hybrid")
    all_ok &= check("page explains the app is not ready",
                    "not ready" in unconfigured)
    all_ok &= check("page names the settings that must be filled in",
                    "DENSE_MODEL_ID" in unconfigured)
    all_ok &= check("no traceback leaked into the page",
                    "Traceback" not in unconfigured)
    print()

    # -- install the stub -----------------------------------------------------
    pipeline, retriever = build_stub_pipeline()
    app.PIPELINE = pipeline
    app.STATE_ERROR = None

    # -- the Blocks object builds at all -------------------------------------
    print("=== Interface construction ===")
    demo = app.build_interface()
    all_ok &= check("build_interface returns a gr.Blocks", demo.__class__.__name__ == "Blocks")
    dependencies = demo.config.get("dependencies", [])
    # Gradio adds its own handlers (gr.Examples registers one), so this is a
    # floor rather than an exact count.
    all_ok &= check("search x2 and clear are all wired",
                    len(dependencies) >= 3, f"got {len(dependencies)}")
    all_ok &= check("every handler writes at least one output",
                    all(len(d.get("outputs", [])) >= 1 for d in dependencies))

    # The clear path is the one that silently broke once, so call it directly
    # rather than trusting the wiring.
    cleared = app.clear_results()
    all_ok &= check("clear returns one plain value per output component",
                    isinstance(cleared, tuple) and len(cleared) == 2,
                    repr(cleared))
    all_ok &= check("clear returns strings, not component config objects",
                    all(isinstance(v, str) for v in cleared), repr(cleared))
    print()

    # -- one search per mode --------------------------------------------------
    print("=== Rendered results, one per mode ===")
    for mode in MODES:
        out = app.search(QUERY, mode)
        all_ok &= check(f"[{mode}] returns HTML text",
                        isinstance(out, str) and bool(out))
        all_ok &= check(f"[{mode}] renders exactly {TOP_K} cards",
                        out.count('class="ky-card"') == TOP_K,
                        f"got {out.count('class=\"ky-card\"')}")
        all_ok &= check(f"[{mode}] renders the provenance panel",
                        'class="ky-panel"' in out)
        all_ok &= check(f"[{mode}] names the mode it ran",
                        MODES[mode]["label"] in out)
        all_ok &= check(f"[{mode}] reports latency", "Latency" in out)
        all_ok &= check(f"[{mode}] no traceback in output", "Traceback" not in out)
        all_ok &= check(f"[{mode}] stylesheet included", "<style>" in out)
    print()

    # -- provenance rendering -------------------------------------------------
    print("=== Missing first-stage ranks render as meaningful ===")
    result = pipeline.run(QUERY, mode="hybrid")

    # RRF merges two pools, so a document found by one retriever alone has no
    # rank from the other. That absence must read as information, not as a bug.
    only_bm25 = Scored(doc_id=result.final[0].doc_id, bm25_rank=3, rrf_rank=8,
                       reranker_score=0.99, final_rank=1)
    card = app._card(only_bm25, pipeline.corpus)
    all_ok &= check("a BM25-only document renders 'Dense &mdash;'",
                    "Dense &mdash;" in card)
    all_ok &= check("its BM25 rank is still shown", "BM25 #3" in card)
    all_ok &= check("no bare 'None' reaches the page", "None" not in card)

    only_dense = Scored(doc_id=result.final[0].doc_id, dense_rank=21, rrf_rank=9,
                        reranker_score=0.5, final_rank=2)
    card = app._card(only_dense, pipeline.corpus)
    all_ok &= check("a dense-only document renders 'BM25 &mdash;'",
                    "BM25 &mdash;" in card)

    plain = Scored(doc_id=result.final[0].doc_id, bm25_rank=1, final_rank=1)
    card = app._card(plain, pipeline.corpus)
    all_ok &= check("a bm25_only result omits the fused and reranker chips",
                    "Fused" not in card and "reranker" not in card)

    panel = app._panel(pipeline.run(QUERY, mode="bm25_only"), pipeline.corpus)
    all_ok &= check("panel renders em-dashes for stages that did not run",
                    "—" in panel)
    all_ok &= check("panel never renders a bare 'None' in a cell",
                    "<td class='ky-mono'>None</td>" not in panel)
    print()

    # -- escaping -------------------------------------------------------------
    print("=== Document text cannot inject markup ===")
    hostile = app.render(QUERY, "hybrid", HostileCorpus())
    all_ok &= check("raw <img> tag does not survive into the page",
                    "<img src=x" not in hostile)
    all_ok &= check("payload appears escaped instead",
                    "&lt;img src=x" in hostile)
    all_ok &= check("no javascript: URL becomes a link",
                    "javascript:" not in hostile)
    all_ok &= check("hostile render still produces all cards",
                    hostile.count('class="ky-card"') == TOP_K)

    safe = app.render(QUERY, "hybrid", SafeCorpus())
    all_ok &= check("a legitimate https source becomes a link",
                    'href="https://example.org/a?b=1&amp;c=2"' in safe)
    all_ok &= check("external link is rel-protected",
                    'rel="noopener noreferrer"' in safe)
    print()

    # -- the real corpus ------------------------------------------------------
    print("=== Against the real corpus ===")
    out = app.search(QUERY, "hybrid")
    all_ok &= check("real corpus renders without error",
                    out.count('class="ky-card"') == TOP_K)

    # Only the source-grounded documents carry a URL; the synthetic majority
    # have none by construction, so a card legitimately renders without a link.
    # Check the invariant on the data, then render one of each kind, rather than
    # hoping a query happens to hit the right sort.
    corpus = pipeline.corpus
    urls = {d: corpus.record(d)["source_url"] for d in corpus.doc_ids}
    with_url = [d for d, u in urls.items() if u.lower().startswith(("http://", "https://"))]
    without_url = [d for d, u in urls.items() if not u]

    all_ok &= check("corpus holds source-grounded documents", bool(with_url),
                    f"got {len(with_url)}")
    all_ok &= check("corpus holds synthetic documents with no URL",
                    bool(without_url), f"got {len(without_url)}")

    grounded = app._card(Scored(doc_id=with_url[0], bm25_rank=1, final_rank=1), corpus)
    all_ok &= check("a source-grounded document renders its link",
                    "View source" in grounded)
    all_ok &= check("a source-grounded document claims no synthetic status",
                    "Synthetic document" not in grounded)

    synthetic = app._card(Scored(doc_id=without_url[0], bm25_rank=1, final_rank=1), corpus)
    all_ok &= check("a synthetic document renders no dead link",
                    "View source" not in synthetic)
    # The `source` column on these rows names the organisation whose material the
    # text was modelled on, not one that published it, so the card has to say so.
    all_ok &= check("a synthetic document says it is synthetic",
                    "Synthetic document" in synthetic)

    all_ok &= check("records expose origin and licence for the UI",
                    corpus.record(corpus.doc_ids[0]).get("origin") != ""
                    and corpus.record(corpus.doc_ids[0]).get("license") != "")
    blank = app.search("   ", "hybrid")
    all_ok &= check("blank query gets a readable prompt, not a traceback",
                    "Enter a question first" in blank)
    all_ok &= check("blank-query response carries no traceback",
                    "Traceback" not in blank)
    print()

    if all_ok:
        print("All app checks passed.")
        return 0
    print("Some checks FAILED.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
