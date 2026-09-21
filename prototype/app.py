"""Team Kinyeti — Agricultural Extension RAG retrieval prototype.

A public demo of the competition retrieval pipeline. A visitor types a
farmer's question and sees the five documents the system retrieves, each
carrying the evidence of how it was found: its BM25 rank, its dense rank, its
fused rank and its cross-encoder score.

That provenance row is the point. It shows, per document, where keyword search
placed it versus where the fine-tuned reranker placed it -- making the team's
actual contribution visible to someone who has never heard of nDCG.

Runs on Hugging Face Spaces (Gradio SDK, ZeroGPU hardware) and on a plain CPU
machine for development; see ``src/accelerator.py`` for how that works.
"""

from __future__ import annotations

import html
import logging
import os

import gradio as gr

from src.bm25 import BM25Index
from src.config import (
    ALPHA,
    DEFAULT_MODE,
    DENSE_MODEL_ID,
    MODES,
    RERANKER_MODEL_ID,
    TOP_K,
    is_placeholder_models,
)
from src.corpus import load_corpus
from src.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("kinyeti")

# Questions a farmer or extension officer might actually ask. Chosen to include
# cases where keyword search visibly fails, so the demo shows the problem rather
# than only the solution.
EXAMPLE_QUESTIONS = [
    "When is the best time to plant rice in northern Nigeria?",
    "My maize leaves are turning yellow, what should I do?",
    "How do I control fall armyworm without chemicals?",
    "How can I improve soil fertility without expensive fertiliser?",
    "What should I do about drought affecting my sorghum?",
]

#: Populated by :func:`build_pipeline`. Stays None if setup fails, in which case
#: the UI explains why rather than crash-looping.
PIPELINE: Pipeline | None = None
STATE_ERROR: str | None = None


def build_pipeline() -> None:
    """Load the corpus, build the BM25 index and instantiate the models.

    Model weights are pulled from Hugging Face Hub by ``sentence-transformers``
    at construction time. The corpus is *not* encoded here -- that requests a
    GPU under ZeroGPU and would claim one before any visitor arrives.
    """
    global PIPELINE, STATE_ERROR

    if is_placeholder_models():
        STATE_ERROR = (
            "Model repositories are not configured. Set the DENSE_MODEL_ID and "
            "RERANKER_MODEL_ID variables in this Space's settings to the "
            "Hugging Face model repos holding the fine-tuned weights."
        )
        logger.error(STATE_ERROR)
        return

    try:
        # Retriever is imported here rather than at module scope: it is the one
        # module that imports torch and sentence-transformers, so a failure to
        # install those should not stop the UI shell from starting and saying so.
        from src.retrieve import Retriever

        corpus = load_corpus()
        bm25 = BM25Index(corpus).build()
        retriever = Retriever(corpus, DENSE_MODEL_ID, RERANKER_MODEL_ID)
        PIPELINE = Pipeline(corpus, bm25, retriever)
        logger.info("Ready: %d documents on %s", len(corpus), retriever.device)
    except Exception as exc:  # noqa: BLE001 - surfaced to the visitor
        STATE_ERROR = f"{type(exc).__name__}: {exc}"
        logger.exception("Failed to initialise the pipeline")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_STYLES = """
<style>
  .ky-wrap { font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; }
  .ky-legend {
    font-size: 13px; opacity: .75; margin: 0 0 14px 2px; line-height: 1.7;
  }
  #results-area {
    scroll-margin-top: 24px;
    min-height: 48px;
  }
  .ky-card {
    border: 1px solid rgba(127,127,127,.28);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 12px;
    background: rgba(127,127,127,.05);
  }
  .ky-head { display: flex; gap: 10px; align-items: baseline; margin-bottom: 6px; }
  .ky-rank {
    font-weight: 700; font-size: 15px; opacity: .55;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    flex: 0 0 auto;
  }
  .ky-title { font-weight: 650; font-size: 15.5px; line-height: 1.35; }
  .ky-meta { font-size: 12.5px; opacity: .7; margin: 2px 0 9px 30px; }
  .ky-chips { margin: 0 0 9px 30px; }
  .ky-chip {
    display: inline-block; padding: 2px 9px; border-radius: 999px;
    font-size: 11.5px; margin: 0 6px 4px 0; border: 1px solid;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    white-space: nowrap;
  }
  .ky-strong  { color:#0d9488; border-color:rgba(13,148,136,.5);  background:rgba(13,148,136,.12); }
  .ky-mid     { color:#d97706; border-color:rgba(217,119,6,.5);   background:rgba(217,119,6,.12); }
  .ky-weak    { color:#6b7280; border-color:rgba(107,114,128,.5); background:rgba(107,114,128,.12); }
  .ky-none    { color:#9ca3af; border-color:rgba(156,163,175,.4); background:transparent;
                border-style:dashed; }
  .ky-score   { color:#6366f1; border-color:rgba(99,102,241,.5);  background:rgba(99,102,241,.12); }
  .ky-excerpt { margin: 0 0 8px 30px; font-size: 13.5px; line-height: 1.6; opacity: .88; }
  .ky-link { margin-left: 30px; font-size: 12.5px; }
  .ky-prov { margin-left: 30px; font-size: 12px; opacity: .6; font-style: italic; }
  .ky-note {
    border-left: 3px solid rgba(127,127,127,.4);
    padding: 7px 12px; margin: 0 0 12px 0; font-size: 13px; opacity: .85;
  }
  .ky-panel {
    border: 1px solid rgba(127,127,127,.22); border-radius: 8px;
    padding: 12px 14px; background: rgba(127,127,127,.04); font-size: 13px;
  }
  .ky-panel table { border-collapse: collapse; width: 100%; margin-top: 8px; }
  .ky-panel th, .ky-panel td {
    text-align: left; padding: 4px 8px; font-size: 12.5px;
    border-bottom: 1px solid rgba(127,127,127,.18);
  }
  .ky-panel th { font-weight: 600; opacity: .7; }
  .ky-mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
</style>
"""


def _rank_chip(label: str, rank: int | None) -> str:
    """Render one retrieval signal as a chip.

    Rank strength is banded, but the number is always shown so the encoding is
    never colour-only. A missing rank is meaningful rather than an error: RRF
    merges two pools, so a document found by only one retriever has no rank from
    the other.
    """
    if rank is None:
        return f'<span class="ky-chip ky-none">{label} &mdash;</span>'
    band = "ky-strong" if rank <= 10 else ("ky-mid" if rank <= 30 else "ky-weak")
    return f'<span class="ky-chip {band}">{label} #{rank}</span>'


def _card(scored, corpus) -> str:
    record = corpus.record(scored.doc_id)

    meta_bits = [b for b in (record["crop"], record["country"], record["source"]) if b]
    meta = html.escape(" · ".join(meta_bits)) or "&mdash;"

    chips = [_rank_chip("BM25", scored.bm25_rank), _rank_chip("Dense", scored.dense_rank)]
    if scored.rrf_rank is not None:
        chips.append(_rank_chip("Fused", scored.rrf_rank))
    if scored.reranker_score is not None:
        chips.append(
            f'<span class="ky-chip ky-score">'
            f"reranker {scored.reranker_score:.3f}</span>"
        )

    excerpt = record["text"]
    if len(excerpt) > 300:
        excerpt = excerpt[:300].rsplit(" ", 1)[0] + "…"

    # Only http(s) URLs become links, and a document without one says why rather
    # than rendering as a dead link. Escaping alone would not stop a
    # `javascript:` URL from executing when clicked, so the scheme is checked too.
    source_url = record["source_url"]
    if source_url.lower().startswith(("http://", "https://")):
        href = html.escape(source_url, quote=True)
        provenance = (
            f'<a class="ky-link" href="{href}" target="_blank" '
            f'rel="noopener noreferrer">View source ↗</a>'
        )
    else:
        # Most of the corpus is synthetic. Those rows still carry a `source`
        # label, but it names the organisation whose material the text was
        # modelled on -- not one that published it. Saying so on the card is the
        # difference between a demo and a misattribution.
        provenance = (
            '<div class="ky-prov">Synthetic document — written for the '
            "competition, so there is no published source to link to.</div>"
        )

    return f"""
    <div class="ky-card">
      <div class="ky-head">
        <span class="ky-rank">{scored.final_rank}</span>
        <span class="ky-title">{html.escape(record['title']) or 'Untitled'}</span>
      </div>
      <div class="ky-meta">{meta}</div>
      <div class="ky-chips">{''.join(chips)}</div>
      <p class="ky-excerpt">{html.escape(excerpt)}</p>
      {provenance}
    </div>
    """


def _panel(result, corpus) -> str:
    """Per-stage detail: any notes, then latency and the provenance table."""
    mode_info = MODES.get(result.mode, {})
    timings = result.timings.as_dict()

    rows = []
    for scored in result.final:
        title = html.escape(corpus.record(scored.doc_id)["title"])[:56]
        reranker = (
            f"{scored.reranker_score:.3f}"
            if scored.reranker_score is not None
            else "—"
        )
        cells = [
            scored.final_rank,
            title,
            scored.bm25_rank or "—",
            scored.dense_rank or "—",
            scored.rrf_rank or "—",
            reranker,
        ]
        rows.append(
            "<tr>" + "".join(f"<td class='ky-mono'>{c}</td>" for c in cells) + "</tr>"
        )

    notes = "".join(f'<div class="ky-note">{html.escape(n)}</div>' for n in result.notes)
    blend_info = ""
    if result.mode == "hybrid":
        blend_info = (
            f"<br/><span style='font-size: 12.5px; opacity: .8;'>"
            f"Score blend (α = {result.alpha:.2f}): {int(result.alpha*100)}% dense + "
            f"{int((1-result.alpha)*100)}% reranker</span>"
        )

    return f"""
    <div class="ky-panel">
      {notes}
      <strong>{html.escape(mode_info.get('label', result.mode))}</strong>{blend_info}<br/>
      {html.escape(mode_info.get('description', ''))}
      <table>
        <tr>
          <th>#</th><th>Document</th><th>BM25</th><th>Dense</th>
          <th>Fused</th><th>Reranker</th>
        </tr>
        {''.join(rows)}
      </table>
      <p style="margin:10px 0 0 0; opacity:.75; font-size:12.5px;">
        Latency &mdash; BM25 {timings['BM25']} ms &middot;
        dense {timings['dense']} ms &middot;
        rerank {timings['rerank']} ms &middot;
        total {timings['total']} ms
      </p>
    </div>
    """


def render(
    query: str,
    mode: str,
    corpus=None,
    pipeline: Pipeline | None = None,
    alpha: float = ALPHA,
) -> str:
    """Run the query and render the full result block as HTML.

    ``pipeline`` defaults to the module-level one the Space loads. ``corpus``
    defaults to that pipeline's own corpus; it is a parameter only so checks can
    substitute a hostile one to prove document text is escaped.
    """
    pipeline = pipeline or PIPELINE
    if pipeline is None:
        raise RuntimeError("No pipeline is loaded.")
    corpus = corpus if corpus is not None else pipeline.corpus

    result = pipeline.run(query, mode=mode, alpha=alpha)
    cards = "".join(_card(s, corpus) for s in result.final)

    # Worth calling out explicitly, since it is the team's contribution: the
    # reranker overruling the fusion order is the whole reason the fine-tuned
    # cross-encoder exists.
    promoted = [
        s for s in result.final if s.rrf_rank and s.final_rank and s.rrf_rank > s.final_rank
    ]
    summary = ""
    if promoted and result.mode == "hybrid":
        summary = (
            f'<div class="ky-legend">Reranking moved '
            f"<strong>{len(promoted)} of {len(result.final)}</strong> of these "
            f"documents above where fusion alone had placed them.</div>"
        )

    return (
        f"{_STYLES}<div class='ky-wrap'>{summary}{cards}"
        f"{_panel(result, corpus)}</div>"
    )


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


def search(query: str, mode: str, alpha: float = ALPHA) -> str:
    """Gradio callback. Returns HTML, or a readable error block."""
    if PIPELINE is None:
        return (
            f"{_STYLES}<div class='ky-wrap'><div class='ky-panel'>"
            "<strong>The demo is not ready.</strong><br/>"
            f"<span class='ky-mono'>{html.escape(STATE_ERROR or 'Unknown error')}</span>"
            "</div></div>"
        )
    try:
        return render(query, mode, alpha=alpha)
    except ValueError as exc:
        return (
            f"{_STYLES}<div class='ky-wrap'><div class='ky-panel'>"
            f"{html.escape(str(exc))}</div></div>"
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the visitor
        logger.exception("Search failed")
        return (
            f"{_STYLES}<div class='ky-wrap'><div class='ky-panel'>"
            f"Something went wrong: <span class='ky-mono'>"
            f"{html.escape(type(exc).__name__)}: {html.escape(str(exc))}</span>"
            "</div></div>"
        )


def clear_results() -> tuple[str, str]:
    """Reset the question box and the rendered results.

    Returns one plain value per output component. This is a named function
    rather than an inline lambda because Gradio needs a *value* here, not a
    component handle: an earlier version returned ``results.clear()``, which
    hands back a component config dict and silently writes that into the page.
    """
    return "", ""


LEGEND = (
    "<div class='ky-wrap'><div class='ky-legend'>"
    "Each document shows where it was found. <strong>BM25</strong> is keyword "
    "search, <strong>Dense</strong> is meaning-based search, <strong>Fused</strong> "
    "combines the two, and <strong>reranker</strong> is the fine-tuned model's "
    "final confidence. A dash means that method never found the document."
    "</div></div>"
)

ATTRIBUTION = """
**About the corpus.** 695 advisory documents covering 13 crops across 21 African
countries, compiled from CGIAR, FAO, Plantwise, ICRISAT, AGRA, IITA and national
extension services. Two kinds of document are mixed here, and the difference
matters when you read a result:

- **637 synthetic documents** (CC0), written for the competition to give the
  retrieval task reliable coverage. No source link — there is no upstream page.
- **58 documents grounded in published sources** (CC-BY), which carry a link to
  the original. These are the ones where you can follow a claim back.

Per-document licences are recorded in the corpus itself. The system retrieves
documents; it does not verify them, and a synthetic document carries no more
authority than the person who wrote it.

Built by **Team Kinyeti** — Israel Olawuyi Mobolaji, Harry Okah,
Edike Jeremiah, Chisom Okafor. Mentors: Oluwaseun Ajayi, Samuel Taiwo,
Adnan Adetunji.
"""


def _header() -> str:
    """Title block. Corpus size is read from the live corpus where possible, so
    the page cannot advertise a document count the index does not actually hold.
    """
    if PIPELINE is not None:
        count = f"{len(PIPELINE.corpus):,}"
        corpus_phrase = f"a corpus of {count} agricultural advisory texts"
    else:
        corpus_phrase = "a corpus of agricultural advisory texts"

    return (
        "# Agricultural Extension RAG\n"
        "**Smart retrieval for farmers.** Ask a farming question in plain "
        f"language and this returns the {TOP_K} most relevant documents from "
        f"{corpus_phrase} — each with its source and the evidence of how it "
        "was found.\n\n"
        "*Built by **Team Kinyeti**. Two BGE models were fine-tuned on this "
        "domain: a dense bi-encoder for retrieval and a cross-encoder for "
        "reranking. The reranker alone took **0.96888 nDCG@5** on the public "
        "leaderboard; the hybrid pipeline selected here took **0.93753** on the "
        "private leaderboard, which is scored on hidden queries.*"
    )


def build_interface() -> gr.Blocks:
    mode_choices = [(info["short"], key) for key, info in MODES.items()]

    with gr.Blocks(title="Agricultural Extension RAG", theme=gr.themes.Soft()) as demo:
        gr.Markdown(_header())

        with gr.Row():
            with gr.Column(scale=3):
                query_box = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. My maize leaves are turning yellow, what should I do?",
                    lines=2,
                    elem_id="query-box",
                )
            with gr.Column(scale=2):
                mode_box = gr.Radio(
                    choices=mode_choices,
                    value=DEFAULT_MODE,
                    label="Retrieval method",
                )
                alpha_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=ALPHA,
                    step=0.05,
                    label="Score blend weight (α)",
                    info="0.0 = 100% Reranker | 1.0 = 100% Dense (used in Hybrid mode)",
                )

        with gr.Row():
            submit = gr.Button("Search", variant="primary", scale=1)
            clear = gr.Button("Clear", scale=1)

        example_pills = [
            ("🌾 Planting rice in northern Nigeria", "When is the best time to plant rice in northern Nigeria?"),
            ("🌽 Maize leaves turning yellow", "My maize leaves are turning yellow, what should I do?"),
            ("🐛 Fall armyworm without chemicals", "How do I control fall armyworm without chemicals?"),
            ("🌱 Soil fertility without fertiliser", "How can I improve soil fertility without expensive fertiliser?"),
            ("☀️ Drought affecting sorghum", "What should I do about drought affecting my sorghum?"),
        ]

        gr.Markdown(
            "<div style='margin-top: 8px; margin-bottom: -2px; font-size: 13px; font-weight: 600; opacity: 0.8;'>"
            "Suggested questions:</div>"
        )
        with gr.Row():
            for label, full_text in example_pills[:3]:
                escaped = full_text.replace("'", "\\'")
                btn = gr.Button(label, size="sm", variant="secondary")
                btn.click(
                    fn=lambda text=full_text: text,
                    outputs=query_box,
                    show_progress="hidden",
                    queue=False,
                    js=f"""() => {{
                        const ta = document.querySelector('#query-box textarea');
                        if (ta) {{
                            ta.value = '{escaped}';
                            ta.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            ta.dispatchEvent(new Event('change', {{ bubbles: true }}));
                            ta.focus();
                        }}
                        return '{escaped}';
                    }}""",
                )
        with gr.Row():
            for label, full_text in example_pills[3:]:
                escaped = full_text.replace("'", "\\'")
                btn = gr.Button(label, size="sm", variant="secondary")
                btn.click(
                    fn=lambda text=full_text: text,
                    outputs=query_box,
                    show_progress="hidden",
                    queue=False,
                    js=f"""() => {{
                        const ta = document.querySelector('#query-box textarea');
                        if (ta) {{
                            ta.value = '{escaped}';
                            ta.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            ta.dispatchEvent(new Event('change', {{ bubbles: true }}));
                            ta.focus();
                        }}
                        return '{escaped}';
                    }}""",
                )

        gr.HTML(LEGEND)
        with gr.Column(elem_id="results-area"):
            results = gr.HTML()

        with gr.Accordion("How this works", open=False):
            gr.Markdown(
                "Every question goes through the same pipeline:\n\n"
                "1. **BM25** finds documents sharing the question's words — fast, "
                "but blind to meaning. Ask about *planting time* and it may return "
                "documents about *nitrogen deficiency* because they share the words "
                "'rice' and 'northern'.\n"
                "2. **Dense retrieval** encodes the question and every document as "
                "vectors, and matches on meaning rather than wording.\n"
                "3. **Reciprocal rank fusion** merges the two ranked lists by "
                "position, so neither method's raw scores need to be comparable.\n"
                "4. **The fine-tuned reranker** reads each surviving candidate "
                "alongside the question and scores how well it actually answers it. "
                "This is the model the team trained, and it is what lifts a document "
                "from a poor first-stage rank to the top.\n\n"
                "Switch the method above to compare: *BM25 only* shows raw keyword "
                "search, *Hybrid* is the full pipeline. The differences between them "
                "are the project's whole contribution.\n\n"
                "The system returns documents, not answers. Interpreting them stays "
                "with extension officers and farmers."
            )

        gr.Markdown(ATTRIBUTION)
        gr.Markdown(
            "*This demo is a competition prototype, not field advice. "
            "Always confirm guidance with a local extension officer.*"
        )

        scroll_js = "() => { document.getElementById('results-area')?.scrollIntoView({ behavior: 'smooth', block: 'start' }); }"

        submit.click(fn=None, js=scroll_js)
        submit.click(
            search,
            inputs=[query_box, mode_box, alpha_slider],
            outputs=results,
            scroll_to_output=True,
        )

        query_box.submit(fn=None, js=scroll_js)
        query_box.submit(
            search,
            inputs=[query_box, mode_box, alpha_slider],
            outputs=results,
            scroll_to_output=True,
        )
        clear.click(clear_results, outputs=[query_box, results])

    return demo


build_pipeline()
demo = build_interface()

if __name__ == "__main__":
    # Spaces sets the port; locally Gradio picks a free one.
    demo.launch(server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"))
