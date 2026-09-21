---
title: Agricultural Extension RAG
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: "5.50.0"
app_file: app.py
pinned: false
license: mit
short_description: Fine-tuned retrieval over 695 agricultural documents
---

# Agricultural Extension RAG — Smart Retrieval for Farmers

Team Kinyeti's retrieval system, built for the *Agricultural Extension RAG: Smart
Retrieval for Farmers* competition. Ask a farming question in plain language; the
app returns the five most relevant documents from a corpus of 695 agricultural
advisory texts.

Every result shows **how it was found** — its BM25 rank, its dense rank, its
fused rank and its cross-encoder score. That provenance row is the point of this
demo: it shows keyword search and semantic search each missing a document that
the fine-tuned reranker then promotes, which is the project's actual
contribution.

## What is being demonstrated

Two BGE models were fine-tuned on this domain: a dense bi-encoder
(`bge-base-en-v1.5`) and a cross-encoder reranker (`bge-reranker-large`).

| Mode | What it does | nDCG@5 |
|---|---|---|
| **Hybrid** *(default)* | BM25 + dense → reciprocal rank fusion → rerank the merged pool | 0.93753 (private) |
| **Pure reranker** | Score all 695 documents with the cross-encoder, no first stage | 0.96888 (public) |
| **Dense only** | Semantic similarity, no lexical matching, no reranking | baseline |
| **BM25 only** | Classic keyword search | baseline |

The two baseline modes exist so a visitor can see what the fine-tuned models
add. Switching between *BM25 only* and *Hybrid* on the same question is the
clearest single view of the project's value.

The default is **Hybrid**, not the higher-scoring *Pure reranker*, because
0.96888 was measured on the public leaderboard while 0.93753 was measured on the
private one — hidden queries, and therefore the better generalisation signal.

### A correctness note worth keeping

The two source notebooks disagree about the reranker's `max_length`: the hybrid
notebook uses 512, the pure-reranker notebook uses 256. Both are inference
settings and both are legitimate — training used 256. They are kept as separate
per-mode constants in `src/config.py` and must not be unified into a shared
default, which would silently change one mode's rankings. See the comment there
before "tidying" it.

## Configuration

Two repository variables must be set in **Settings → Variables and secrets**, or
the app starts and reports that it is not configured rather than failing
silently:

| Variable | Value |
|---|---|
| `DENSE_MODEL_ID` | HF repo holding the fine-tuned bi-encoder |
| `RERANKER_MODEL_ID` | HF repo holding the fine-tuned cross-encoder |

They are variables rather than secrets: the weights are public, and this keeps
the same commit deployable against a different account.

## How it runs

Built for **ZeroGPU** hardware. Both models are loaded onto `cuda` at module
level, as ZeroGPU requires, and the corpus is encoded lazily on the first
question so that no GPU is claimed before a visitor arrives.

```python
@spaces.GPU(duration=120)
def rerank(self, query, doc_ids, max_length): ...
```

`@spaces.GPU` is documented as effect-free outside ZeroGPU, so the identical
code runs on a plain CPU machine for development. That is a convenience, not a
guarantee of equivalence — see below.

### The ZeroGPU process boundary

A `@gpu` function does **not** run in the calling process. ZeroGPU forks a worker
(`multiprocessing.get_context('fork')`), pickles the arguments in, and pickles
only the **return value** back:

```python
worker.arg_queue.put(((args, kwargs), ...))   # parent -> child
res = task(*args, **kwargs)                   # runs in the child
res_queue.put(OkResult(res))                  # child -> parent, return value only
```

So a `@gpu` method must *return* its result and let the caller assign it. An
assignment to `self` inside the `@gpu` function writes to the child's copy and is
discarded when the call returns — the parent's object is unchanged, and the
failure surfaces later as something unrelated, such as
`RuntimeError: Corpus not encoded`.

This is invisible locally, because off ZeroGPU the decorator degrades to a
passthrough and every local test passes. Hence a static check:

```bash
python scripts/check_gpu_purity.py
```

It parses `src/` and fails if a `@gpu` function assigns to `self` or declares
`global`. A deliberately same-call side effect can be marked with a `# gpu-ok:`
comment, which reports it as intentional rather than failing.

**Cold starts are slow and that is not a bug.** Free Spaces sleep after
inactivity and their storage is ephemeral, so the first visit after a quiet
period re-downloads ~2.7 GB of weights before serving anything.

## The corpus

695 documents, 13 crops, 21 African countries, compiled from CGIAR, FAO,
Plantwise, ICRISAT, AGRA, IITA and national extension services.

Two kinds of document are mixed, and the difference matters when reading a
result:

- **637 synthetic documents** (CC0), written for the competition. No source link.
- **58 documents grounded in published sources** (CC-BY), which carry a link to
  the original.

Per-document licences are recorded in the corpus itself. See
[data/ATTRIBUTION.md](data/ATTRIBUTION.md) for full provenance. The system
retrieves documents; it does not verify them.

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

Without `DENSE_MODEL_ID` and `RERANKER_MODEL_ID` set, the page loads and explains
that it is unconfigured. With them set, the models download on first run.

There are also checks that need no model weights at all, because the retriever is
swappable for a deterministic stub:

```bash
python scripts/check_pipeline.py     # fusion, ordering, trace, lazy encoding
python scripts/check_app.py          # rendering, escaping, error states
python scripts/check_gpu_purity.py   # no @gpu function relies on a lost side effect
python scripts/make_preview.py       # writes preview.html for design review
```

And one that needs the Space to exist, so it runs after deploying:

```bash
python scripts/check_deployed.py --repo you/your-space
```

That one asks the running app a real question in every mode. It exists because a
Space can be green on every dashboard — `RUNNING`, no build errors, hardware
attached — and still answer every visitor with an error. Nothing short of
querying it distinguishes the two.

## Deploying

See [DEPLOY.md](DEPLOY.md) for the full walkthrough. In short:

```bash
python scripts/upload_models.py --dry-run   # Kaggle Models -> HF Hub, once
python scripts/deploy_space.py --dry-run --repo your-username/kinyeti-rag
python scripts/deploy_space.py --repo your-username/kinyeti-rag --watch
```

## Layout

```
app.py                 Gradio UI, rendering and event wiring
src/
  config.py            Model IDs, per-mode constants, document flattening
  corpus.py            Loads documents.csv, applies create_search_content()
  bm25.py              Okapi BM25 index
  retrieve.py          Dense bi-encoder search and cross-encoder reranking
  scoring.py           RRF, blending, result types (no torch import)
  pipeline.py          Runs a mode end to end, records the full trace
  accelerator.py       ZeroGPU shim; no-op fallback off-platform
data/
  documents.csv        The corpus
  ATTRIBUTION.md       Provenance and per-document licensing
scripts/               Checks and the preview generator
```

`scoring.py` and `bm25.py` deliberately avoid importing torch, so the ranking
logic stays testable without downloading several gigabytes of weights.

## Scope and limitations

This is a **retrieval** demo, not an advisory service. It returns documents, not
answers, and it does not verify their contents. Interpreting the results — and
deciding what to act on — stays with extension officers and farmers. Confirm
guidance locally before acting on it.

## Team

**Team Kinyeti** — Israel Olawuyi Mobolaji, Harry Okah, Edike Jeremiah,
Chisom Okafor.

**Mentors** — Oluwaseun Ajayi, Samuel Taiwo, Adnan Adetunji.

Code is MIT licensed ([LICENSE](LICENSE)); the bundled corpus keeps its own
per-document terms ([data/ATTRIBUTION.md](data/ATTRIBUTION.md)). This is a
non-commercial demonstration built for a competition.
