"""Generate the Kaggle verification notebook.

The notebook runs the *original* hybrid pipeline from
``scripts/running-on-fine-tune-of-bge-rera-bge-dense-bm25.ipynb`` and the
*ported* pipeline in ``prototype/src/`` over the same test queries, then reports
whether they rank documents identically.

The ported source is embedded by reading the real files at generation time, so
the notebook exercises the actual shipped code rather than a hand-copied
paraphrase. Regenerate after changing anything in ``prototype/src/``:

    python prototype/scripts/make_verification_notebook.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = ROOT / "notebooks" / "verify_parity.ipynb"


def discover_modules() -> list[str]:
    """Every ``.py`` under ``src/``.

    Auto-discovered rather than hardcoded. The first draft listed the modules by
    hand and omitted ``accelerator.py``, which only surfaced on Kaggle as a
    ``ModuleNotFoundError`` from deep inside an import chain.
    """
    return sorted(p.name for p in SRC.glob("*.py"))


def check_relative_imports(sources: dict[str, str]) -> None:
    """Fail loudly if an embedded module imports a sibling we did not embed.

    Catches exactly the omission above, at generation time rather than after a
    Kaggle run.
    """
    embedded = {Path(name).stem for name in sources}
    problems = []
    for name, body in sources.items():
        for target in re.findall(r"^\s*from \.(\w+) import", body, flags=re.MULTILINE):
            # Skip the package-relative form (`from . import x`), which is fine.
            if target not in embedded:
                problems.append(f"{name} imports .{target}, which is not embedded")
    if problems:
        raise RuntimeError("Unresolved relative imports:\n  " + "\n  ".join(problems))


def read_sources() -> dict[str, str]:
    sources = {}
    for name in discover_modules():
        sources[name] = (SRC / name).read_text(encoding="utf-8")
    if not sources:
        raise FileNotFoundError(f"No Python modules found under {SRC}")
    check_relative_imports(sources)
    return sources


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def build() -> dict:
    sources = read_sources()

    write_cell = code(
        "import pathlib\n\n"
        "SRC_ROOT = pathlib.Path('src')\n"
        "SRC_ROOT.mkdir(exist_ok=True)\n\n"
        "SOURCES = " + json.dumps(sources, indent=1) + "\n\n"
        "for _name, _body in SOURCES.items():\n"
        "    (SRC_ROOT / _name).write_text(_body, encoding='utf-8')\n"
        "    print(f'wrote src/{_name} ({len(_body.splitlines())} lines)')\n"
    )

    cells = [
        md(
            "# Verification: ported pipeline vs original notebook\n"
            "\n"
            "**What this does.** Runs the same 200 test queries through two\n"
            "implementations of the hybrid retrieval pipeline and reports whether\n"
            "they rank documents identically:\n"
            "\n"
            "1. **Original** — the pipeline exactly as it appears in\n"
            "   `running-on-fine-tune-of-bge-rera-bge-dense-bm25.ipynb`, the code that\n"
            "   produced the competition submission.\n"
            "2. **Ported** — the re-housed version in `prototype/src/`, embedded below\n"
            "   from the real source files, which is what the web prototype will run.\n"
            "\n"
            "**Why.** The port moved the same maths into new files so it could answer\n"
            "one question at a time and load models from Hugging Face. That\n"
            "restructuring is where a silent bug would hide — most likely in the\n"
            "reranker's `max_length`, which the two original notebooks disagree about\n"
            "(512 for hybrid, 256 for pure reranker).\n"
            "\n"
            "**Setup required before running:**\n"
            "\n"
            "- Add the competition dataset as an input (for `documents.csv` and\n"
            "  `test_queries.csv`).\n"
            "- Add both fine-tuned models as inputs.\n"
            "- Set **Accelerator → GPU** (T4 or P100). CPU will take far too long.\n"
            "- Turn **Internet on** if you want the pip install cell to run.\n"
            "\n"
            "**Result.** The final cell prints PASS or FAIL. PASS means the port is\n"
            "faithful and safe to deploy.\n"
        ),
        code(
            "import os\n"
            "import subprocess\n"
            "import sys\n"
            "\n"
            "subprocess.run(\n"
            "    [sys.executable, '-m', 'pip', 'install', '-q', 'rank-bm25',\n"
            "     'sentence-transformers'],\n"
            "    check=False,\n"
            ")\n"
            "\n"
            "import torch\n"
            "print('torch', torch.__version__)\n"
            "print('cuda available:', torch.cuda.is_available())\n"
            "if not torch.cuda.is_available():\n"
            "    print('WARNING: no GPU. This will be extremely slow.')\n"
        ),
        md(
            "## 1. Load data and models\n"
            "\n"
            "Paths are auto-detected by searching `/kaggle/input`, because the mount\n"
            "folder names change depending on how inputs were attached.\n"
        ),
        code(
            "import glob\n"
            "import pandas as pd\n"
            "\n"
            "\n"
            "def find_one(pattern, what):\n"
            "    hits = glob.glob(f'/kaggle/input/**/{pattern}', recursive=True)\n"
            "    if not hits:\n"
            "        raise FileNotFoundError(\n"
            "            f'Could not find {what} ({pattern}) under /kaggle/input. '\n"
            "            'Did you add it as an input to this notebook?'\n"
            "        )\n"
            "    print(f'{what}: {hits[0]}')\n"
            "    return hits[0]\n"
            "\n"
            "\n"
            "DOCS_CSV = find_one('documents.csv', 'documents.csv')\n"
            "DENSE_DIR = find_one('fine_tuned_bge_base_agri', 'dense bi-encoder')\n"
            "RERANK_DIR = find_one('fine_tuned_bge_reranker', 'cross-encoder reranker')\n"
            "\n"
            "# test_queries.csv may sit alongside documents.csv or in its own folder.\n"
            "try:\n"
            "    TEST_CSV = find_one('test_queries.csv', 'test_queries.csv')\n"
            "    test = pd.read_csv(TEST_CSV)\n"
            "except FileNotFoundError:\n"
            "    TEST_CSV = None\n"
            "    print('test_queries.csv not found; will fall back to training queries.')\n"
            "\n"
            "df = pd.read_csv(DOCS_CSV, index_col='document_id')\n"
            "print(f'corpus: {len(df)} documents')\n"
            "\n"
            "# Uncomment to shorten the run while debugging.\n"
            "# QUERY_LIMIT = 20\n"
            "QUERY_LIMIT = None\n"
        ),
        md("## 2. Original pipeline (verbatim from the competition notebook)"),
        code(
            "import numpy as np\n"
            "import torch\n"
            "from rank_bm25 import BM25Okapi\n"
            "from sentence_transformers import CrossEncoder, SentenceTransformer\n"
            "from tqdm.auto import tqdm\n"
            "\n"
            "# --- flattening: identical to the competition notebook ---\n"
            "def create_search_content(row):\n"
            "    title = str(row.get('title', ' ')).strip()\n"
            "    text = str(row.get('text', ' ')).strip()\n"
            "    source = str(row.get('source', ' ')).strip()\n"
            "    crop = str(row.get('crop', ' ')).strip()\n"
            "    country = str(row.get('country', ' ')).strip()\n"
            "    source_url = str(row.get('source_url', ' ')).strip()\n"
            "    return f'Crop {crop} | Country {country} | Title {title} | Text {text} | Source {source}'\n"
            "\n"
            "\n"
            "df['search_text'] = df.apply(create_search_content, axis=1)\n"
            "docs_ids = df.index.to_list()          # NOTE: ints here\n"
            "corpus_texts = df['search_text'].tolist()\n"
            "\n"
            "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
            "print('device:', device)\n"
            "\n"
            "orig_dense = SentenceTransformer(DENSE_DIR, device=device)\n"
            "orig_reranker = CrossEncoder(RERANK_DIR, max_length=512, device=device)\n"
            "\n"
            "id_to_text = dict(zip(docs_ids, corpus_texts))\n"
            "num_docs = len(corpus_texts)\n"
            "\n"
            "tokenized_corpus = [doc.lower().split() for doc in corpus_texts]\n"
            "bm25 = BM25Okapi(tokenized_corpus)\n"
            "\n"
            "print('Encoding corpus with the dense bi-encoder...')\n"
            "doc_embeddings = orig_dense.encode(\n"
            "    corpus_texts, batch_size=64, show_progress_bar=True,\n"
            "    normalize_embeddings=True, convert_to_tensor=True, device=device,\n"
            ")\n"
            "\n"
            "\n"
            "def search_bm25(query_text, top_k=50):\n"
            "    tokenized_query = query_text.lower().split()\n"
            "    scores = bm25.get_scores(tokenized_query)\n"
            "    top_indices = np.argsort(scores)[::-1][:top_k]\n"
            "    return [docs_ids[i] for i in top_indices]\n"
            "\n"
            "\n"
            "def search_dense(query_text, top_k=50):\n"
            "    prefixed_query = f'Represent this sentence for searching relevant passages: {query_text}'\n"
            "    q_emb = orig_dense.encode(\n"
            "        prefixed_query, normalize_embeddings=True,\n"
            "        convert_to_tensor=True, device=device,\n"
            "    )\n"
            "    sim_scores = torch.matmul(doc_embeddings, q_emb)\n"
            "    top_res = torch.topk(sim_scores, k=min(top_k, num_docs))\n"
            "    indices = top_res.indices.cpu().numpy()\n"
            "    scores = top_res.values.cpu().numpy()\n"
            "    dense_scores_dict = {docs_ids[i]: float(scores[idx]) for idx, i in enumerate(indices)}\n"
            "    return list(dense_scores_dict.keys()), dense_scores_dict\n"
            "\n"
            "\n"
            "def reciprocal_rank_fusion(bm25_ids, dense_ids, k=60):\n"
            "    rrf_scores = {}\n"
            "    for rank, doc_id in enumerate(bm25_ids):\n"
            "        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)\n"
            "    for rank, doc_id in enumerate(dense_ids):\n"
            "        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)\n"
            "    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)\n"
            "    return [doc_id for doc_id, _ in sorted_docs]\n"
            "\n"
            "\n"
            "CANDIDATE_POOL_SIZE = 50\n"
            "ALPHA = 0.0\n"
            "\n"
            "\n"
            "def original_rank(q_text):\n"
            "    bm25_top = search_bm25(q_text, top_k=CANDIDATE_POOL_SIZE)\n"
            "    dense_top_ids, dense_score_map = search_dense(q_text, top_k=CANDIDATE_POOL_SIZE)\n"
            "    candidate_ids = reciprocal_rank_fusion(bm25_top, dense_top_ids, k=60)[:CANDIDATE_POOL_SIZE]\n"
            "    pairs = [[q_text, id_to_text[d]] for d in candidate_ids]\n"
            "    raw = orig_reranker.predict(pairs, batch_size=32, convert_to_numpy=True)\n"
            "    norm_reranker = 1.0 / (1.0 + np.exp(-raw))\n"
            "    blended = []\n"
            "    for idx, d in enumerate(candidate_ids):\n"
            "        d_score = dense_score_map.get(d, 0.0)\n"
            "        blended.append((ALPHA * d_score) + ((1.0 - ALPHA) * norm_reranker[idx]))\n"
            "    blended = np.array(blended)\n"
            "    top_5 = np.argsort(blended)[::-1][:5]\n"
            "    return [candidate_ids[i] for i in top_5]\n"
        ),
        md(
            "## 3. Ported pipeline\n"
            "\n"
            "The cells below write the real `prototype/src/` modules to disk and import\n"
            "them. The source text is embedded verbatim at generation time, so this\n"
            "tests the shipped code.\n"
        ),
        write_cell,
        md("Now import the ported modules and build the index."),
        code(
            "import importlib\n"
            "import sys\n"
            "\n"
            "sys.path.insert(0, '.')\n"
            "for _name in list(sys.modules):\n"
            "    if _name == 'src' or _name.startswith('src.'):\n"
            "        del sys.modules[_name]\n"
            "\n"
            "from src.config import RERANKER_MAX_LENGTH\n"
            "from src.corpus import load_corpus\n"
            "from src.bm25 import BM25Index\n"
            "from src.retrieve import Retriever\n"
            "from src.scoring import blend_scores, reciprocal_rank_fusion as ported_rrf\n"
            "\n"
            "# The ported corpus loader would look for a local documents.csv; point it\n"
            "# at the Kaggle path explicitly.\n"
            "import os\n"
            "os.environ['DOCUMENTS_CSV'] = DOCS_CSV\n"
            "corpus = load_corpus(DOCS_CSV)\n"
            "print(f'ported corpus: {len(corpus)} documents')\n"
            "print(f\"reranker max_length by mode: {RERANKER_MAX_LENGTH}\")\n"
            "\n"
            "ported_bm25 = BM25Index(corpus).build()\n"
            "retriever = Retriever(corpus, DENSE_DIR, RERANK_DIR, device=device)\n"
            "retriever.encode_corpus()\n"
            "print('ported index ready')\n"
        ),
        code(
            "def ported_rank(q_text):\n"
            "    bm25_hits = ported_bm25.search(q_text, top_k=50)\n"
            "    dense_hits = retriever.search_dense(q_text, top_k=50)\n"
            "\n"
            "    bm25_ids = [d for d, _ in bm25_hits]\n"
            "    dense_ids = [d for d, _ in dense_hits]\n"
            "    dense_scores = dict(dense_hits)\n"
            "\n"
            "    fused = ported_rrf(bm25_ids, dense_ids, k=60)[:50]\n"
            "    candidate_ids = [d for d, _ in fused]\n"
            "\n"
            "    logits = retriever.rerank(\n"
            "        q_text, candidate_ids, max_length=RERANKER_MAX_LENGTH['hybrid']\n"
            "    )\n"
            "    rerank_scores = dict(logits)\n"
            "\n"
            "    blended = blend_scores(\n"
            "        [rerank_scores[d] for d in candidate_ids],\n"
            "        [dense_scores.get(d, 0.0) for d in candidate_ids],\n"
            "        alpha=0.0,\n"
            "    )\n"
            "    order = np.argsort(np.array(blended))[::-1][:5]\n"
            "    return [candidate_ids[i] for i in order]\n"
        ),
        md(
            "## 4. Compare\n"
            "\n"
            "Document IDs are compared as strings: the original notebook keeps integer\n"
            "IDs while the port uses strings, which is a representation difference and\n"
            "not a ranking difference.\n"
        ),
        code(
            "if TEST_CSV is None:\n"
            "    # Fall back to the corpus titles as pseudo-queries so the cell still runs.\n"
            "    queries = df['title'].dropna().astype(str).head(50).tolist()\n"
            "    query_ids = [f'q{i}' for i in range(len(queries))]\n"
            "    print('Using corpus titles as stand-in queries.')\n"
            "else:\n"
            "    qid_col = 'QueryId' if 'QueryId' in test.columns else test.columns[0]\n"
            "    qtext_col = 'Query' if 'Query' in test.columns else test.columns[1]\n"
            "    subset = test if QUERY_LIMIT is None else test.head(QUERY_LIMIT)\n"
            "    query_ids = [str(v).replace('.0', '') for v in subset[qid_col]]\n"
            "    queries = [str(v).strip() for v in subset[qtext_col]]\n"
            "\n"
            "print(f'Comparing {len(queries)} queries...')\n"
            "\n"
            "mismatches = []\n"
            "for qid, qtext in tqdm(list(zip(query_ids, queries))):\n"
            "    try:\n"
            "        a = [str(x) for x in original_rank(qtext)]\n"
            "        b = [str(x) for x in ported_rank(qtext)]\n"
            "    except Exception as exc:\n"
            "        mismatches.append((qid, qtext, f'ERROR: {exc}', '', ''))\n"
            "        continue\n"
            "    if a != b:\n"
            "        mismatches.append((qid, qtext, a, b, len(set(a) & set(b))))\n"
            "\n"
            "print()\n"
            "print('=' * 70)\n"
            "if not mismatches:\n"
            "    print(f'PASS - all {len(queries)} queries ranked identically.')\n"
            "    print('The port is faithful; safe to deploy.')\n"
            "else:\n"
            "    print(f'FAIL - {len(mismatches)} of {len(queries)} queries differ.')\n"
            "    print('=' * 70)\n"
            "    for qid, qtext, a, b, overlap in mismatches[:10]:\n"
            "        print(f'\\nQuery {qid}: {qtext}')\n"
            "        print(f'  original: {a}')\n"
            "        print(f'  ported:   {b}')\n"
            "        if overlap != '':\n"
            "            print(f'  shared documents: {overlap}/5')\n"
            "print('=' * 70)\n"
        ),
        md(
            "## Reading the result\n"
            "\n"
            "- **PASS** — the port ranks identically. Deploy it.\n"
            "- **FAIL with high overlap** (4/5 shared, different order) — a tie-breaking\n"
            "  or floating-point difference. Usually harmless, but worth a look.\n"
            "- **FAIL with low overlap** — a real bug. The `max_length` setting is the\n"
            "  first thing to check, since the hybrid and pure-reranker notebooks use\n"
            "  different values.\n"
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
                "mimetype": "text/x-python",
                "file_extension": ".py",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    notebook = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")

    n_code = sum(1 for c in notebook["cells"] if c["cell_type"] == "code")
    print(f"Wrote {OUT}")
    print(f"  {len(notebook['cells'])} cells ({n_code} code)")
    print(f"  embedded modules: {', '.join(discover_modules())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
