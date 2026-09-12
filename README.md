# C10-Team-Kinyeti — Agricultural Extension RAG: Smart Retrieval for Farmers

Team Kinyeti's solution to the **Agricultural Extension RAG: Smart Retrieval for Farmers** competition. The task: given a farmer's natural-language question, retrieve the top 5 most relevant documents from a corpus of agricultural advisory content. Our system **fine-tuned BGE embedding and reranker models** and scored up to **0.96888 nDCG@5** on the public leaderboard.

## Dataset

The competition provided the data (access by invitation — contact **cohort-10@tri-ai.org**):

| File | Contents |
|---|---|
| `documents.csv` | 695 agricultural advisory documents (`title`, `text`, `source`, `crop`, `country`, `source_url`), indexed by `document_id` |
| `train_queries.csv` | Training queries (`QueryId`, query text) |
| `qrels_train.csv` | Graded relevance judgments (`QueryId`, `DocumentId`, `relevance`) |
| `test_queries.csv` | 200 held-out test queries |


Aside from the corpus provided by the competition, we also documented these as candidate collection sources in our data card when completing the Data Card Challenge ([docs/data_card.pdf](docs/data_card.pdf), [data/README.md](data/README.md)).

**Preprocessing.** Each document is flattened into a single searchable string — `Crop {crop} | Country {country} | Title {title} | Text {text} | Source {source}` — used consistently for training and retrieval.

## Training Pipeline

Two models were fine-tuned (notebooks run on Kaggle with GPU accelerators, multi-GPU via `DataParallel`):

### 1. Dense bi-encoder — [fine-tuning-bge-base-dense-model.ipynb](scripts/fine-tuning-bge-base-dense-model.ipynb)

Fine-tunes `BAAI/bge-base-en-v1.5` (mean pooling + L2 normalization).

- **Training data:** 6,108 (query, positive, negative) triplets built from the qrels — documents with relevance ≥ 2 are positives, relevance = 0 are hard negatives; queries get the BGE instruction prefix `"Represent this sentence for searching relevant passages: "`.
- **Loss:** InfoNCE contrastive loss with in-batch negatives (cosine similarity × scale 20.0).
- **Hyperparameters:** AdamW lr 2e-5, 3 epochs, effective batch size 32, max sequence length 256, 10% linear warmup.

### 2. Cross-encoder reranker — [fine-tuning-bge-reranker-model.ipynb](scripts/fine-tuning-bge-reranker-model.ipynb)

Fine-tunes `BAAI/bge-reranker-large` as a sequence-classification model producing a single relevance logit per query–document pair.

- **Training data:** pairwise preference pairs — for every query, all ordered document pairs (A, B) from the graded qrels where A is more relevant than B.
- **Loss:** `MarginRankingLoss` (margin 0.2), optimizing the model to score the more relevant document higher.
- **Hyperparameters:** AdamW lr 2e-5, 4 epochs, batch size 2 per GPU, max sequence length 256, 10% linear warmup.

**Key design choices.** We submitted two configurations: the pure fine-tuned reranker (best public score) and a hybrid pipeline (BM25 + dense + reranking, best private score). Both rely on the same two fine-tuned models.

**Hyperparameter search.** We grid-searched the dense/reranker score-blending weight α ∈ {0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0} on a 20% held-out query split using [ranx](https://github.com/AmenRa/ranx); α = 0.0 (pure reranker) performed best.

## Evaluation

Evaluation with ranx on 20% held-out training queries (dense retrieval → top-50 candidates → α-blended reranker scores), with a Student's t-test for significance. This served as our model-selection tool for comparing configurations; since the models were fine-tuned on the full training set and evaluation candidates were capped at the dense top-50, the resulting numbers are indicative rather than unbiased:

| Configuration | nDCG@5 | nDCG@10 | MRR@10 | Recall@50 |
|---|---|---|---|---|
| **Pure reranker (α=0.0)** | **0.905** | **0.889** | 0.951 | 0.799 |
| α=0.2 | 0.900 | 0.885 | 0.961 | 0.799 |
| α=0.4 | 0.877 | 0.871 | 0.958 | 0.799 |
| Pure dense (α=1.0) | 0.825 | 0.820 | 0.958 | 0.799 |

**Final results** (nDCG@5; the private set is 47% of the test data):

| Submission | Public LB | Private LB |
|---|---|---|
| Pure fine-tuned reranker | **0.96888** | 0.92877 |
| Hybrid (BM25 + dense + RRF + reranker) | 0.95918 | **0.93753** |

Submissions were format-validated before export (5 unique documents per query, correct row count) with hard assertions and checks in the notebooks.

## Reproduction

All notebooks are designed for the Kaggle environment (GPU accelerator required; 2× T4 recommended). Dependencies are in [scripts/requirements.txt](scripts/requirements.txt) — see [scripts/README.md](scripts/README.md) for details.

1. **Get access** — request a competition invitation from cohort-10@tri-ai.org.
2. **Fine-tune the dense bi-encoder** — run [fine-tuning-bge-base-dense-model.ipynb](scripts/fine-tuning-bge-base-dense-model.ipynb); saves `fine_tuned_bge_base_agri`.
3. **Fine-tune the reranker** — run [fine-tuning-bge-reranker-model.ipynb](scripts/fine-tuning-bge-reranker-model.ipynb); saves `fine_tuned_bge_reranker`.
4. **Upload both models to Kaggle Models** (already published: [dense](https://www.kaggle.com/models/israelolawuyi/baai-bge-base-finetune-agric-doctri-ai), [reranker](https://www.kaggle.com/models/israelolawuyi/bge-reranker-largetri-ai-agri)).
5. **Generate the primary submission** — run [running-inference-on-fine-tune-of-bge-reranker.ipynb](scripts/running-inference-on-fine-tune-of-bge-reranker.ipynb): the reranker scores all 139,000 query×document pairs and outputs the top 5 per query → `submission_finetuned_reranker.csv`.
6. **Generate the second submission (hybrid pipeline)** — run [running-on-fine-tune-of-bge-rera-bge-dense-bm25.ipynb](scripts/running-on-fine-tune-of-bge-rera-bge-dense-bm25.ipynb): BM25 + dense retrieval fused via reciprocal rank fusion (k=60) → top 50 → cross-encoder rerank → top 5 → `submission_hybrid_finetuned_reranker.csv`.

## Appendix

**Team Kinyeti** (contributors):

- *Israel Olawuyi Mobolaji*
- *Harry Okah*
- *Edike jeremiah*
- *Chisom Okafor*

**Mentors:**

- *Oluwaseun Ajayi*
- *Samuel Taiwo*
- *Adnan Adetunji*
