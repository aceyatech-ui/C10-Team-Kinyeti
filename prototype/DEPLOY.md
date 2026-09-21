# Deploying the prototype to Hugging Face Spaces

This is the whole path from "the code is on my laptop" to "there is a public URL
the newsletter can link to". Two steps need you specifically — uploading the
model weights and having an eligible account — and the rest is one command.

Estimated time: **20–30 minutes**, most of it waiting for uploads. Plus one hard
wait if your HF account is new (see below).

## What you need first

| | Why | How to check |
|---|---|---|
| Kaggle account that owns the two fine-tuned models | They are the only copy of the weights | You published them, so you have this |
| HF account, **email verified, older than 30 days** | Free ZeroGPU hosting requires both | `huggingface.co/settings/account` |
| HF **write** token | Needed to create repos and push | `huggingface.co/settings/tokens` |
| `documents.csv` | The corpus the Space serves | Already at `data/documents.csv` |

⚠️ **The 30-day rule is the only thing that cannot be worked around.** Free
personal accounts can host up to 2 ZeroGPU Spaces, but only if the account is in
good standing, and HF defines that as verified email *and* older than 30 days. If
your account is newer, everything below still works — the Space just runs on CPU
until it qualifies, which makes the *Pure reranker* mode take minutes instead of
seconds. Start the account clock now if you have not.

## Step 0 — install and log in

```bash
pip install huggingface_hub kagglehub pyyaml
hf auth login                  # paste a token with WRITE access
```

The token must have the **write** role, not `read` — creating repos and pushing
files both need it.

> On `huggingface_hub` **1.x** the CLI is `hf`, and login sits under `auth`. On
> **0.x** the same command is `huggingface-cli login`, with no `auth`. Check with
> `hf version`; if `hf` is not found, you are on 0.x.

Kaggle credentials come from either `~/.kaggle/kaggle.json` (Kaggle → Account →
Create New API Token) or the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment
variables. Kaggle serves *your own* models only to you, so this must be the
account that owns them.

## Step 1 — upload the models to the Hub

The Space runs on HF infrastructure and cannot authenticate against Kaggle. The
competition notebooks read weights from hardcoded `/kaggle/input/...` paths,
which resolve nowhere else. So the weights have to move first.

```bash
python prototype/scripts/upload_models.py --dry-run   # check the plan
python prototype/scripts/upload_models.py             # ~2.7 GB, takes a while
```

`MODELS` at the top of [scripts/upload_models.py](scripts/upload_models.py)
already points at both Kaggle models and at the destination repos on the Hub. A
Kaggle handle needs four parts — `owner/model/framework/variation` — and the last
two are not guessable from the model name, so the script asks Kaggle's API for
them rather than relying on a hardcoded value. The dry run prints the resolved
handles; **check that line**, because a wrong one fails only at download time,
after the transfer has started.

The script validates each download before uploading — if the weights are not
actually in the directory, or `config.json` is missing, it refuses rather than
publishing a repo that cannot load. That check exists because the alternative is
a Space that builds cleanly and then fails on every request with an error that
points nowhere useful.

It prints the two repo IDs at the end. They are what the Space needs next.

## Step 2 — deploy the Space

```bash
python prototype/scripts/deploy_space.py --dry-run --repo Overwatch886/kinyeti-rag
python prototype/scripts/deploy_space.py --repo Overwatch886/kinyeti-rag --watch
```

The dry run checks everything it can without touching the Hub: that the Space
card parses and pins the same Gradio version you have installed locally, that
every Python file compiles, that the required files are present, and that the
corpus is where it should be. **Do not skip it** — a Gradio version mismatch
between the frontmatter and your local install is a deploy that succeeds and then
fails at runtime.

The real run then:

1. **Stages the corpus.** The Space repo root is `prototype/`, so it cannot see
   `data/documents.csv` at the repository root. The script copies it in.
   That copy is gitignored — one authoritative file, staged at deploy time,
   rather than two that drift.
2. **Creates the Space** as a public Gradio Space and requests ZeroGPU hardware.
3. **Sets `DENSE_MODEL_ID` and `RERANKER_MODEL_ID`** as Space variables, from the
   repo IDs. They are variables rather than secrets because the weights are
   public, which keeps the same commit deployable against a different account.
4. **Verifies the model repos exist** before uploading. This is the check that
   matters most: missing weights are the one failure that produces a Space that
   builds fine and errors on every single request.
5. **Uploads `prototype/`** and, with `--watch`, polls until the build finishes —
   printing the build logs if it fails rather than leaving you to find them.

If hardware allocation is refused (usually the 30-day rule), the script says so,
creates the Space anyway, and tells you to set it by hand. The app runs on CPU
until then.

## Step 3 — check it

`https://huggingface.co/spaces/Overwatch886/team-kinyeti-agricultural-extension-rag`

**Do not trust the build status.** `RUNNING` means the container started, and
nothing more. A Space that never received the corpus, or never got its model
variables, also reaches `RUNNING` and then answers every visitor with
"The demo is not ready" — a link that is broken for everyone while looking
perfectly healthy from outside. Ask it a real question instead:

```bash
python prototype/scripts/check_deployed.py --repo Overwatch886/team-kinyeti-agricultural-extension-rag
```

That waits for the Space to come up, then queries all four modes and confirms
documents come back. It covers the whole path — variables, weights, GPU, corpus,
rendering — which no amount of checking from the outside can.

Then, by hand:

- **Build logs:** append `?logs=build`.
- **First load is slow, and that is not a bug.** Free Spaces sleep after
  inactivity and their storage is ephemeral, so the first visit after a quiet
  period re-downloads ~2.7 GB of weights. Give it several minutes. Later visits
  are fast.
- **Test in a logged-out browser**, because that is how newsletter readers will
  arrive, and it exercises the different quota tier.
- Try a question in each mode. Switching *BM25 only* → *Hybrid* on the same
  question is the demo's whole argument, so make sure that contrast actually
  renders.

## What visitors will experience

ZeroGPU quota is per-visitor, per-day: **2 minutes** unauthenticated, **5 minutes**
for a free signed-in account. A query costs roughly 1–3 seconds, so a normal
visitor will not come close — but expect queuing on a busy newsletter day, and
expect the first visitor after a cold start to wait.

Budget your own testing accordingly: **5 minutes a day is easy to burn while
debugging**, and being locked out of your own demo mid-check is irritating. Do
your iteration on the local checks (`scripts/check_*.py`), which need no GPU at
all, and spend GPU minutes only on confirming the deployed thing works.

## Updating the Space later

Edit locally, re-run the same command with the same `--repo`. There is no cache
to clear; the Space rebuilds from the upload. Runtime state is wiped on every
rebuild.

If you prefer git, the Space is an ordinary repo and `git subtree push` works,
but you must first place the corpus at `prototype/data/documents.csv` by hand,
since the deploy script normally stages it.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Build logs show `Could not find documents.csv`, or the page says "The demo is not ready" with a `FileNotFoundError` | A `.gitignore` committed to the Space excludes the corpus. The Hub applies a committed `.gitignore` to the commit **server-side**, so the file is dropped even though the upload succeeds. | Never add `data/documents.csv` to `prototype/.gitignore` — the staged copy is kept out of *git* by the repository-root `.gitignore`. Re-run the deploy script; it now reads the repo back and fails if the corpus is missing. |
| Build succeeds, every search errors | Model repos missing or private | Re-run `deploy_space.py`; it verifies them before uploading |
| `AttributeError` / import error from Gradio at runtime | Frontmatter `sdk_version` differs from the version the code was written against | Correct `sdk_version`, or the local install, so they match |
| Space runs but is slow, no GPU | ZeroGPU not assigned, or account not yet eligible | Settings → Hardware → ZeroGPU |
| "The demo is not ready" in the page, naming `DENSE_MODEL_ID` / `RERANKER_MODEL_ID` | The Space variables are unset. `create_repo(space_variables=...)` accepts the argument but does **not** persist it | Space → Settings → Variables, or re-run the deploy script, which now sets them explicitly and checks the API's response |
| First request after idle takes minutes | Cold start re-downloading weights | Expected; not fixable on free storage |
| `torch` install errors during build | A pinned torch outside ZeroGPU's supported range | `requirements.txt` deliberately does not pin torch — do not add one |

## What this deliberately does not do

The app returns **documents, not answers**. A small multilingual LLM could
generate cited answers from the retrieved passages, and `RetrievalResult` in
[src/scoring.py](src/scoring.py) is the seam where that would attach without
touching retrieval. It is out of scope here for a reason worth stating: the
project's own impact statement is explicit that extension officers, not the
system, remain the interpreters of results. Adding a generator would blur the
thing the team actually built — a retrieval system that won on hidden queries.
