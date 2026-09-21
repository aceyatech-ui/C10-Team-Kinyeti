"""Copy the two fine-tuned models from Kaggle Models to the Hugging Face Hub.

Why this is necessary: the Space runs on HF infrastructure and cannot
authenticate against Kaggle, so anything it needs at runtime has to live on the
Hub. The competition notebooks read their weights from hardcoded
``/kaggle/input/...`` paths, which resolve nowhere else.

Both models are small as models go (438 MB and 2.2 GB), so this is a plain
copy -- no conversion, no quantization, no re-saving. Uploading the artefacts
unmodified is what keeps the deployed rankings identical to the leaderboard
ones.

    python prototype/scripts/upload_models.py --dry-run   # show the plan
    python prototype/scripts/upload_models.py             # do it

Authentication, both needed before running for real:

* Kaggle  -- ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` in the environment, or
  ``~/.kaggle/kaggle.json``. Kaggle only serves *your own* models to you, so
  this must be the account that owns them.
* Hub     -- ``hf auth login`` on huggingface_hub 1.x (``huggingface-cli login``
  on 0.x), or ``HF_TOKEN`` in the environment. The token needs write access.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# --- What to copy -----------------------------------------------------------
# Just owner/model is enough: the framework and variation segments are resolved
# from Kaggle's public API, because they are not guessable from the model name.
# (They are "Transformers"/"default" for both of these, but hardcoding that here
# would be a guess that fails silently the day either model is re-published
# under a different framework.)
#
# A full handle or model page URL works too, and skips the lookup:
#   https://www.kaggle.com/models/owner/model/Transformers/default
#
# `hf_repo` is the destination on the Hub, created if it does not exist. It must
# be under an account or org you can write to.
MODELS = [
    {
        "role": "dense bi-encoder",
        "kaggle": "israelolawuyi/baai-bge-base-finetune-agric-doctri-ai",
        "hf_repo": "Overwatch886/bge-base-agri",
    },
    {
        "role": "cross-encoder reranker",
        "kaggle": "israelolawuyi/bge-reranker-largetri-ai-agri",
        "hf_repo": "Overwatch886/bge-reranker-large-agri",
    },
]

# Files that must be present for the model to load. Missing weights are the one
# failure that would otherwise surface as a confusing error inside the Space
# build, minutes after the upload looked like it worked.
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".onnx")
CORE_FILES = ("config.json",)

KAGGLE_MODELS_API = "https://www.kaggle.com/api/v1/models/list?owner={owner}"


def normalise_kaggle_ref(value: str) -> list[str]:
    """Strip a Kaggle URL down to its path segments.

    Returns the segments after ``models/``, so 2 to 5 of them depending on how
    much of the handle the caller supplied.
    """
    text = value.strip()
    text = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", text)  # scheme
    text = re.sub(r"^(www\.)?kaggle\.com/", "", text)  # host
    text = re.sub(r"^models/", "", text)
    text = text.split("?", 1)[0].split("#", 1)[0]
    return [p for p in text.strip("/").split("/") if p]


def fetch_model_instances(owner: str, model: str) -> list[dict]:
    """Ask Kaggle's public API which framework/variation instances exist.

    The ``models/list`` endpoint serves public models without authentication, so
    this works before Kaggle credentials are configured -- which is the point:
    the handle can be resolved and reviewed in a dry run.
    """
    import json
    import urllib.error
    import urllib.request

    url = KAGGLE_MODELS_API.format(owner=owner)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LookupError(
            f"Could not reach Kaggle to resolve {owner}/{model} ({type(exc).__name__}). "
            "Supply the full handle instead, e.g. owner/model/framework/variation."
        ) from exc

    ref = f"{owner}/{model}"
    for entry in payload.get("models", []):
        if entry.get("ref") == ref:
            return entry.get("instances", [])
    return []


def resolve_handle(value: str) -> str:
    """Turn whatever the caller supplied into a full kagglehub handle.

    A 4- or 5-segment value is taken at face value. A 2-segment ``owner/model``
    is resolved against the Kaggle API, and a 3-segment value is resolved down to
    a variation. Where the API offers more than one candidate the error names all
    of them, so the fix is a copy-paste rather than a guess.
    """
    parts = normalise_kaggle_ref(value)
    if len(parts) in (4, 5):
        return "/".join(parts)
    if len(parts) not in (2, 3):
        raise ValueError(
            f"Could not read a Kaggle model reference out of {value!r}. Expected "
            "owner/model, owner/model/framework, or the full "
            "owner/model/framework/variation. Copy the URL from the model's "
            "Kaggle page."
        )

    owner, model = parts[0], parts[1]
    instances = fetch_model_instances(owner, model)
    if not instances:
        raise ValueError(
            f"Kaggle lists no model named {owner}/{model} with public instances. "
            "Check the name, or pass the full handle from the model page URL."
        )

    framework_filter = parts[2].lower() if len(parts) == 3 else None
    candidates = [
        i for i in instances
        if framework_filter is None or str(i.get("framework", "")).lower() == framework_filter
    ]
    if not candidates:
        available = sorted({str(i.get("framework", "?")) for i in instances})
        raise ValueError(
            f"{owner}/{model} has no {parts[2]!r} framework. Available: "
            + ", ".join(available)
        )

    if len(candidates) > 1:
        options = "\n".join(
            f"    {owner}/{model}/{i.get('framework')}/{i.get('slug')}"
            + (f"  (version {i['versionNumber']})" if i.get("versionNumber") else "")
            for i in candidates
        )
        raise ValueError(
            f"{owner}/{model} has more than one variation; pick one and put the "
            f"full handle in MODELS:\n{options}"
        )

    chosen = candidates[0]

    # Prefer the instance's own `url`: it is the canonical, correctly-cased path.
    # The `framework` field comes back lowercased ("transformers") while the URL
    # uses "Transformers", and kagglehub resolves handles by that path -- so
    # building the handle from `framework` risks a case mismatch at download time.
    url = str(chosen.get("url", ""))
    url_parts = normalise_kaggle_ref(url) if url else []
    if len(url_parts) >= 4:
        return "/".join(url_parts[:4])
    return f"{owner}/{model}/{chosen.get('framework')}/{chosen.get('slug')}"


def inspect_model_dir(path: Path) -> tuple[list[str], list[str]]:
    """Return ``(problems, notes)`` for a downloaded model directory.

    A hard problem means the upload would produce a repo that cannot load, which
    is worth catching here rather than inside a failing Space build.
    """
    problems: list[str] = []
    notes: list[str] = []

    if not path.is_dir():
        return [f"{path} is not a directory"], notes

    files = [p for p in path.rglob("*") if p.is_file()]
    if not files:
        return [f"{path} is empty"], notes

    names = {p.name for p in files}
    top = sorted(p.name for p in path.iterdir() if p.is_file())

    if not any(n.endswith(WEIGHT_SUFFIXES) for n in names):
        problems.append(
            "no model weights found (looked for "
            + ", ".join(WEIGHT_SUFFIXES)
            + ")"
        )
    for required in CORE_FILES:
        if required not in names:
            problems.append(f"missing {required}")

    # sentence-transformers wraps a transformer plus pooling config; a
    # CrossEncoder is loaded straight from the transformer files.
    if "modules.json" in names:
        notes.append("looks like a sentence-transformers model")
    elif "config_sentence_transformers.json" in names:
        notes.append("has sentence-transformers config")
    else:
        notes.append("no sentence-transformers wrapper files; treating as a plain transformer")

    if not any(n.startswith("tokenizer") or n in {"vocab.txt", "spiece.model"} for n in names):
        notes.append("no tokenizer files -- fine for a CrossEncoder, a problem for a bi-encoder")

    total = sum(p.stat().st_size for p in files)
    notes.append(f"{len(files)} files, {total / 1e6:.0f} MB")
    notes.append("top level: " + (", ".join(top[:8]) or "(all nested)"))
    return problems, notes


def check_credentials() -> list[str]:
    """Report which credentials are missing, as a list of problems."""
    problems: list[str] = []

    kaggle_env = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    kaggle_file = (Path.home() / ".kaggle" / "kaggle.json").exists()
    if not (kaggle_env or kaggle_file):
        problems.append(
            "no Kaggle credentials: set KAGGLE_USERNAME and KAGGLE_KEY, or place "
            "kaggle.json in ~/.kaggle/"
        )

    hf_token = bool(os.getenv("HF_TOKEN"))
    hf_file = (Path.home() / ".cache" / "huggingface" / "token").exists()
    if not (hf_token or hf_file):
        problems.append(
            "no Hugging Face token: run `hf auth login`, or set HF_TOKEN"
        )

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and validate config without downloading or uploading",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="re-download from Kaggle even if a cached copy exists",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        default=True,
        help="create the Hub model repos as public (default)",
    )
    parser.add_argument(
        "--private",
        dest="public",
        action="store_false",
        help="create the Hub model repos as private (the Space then needs a token)",
    )
    args = parser.parse_args(argv)

    # -- validate the config before touching the network ----------------------
    print("=== Plan ===")
    parsed = []
    config_ok = True
    for entry in MODELS:
        try:
            handle = resolve_handle(entry["kaggle"])
        except (ValueError, LookupError) as exc:
            print(f"  FAIL  {entry['role']}: {exc}")
            config_ok = False
            continue
        if "/" not in entry["hf_repo"]:
            print(f"  FAIL  {entry['role']}: hf_repo must be owner/name")
            config_ok = False
            continue
        parsed.append({**entry, "handle": handle})
        print(f"  {entry['role']}")
        print(f"    kaggle handle : {handle}")
        print(f"    -> hub repo   : {entry['hf_repo']}  ({'public' if args.public else 'private'})")

    if not config_ok:
        print("\nFix the entries in MODELS before running this for real.")
        return 1

    print("\n=== Credentials ===")
    cred_problems = check_credentials()
    if cred_problems:
        for problem in cred_problems:
            print(f"  {'warn' if args.dry_run else 'FAIL'}  {problem}")
        if args.dry_run:
            print("  (not needed for a dry run -- a real run will require both)")
        else:
            print("\nSort these out, then run again.")
            return 1
    else:
        print("  ok    Kaggle and Hugging Face credentials both present")

    if args.dry_run:
        print(
            "\nDry run: nothing downloaded, nothing uploaded.\n"
            "Re-run without --dry-run to copy the models."
        )
        return 0

    # -- imports deferred so --dry-run works without the packages installed ---
    try:
        import kagglehub
        from huggingface_hub import HfApi
    except ImportError as exc:
        print(f"\nMissing dependency: {exc.name}. Install it with:")
        print("  pip install kagglehub huggingface_hub")
        return 1

    api = HfApi()
    uploaded: list[tuple[str, str]] = []

    for entry in parsed:
        role = entry["role"]
        print(f"\n=== {role} ===")

        print(f"  downloading {entry['handle']} from Kaggle ...")
        try:
            local = Path(kagglehub.model_download(
                entry["handle"], force_download=args.force_download
            ))
        except Exception as exc:  # noqa: BLE001 - surfaced to the operator
            print(f"  FAIL  Kaggle download failed: {type(exc).__name__}: {exc}")
            print(
                "        Check the handle against the model's Kaggle page, and that\n"
                "        the credentials belong to the account that owns it."
            )
            return 1
        print(f"  downloaded to {local}")

        problems, notes = inspect_model_dir(local)
        for note in notes:
            print(f"        {note}")
        if problems:
            for problem in problems:
                print(f"  FAIL  {problem}")
            print(
                "        Refusing to upload: the repo would not load. Check that the\n"
                "        Kaggle variation you downloaded actually holds the weights."
            )
            return 1
        print("  contents look loadable")

        print(f"  creating {entry['hf_repo']} on the Hub ...")
        api.create_repo(
            repo_id=entry["hf_repo"],
            repo_type="model",
            exist_ok=True,
            private=not args.public,
        )

        print("  uploading ...")
        api.upload_folder(
            folder_path=str(local),
            repo_id=entry["hf_repo"],
            repo_type="model",
            commit_message="Add fine-tuned weights from Kaggle Models",
        )
        print(f"  done: https://huggingface.co/{entry['hf_repo']}")
        uploaded.append((role, entry["hf_repo"]))

    if not uploaded:
        print("\nNothing uploaded.")
        return 1

    dense = next((r for role, r in uploaded if "dense" in role), "<dense-repo>")
    reranker = next((r for role, r in uploaded if "rerank" in role), "<reranker-repo>")

    print("\n" + "=" * 68)
    print("Both models are on the Hub. Next: set these Space variables.")
    print("=" * 68)
    print(f"  DENSE_MODEL_ID     = {dense}")
    print(f"  RERANKER_MODEL_ID  = {reranker}")
    print(
        "\nSpace -> Settings -> Variables and secrets -> New variable.\n"
        "They are variables, not secrets: the weights are public.\n"
        "\nprototype/scripts/deploy_space.py sets them automatically if you let it\n"
        "create the Space."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
