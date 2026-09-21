"""Create the Hugging Face Space and push the prototype to it.

This does with the Hub API what would otherwise be a git dance: creates the
Space repo, sets its hardware and configuration variables, uploads
``prototype/``, and then watches the build so a failure is reported here rather
than discovered in a browser.

    python prototype/scripts/deploy_space.py --dry-run      # check everything
    python prototype/scripts/deploy_space.py --repo you/kinyeti-rag
    python prototype/scripts/deploy_space.py --repo you/kinyeti-rag --update

Requires a Hub token with **write** access (``hf auth login`` on huggingface_hub
1.x, ``huggingface-cli login`` on 0.x, or ``HF_TOKEN``). The Space is created
public by default; pass ``--private`` to make it private while you check it over.

Two things only you can do, and this script checks for them rather than failing
obscurely later:

1. **Upload the fine-tuned models first** (``upload_models.py``). If the model
   repos do not exist on the Hub, the Space build will fail at import.
2. **An account eligible for free ZeroGPU** -- verified email, older than 30
   days. If hardware allocation is refused, the Space still builds; set the
   hardware by hand in Settings and it will start using the GPU.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - reported cleanly in main()
    yaml = None

# `prototype/` is the Space repo root; this file lives in `prototype/scripts/`.
SPACE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SPACE_ROOT.parent

# Staged into the Space at build time rather than committed twice: the
# authoritative copy stays at the repository root, next to the notebooks.
CORPUS_SOURCE = REPO_ROOT / "data" / "documents.csv"
CORPUS_DEST = SPACE_ROOT / "data" / "documents.csv"

# Never uploaded. `preview.html` is a local review artefact; the rest is noise
# the Hub would otherwise serve from the Space's file browser.
UPLOAD_IGNORE = [
    "preview.html",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.ipynb_checkpoints/**",
    "**/.pytest_cache/**",
]


def check_frontmatter(problems: list[str]) -> dict:
    """Validate the Space card, which is what the Hub reads to build the app."""
    readme = SPACE_ROOT / "README.md"
    if not readme.exists():
        problems.append("prototype/README.md is missing; the Hub needs it")
        return {}

    text = readme.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not match:
        problems.append("prototype/README.md has no YAML frontmatter block")
        return {}

    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        problems.append(f"frontmatter does not parse: {exc}")
        return {}

    if meta.get("sdk") != "gradio":
        problems.append(f"sdk must be 'gradio' for ZeroGPU, got {meta.get('sdk')!r}")
    app_file = meta.get("app_file")
    if not app_file:
        problems.append("frontmatter has no app_file")
    elif not (SPACE_ROOT / app_file).exists():
        problems.append(f"frontmatter app_file {app_file!r} does not exist")
    if not meta.get("title"):
        problems.append("frontmatter has no title")
    if len(meta.get("short_description", "")) > 60:
        problems.append(
            f"short_description is {len(meta['short_description'])} chars; "
            "the Hub truncates at 60"
        )
    return meta


def check_gradio_version(meta: dict, problems: list[str], notes: list[str]) -> None:
    """Warn when the pinned SDK version will not run the installed Gradio.

    The Space builds with `sdk_version` from the frontmatter, so a mismatch here
    means the deploy succeeds and then the app fails at runtime with an API error
    that points nowhere useful.
    """
    pinned = str(meta.get("sdk_version", "")).strip()
    if not pinned:
        notes.append("no sdk_version pinned; the Hub will pick one")
        return
    try:
        import gradio
    except ImportError:
        notes.append("gradio not installed locally; skipping the version cross-check")
        return

    installed = gradio.__version__
    if pinned != installed:
        problems.append(
            f"frontmatter pins gradio {pinned} but {installed} is installed "
            "locally. They must match, or code that works here will fail in the "
            "Space. Update sdk_version in prototype/README.md."
        )
    else:
        notes.append(f"gradio {pinned} matches the local install")


def check_sources_compile(problems: list[str], notes: list[str]) -> None:
    """Byte-compile every Python file that ships, so syntax errors surface here."""
    files = sorted(SPACE_ROOT.glob("*.py")) + sorted((SPACE_ROOT / "src").glob("*.py"))
    if not files:
        problems.append("no Python sources found to deploy")
        return
    for path in files:
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            problems.append(f"{path.relative_to(SPACE_ROOT)} line {exc.lineno}: {exc.msg}")
    notes.append(f"{len(files)} Python files compile")


def check_gpu_purity(problems: list[str], notes: list[str]) -> None:
    """Fail the deploy if a ``@gpu`` function relies on a side effect.

    This is the last gate before a public URL, and it is the one failure that no
    local run can catch: off ZeroGPU ``@gpu`` degrades to a passthrough, so code
    that assigns to ``self`` inside a decorated method passes every test here and
    then fails on every request in the Space. Run before the push so the Space is
    never rebuilt into a known-broken state.
    """
    try:
        import check_gpu_purity as purity
    except ImportError:  # pragma: no cover - only if the script is moved alone
        notes.append("gpu purity check unavailable; run scripts/check_gpu_purity.py by hand")
        return

    found: list[str] = []
    noted = 0
    for path in sorted((SPACE_ROOT / "src").rglob("*.py")):
        file_problems, file_acknowledged = purity.check_file(path)
        noted += len(file_acknowledged)
        for lineno, func, description in file_problems:
            found.append(f"{path.name}:{lineno} in {func}() -- {description}")

    if found:
        problems.append(
            "these @gpu functions write to state that cannot survive the ZeroGPU "
            "process boundary, so the write is discarded and the Space will fail "
            "on every request:"
        )
        problems.extend(f"  {item}" for item in found)
        problems.append("  fix: return the value and assign it in the caller")
        return

    suffix = f", {noted} declared intentional" if noted else ""
    notes.append(f"no @gpu function relies on a discarded side effect{suffix}")


def stage_corpus(problems: list[str], notes: list[str], dry_run: bool) -> None:
    """Copy the corpus into the Space tree.

    The Space repo root is ``prototype/``, so a repo-root ``data/documents.csv``
    is invisible to it. Rather than commit the same 375 KB file twice and let
    the copies drift, the authoritative one lives at the repo root and is staged
    here on each deploy.
    """
    if CORPUS_DEST.exists() and CORPUS_DEST.stat().st_size == CORPUS_SOURCE.stat().st_size:
        notes.append(f"corpus already staged ({CORPUS_DEST.stat().st_size / 1024:.0f} KB)")
        return
    if not CORPUS_SOURCE.exists():
        problems.append(
            f"corpus not found at {CORPUS_SOURCE}. Download documents.csv from the "
            "competition and place it in the repository's data/ folder."
        )
        return
    size_kb = CORPUS_SOURCE.stat().st_size / 1024
    if dry_run:
        notes.append(f"would stage corpus ({size_kb:.0f} KB) into prototype/data/")
        return
    CORPUS_DEST.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_DEST.write_bytes(CORPUS_SOURCE.read_bytes())
    notes.append(f"staged corpus ({size_kb:.0f} KB) into prototype/data/")


def check_model_repos(api, dense: str, reranker: str, problems: list[str], notes: list[str]) -> None:
    """Confirm the fine-tuned weights are actually on the Hub.

    This is the check that matters most: missing model repos are the one failure
    that produces a Space which builds cleanly and then errors on every request.
    """
    for label, repo_id in (("dense", dense), ("reranker", reranker)):
        try:
            api.model_info(repo_id)
        except Exception as exc:  # noqa: BLE001 - reported per repo
            problems.append(
                f"{label} model repo {repo_id!r} is not readable on the Hub "
                f"({type(exc).__name__}). Run upload_models.py first, or correct "
                "the id."
            )
        else:
            notes.append(f"{label} model repo found: {repo_id}")


def _variable_present(payload, key: str) -> bool | None:
    """Whether the API's response to a variable write confirms ``key`` is set.

    Returns ``None`` when the response is not a shape we recognise, so the caller
    can report "unverified" rather than claim a success it cannot actually see.
    """
    entries = payload.get("variables", payload) if isinstance(payload, dict) else payload
    if isinstance(entries, dict):
        return key in entries
    if isinstance(entries, list):
        return any(isinstance(e, dict) and e.get("key") == key for e in entries)
    return None


def verify_upload(api, repo_id: str, problems: list[str], notes: list[str]) -> None:
    """Read the Space repo back and confirm the files the app needs are in it.

    This exists because an upload can report success while silently dropping a
    file. The Hub applies a ``.gitignore`` committed to a repo to that commit
    server-side, so a ``.gitignore`` that excludes the corpus strips it out of
    the Space -- which then builds cleanly, reaches RUNNING, and answers every
    visitor with "Could not find documents.csv". Nothing in the upload's return
    value reveals that, so it has to be read back.
    """
    required = {
        "app.py": "the Gradio entrypoint",
        "data/documents.csv": "the corpus",
        "data/ATTRIBUTION.md": "corpus provenance",
        "requirements.txt": "dependencies",
    }
    try:
        present = set(api.list_repo_files(repo_id, repo_type="space"))
    except Exception as exc:  # noqa: BLE001
        problems.append(
            f"could not read the Space repo back to verify the upload "
            f"({type(exc).__name__}). Check it by hand at "
            f"https://huggingface.co/spaces/{repo_id}/tree/main"
        )
        return

    for path, what in required.items():
        if path not in present:
            problems.append(
                f"{path} is missing from the Space ({what}). If it exists "
                "locally, a committed .gitignore is probably excluding it -- the "
                "Hub applies those server-side. See the note in prototype/.gitignore."
            )
    if not problems:
        notes.append(f"verified {len(required)} required files are in the Space repo")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repo",
        required=False,
        help="Space repo id, e.g. your-username/kinyeti-rag",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="run every check without creating or uploading anything")
    parser.add_argument("--update", action="store_true",
                        help="upload into an existing Space instead of creating it")
    parser.add_argument("--private", action="store_true",
                        help="create the Space private instead of public")
    parser.add_argument("--no-zerogpu", action="store_true",
                        help="skip requesting ZeroGPU hardware (set it by hand later)")
    parser.add_argument("--dense-model", default=os.getenv("DENSE_MODEL_ID", ""),
                        help="Hub repo id of the fine-tuned bi-encoder")
    parser.add_argument("--reranker-model", default=os.getenv("RERANKER_MODEL_ID", ""),
                        help="Hub repo id of the fine-tuned cross-encoder")
    parser.add_argument("--watch", action="store_true",
                        help="poll the build and print logs if it fails")
    args = parser.parse_args(argv)

    if yaml is None:
        print("Missing dependency: pyyaml.  pip install pyyaml")
        return 1

    problems: list[str] = []
    notes: list[str] = []

    if not args.repo:
        problems.append("--repo is required, e.g. --repo your-username/kinyeti-rag")
    elif "/" not in args.repo:
        problems.append("--repo must be owner/name")

    print("=== Checking the Space tree ===")
    meta = check_frontmatter(problems)
    if meta:
        check_gradio_version(meta, problems, notes)
    check_sources_compile(problems, notes)
    check_gpu_purity(problems, notes)

    for name in ("app.py", "requirements.txt", "LICENSE", "data/ATTRIBUTION.md"):
        if (SPACE_ROOT / name).exists():
            notes.append(f"{name} present")
        else:
            problems.append(f"prototype/{name} is missing")

    for note in notes:
        print(f"  ok    {note}")
    for problem in problems:
        print(f"  FAIL  {problem}")
    if problems:
        print("\nFix these before deploying.")
        return 1

    print("\n=== Staging the corpus ===")
    stage_corpus(problems, notes, args.dry_run)
    for note in notes[-1:]:
        print(f"  {note}")
    if problems:
        for problem in problems:
            print(f"  FAIL  {problem}")
        return 1

    # -- the Hub-side checks need credentials --------------------------------
    print("\n=== Hub checks ===")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("  FAIL  huggingface_hub is not installed.  pip install huggingface_hub")
        return 1

    api = HfApi()
    owner = args.repo.split("/", 1)[0]
    authenticated = True
    try:
        who = api.whoami()
        owner = who.get("name") or owner
        print(f"  ok    authenticated as {owner}")
    except Exception as exc:  # noqa: BLE001
        if args.dry_run:
            # A dry run exists so the plan can be reviewed before credentials are
            # sorted out, so fall back to the owner in --repo and carry on.
            authenticated = False
            print(f"  warn  not authenticated ({type(exc).__name__}); "
                  "using the owner from --repo")
        else:
            print(
                f"  FAIL  not authenticated ({type(exc).__name__}). "
                "Run `hf auth login`."
            )
            return 1

    dense = args.dense_model or f"{owner}/bge-base-agri"
    reranker = args.reranker_model or f"{owner}/bge-reranker-large-agri"
    if not args.dense_model:
        print(f"  note  --dense-model not given; assuming {dense}")
    if not args.reranker_model:
        print(f"  note  --reranker-model not given; assuming {reranker}")

    if not authenticated:
        print("  (not authenticated: skipping the model-repo check)")
    else:
        hub_problems: list[str] = []
        hub_notes: list[str] = []
        check_model_repos(api, dense, reranker, hub_problems, hub_notes)
        for note in hub_notes:
            print(f"  ok    {note}")
        for problem in hub_problems:
            print(f"  {'warn' if args.dry_run else 'FAIL'}  {problem}")
        if hub_problems and not args.dry_run:
            print(
                "\nThe Space would build and then fail on every request. Upload the\n"
                "models first (upload_models.py), or pass --dense-model/--reranker-model."
            )
            return 1

    if args.dry_run:
        print("\n=== Would do ===")
        print(f"  create  {args.repo}  (space, gradio, "
              f"{'private' if args.private else 'public'}"
              f"{'' if args.no_zerogpu else ', zero-a10g'})")
        print(f"  set     DENSE_MODEL_ID    = {dense}")
        print(f"  set     RERANKER_MODEL_ID = {reranker}")
        print(f"  upload  {SPACE_ROOT}  (excluding {', '.join(UPLOAD_IGNORE)})")
        print("\nDry run: nothing created, nothing uploaded.")
        return 0

    # -- create the Space -----------------------------------------------------
    print(f"\n=== Creating {args.repo} ===")
    create_kwargs = dict(
        repo_id=args.repo,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
        private=args.private,
    )
    try:
        api.create_repo(**create_kwargs, space_hardware="zero-a10g")
        print("  ok    created with ZeroGPU hardware requested")
    except Exception as exc:  # noqa: BLE001
        # Hardware is gated on account standing (verified email, >30 days old).
        # The Space is still worth creating; hardware can be set by hand.
        print(f"  warn  could not request ZeroGPU hardware ({type(exc).__name__})")
        print("        Creating the Space without it. Set Hardware = ZeroGPU in")
        print("        Settings once the account is eligible; the app runs on CPU")
        print("        until then, slowly.")
        try:
            api.create_repo(**create_kwargs)
            print("  ok    created")
        except Exception as inner:  # noqa: BLE001
            print(f"  FAIL  could not create the Space: {type(inner).__name__}: {inner}")
            return 1

    # -- upload ---------------------------------------------------------------
    print("\n=== Uploading prototype/ ===")
    try:
        api.upload_folder(
            folder_path=str(SPACE_ROOT),
            repo_id=args.repo,
            repo_type="space",
            ignore_patterns=UPLOAD_IGNORE,
            commit_message="Deploy Agricultural Extension RAG prototype",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  upload failed: {type(exc).__name__}: {exc}")
        return 1
    print("  ok    uploaded")

    # -- verify the upload ----------------------------------------------------
    # An upload reporting success is not evidence that the files arrived.
    print("\n=== Verifying the upload ===")
    upload_problems: list[str] = []
    upload_notes: list[str] = []
    verify_upload(api, args.repo, upload_problems, upload_notes)
    for note in upload_notes:
        print(f"  ok    {note}")
    if upload_problems:
        for problem in upload_problems:
            print(f"  FAIL  {problem}")
        return 1

    # -- configuration variables ----------------------------------------------
    # Set explicitly, NOT via ``create_repo(space_variables=...)``. That argument
    # is accepted by the signature and silently dropped: the Space is created
    # with no variables, starts cleanly, and then answers every visitor with
    # "The demo is not ready" while looking completely healthy from outside.
    # Verify the write against the API's own response rather than assuming it.
    print("\n=== Setting variables ===")
    for key, value in (("DENSE_MODEL_ID", dense), ("RERANKER_MODEL_ID", reranker)):
        try:
            returned = api.add_space_variable(repo_id=args.repo, key=key, value=value)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL  could not set {key}: {type(exc).__name__}: {exc}")
            print("        Set it by hand in Space -> Settings -> Variables, then")
            print(f"        confirm with: python scripts/check_deployed.py --repo {args.repo}")
            return 1
        confirmed = _variable_present(returned, key)
        if confirmed is False:
            print(f"  FAIL  {key} did not persist")
            return 1
        note = "" if confirmed else "  (write not confirmed by the API response)"
        print(f"  ok    {key} = {value}{note}")

    url = f"https://huggingface.co/spaces/{args.repo}"
    print(f"\nSpace: {url}")
    print(f"Build logs: {url}?logs=build")

    if args.watch:
        print("\n=== Watching the build ===")
        watch_build(api, args.repo)

    print(
        "\nFirst load after the build finishes downloads ~2.7 GB of weights, so give\n"
        "it a few minutes before deciding it is broken. Then check it in a logged-out\n"
        "browser, which is how newsletter readers will arrive."
    )
    return 0


def watch_build(api, repo_id: str, timeout: int = 900) -> None:
    """Poll the Space until it runs or fails, and print the logs if it fails."""
    import time

    started = time.time()
    last = None
    while time.time() - started < timeout:
        try:
            runtime = api.get_space_runtime(repo_id)
            stage = str(runtime.stage)
        except Exception as exc:  # noqa: BLE001 - transient API errors are normal
            print(f"  (status check failed: {type(exc).__name__}; retrying)")
            time.sleep(15)
            continue

        if stage != last:
            print(f"  stage: {stage}")
            last = stage

        # Only RUNNING means ready. RUNNING_BUILDING is a rebuild still in
        # progress -- treating it as done declares success while the container is
        # still starting, which is how a broken deploy gets waved through.
        if stage == "RUNNING":
            print("  build finished; the Space is up")
            return
        if stage.endswith("ERROR"):
            print(f"  build failed ({stage}). Logs:\n")
            try:
                print(api.fetch_space_logs(repo_id, build=True))
            except Exception as exc:  # noqa: BLE001
                print(f"  (could not fetch logs: {type(exc).__name__})")
                print(f"  Read them at https://huggingface.co/spaces/{repo_id}?logs=build")
            return
        time.sleep(15)

    print(f"  still building after {timeout}s; check the logs tab")


if __name__ == "__main__":
    raise SystemExit(main())
