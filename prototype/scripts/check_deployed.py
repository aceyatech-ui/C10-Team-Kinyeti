"""Verify a *deployed* Space actually answers queries.

The build reaching ``RUNNING`` proves very little. A Space with no
``DENSE_MODEL_ID`` set also reaches ``RUNNING`` -- the app starts, finds no
models configured, and serves "The demo is not ready" to every visitor. That is
the worst failure mode available, because everything looks healthy from the
outside: green status, no build errors, and a link that is broken for everyone.

So this asks the running app a real question and checks that documents come
back. It is the only check that covers the whole path -- variables, weights,
GPU, corpus, rendering.

    python prototype/scripts/check_deployed.py --repo you/your-space

Needs no credentials unless the Space is private, in which case set ``HF_TOKEN``.
A cold Space can take minutes to load weights, so this waits rather than
declaring failure on the first attempt.
"""

from __future__ import annotations

import argparse
import re
import sys
import time

#: A question with a clear right answer in the corpus, used for every mode.
PROBE_QUERY = "How do I control fall armyworm in my maize field?"

#: Every mode the app exposes, and whether it needs the fine-tuned models.
#: BM25 works with no weights at all, so it is a useful control: if BM25
#: succeeds but the others do not, the corpus is fine and the models are not.
MODES = [
    ("bm25_only", False),
    ("dense_only", True),
    ("hybrid", True),
    ("pure_reranker", True),
]

#: Substrings that mean the app returned an error block rather than documents.
#: Checking the *text* is not enough on its own: an error block is comfortably
#: longer than a naive length threshold, so it slips through as a pass. That is
#: exactly how a Space that failed on every request was once reported healthy.
ERROR_MARKERS = ("not ready", "something went wrong", "traceback (most recent")

#: The app renders TOP_K cards per query.
EXPECTED_CARDS = 5


def html_to_text(markup: str) -> str:
    """Crude tag strip -- enough to assert on and read in a terminal."""
    text = re.sub(r"<style.*?</style>", " ", markup, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def wait_for_running(api, repo: str, timeout_s: int) -> str:
    """Poll the runtime until the Space is RUNNING, or give up."""
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        try:
            stage = api.get_space_runtime(repo).stage
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            stage = f"unreadable ({type(exc).__name__})"
        if stage != last:
            print(f"  stage: {stage}")
            last = stage
        if stage == "RUNNING":
            return stage
        time.sleep(10)
    return last or "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="Space repo, e.g. you/my-space")
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="seconds to wait for RUNNING and for the first answer (default 900)",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        default=None,
        help="subset of modes to test (default: all)",
    )
    args = parser.parse_args(argv)

    try:
        from gradio_client import Client
        from huggingface_hub import HfApi
    except ImportError as exc:
        print(f"Missing dependency: {exc.name}. Install it with:")
        print("  pip install gradio_client huggingface_hub")
        return 1

    api = HfApi()
    print(f"=== Waiting for {args.repo} ===")
    stage = wait_for_running(api, args.repo, args.timeout)
    if stage != "RUNNING":
        print(f"\nSpace never reached RUNNING (last stage: {stage}).")
        print("Check the build logs: "
              f"https://huggingface.co/spaces/{args.repo}?logs=build")
        return 1

    print("\n=== Connecting ===")
    try:
        client = Client(args.repo, verbose=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  could not connect: {type(exc).__name__}: {exc}")
        return 1
    print("  connected")

    wanted = args.modes or [m for m, _ in MODES]
    unknown = [m for m in wanted if m not in {m for m, _ in MODES}]
    if unknown:
        print(f"  FAIL  unknown mode(s): {', '.join(unknown)}")
        print(f"        known: {', '.join(m for m, _ in MODES)}")
        return 1

    print("\n=== Querying ===")
    results: dict[str, str] = {}
    for mode, needs_models in MODES:
        if mode not in wanted:
            continue
        label = "needs weights" if needs_models else "no weights needed"
        print(f"  {mode} ({label}) ...", end=" ", flush=True)
        start = time.monotonic()
        try:
            # The first call of a session encodes the 695-document corpus on the
            # GPU, so this one is legitimately slow. Later calls are quick.
            out = client.predict(PROBE_QUERY, mode, api_name="/search")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL ({type(exc).__name__}: {exc})")
            results[mode] = "error"
            continue
        elapsed = time.monotonic() - start

        text = html_to_text(out)
        # Count rendered cards, not the CSS rule that also contains "ky-card".
        cards = out.count('class="ky-card"')

        marker = next((m for m in ERROR_MARKERS if m in text.lower()), None)
        if marker is not None:
            print(f"ERROR ({elapsed:.0f}s)")
            print(f"        {text[:300]}")
            results[mode] = "unconfigured" if marker == "not ready" else "error"
        elif cards == 0:
            print(f"NO RESULTS ({elapsed:.0f}s) -- {len(text)} chars, no result cards")
            print(f"        {text[:300]}")
            results[mode] = "empty"
        elif cards < EXPECTED_CARDS:
            print(f"PARTIAL -- only {cards} of {EXPECTED_CARDS} cards ({elapsed:.0f}s)")
            results[mode] = f"partial ({cards} cards)"
        else:
            print(f"ok -- {cards} cards, {len(text)} chars ({elapsed:.0f}s)")
            results[mode] = f"ok ({cards} cards)"

    print("\n=== Summary ===")
    for mode, status in results.items():
        print(f"  {mode:<14} {status}")

    bad = {m: s for m, s in results.items() if s != "ok" and not s.startswith("ok (")}
    if bad:
        print()
        if all(s == "unconfigured" for s in bad.values()):
            print("The Space is running but has no models configured.")
            print("Set these in Space -> Settings -> Variables and secrets:")
            print("  DENSE_MODEL_ID, RERANKER_MODEL_ID")
        elif len(bad) == len(results):
            print("Every mode failed identically, so this is almost certainly a")
            print("code fault rather than quota or a cold start. Most likely, in order:")
            print("  1. a @gpu function writes to `self` -- under ZeroGPU it runs in a")
            print("     forked worker, so the write is discarded. Check with:")
            print("       python prototype/scripts/check_gpu_purity.py")
            print("  2. a model repo is missing, private, or renamed")
            print("  3. the corpus is absent from the Space repo")
            print("Runtime logs:")
            print(f"  https://huggingface.co/spaces/{args.repo}?logs=container")
        else:
            print("Some modes failed. The usual causes, in order:")
            print("  1. ZeroGPU quota exhausted (5 min/day signed in, 2 min out)")
            print("  2. a cold start still loading weights -- retry in a few minutes")
            print("  3. an exception in that mode only -- see the runtime logs:")
            print(f"     https://huggingface.co/spaces/{args.repo}?logs=container")
        return 1

    print("\nAll tested modes returned documents.")
    print(f"  https://huggingface.co/spaces/{args.repo}")
    print("\nStill worth doing by hand: open it in a logged-out browser, since")
    print("that is how newsletter readers arrive and it is a different quota tier.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
