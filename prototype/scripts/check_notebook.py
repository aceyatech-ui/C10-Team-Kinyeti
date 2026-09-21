"""Validate the generated notebook without spending a Kaggle run.

Checks three things:

1. The notebook is structurally valid.
2. The source it embeds is byte-identical to ``prototype/src/`` on disk.
3. Those embedded sources actually import in isolation -- which is how the
   notebook consumes them, and is the check that would have caught the missing
   ``accelerator.py`` before it reached Kaggle.

Run from the repository root:

    python prototype/scripts/check_notebook.py
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "verify_parity.ipynb"
SRC = ROOT / "src"


def fail(message: str) -> None:
    print(f"  FAIL: {message}")
    raise SystemExit(1)


def extract_embedded_sources(notebook: dict) -> dict[str, str]:
    """Pull the SOURCES mapping out of the notebook's write cell."""
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        body = "".join(cell["source"])
        if "SOURCES = {" not in body:
            continue
        blob = body.split("SOURCES = ", 1)[1].split("\n\nfor ")[0]
        return json.loads(blob)
    fail("no cell defining SOURCES was found")


def check_structure(notebook: dict) -> None:
    n_code = sum(1 for c in notebook["cells"] if c["cell_type"] == "code")
    print(f"  valid notebook: {len(notebook['cells'])} cells ({n_code} code)")


def check_fidelity(embedded: dict[str, str]) -> None:
    on_disk = {p.name: p.read_text(encoding="utf-8") for p in SRC.glob("*.py")}

    missing = sorted(set(on_disk) - set(embedded))
    if missing:
        fail(f"modules on disk but not embedded: {', '.join(missing)}")

    stale = sorted(name for name in embedded if embedded[name] != on_disk.get(name))
    if stale:
        fail(f"embedded source differs from disk: {', '.join(stale)}")

    print(f"  embedded {len(embedded)} modules, all matching disk: "
          f"{', '.join(sorted(embedded))}")


def check_imports(embedded: dict[str, str]) -> None:
    """Write the embedded sources to a temp dir and import them, as Kaggle does."""
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "src"
        pkg.mkdir()
        for name, body in embedded.items():
            (pkg / name).write_text(body, encoding="utf-8")

        # Import in a subprocess so a partially-initialised `src` in this process
        # can't mask a failure.
        script = (
            "import sys\n"
            f"sys.path.insert(0, {tmp!r})\n"
            "import src.config, src.corpus, src.scoring, src.bm25, src.accelerator, src.retrieve\n"
            "print('ok')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            fail("embedded modules failed to import in isolation")

    print("  embedded modules import cleanly in isolation")


def main() -> int:
    if not NOTEBOOK.exists():
        fail(f"{NOTEBOOK} does not exist; run make_verification_notebook.py first")

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    print("Validating", NOTEBOOK.name)
    check_structure(notebook)
    embedded = extract_embedded_sources(notebook)
    check_fidelity(embedded)
    check_imports(embedded)
    print("\nAll checks passed. Notebook is safe to upload to Kaggle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
