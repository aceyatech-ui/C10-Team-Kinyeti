"""Fail if a ``@gpu`` function relies on side effects instead of its return value.

Why this exists: under ZeroGPU, a ``@gpu``-decorated function does **not** run in
the calling process. ``spaces/zero/wrappers.py`` forks a worker with
``multiprocessing.get_context('fork')``, pickles the arguments in, and pickles
only the *return value* back:

    worker.arg_queue.put(((args, kwargs), ...))   # parent -> child
    res = task(*args, **kwargs)                   # runs in the child
    res_queue.put(OkResult(res))                  # child -> parent, return only

So an assignment to ``self`` inside a ``@gpu`` function is written to the child's
copy and thrown away when the call returns. The parent's object is unchanged.
That failure is invisible locally, because ``spaces`` is not installed off
ZeroGPU and ``@gpu`` degrades to a passthrough -- every local check passes while
the deployed Space fails on every request.

Hence a *static* check. It cannot tell whether a GPU call is correct, only
whether it is written in the shape that survives the process boundary.

    python prototype/scripts/check_gpu_purity.py
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"

#: Put this in a comment on the offending line to declare the write intentional.
#: Only legitimate when the value is *read back in the same call*, since the
#: child process is discarded -- it is never a way to persist something.
GPU_OK_MARKER = "gpu-ok:"


def is_gpu_decorator(node: ast.expr) -> bool:
    """Whether a decorator expression is ``gpu`` or ``something.gpu``."""
    if isinstance(node, ast.Name):
        return node.id == "gpu"
    if isinstance(node, ast.Attribute):
        return node.attr == "gpu"
    if isinstance(node, ast.Call):
        return is_gpu_decorator(node.func)
    return False


def _dotted(node: ast.Attribute) -> str:
    """Render an attribute chain as ``self.reranker.max_length``."""
    parts: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return ""
    parts.append(current.id)
    return ".".join(reversed(parts))


def _self_attributes(node: ast.AST) -> list[str]:
    """Names assigned through ``self`` in a single assignment target."""
    if isinstance(node, ast.Attribute):
        dotted = _dotted(node)
        return [dotted] if dotted.startswith("self.") else []
    if isinstance(node, (ast.Tuple, ast.List)):
        found: list[str] = []
        for elt in node.elts:
            found.extend(_self_attributes(elt))
        return found
    return []


def inspect_function(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[tuple[int, str]]:
    """Return ``(lineno, description)`` for every side effect that cannot survive.

    Deliberately checks the function body only, not nested functions: a closure
    that never crosses the process boundary is fine.
    """
    findings: list[tuple[int, str]] = []

    for node in ast.walk(func):
        if isinstance(node, ast.Global):
            findings.append((node.lineno, f"`global {', '.join(node.names)}`"))
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for attr in _self_attributes(target):
                    kind = "augmented" if isinstance(node, ast.AugAssign) else "assigned"
                    findings.append((node.lineno, f"`{attr}` {kind}"))

    return sorted(set(findings))


def check_file(path: Path) -> tuple[list[tuple[int, str, str]], list[tuple[int, str, str]]]:
    """Return ``(problems, acknowledged)``, each ``(lineno, function, description)``.

    A write marked with :data:`GPU_OK_MARKER` on its own line is treated as a
    deliberate, same-call side effect and reported separately rather than failed.
    """
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    lines = text.splitlines()

    problems: list[tuple[int, str, str]] = []
    acknowledged: list[tuple[int, str, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(is_gpu_decorator(d) for d in node.decorator_list):
            continue
        for lineno, description in inspect_function(node):
            # A comment usually sits on the line above, so check both it and the
            # line the assignment is on.
            window = lines[max(0, lineno - 2) : lineno]
            target = acknowledged if any(GPU_OK_MARKER in ln for ln in window) else problems
            target.append((lineno, node.name, description))

    return problems, acknowledged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        type=Path,
        default=SRC,
        help=f"directory to scan (default: {SRC})",
    )
    args = parser.parse_args(argv)

    if not args.src.is_dir():
        print(f"FAIL  no source directory at {args.src}")
        return 1

    files = sorted(args.src.rglob("*.py"))
    gpu_functions = 0
    problems: list[tuple[Path, int, str, str]] = []
    acknowledged: list[tuple[Path, int, str, str]] = []

    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            print(f"FAIL  {path.name} does not parse: {exc}")
            return 1

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                is_gpu_decorator(d) for d in node.decorator_list
            ):
                gpu_functions += 1

        file_problems, file_acked = check_file(path)
        for lineno, func, description in file_problems:
            problems.append((path, lineno, func, description))
        for lineno, func, description in file_acked:
            acknowledged.append((path, lineno, func, description))

    print(f"Scanned {len(files)} file(s); found {gpu_functions} @gpu function(s).")

    for path, lineno, func, description in acknowledged:
        rel = path.relative_to(path.parent.parent)
        print(f"  note  {rel}:{lineno}  {func}()  --  {description} (declared intentional)")

    if not problems:
        print("\nAll @gpu functions communicate through their return value only.")
        return 0

    print("\nFAIL  these @gpu functions write to state that cannot survive the")
    print("      ZeroGPU process boundary, so the write is silently discarded:\n")
    for path, lineno, func, description in problems:
        rel = path.relative_to(path.parent.parent)
        print(f"  {rel}:{lineno}  in {func}()  --  {description}")
    print(
        "\nFix: return the value from the @gpu function and assign it in the\n"
        "caller, which runs in the parent process. See Retriever.encode_corpus\n"
        "in src/retrieve.py for the pattern."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
