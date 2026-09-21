"""ZeroGPU compatibility shim.

Hugging Face's ``@spaces.GPU`` decorator requests a GPU for the duration of the
decorated call and releases it afterwards. It is documented as *effect-free* in
non-ZeroGPU environments, but the ``spaces`` package is only installed there --
so importing it unconditionally would break local development.

This module provides a ``gpu`` decorator that is the real thing on the Space and
a transparent no-op everywhere else, letting one codebase run both places.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:  # pragma: no cover - environment dependent
    import spaces

    _HAS_SPACES = True
    logger.info("ZeroGPU detected; @gpu will request accelerator time.")
except ImportError:  # pragma: no cover - local development
    spaces = None  # type: ignore[assignment]
    _HAS_SPACES = False
    logger.info("ZeroGPU unavailable; @gpu is a no-op. Expect slow CPU inference.")


def _noop_decorator(func: Callable) -> Callable:
    """Pass the wrapped function through, preserving metadata."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    wrapper.gpu_disabled = True  # type: ignore[attr-defined]
    return wrapper


def gpu(*dargs: Any, **dkwargs: Any) -> Callable:
    """Request GPU time for a call, or no-op when ZeroGPU is unavailable.

    Supports both spellings used in HF's documentation::

        @gpu
        def f(): ...

        @gpu(duration=120)
        def g(): ...

    ``duration`` caps the accelerator runtime in seconds and is passed straight
    through to ZeroGPU. Shorter durations earn better queue priority, so set it
    to something realistic rather than leaving the 60s default everywhere.
    """
    if not _HAS_SPACES:
        # Called bare (@gpu) -> dargs holds the function itself.
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return _noop_decorator(dargs[0])

        def decorate(func: Callable) -> Callable:
            return _noop_decorator(func)

        return decorate

    return spaces.GPU(*dargs, **dkwargs)


def cuda_emulation_active() -> bool:
    """Whether module-level ``.to("cuda")`` is running against emulated CUDA.

    Under ZeroGPU a real GPU only exists inside a ``@gpu`` call, but models are
    expected to be placed on ``cuda`` at import time. False here means we are on
    an ordinary machine with no accelerator at all.
    """
    return _HAS_SPACES
