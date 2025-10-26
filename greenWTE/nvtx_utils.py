"""Utility functions for NVTX annotations."""

from types import TracebackType

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

try:
    import nvtx

    NVTX_ENABLED = True
except ImportError:
    nvtx = None
    NVTX_ENABLED = False


class _DummyAnnotateResult:
    def __call__(self, fn: callable) -> callable:
        # used as @nvtx.annotate(...): just return the function
        return fn

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, type_: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None
    ) -> bool | None:
        return False  # don't suppress exceptions


def annotate(*args, **kwargs):
    """Drop-in for nvtx.annotate, does nothing if NVTX is not available.

    Can be used as a decorator or context manager and signature is the same as :func:`nvtx.annotate`.
    """
    if NVTX_ENABLED:
        return nvtx.annotate(*args, **kwargs)

    return _DummyAnnotateResult()
