"""Tests for nvtx_utils module."""

import importlib
import sys
import types


def import_fresh_nvtx_utils(with_nvtx: bool):
    """Import nvtx_utils with a clean module state, optionally providing a fake nvtx.

    Parameters
    ----------
    with_nvtx
        If True, a fake 'nvtx' module is provided so that nvtx_utils behaves as if nvtx is installed.

    Returns
    -------
    module
        The imported nvtx_utils module.
    module or None
        The fake nvtx module if with_nvtx is True, else None.

    """
    # Drop cached modules so import runs the top-level try/except again
    sys.modules.pop("greenWTE.nvtx_utils", None)

    fake_nvtx = None
    if with_nvtx:
        # Provide a fake 'nvtx' module before importing nvtx_utils
        fake_nvtx = types.ModuleType("nvtx")

        # --- Fake behavior for nvtx.annotate ---
        calls = []

        class _CtxMgrOrDecorator:
            def __init__(self, *a, **kw):
                self.args = a
                self.kwargs = kw
                calls.append(("construct", a, kw))

            # Decorator mode
            def __call__(self, fn):
                calls.append(("decorate", self.args, self.kwargs, fn))

                # Wrap to prove delegation happened
                def _wrapped(*fa, **fkw):
                    calls.append(("call_wrapped", fn, fa, fkw))
                    return fn(*fa, **fkw)

                _wrapped.__wrapped__ = fn  # help introspection
                return _wrapped

            # Context-manager mode
            def __enter__(self):
                calls.append(("enter", self.args, self.kwargs))
                return self

            def __exit__(self, exc_type, exc, tb):
                calls.append(("exit", self.args, self.kwargs))
                return False

        def annotate(*a, **kw):
            calls.append(("annotate_fn", a, kw))
            # Support both bare decorator and decorator-with-args
            return _CtxMgrOrDecorator(*a, **kw)

        fake_nvtx.annotate = annotate
        fake_nvtx._calls = calls
        sys.modules["nvtx"] = fake_nvtx

        nvtx_utils = importlib.import_module("greenWTE.nvtx_utils")

        return nvtx_utils, fake_nvtx

    class _BlockNvtxFinder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "nvtx":
                raise ModuleNotFoundError("blocked by test")
            return None

    blocker = _BlockNvtxFinder()
    sys.modules.pop("nvtx", None)
    sys.meta_path.insert(0, blocker)
    try:
        nvtx_utils = importlib.import_module("greenWTE.nvtx_utils")
    finally:
        sys.meta_path.remove(blocker)

    return nvtx_utils, None


def test_annotate_no_nvtx():
    """Test nvtx_utils.annotate when nvtx is not available."""
    nvtx_utils, _ = import_fresh_nvtx_utils(with_nvtx=False)
    assert nvtx_utils.NVTX_ENABLED is False

    # decorator
    @nvtx_utils.annotate("increment", color="red")
    def increment(x):
        return x + 1

    assert increment(1) == 2
    assert not hasattr(increment, "__wrapped__")  # make sure nvtx path not taken

    # context manager
    x = 1
    with nvtx_utils.annotate("increment", color="red"):
        x += 1

    assert x == 2


def test_annotate_with_nvtx():
    """Test nvtx_utils.annotate when nvtx is available."""
    nvtx_utils, fake_nvtx = import_fresh_nvtx_utils(with_nvtx=True)
    assert nvtx_utils.NVTX_ENABLED is True

    # decorator
    @nvtx_utils.annotate("increment", color="red")
    def increment(x):
        return x + 1

    assert increment(1) == 2
    assert hasattr(increment, "__wrapped__")  # make sure nvtx path taken
    kinds = [call[0] for call in fake_nvtx._calls]
    assert "annotate_fn" in kinds
    assert "decorate" in kinds
    assert "call_wrapped" in kinds

    # context manager
    x = 1
    fake_nvtx._calls.clear()
    with nvtx_utils.annotate("increment", color="red"):
        x += 1

    assert x == 2
    kinds = [call[0] for call in fake_nvtx._calls]
    assert "annotate_fn" in kinds
    assert "construct" in kinds
    assert "enter" in kinds
    assert "exit" in kinds
