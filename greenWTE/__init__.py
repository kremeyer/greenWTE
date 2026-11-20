"""greenWTE — frequency-domain solvers for the phonon Wigner Transport Equation (WTE)."""

import importlib.util
import os
import warnings

import hdf5plugin

# triggered when cupyx.scipy.sparse.linalg.gmres
# invokes np.linalg.lstsq under the hood
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=r".*`rcond` parameter will change to the default of machine precision.*"
)

__version__ = "1.1.0"


FORCE_CPU = os.getenv("GREENWTE_BACKEND", "").lower() in ["cpu", "numpy"]


def to_cpu(a):
    """Move array-like `a` to CPU memory.

    Parameters
    ----------
    a : array-like
        Input array, possibly on GPU.

    Returns
    -------
    array-like
        Input array moved to CPU memory.

    """
    return a.get() if hasattr(a, "get") else a


try:  # pragma: no cover
    if FORCE_CPU:
        raise ImportError("Forcing CPU backend.")

    import cupy as xp
    from cupyx.scipy.interpolate import PchipInterpolator as xp_PchipInterpolator
    from cupyx.scipy.sparse.linalg import gmres as xp_gmres

    # CuPy can be imported without a usable GPU (no driver, no device), check that one is available
    try:
        xp.cuda.runtime.getDeviceCount()
    except xp.cuda.runtime.CUDARuntimeError as e:
        raise ImportError(e) from e

    HAVE_GPU = True

except ImportError as e:  # pragma: no cover
    if not FORCE_CPU and importlib.util.find_spec("cupy") is not None:
        # CuPy is installed but cannot be used (e.g. missing CUDA libraries or no GPU), don't fall back silently
        warnings.warn(
            f"CuPy is installed but failed to load ({e}). Falling back to the CPU backend. "
            "Install greenWTE[cuda12x] or greenWTE[cuda13x] to get the CUDA libraries from PyPI, "
            "or make a system CUDA toolkit available. Set GREENWTE_BACKEND=CPU to silence this warning.",
            RuntimeWarning,
            stacklevel=2,
        )

    import numpy as xp
    from scipy.interpolate import PchipInterpolator as xp_PchipInterpolator
    from scipy.sparse.linalg import gmres as xp_gmres

    HAVE_GPU = False

hdf5plugin.register("bshuf")  # ensure Bitshuffle filter is registered before import of h5py

__all__ = [
    "hdf5plugin",
    "xp",
    "xp_PchipInterpolator",
    "xp_gmres",
    "to_cpu",
    "HAVE_GPU",
]
