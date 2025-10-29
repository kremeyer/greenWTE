"""greenWTE — frequency-domain solvers for the phonon Wigner Transport Equation (WTE)."""

import os
import warnings

# triggered when cupyx.scipy.sparse.linalg.gmres
# invokes np.linalg.lstsq under the hood
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=r".*`rcond` parameter will change to the default of machine precision.*"
)

__version__ = "0.3.1"


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

    HAVE_GPU = True

except ImportError:
    import numpy as xp
    from scipy.interpolate import PchipInterpolator as xp_PchipInterpolator
    from scipy.sparse.linalg import gmres as xp_gmres

    HAVE_GPU = False

__all__ = [
    "xp",
    "xp_PchipInterpolator",
    "xp_gmres",
    "to_cpu",
    "HAVE_GPU",
]
