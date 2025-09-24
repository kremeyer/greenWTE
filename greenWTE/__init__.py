"""greenWTE — frequency-domain solvers for the phonon Wigner Transport Equation (WTE)."""

import warnings

# triggered when cupyx.scipy.sparse.linalg.gmres
# invokes np.linalg.lstsq under the hood
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=r".*`rcond` parameter will change to the default of machine precision.*"
)

__version__ = "0.2.2"
