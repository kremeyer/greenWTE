"""Test suite for the greenWTE package.

This file contains some helper functions.
"""

import cupy as cp
from greenWTE.base import N_to_dT


def _final_residual_and_scale(solver, material):
    """Compute |F(dT)| and the scale used by the solver's combined tolerance."""
    dT_final = solver.dT[0]
    omg_ft = float(cp.asnumpy(solver.omg_ft_array[0]))
    # Evaluate F at the final iterate to avoid Aitken mismatch
    n_next, _ = solver._dT_to_N(dT=dT_final, omg_ft=omg_ft, omg_idx=0, sol_guess=None)
    dT_next = N_to_dT(n_next, material)

    r_abs = float(cp.abs(dT_final - dT_next).item())
    scale = float(max(cp.abs(dT_final).item(), cp.abs(dT_next).item(), 1.0))
    return r_abs, scale, dT_final, dT_next
