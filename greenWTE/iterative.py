r"""Library module for solving the Wigner Transport Equation (WTE) with a source term.

Provides :py:class:`~greenWTE.iterative.IterativeWTESolver`, an outer solver that maps a complex temperature change
``dT`` to the Wigner distribution function ``n`` at each temporal Fourier frequency :math:`\omega`. The inner mapping
is performed by an iterative Krylov method (e.g. :py:func:`cupyx.scipy.sparse.linalg.gmres`) or a direct GPU solve
(``cgesv``), see :py:func:`~greenWTE.base.dT_to_N_iterative`.


See also :py:class:`~greenWTE.base.SolverBase` for shared solver infrastructure.

"""

from argparse import Namespace

from . import xp
from .base import Material, SolverBase, dT_to_N_iterative


class IterativeWTESolver(SolverBase):
    r"""Wigner Transport Equation solver using iterative or direct linear solvers.

    This solver computes the mapping from a temperature change ``dT`` to the Wigner distribution function ``n`` by
    solving the underlying linear system for each temporal Fourier frequency. The inner solver can be an iterative
    Krylov method such as GMRES or a direct solver (cgesv), and residuals from the inner solver are recorded for
    convergence analysis.

    Parameters
    ----------
    omg_ft_array : cupy.ndarray
        1D array of temporal Fourier variables in rad/s for which the WTE will be solved.
    k_ft : float
        Magnitude of the spatial Fourier variable in rad/m.
    material : :py:class:`~greenWTE.base.Material`
        Material object containing the necessary material properties.
    source : cupy.ndarray
        Source term of the WTE, with shape (nq, nat3, nat3).
    source_type : str
        The type of the source term, either "energy" or "gradient". When injecting energy through the source term, there
        is no additional factor of dT for the offdiagonals of the source. For the temperature gradient type source
        terms, the offdiagonal elements are scaled by dT.
    dT_init : complex, optional
        Initial guess for :math:`\Delta T` used by the outer solver.
    max_iter : int, optional
        Maximum number of iterations for the outer solver.
    conv_thr_rel : float, optional
        The relative convergence threshold for the solver.
    conv_thr_abs : float, optional
        The absolute convergence threshold for the solver.
    outer_solver : {'plain', 'aitken', 'root', 'none'}, optional
        Outer-solver strategy. ``'root'`` uses :func:`scipy.optimize.root`, ``'plain'`` is fixed-point, ``'aitken'``
        applies :class:`~greenWTE.base.AitkenAccelerator`, and ``'none'`` performs a single mapping.
    inner_solver : {'gmres', 'cgesv'}, optional
        Inner solver for mapping ``dT`` to ``n``.
        - ``'gmres'``: Iterative Krylov solver with residual history tracking.
        - ``'cgesv'``: Direct CuSolver dense complex equation solver.
        Default is ``'gmres'``.
    command_line_args : argparse.Namespace, optional
        Optional namespace of parsed command-line arguments to be added to the results file.
    print_progress : bool, optional
        If ``True``, prints progress while solving.

    Attributes
    ----------
    inner_solver : str
        The chosen inner method for solving the linear system in the :py:meth:`_dT_to_N` step.

    See Also
    --------
    :py:class:`~greenWTE.base.SolverBase` : Parent class that provides the outer-solver infrastructure.
    :py:class:`~greenWTE.green.GreenWTESolver` : WTE solver using a direct linear solver.

    """

    def __init__(
        self,
        omg_ft_array: xp.ndarray,
        k_ft: xp.ndarray,
        material: Material,
        source: xp.ndarray,
        source_type: str = "energy",
        dT_init: complex = 1.0 + 1.0j,
        max_iter: int = 100,
        conv_thr_rel: float = 1e-12,
        conv_thr_abs: float = 0,
        outer_solver: str = "aitken",
        inner_solver: str = "gmres",
        command_line_args: Namespace = Namespace(),
        print_progress: bool = False,
    ) -> None:
        """Initialize IterativeWTESolver."""
        super().__init__(
            omg_ft_array=omg_ft_array,
            k_ft=k_ft,
            material=material,
            source=source,
            source_type=source_type,
            max_iter=max_iter,
            conv_thr_rel=conv_thr_rel,
            conv_thr_abs=conv_thr_abs,
            outer_solver=outer_solver,
            command_line_args=command_line_args,
            dT_init=dT_init,
            print_progress=print_progress,
        )
        self.inner_solver = inner_solver

    def _dT_to_N(
        self,
        dT: complex,
        omg_ft: float,
        omg_idx: int,
        sol_guess: xp.ndarray = None,
    ) -> tuple[xp.ndarray, list]:
        return dT_to_N_iterative(
            dT=dT,
            omg_ft=omg_ft,
            k_ft=self.k_ft,
            material=self.material,
            source=self.source,
            source_type=self.source_type,
            sol_guess=sol_guess,
            solver=self.inner_solver,
            conv_thr_rel=self.conv_thr_rel,
            conv_thr_abs=self.conv_thr_abs,
            progress=self.verbose,
        )
