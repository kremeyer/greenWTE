"""Library module for solving the Wigner Transport Equation (WTE) with a source term."""

from argparse import Namespace

import cupy as cp

from .base import Material, SolverBase, dT_to_N_iterative


class IterativeWTESolver(SolverBase):
    r"""Wigner Transport Equation solver using iterative or direct linear solvers.

    This solver computes the mapping from a temperature change ``dT`` to the
    Wigner distribution function ``n`` by solving the underlying linear
    system for each temporal Fourier frequency. The inner solver can be an
    iterative Krylov method such as GMRES or a direct solver (cgesv), and
    residuals from the inner solver are recorded for convergence analysis.

    Parameters
    ----------
    omg_ft_array : cupy.ndarray
        1D array of temporal Fourier variables :math:`\omega` [Hz] for which
        the WTE will be solved.
    k_ft : cupy.ndarray
        Magnitude of the spatial Fourier variable :math:`k` [m⁻¹].
    material : Material
        Material object containing the necessary physical properties
        (phonon frequencies, linewidths, heat capacities, etc.).
    source : cupy.ndarray
        Source term of the WTE, with shape ``(nq, nat3, nat3)``.
    max_iter : int, optional
        Maximum number of iterations for the **outer** solver (default is 100).
    conv_thr : float, optional
        Convergence threshold for both outer and inner solvers (default is 1e-12).
    outer_solver : {'root', 'aitken', 'plain'}, optional
        Outer solver strategy to use. One of:
        - ``'root'``: nonlinear root finding in 2D space (real & imaginary parts)
        - ``'aitken'``: Aitken Δ² acceleration applied to fixed-point iteration
        - ``'plain'``: plain fixed-point iteration without acceleration
        Default is ``'aitken'``.
    inner_solver : {'gmres', 'cgesv'}, optional
        Inner solver for mapping ``dT`` to ``n``.
        - ``'gmres'``: Iterative Krylov solver with residual history tracking.
        - ``'cgesv'``: Direct CuSolver dense complex equation solver.
        Default is ``'gmres'``.
    command_line_args : argparse.Namespace, optional
        Optional namespace of parsed command-line arguments for controlling
        solver behavior or I/O (default is an empty Namespace).

    Attributes
    ----------
    inner_solver : str
        The chosen inner method for solving the linear system in the
        :meth:`_dT_to_N` step.
    gmres_residual_list : list
        Nested list containing GMRES (or other inner solver) residual
        histories for each frequency, each outer iteration, and each
        ``q``-point.

    Notes
    -----
    - This implementation calls :func:`dT_to_N_iterative` to perform the
      temperature-to-Wigner mapping. The solver may use a previous solution
      as an initial guess (`sol_guess`) to accelerate convergence.
    - ``cgesv`` is a direct solver and does not produce iterative residuals.
    - ``gmres`` convergence is influenced by `conv_thr`, the preconditioning
      strategy, and the choice of initial guess.
    - Performance depends on the conditioning of the WTE system matrix and
      the selected inner solver.

    See Also
    --------
    SolverBase : Parent class that provides the outer-solver infrastructure.
    GreenWTESolver : Alternative solver using precomputed Green's operators.

    """

    def __init__(
        self,
        omg_ft_array: cp.ndarray,
        k_ft: cp.ndarray,
        material: Material,
        source: cp.ndarray,
        max_iter=100,
        conv_thr=1e-12,
        outer_solver="aitken",
        inner_solver="gmres",
        command_line_args=Namespace(),
        residual_weights: tuple[float, float] = (1.0, 0.0),
    ) -> None:
        """Initialize IterativeWTESolver."""
        super().__init__(
            omg_ft_array=omg_ft_array,
            k_ft=k_ft,
            material=material,
            source=source,
            max_iter=max_iter,
            conv_thr=conv_thr,
            outer_solver=outer_solver,
            command_line_args=command_line_args,
            residual_weights=residual_weights,
        )
        self.inner_solver = inner_solver

    def _dT_to_N(
        self,
        dT: complex,
        omg_ft: float,
        omg_idx: int,
        sol_guess: cp.ndarray = None,
    ) -> tuple[cp.ndarray, list]:
        return dT_to_N_iterative(
            dT=dT,
            omg_ft=omg_ft,
            k_ft=self.k_ft,
            material=self.material,
            source=self.source,
            sol_guess=sol_guess,
            solver=self.inner_solver,
            conv_thr=self.conv_thr,
            progress=self.progress,
        )
