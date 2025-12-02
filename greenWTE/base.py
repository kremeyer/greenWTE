"""Base classes and methods for the greenWTE package."""

import time
import warnings
from abc import ABC, abstractmethod
from argparse import Namespace

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from typing import Sequence

import numpy as np
from scipy.constants import hbar
from scipy.optimize import root

from . import HAVE_GPU, nvtx_utils, to_cpu, xp, xp_gmres, xp_PchipInterpolator


class Material:
    """Container for material-specific properties used by WTE solvers.

    A :py:class:`~greenWTE.base.Material` holds the arrays and scalars that define a crystalline model at a given
    temperature, including the velocity operator, phonon frequencies, linewidths, heat capacities and the simulation
    cell volume. Instances can be created directly from CuPy arrays or loaded from a :py:mod:`phono3py` result via
    :py:meth:`~greenWTE.base.Material.from_phono3py`.

    Parameters
    ----------
    temperature : float
        Background temperature [K]
    velocity_operator : cupy.ndarray
        The velocity operator in m/s, shape (nq, nat3, nat3).
    phonon_freq : cupy.ndarray
        The phonon frequencies in rad/s, shape (nq, nat3).
    linewidth : cupy.ndarray
        The linewidths in rad/s, shape (nq, nat3).
    heat_capacity : cupy.ndarray
        The heat capacities in J/m^3/K, shape (nq, nat3).
    volume : float
        The volume of the unit cell in m^3.
    qpoint : cupy.ndarray, optional
        The q-point coordinates, shape (3). Not required to solve the WTE.
    name : str, optional
        The name of the material.

    Attributes
    ----------
    dtyper : cupy.dtype
        The data type for the real-valued arrays.
    dtypec : cupy.dtype
        The data type for the complex-valued arrays.
    nq : int
        The number of q-points.
    nat3 : int
        The number of phonon modes (3 times the number of atoms in the unit cell).

    Notes
    -----
    All array attributes are expected to be :py:class:`cupy.ndarray` so that downstream kernels run on the GPU. If you
    pass NumPy arrays, they should be converted by the caller before construction.

    """

    def __init__(
        self,
        temperature: float,
        velocity_operator: xp.ndarray,
        phonon_freq: xp.ndarray,
        linewidth: xp.ndarray,
        heat_capacity: xp.ndarray,
        volume: float,
        qpoint: xp.ndarray | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the Material class with physical properties."""
        self.temperature = temperature
        self.velocity_operator = velocity_operator
        self.phonon_freq = phonon_freq
        self.linewidth = linewidth
        self.heat_capacity = heat_capacity
        self.volume = volume
        self.dtypec = self.velocity_operator.dtype
        self.dtyper = self.phonon_freq.dtype
        self.qpoint = qpoint
        self.name = name
        self.nq = self.phonon_freq.shape[0]
        self.nat3 = self.phonon_freq.shape[1]

    @classmethod
    def from_phono3py(
        cls: type[Self],
        filename: str,
        temperature: float,
        dir_idx: int = 0,
        exclude_gamma: bool = True,
        dtyper: xp.dtype = xp.float64,
        dtypec: xp.dtype = xp.complex128,
    ) -> Self:
        """Construct a :py:class:`~greenWTE.base.Material` from a :py:mod:`phono3py` HDF5 output.

        Parameters
        ----------
        filename : str
            The path to the phono3py output file.
        temperature : float
            The temperature at which to load the data.
        dir_idx : int, optional
            The index of the directory containing the output files. Default is 0, which corresponds to the x-direction.
        exclude_gamma : bool, optional
            Whether to exclude the gamma point from the calculations. This will skip the loading the first
            q-point for each quantity. Default is True, because the acoustic phonons have zero frequency at the Gamma
            point.
        dtyper : cupy.dtype, optional
            The data type for the real parts of the matrices.
        dtypec : cupy.dtype, optional
            The data type for the complex matrices.

        Returns
        -------
        Material
            An instance of the Material class with the loaded properties.

        """
        from .io import load_phono3py_data

        qpoint, velocity_operator, phonon_freq, linewidth, heat_capacity, volume, _ = load_phono3py_data(
            filename,
            temperature=temperature,
            dir_idx=dir_idx,
            exclude_gamma=exclude_gamma,
            dtyper=dtyper,
            dtypec=dtypec,
        )
        return cls(
            temperature,
            velocity_operator,
            phonon_freq,
            linewidth,
            heat_capacity,
            volume,
            qpoint=qpoint,
            name=filename,
        )

    def __repr__(self) -> str:
        """Return a string representation of the Material."""
        return f"{self.name}@{self.temperature}K with {self.nq} qpoints and {self.nat3} modes"

    def __getitem__(self, iq: int) -> Self:
        """Return a single-q-point :py:class:`~greenWTE.base.Material`.

        Parameters
        ----------
        iq : int
            The index of the q-point to retrieve.

        Returns
        -------
        Material
            A shallow view where all array attributes are sliced at `iq` and scalars are preserved.

        """
        return Material(
            temperature=self.temperature,
            velocity_operator=self.velocity_operator[iq][None, ...],
            phonon_freq=self.phonon_freq[iq][None, ...],
            linewidth=self.linewidth[iq][None, ...],
            heat_capacity=self.heat_capacity[iq][None, ...],
            volume=self.volume,
            name=self.name,
        )

    def __iter__(self) -> Self:
        """Allow iteration over the Materials q-points."""
        self._iter_index = 0
        return self

    def __next__(self) -> Self:
        """Return the next q-point in the iteration."""
        if self._iter_index < self.nq:
            result = self[self._iter_index]
            self._iter_index += 1
            return result
        raise StopIteration


class AitkenAccelerator:
    r"""Delta-squared acceleration for fixed-point iterations.

    The accelerator stores the last few iterates of the scalar complex sequence :math:`\{\Delta T^{(k)}\}` and applies
    Aitken's :math:`\Delta^2` formula to propose an improved iterate.

    Attributes
    ----------
    history : list
        A list to store the history of temperature changes dT. The last three entries are used for the acceleration.

    Notes
    -----
    See `Wikipedia`_ for details.

    .. _Wikipedia: https://en.wikipedia.org/wiki/Aitken%27s_delta-squared_process

    """

    def __init__(self) -> None:
        """Initialize the AitkenAccelerator."""
        self.history = []

    def reset(self) -> None:
        """Clear the AitkenAccelerator history."""
        self.history.clear()

    def update(self, dT_new: complex) -> complex:
        """Update the AitkenAccelerator with a new temperature change.

        Parameters
        ----------
        dT_new : complex
            The new temperature change dT [K] to be added to the history.

        Returns
        -------
        complex
            The accelerated temperature change dT using Aitken's delta-squared process.

        """
        self.history.append(dT_new)
        if len(self.history) < 3:
            return dT_new
        dT0, dT1, dT2 = self.history[-3:]
        denom = dT2 - 2 * dT1 + dT0
        if abs(denom) < 1e-14:
            return dT2
        dT_accel = dT0 - ((dT1 - dT0) ** 2) / denom
        self.history[-1] = dT_accel

        return dT_accel


def estimate_initial_dT(
    omg_ft: float, history: Sequence[tuple[float, complex]], dtyper=xp.float64, dtypec=xp.complex128
) -> complex:
    r"""Estimate the initial guess for :math:`\Delta T(\omega)`.

    Try to interpolate the values of :math:`\Delta T(\omega)` from history of previous values. If no history is
    available, we return a default of ``1.0 + 1.0j``.

    Parameters
    ----------
    omg_ft : float
        Temporal Fourier variable in rad/s.
    history : Sequence[tuple[float, complex]]
        Sequence of previously solve ``(omg_ft, dT)`` pairs.
    dtyper : cupy.dtype, optional
        Data type for the real parts of the matrices.
    dtypec : cupy.dtype, optional
        Data type for the complex matrices.

    Returns
    -------
    complex
        Interpolated guess for :math:`\Delta T(\omega)`. If fewer than two history points are available, returns
        ``1.0 + 1.0j``.

    Notes
    -----
    Uses PCHIP interlator from :py:class:`cupyx.scipy.interpolate.PchipInterpolator`.

    """
    if not history:
        return xp.asarray((1.0 + 1.0j))

    omg_fts, dTs = zip(*sorted(history), strict=True)
    omg_fts = xp.asarray(omg_fts, dtype=dtyper)
    dTs = xp.asarray(dTs, dtype=dtypec)

    if len(omg_fts) == 1:
        return dTs[0]

    interp_real = xp_PchipInterpolator(omg_fts, xp.real(dTs), extrapolate=True)
    interp_imag = xp_PchipInterpolator(omg_fts, xp.imag(dTs), extrapolate=True)
    dT_guess = interp_real(omg_ft) + 1j * interp_imag(omg_ft)

    return dT_guess


class SolverBase(ABC):
    r"""Abstract base class for Wigner Transport Equation solvers.

    This class defines the common interface, data structures, and solver control logic for computing the Wigner
    distribution function `n` and derived transport quantities such as flux and thermal conductivity from a given
    material model, source term, and temporal Fourier frequency grid.

    Concrete subclasses must implement the :py:meth:`_dT_to_N` method to perform the actual mapping from a temperature
    change ``dT`` to the Wigner distribution function ``n`` for a specific solver implementation.

    The class supports multiple outer-solver strategies:
    - **plain**: fixed-point iteration
    - **aitken**: Aitken's Δ² acceleration of fixed-point iteration
    - **root**: root finding on the complex plane using :py:func:`scipy.optimize.root`
    - **none**: single mapping without iteration

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
    command_line_args : argparse.Namespace, optional
        Optional namespace of parsed command-line arguments to be added to the results file.
    print_progress : bool, optional
        If ``True``, prints progress while solving.

    Attributes
    ----------
    dT : cupy.ndarray
        Converged complex temperature changes, shape (n_omg_ft,).
    n : cupy.ndarray
        Computed Wigner distributions, shape (n_omg_ft, nq, nat3, nat3).

    Notes
    -----
    - The solver stores intermediate convergence data in lists during the run. After solving all frequencies,
      :py:meth:`_solution_lists_to_arrays` can be used to convert them into CuPy arrays with NaN padding for easier
      post-processing.
    - The actual numerical strategy for mapping ``dT`` to ``n`` is deferred to subclasses via the :py:meth:`_dT_to_N`
      method.

    See Also
    --------
    :py:class:`~greenWTE.iterative.IterativeWTESolver` : WTE solver using iterative methods.
    :py:class:`~greenWTE.green.GreenWTESolver` : WTE solver using precomputed Green's operators.

    """

    _flux = None
    _kappa = None
    _kappa_p = None
    _kappa_c = None

    def __init__(
        self,
        omg_ft_array: xp.ndarray,
        k_ft: float,
        material: Material,
        source: xp.ndarray,
        source_type: str = "energy",
        dT_init: complex = 1.0 + 1.0j,
        max_iter: int = 100,
        conv_thr_rel: float = 1e-12,
        conv_thr_abs: float = 0,
        outer_solver: str = "root",
        command_line_args=Namespace(),
        print_progress: bool = False,
    ) -> None:
        """Initialize SolverBase."""
        self.omg_ft_array = xp.asarray(omg_ft_array)
        self.k_ft = k_ft
        self.material = material
        self.source = source
        if source_type in ["energy", "gradient"]:
            self.source_type = source_type
        else:
            raise ValueError(f"Unknown source type: {source_type}")
        self.max_iter = max_iter
        self.conv_thr_rel = conv_thr_rel
        self.conv_thr_abs = conv_thr_abs
        self.outer_solver = outer_solver
        self.command_line_args = command_line_args
        self.dtyper = material.dtyper
        self.dtypec = material.dtypec
        self.nq = material.nq
        self.nat3 = material.nat3

        self.history = []
        self.dT = xp.zeros_like(self.omg_ft_array, dtype=self.dtypec)
        self.dT_init = xp.zeros_like(self.omg_ft_array, dtype=self.dtypec)
        self.dT_init_user = dT_init
        self.n = xp.zeros((self.omg_ft_array.shape[0], self.nq, self.nat3, self.nat3), dtype=self.dtypec)
        self.niter = xp.zeros(self.omg_ft_array.shape[0], dtype=xp.int32)
        self.iter_time_list = []
        self.dT_iterates_list = []
        self.n_norms_list = []
        self.gmres_residual_list = []
        self.verbose = self.omg_ft_array.shape[0] == 1 and print_progress
        self.print_progress = print_progress

    @abstractmethod
    def _dT_to_N(
        self,
        dT: complex,
        omg_ft: float,
        omg_idx: int,
        sol_guess: xp.ndarray | None = None,
    ) -> tuple[xp.ndarray, list]:
        """Map a temperature change to a Wigner distribution ``n``.

        Subclasses must implement this method.
        """

    def _dT_converged(self, dT: complex, dT_new: complex) -> bool:
        r"""Check whether two successive :math:`\Delta T` iterates meet the thresholds.

        Parameters
        ----------
        dT, dT_new : complex
            The previous and current temperature changes dT.

        Returns
        -------
        bool
            ``True`` absolute and relative convergence criteria are met.

        """
        r = dT - dT_new
        r_abs = float(xp.abs(r))
        scale = float(max(xp.abs(dT), xp.abs(dT_new), 1.0))
        thresh = self.conv_thr_abs + self.conv_thr_rel * scale
        return r_abs <= thresh

    def run(self) -> None:
        r"""Run the WTE solver at each :math:`\omega \in` :attr:`omg_ft_array`.

        The outer iteration chosen by ``outer_solver`` is used to find self-consistent solutions for the temperature
        changes :math:`\Delta T(\omega)` and the Wigner distribution. After running, the results are stored in the class
        attributes dT, dT_init, n, niter, n_norms, iter_time, dT_iterates, and gmres_residual.
        """
        if self.outer_solver == "aitken":
            run_func = self._run_solver_aitken
        elif self.outer_solver == "plain":
            run_func = self._run_solver_plain
        elif self.outer_solver == "root":
            run_func = self._run_solver_root
        elif self.outer_solver == "none":
            run_func = self._run_solver_none
            self.max_iter = 1  # just for string formatting
        else:
            raise ValueError(f"Unknown outer solver: {self.outer_solver}")

        for i, omg_ft in enumerate(self.omg_ft_array):
            ret = run_func(i, omg_ft)
            self.dT[i] = ret[1]
            self.dT_init[i] = ret[2]
            self.n[i] = ret[3]
            self.niter[i] = ret[4]
            self.iter_time_list.append(ret[5])
            self.dT_iterates_list.append(ret[6])
            self.n_norms_list.append(ret[7])
            self.gmres_residual_list.append(ret[8])

            if self.outer_solver != "none":
                # check self-consistency by checking norm(n(dT) - n)
                last_n = ret[3]
                theoretical_next_n, _ = self._dT_to_N(
                    dT=ret[1],
                    omg_ft=omg_ft,
                    omg_idx=i,
                    sol_guess=None,
                )
                n_step_norm = xp.linalg.norm(theoretical_next_n - last_n) / (xp.linalg.norm(last_n) + 1e-300)
                theoretical_next_dT = N_to_dT(theoretical_next_n, self.material)
                self.n_norms_list[i].append(n_step_norm)
            else:
                n_step_norm = 0
                theoretical_next_dT = 0
                self.n_norms_list[i].append(xp.nan)

            if self.print_progress:
                if self.verbose:
                    print("")
                width = len(str(len(self.omg_ft_array)))
                print(
                    f"[{i + 1:{width}d}/{len(self.omg_ft_array)}] "
                    f"k={self.k_ft:.2e} "
                    f"w={omg_ft:.2e} "
                    f"dT={self.dT[i]: .2e} "
                    f"n_it={self.niter[i]:{int(xp.log10(self.max_iter)) + 1}d} "
                    f"it_time={xp.mean(xp.array(self.iter_time_list[-1])):.2f} "
                    f"n_conv={n_step_norm:.1e} "
                    f"dT_conv={xp.abs(self.dT[i] - theoretical_next_dT) / xp.abs(self.dT[i]):.1e} "
                    f"dT_next={theoretical_next_dT: .1e}"
                )

        self._solution_lists_to_arrays()

    def _run_solver_aitken(
        self, omg_idx: int, omg_ft: float
    ) -> tuple[float, complex, complex, xp.ndarray, int, list[float], list[complex], list[float], list[float]]:
        """Run the WTE solver using Aitken's delta-squared process for acceleration.

        Parameters
        ----------
        omg_idx : int
            The index of the temporal Fourier variable for which the WTE will be solved.
        omg_ft : float
            The temporal Fourier variable in rad/s for which the WTE will be solved.

        Returns
        -------
        omg_ft : float
            The input temporal Fourier variable in rad/s.
        dT : complex
            The calculated temperature change dT in K for the given omg_ft.
        dT_init : complex
            The initial temperature change dT in K estimated for the given omg_ft.
        n : cupy.ndarray
            The wigner distribution function n for the given omg_ft, shape (nq, nat3, nat3).
        niter : int
            The number of iterations taken for the outer solver to converge for the given omg_ft.
        iter_times : list
            A list of iteration times for the outer solver for the given omg_ft.
        dT_iterates : list
            A list of iteration values for the temperature changes dT for the given omg_ft.
        n_norms : list
            A list of norms for the wigner distribution function n for the given omg_ft.
        gmres_residual : list
            A list of GMRES residuals for the given omg_ft. Empty if the inner solver is not GMRES.

        """
        accelerator = AitkenAccelerator()
        if self.history:
            dT_init = estimate_initial_dT(
                omg_ft=omg_ft, history=self.history, dtyper=self.material.dtyper, dtypec=self.material.dtypec
            )
        else:
            dT_init = self.dT_init_user
        dT = dT_init
        n = None
        iter_times = []
        dT_iterates = []
        n_norms = []
        gmres_residual = []

        iterations = 0
        for _ in range(self.max_iter):
            iterations += 1
            iter_start = time.time()
            n_prev = n

            n, resid = self._dT_to_N(
                dT=dT,
                omg_ft=omg_ft,
                omg_idx=omg_idx,
                sol_guess=n_prev,
            )
            gmres_residual.append(resid)
            dT_new = N_to_dT(n, self.material)
            converged = self._dT_converged(dT, dT_new)
            dT_iterates.append(dT_new)
            dT = accelerator.update(dT_new)

            iter_times.append(time.time() - iter_start)

            if n_prev is not None:
                n_step_norm = xp.linalg.norm(n - n_prev) / (xp.linalg.norm(n_prev) + 1e-300)
                n_norms.append(n_step_norm)

            if converged:
                self.history.append((omg_ft, dT_new))
                break

        return omg_ft, dT, dT_init, n, iterations, iter_times, dT_iterates, n_norms, gmres_residual

    def _run_solver_plain(
        self, omg_idx: int, omg_ft: float
    ) -> tuple[float, complex, complex, xp.ndarray, int, list[float], list[complex], list[float], list[float]]:
        """Run the WTE solver using simple fixed-point iteration.

        Parameters
        ----------
        omg_idx : int
            The index of the temporal Fourier variable for which the WTE will be solved.
        omg_ft : float
            The temporal Fourier variable in rad/s for which the WTE will be solved.

        Returns
        -------
        omg_ft : float
            The input temporal Fourier variable in rad/s.
        dT : complex
            The calculated temperature change dT in K for the given omg_ft.
        dT_init : complex
            The initial temperature change dT in K estimated for the given omg_ft.
        n : cupy.ndarray
            The wigner distribution function n for the given omg_ft, shape (nq, nat3, nat3).
        niter : int
            The number of iterations taken for the outer solver to converge for the given omg_ft.
        iter_times : list
            A list of iteration times for the outer solver for the given omg_ft.
        dT_iterates : list
            A list of iteration values for the temperature changes dT for the given omg_ft.
        n_norms : list
            A list of norms for the wigner distribution function n for the given omg_ft.
        gmres_residual : list
            A list of GMRES residuals for the given omg_ft. Empty if the inner solver is not GMRES.

        """
        if self.history:  # pragma: no cover
            dT_init = estimate_initial_dT(
                omg_ft=omg_ft, history=self.history, dtyper=self.material.dtyper, dtypec=self.material.dtypec
            )
        else:
            dT_init = self.dT_init_user
        dT = dT_init
        n = None
        iter_times = []
        dT_iterates = []
        n_norms = []
        gmres_residual = []

        iterations = 0
        for _ in range(self.max_iter):
            iterations += 1
            iter_start = time.time()
            n_prev = n

            n, resid = self._dT_to_N(
                dT=dT,
                omg_ft=omg_ft,
                omg_idx=omg_idx,
                sol_guess=n_prev,
            )
            gmres_residual.append(resid)

            dT_new = N_to_dT(n, self.material)
            converged = self._dT_converged(dT, dT_new)
            dT_iterates.append(dT_new)
            dT = dT_new

            iter_times.append(time.time() - iter_start)

            if n_prev is not None:
                n_step_norm = xp.linalg.norm(n - n_prev) / (xp.linalg.norm(n_prev) + 1e-300)
                n_norms.append(n_step_norm)

            if converged:
                self.history.append((omg_ft, dT))
                break

        return omg_ft, dT, dT_init, n, iterations, iter_times, dT_iterates, n_norms, gmres_residual

    def _run_solver_none(
        self, omg_idx: int, omg_ft: float
    ) -> tuple[float, complex, complex, xp.ndarray, int, list[float], list[complex], list[float], list[float]]:
        """Run the WTE solver without outer iterations, performing a single mapping from dT to n.

        Parameters
        ----------
        omg_idx : int
            The index of the temporal Fourier variable for which the WTE will be solved.
        omg_ft : float
            The temporal Fourier variable in rad/s for which the WTE will be solved.

        Returns
        -------
        omg_ft : float
            The input temporal Fourier variable in rad/s.
        dT : complex
            The calculated temperature change dT in K for the given omg_ft.
        dT_init : complex
            The initial temperature change dT in K estimated for the given omg_ft.
        n : cupy.ndarray
            The wigner distribution function n for the given omg_ft, shape (nq, nat3, nat3).
        niter : int
            The number of iterations taken for the outer solver to converge for the given omg_ft.
        iter_times : list
            A list of iteration times for the outer solver for the given omg_ft.
        dT_iterates : list
            A list of iteration values for the temperature changes dT for the given omg_ft.
        n_norms : list
            A list of norms for the wigner distribution function n for the given omg_ft.
        gmres_residual : list
            A list of GMRES residuals for the given omg_ft. Empty if the inner solver is not GMRES.

        Notes
        -----
        The lengthy return signature is to maintain compatibility with the other outer solvers.

        """
        dT = self.dT_init_user
        n = None
        iter_times = []
        dT_iterates = []
        n_norms = []
        gmres_residual = []

        iterations = 1

        iter_start = time.time()
        n_prev = n

        n, resid = self._dT_to_N(
            dT=dT,
            omg_ft=omg_ft,
            omg_idx=omg_idx,
            sol_guess=n_prev,
        )
        gmres_residual.append(resid)

        dT_new = N_to_dT(n, self.material)
        dT_iterates.append(dT_new)

        iter_times.append(time.time() - iter_start)

        return omg_ft, dT, dT, n, iterations, iter_times, dT_iterates, n_norms, gmres_residual

    def _run_solver_root(
        self, omg_idx: int, omg_ft: float
    ) -> tuple[float, complex, complex, xp.ndarray, int, list[float], list[complex], list[float], list[float]]:
        """Run the WTE solver using SciPy's root-finding algorithm :py:mod:scipy.optimize.root`.

        Parameters
        ----------
        omg_idx : int
            The index of the temporal Fourier variable for which the WTE will be solved.
        omg_ft : float
            The temporal Fourier variable in rad/s for which the WTE will be solved.

        Returns
        -------
        omg_ft : float
            The input temporal Fourier variable in rad/s.
        dT : complex
            The calculated temperature change dT in K for the given omg_ft.
        dT_init : complex
            The initial temperature change dT in K estimated for the given omg_ft.
        n : cupy.ndarray
            The wigner distribution function n for the given omg_ft, shape (nq, nat3, nat3).
        niter : int
            The number of iterations taken for the outer solver to converge for the given omg_ft.
        iter_times : list
            A list of iteration times for the outer solver for the given omg_ft.
        dT_iterates : list
            A list of iteration values for the temperature changes dT for the given omg_ft.
        n_norms : list
            A list of norms for the wigner distribution function n for the given omg_ft.
        gmres_residual : list
            A list of GMRES residuals for the given omg_ft. Empty if the inner solver is not GMRES.

        """
        if self.history:  # pragma: no cover
            dT_init = estimate_initial_dT(
                omg_ft=omg_ft, history=self.history, dtyper=self.material.dtyper, dtypec=self.material.dtypec
            )
        else:
            dT_init = self.dT_init_user

        iter_times = []
        dT_iterates = []
        n_norms = []
        gmres_residual = []
        n = None
        n_old = None
        last_eval_dT = None
        last_eval_n = None
        last_eval_dTnew = None

        def residual_vec(x: np.ndarray) -> np.ndarray:
            # x is np.array([Re(dT), Im(dT)]) on CPU (necessary for scipy)
            nonlocal n, n_old, dT_iterates, gmres_residual, iter_times
            nonlocal last_eval_dT, last_eval_n, last_eval_dTnew
            dT = xp.asarray(x[0] + 1j * x[1], dtype=self.material.dtypec)
            iter_start = time.time()
            n_old = n

            n, resid = self._dT_to_N(
                dT=dT,
                omg_ft=omg_ft,
                omg_idx=omg_idx,
                sol_guess=n_old,
            )
            gmres_residual.append(resid)

            dT_new = N_to_dT(n, self.material)
            dT_iterates.append(dT_new)
            if n_old is not None:
                n_step_norm = xp.linalg.norm(n - n_old) / (xp.linalg.norm(n_old) + 1e-300)
                n_norms.append(n_step_norm)

            iter_times.append(time.time() - iter_start)

            # early exit hint: our definition of convergence is less tight
            if self._dT_converged(dT, dT_new):
                last_eval_dT = (x[0], x[1])
                last_eval_n = n
                last_eval_dTnew = dT_new
                return np.array([0.0, 0.0], dtype=self.material.dtyper)

            scale = float(max(xp.abs(dT), xp.abs(dT_new), 1.0))
            rR = float(xp.real(dT - dT_new)) / scale
            rI = float(xp.imag(dT - dT_new)) / scale

            last_eval_dT = (x[0], x[1])
            last_eval_n = n
            last_eval_dTnew = dT_new

            return np.array([rR, rI], dtype=self.material.dtyper)

        x0 = np.array([to_cpu(xp.real(dT_init)), to_cpu(xp.imag(dT_init))], dtype=self.material.dtyper)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            sol = root(
                residual_vec,
                x0=x0,
                method="hybr",
                options={"xtol": self.conv_thr_rel, "maxfev": self.max_iter + 2},
            )

        dT = xp.asarray(sol.x[0] + 1j * sol.x[1], dtype=self.material.dtypec)

        # Reuse the last expensive evaluation if it matches the final x
        if last_eval_dT is not None and np.allclose(
            np.array(last_eval_dT),
            np.array([float(xp.real(dT)), float(xp.imag(dT))]),
            rtol=1e-12,
            atol=0.0,
        ):
            n_final = last_eval_n
            dT_final_new = last_eval_dTnew
        else:
            # one fallback call
            n_final, _ = self._dT_to_N(dT=dT, omg_ft=omg_ft, omg_idx=omg_idx, sol_guess=n)
            dT_final_new = N_to_dT(n_final, self.material)

        converged = self._dT_converged(dT, dT_final_new)

        if not converged:
            print(RuntimeWarning(f"Root finding did not meet tolerance: success={sol.success}"))

        self.history.append((omg_ft, dT))
        niter = len(iter_times)
        return omg_ft, dT, dT_init, n, niter, iter_times, dT_iterates, n_norms, gmres_residual

    def _solution_lists_to_arrays(self) -> None:
        """Convert the lists of iteration times, dT iterates, n norms, and GMRES residuals into cupy arrays.

        We can only do that after all omg_ft have been solved, since the lengths of the lists can vary. All entries in
        the lists are padded with NaNs.
        """
        max_len = max(len(times) for times in self.iter_time_list)
        iter_times_array = xp.full((len(self.iter_time_list), max_len), np.nan, dtype=self.material.dtyper)
        for i, times in enumerate(self.iter_time_list):
            times = xp.asarray(times, dtype=self.material.dtyper)
            iter_times_array[i, : len(times)] = times
        self.iter_time = iter_times_array

        max_len = max(len(dTs) for dTs in self.dT_iterates_list)
        dT_iterates_array = xp.full((len(self.dT_iterates_list), max_len), np.nan, dtype=self.material.dtypec)
        for i, dTs in enumerate(self.dT_iterates_list):
            dTs = xp.asarray(dTs, dtype=self.material.dtypec)
            dT_iterates_array[i, : len(dTs)] = dTs
        self.dT_iterates = dT_iterates_array

        max_len = max(len(n_convs) for n_convs in self.n_norms_list)
        n_norms_array = xp.full((len(self.n_norms_list), max_len), np.nan, dtype=self.material.dtyper)
        for i, n_convs in enumerate(self.n_norms_list):
            n_convs = xp.asarray(n_convs, dtype=self.material.dtyper)
            n_norms_array[i, : len(n_convs)] = n_convs
        self.n_norms = n_norms_array

        n_omega = len(self.omg_ft_array)
        max_outer = max(len(outer) for outer in self.gmres_residual_list)
        max_q = max(len(q_list) for outer in self.gmres_residual_list for q_list in outer)
        max_gmres = max(len(res_list) for outer in self.gmres_residual_list for q_list in outer for res_list in q_list)
        gmres_residual_array = xp.full((n_omega, max_outer, max_q, max_gmres), np.nan, dtype=self.material.dtyper)
        for i, outer in enumerate(self.gmres_residual_list):
            for j, q_list in enumerate(outer):
                for k, res_list in enumerate(q_list):
                    res_list = xp.asarray(res_list, dtype=self.material.dtyper)
                    gmres_residual_array[i, j, k, : len(res_list)] = res_list
        self.gmres_residual = gmres_residual_array

    @property
    def flux(self) -> np.ndarray:
        """Energy flux tensor J computed from :py:attr:`n` and the velocity operator.

        Returns
        -------
        cupy.ndarray
            The thermal flux in W/m^2 for each omg_ft, shape (n_omg_ft, nq, nat3, nat3).

        """
        if self._flux is not None:
            return self._flux
        if not self.iter_time_list:
            raise RuntimeError("Solver has not been run yet. Please run the solver first.")

        # at this point all the heavy lifting was done by the GPU
        # we calculate the flux on the CPU to save GPU memory
        self._flux = np.zeros(self.n.shape, dtype=self.material.dtypec)
        for i, _ in enumerate(self.omg_ft_array):
            n = self.n[i]
            self._flux[i] = to_cpu(flux_from_n(n, self.material))
        return self._flux

    @property
    def kappa(self) -> np.ndarray:
        r"""Total thermal conductivity :math:`\kappa(\omega)`.

        Compute kappa from the Wigner distribution function :py:attr:`n` and the temperature change :py:attr:`dT`.

        Returns
        -------
        cupy.ndarray
            The thermal conductivity in W/m/K for each omg_ft, shape (n_omg_ft,).

        """
        if self._kappa is not None:
            return self._kappa
        if not self.iter_time_list:
            raise RuntimeError("Solver has not been run yet. Please run the solver first.")

        num = np.einsum("wqij->w", self.flux)
        den = 1j * self.k_ft * self.dT
        self._kappa = _safe_divide(num, den)
        return self._kappa

    @property
    def kappa_p(self) -> np.ndarray:
        r"""Ppopulation thermal conductivity :math:`\kappa_\mathrm{P}(\omega)`.

        Compute kappa_P from the Wigner distribution function :py:attr:`n` and the temperature change :py:attr:`dT`.

        Returns
        -------
        cupy.ndarray
            The population thermal conductivity in W/m/K for each omg_ft, shape (n_omg_ft,).

        """
        if self._kappa_p is not None:
            return self._kappa_p
        if not self.iter_time_list:
            raise RuntimeError("Solver has not been run yet. Please run the solver first.")

        flux_diag = np.einsum("wqii->w", self.flux)
        den = 1j * self.k_ft * self.dT
        self._kappa_p = _safe_divide(flux_diag, den)
        return self._kappa_p

    @property
    def kappa_c(self) -> np.ndarray:
        r"""Coherence thermal conductivity :math:`\kappa_\mathrm{C}(\omega)`.

        Compute kappa_C from the Wigner distribution function :py:attr:`n` and the temperature change :py:attr:`dT`.

        Returns
        -------
        cupy.ndarray
            The coherence thermal conductivity in W/m/K for each omg_ft, shape (n_omg_ft,).

        """
        if self._kappa_c is not None:
            return self._kappa_c
        if not self.iter_time_list:
            raise RuntimeError("Solver has not been run yet. Please run the solver first.")

        flux_total = np.einsum("wqij->w", self.flux)
        flux_diag = np.einsum("wqii->w", self.flux)
        flux_offdiag = flux_total - flux_diag
        den = 1j * self.k_ft * self.dT
        self._kappa_c = _safe_divide(flux_offdiag, den)
        return self._kappa_c


def _safe_divide(
    num: xp.ndarray | np.ndarray, den: xp.ndarray | np.ndarray, eps: float = 1e-300
) -> xp.ndarray | np.ndarray:
    """Element-wise division with broadcasting and protection against zeros.

    Parameters
    ----------
    num, den : array-like
        Numerator and denominator. Broadcastable to a common shape.
    eps : float, optional
        Small real added to ``|den|`` to avoid division by zero.

    Returns
    -------
    cupy.ndarray, numpy.ndarray
        The elementwise division result.

    """
    if hasattr(num, "__cuda_array_interface__") and hasattr(den, "__cuda_array_interface__"):  # pragma: no cover
        _xp = xp
    else:
        num = to_cpu(num)
        den = to_cpu(den)
        _xp = np

    num = _xp.asarray(num)
    den = _xp.asarray(den)

    out_shape = _xp.broadcast(num, den).shape
    num_b = _xp.broadcast_to(num, out_shape)
    den_b = _xp.broadcast_to(den, out_shape)

    out_dtype = _xp.result_type(num_b, den_b)

    mask = _xp.abs(den_b) > eps
    out = _xp.zeros(out_shape, dtype=out_dtype)
    out[mask] = num_b[mask] / den_b[mask]

    return out


def N_to_dT(n: xp.ndarray, material: Material) -> complex:
    r"""Compute :math:`\Delta T` from a Wigner distribution.

    Parameters
    ----------
    n : cupy.ndarray
        Wigner distribution function n, shape (nq, nat3, nat3).
    material : :py:class:`~greenWTE.base.Material`
        Material instance.

    Returns
    -------
    complex
        The temperature change dT.

    See Also
    --------
    _N_to_dT : Core function that computes dT from n and material properties.

    """
    return _N_to_dT(n, material.phonon_freq, material.heat_capacity, material.volume)


def _N_to_dT(n: xp.ndarray, phonon_freq: xp.ndarray, heat_capacity: xp.ndarray, volume: float) -> complex:
    r"""Compute :math:`\Delta T` from a Wigner distribution.

    Parameters
    ----------
    n : cupy.ndarray
        The wigner distribution function n, shape (nq, nat3, nat3).
    phonon_freq : cupy.ndarray
        The phonon frequencies in rad/s, shape (nq, nat3).
    heat_capacity : cupy.ndarray
        The heat capacity in J/m^3/K of the phonon modes, shape (nq, nat3).
    volume : float
        The volume of the cell in m^3.

    Returns
    -------
    complex
        The temperature change dT.

    See Also
    --------
    N_to_dT : Wrapper function that extracts material properties from a Material instance.

    """
    nq = n.shape[0]
    dT = 0
    for i in range(nq):
        dT += xp.sum(phonon_freq[i] * xp.diag(n[i]))
    dT *= hbar / volume / nq
    dT /= xp.sum(heat_capacity)
    return dT


def dT_to_N_iterative(
    dT: complex,
    omg_ft: float,
    k_ft: float,
    material: Material,
    source: xp.ndarray,
    source_type="energy",
    sol_guess=None,
    solver="gmres",
    conv_thr_rel=1e-12,
    conv_thr_abs=0,
    progress=False,
) -> tuple[xp.ndarray, list]:
    r"""Fixed-point mapping :math:`\Delta T \mapsto n` using iterative linear solves.

    This function solves the linear system of equations that arises from the WTE for the wigner distribution function n
    for a given temperature change dT.

    Parameters
    ----------
    dT : complex
        The temperature change dT in K.
    omg_ft : float
        The temporal Fourier variable in rad/s.
    k_ft : float
        The thermal grating wavevector in rad/m.
    material : :py:class:`~greenWTE.base.Material`
        Material instance.
    source : cupy.ndarray
        The source term of the WTE, shape (nq, nat3, nat3).
    source_type : str
        The type of the source term, either "energy" or "gradient". When injecting energy through the source term, there
        is no additional factor of dT for the offdiagonals of the source. For the temperature gradient type source
        terms, the offdiagonal elements are scaled by dT.
    sol_guess : cupy.ndarray, optional
        The initial guess for the solution, shape (nq, nat3, nat3).
    solver : {'gmres', 'direct'}, optional
        Inner linear solver. ``'gmres'`` uses :py:func:`cupyx.scipy.sparse.linalg.gmres`; ``'direct'`` uses
        :py:func:`cupy.linalg.solve` to perform a dense factorization on the GPU.
    conv_thr_rel : float, optional
        The relative convergence threshold for the solver.
    conv_thr_abs : float, optional
        The absolute convergence threshold for the solver.
    progress : bool, optional
        If True, a `.` is printed after each iteration to indicate progress.

    Returns
    -------
    cupy.ndarray
        The Wigner distribution function n, shape (nq, nat3, nat3).
    list
        A list of residuals for each iteration of the solver.

    See Also
    --------
    _dT_to_N_iterative : Core function that computes n from dT and material properties.

    """
    return _dT_to_N_iterative(
        dT=dT,
        omg_ft=omg_ft,
        k_ft=k_ft,
        phonon_freq=material.phonon_freq,
        linewidth=material.linewidth,
        velocity_operator=material.velocity_operator,
        heat_capacity=material.heat_capacity,
        volume=material.volume,
        source=source,
        source_type=source_type,
        sol_guess=sol_guess,
        dtyper=material.dtyper,
        dtypec=material.dtypec,
        solver=solver,
        conv_thr_rel=conv_thr_rel,
        conv_thr_abs=conv_thr_abs,
        progress=progress,
    )


def _dT_to_N_iterative(
    dT: complex,
    omg_ft: float,
    k_ft: float,
    phonon_freq: xp.ndarray,
    linewidth: xp.ndarray,
    velocity_operator: xp.ndarray,
    heat_capacity: xp.ndarray,
    volume: float,
    source: xp.ndarray,
    source_type="energy",
    sol_guess=None,
    dtyper=xp.float64,
    dtypec=xp.complex128,
    solver="gmres",
    conv_thr_rel=1e-12,
    conv_thr_abs=0,
    progress=False,
) -> tuple[xp.ndarray, list]:
    r"""Fixed-point mapping :math:`\Delta T \mapsto n` using iterative linear solves.

    This function solves the linear system of equations that arises from the WTE for the wigner distribution function n
    for a given temperature change dT.

    Parameters
    ----------
    dT : complex
        The temperature change dT in K.
    omg_ft : float
        The temporal Fourier variable in rad/s.
    k_ft : float
        The thermal grating wavevector in rad/m.
    phonon_freq : cupy.ndarray
        The phonon frequencies in rad/s, shape (nq, nat3).
    linewidth : cupy.ndarray
        The linewidths in rad/s, shape (nq, nat3).
    velocity_operator : cupy.ndarray
        The velocity operator for the phonon modes, shape (nq, nat3, nat3).
    heat_capacity : cupy.ndarray
        The heat capacity in J/m^3/K of the phonon modes, shape (nq, nat3).
    volume : float
        The volume of the cell in m^3.
    source : cupy.ndarray
        The source term of the WTE, shape (nq, nat3, nat3).
    source_type : str
        The type of the source term, either "energy" or "gradient". When injecting energy through the source term, there
        is no additional factor of dT for the offdiagonals of the source. For the temperature gradient type source
        terms, the offdiagonal elements are scaled by dT.
    sol_guess : cupy.ndarray, optional
        The initial guess for the solution, shape (nq, nat3, nat3).
    dtyper : cupy.dtype, optional
        The real dtype to use.
    dtypec : cupy.dtype, optional
        The complex dtype to use.
    solver : {'gmres', 'direct'}, optional
        Inner linear solver. ``'gmres'`` uses :py:func:`cupyx.scipy.sparse.linalg.gmres`; ``'direct'`` uses
        :py:func:`cupy.linalg.solve` to perform a dense factorization on the GPU.
    conv_thr_rel : float, optional
        The relative convergence threshold for the solver.
    conv_thr_abs : float, optional
        The absolute convergence threshold for the solver.
    progress : bool, optional
        If True, a `.` is printed after each iteration to indicate progress.

    Returns
    -------
    cupy.ndarray
        The Wigner distribution function n, shape (nq, nat3, nat3).
    list
        A list of residuals for each iteration of the solver.

    """
    with nvtx_utils.annotate("init dT_to_N", color="blue"):
        nq = phonon_freq.shape[0]
        nat3 = phonon_freq.shape[1]
        n = xp.zeros((nq, nat3, nat3), dtype=dtypec)

        I_small = xp.eye(nat3, dtype=dtyper)
        I_big = xp.eye(nat3**2, dtype=dtypec)
        OMG = xp.zeros((nat3, nat3), dtype=dtyper)
        GAM = xp.zeros((nat3, nat3), dtype=dtyper)

        outer_residuals = []

    for ii in range(nq):
        with nvtx_utils.annotate("init q", color="purple"):
            xp.fill_diagonal(OMG, phonon_freq[ii])
            xp.fill_diagonal(GAM, linewidth[ii])

            gv_op = velocity_operator[ii]

            # term1 = xp.kron(I, OMG) - xp.kron(OMG, I) - (omg_ft * I_big)
            term1 = (
                xp.einsum("ij,kl->ikjl", I_small, OMG).reshape(nat3**2, nat3**2)
                - xp.einsum("ij,kl->ikjl", OMG, I_small).reshape(nat3**2, nat3**2)
                - omg_ft * I_big
            )
            # term2 = (k_ft / 2) * (xp.kron(I, gv_op) + xp.kron(gv_op.T, I))
            term2 = (k_ft / 2) * (
                xp.einsum("ij,kl->ikjl", I_small, gv_op).reshape(nat3**2, nat3**2)
                + xp.einsum("ij,kl->ikjl", gv_op.T, I_small).reshape(nat3**2, nat3**2)
            )
            # term3 = 0.5 * (xp.kron(I, GAM) + xp.kron(GAM, I))
            term3 = 0.5 * (
                xp.einsum("ij,kl->ikjl", I_small, GAM).reshape(nat3**2, nat3**2)
                + xp.einsum("ij,kl->ikjl", GAM, I_small).reshape(nat3**2, nat3**2)
            )

            lhs = (1j * (term1 - term2)) + term3

            nbar_deltat = volume * nq / hbar / phonon_freq[ii] * heat_capacity[ii] * xp.array(dT, dtype=dtypec)
            if source_type == "energy":
                rhs = xp.copy(source[ii])
            elif source_type == "gradient":
                rhs = (xp.array(dT, dtype=dtypec) * source[ii]).copy()
            else:
                raise ValueError(f"Unknown source type: {source_type}")
            xp.fill_diagonal(rhs, xp.diag(rhs) + linewidth[ii] * nbar_deltat)
            rhs = rhs.flatten(order="F")

            residuals = []

            def gmres_callback(residual):
                residuals.append(residual)

        if solver == "gmres":
            with nvtx_utils.annotate("gmres", color="green"):
                guess = sol_guess[ii].flatten(order="F") if sol_guess is not None else xp.zeros_like(rhs, dtype=dtypec)
                gmres_kwargs = {
                    "x0": guess,
                    "callback": gmres_callback,
                    "callback_type": "pr_norm",
                    "atol": conv_thr_abs,
                    "M": xp.diag(1 / lhs.diagonal()),
                }
                if HAVE_GPU:  # pragma: no cover
                    # cupyx GMRES uses 'tol' for relative tolerance
                    gmres_kwargs["tol"] = conv_thr_rel
                else:  # pragma: no cover
                    # scipy GMRES uses 'rtol' for relative tolerance
                    gmres_kwargs["rtol"] = conv_thr_rel
                sol, info = xp_gmres(
                    lhs,
                    rhs,
                    **gmres_kwargs,
                )
            if info != 0:
                print(f"GMRES failed to converge: {info}")
        elif solver == "cgesv":
            sol = xp.linalg.solve(lhs, rhs)
        else:
            raise ValueError(f"Unknown inner solver: {solver}")
        outer_residuals.append(residuals)
        sol = sol.reshape(nat3, nat3, order="F")
        n[ii] = sol
    if progress:
        print(".", end="")
    return n, outer_residuals


def dT_to_N_matmul(
    dT: complex,
    material: Material,
    green: xp.ndarray,
    source: xp.ndarray,
    source_type: str = "energy",
) -> xp.ndarray:
    r"""Fixed-point mapping :math:`\Delta T \mapsto n` using precomputed Green operators.

    This variant assumes a Green operator has been precomputed and can be applied via batched matrix-matrix products.

    Parameters
    ----------
    dT : complex
        The temperature change dT in K.
    material : :py:class:`~greenWTE.base.Material`
        Material instance.
    source : cupy.ndarray
        The source term of the WTE, shape (nq, nat3, nat3).
    source_type : str
        The type of the source term, either "energy" or "gradient". When injecting energy through the source term, there
        is no additional factor of dT for the offdiagonals of the source. For the temperature gradient type source
        terms, the offdiagonal elements are scaled by dT.
    green : cupy.ndarray
        The Green's function, shape (nq, nat3, nat3).

    Returns
    -------
    cupy.ndarray
        The Wigner distribution function n, shape (nq, nat3, nat3).

    Notes
    -----
    This function is a vectorized equivalent of the following more legible loop-based implementation:

    .. code-block:: python

        nq, nat3 = material.nq, material.nat3
        n = xp.zeros((nq, nat3, nat3), dtype=material.dtypec)
        for iq in range(nq):
            nbar_deltat = (
                material.volume * nq / hbar
                / material.phonon_freq[iq]
                * material.heat_capacity[iq]
                * xp.array(dT, dtype=material.dtypec)
            )
            rhs = xp.copy(source[iq])
            xp.fill_diagonal(
                rhs,
                xp.diag(source[iq]) + material.linewidth[iq] * nbar_deltat
            )
            n[iq] = (green[iq] @ rhs.flatten(order="F")).reshape(nat3, nat3, order="F")
        return n

    The vectorized form avoids explicit Python loops and uses batched
    matrix multiplication for improved performance on the GPU.

    """
    with nvtx_utils.annotate("dT_to_N_matmul", color="orange"):
        nq, nat3 = material.nq, material.nat3
        m = nat3**2

        green = xp.asarray(green)
        green = xp.ascontiguousarray(green).reshape(nq, m, m)
        if source_type == "energy":
            rhs = xp.ascontiguousarray(source).reshape(nq, nat3, nat3).copy()
        elif source_type == "gradient":
            rhs = (xp.array(dT, dtype=material.dtypec) * xp.ascontiguousarray(source).reshape(nq, nat3, nat3)).copy()
        else:
            raise ValueError(f"Unknown source type: {source_type}")
        prefac = material.volume * nq / hbar * material.heat_capacity / material.phonon_freq * dT

        i = xp.arange(nat3)
        rhs[:, i, i] += material.linewidth * prefac

        rhs_flat = rhs.reshape(nq, m, order="F")
        n_flat = (green @ rhs_flat[..., None]).squeeze(-1)

    return n_flat.reshape(nq, nat3, nat3, order="F")


def flux_from_n(n: xp.ndarray, material: Material) -> xp.ndarray:
    """Energy flux tensor J computed from :py:attr:`n` and the velocity operator.

    It corresponds to Equation (42) in `Phys. Rev. X 12, 041011`_.

    .. _Phys. Rev. X 12, 041011: https://doi.org/10.1103/PhysRevX.12.041011

    Parameters
    ----------
    n : cupy.ndarray
        The wigner distribution function n, shape (nq, nat3, nat3).
    material : :py:class:`~greenWTE.base.Material`
        Material instance.

    Returns
    -------
    cupy.ndarray
        The thermal flux calculated from the Wigner distribution function n, shape (nq, nat3, nat3).

    See Also
    --------
    _flux_from_n : Core function that computes the flux from n and material properties.

    """
    return _flux_from_n(n, material.velocity_operator, material.phonon_freq, material.volume)


def _flux_from_n(n: xp.ndarray, velocity_operator: xp.ndarray, phonon_freq: xp.ndarray, volume: float) -> xp.ndarray:
    """Energy flux tensor J computed from :py:attr:`n` and the velocity operator.

    It corresponds to Equation (42) in `Phys. Rev. X 12, 041011`_.

    .. _Phys. Rev. X 12, 041011: https://doi.org/10.1103/PhysRevX.12.041011

    Parameters
    ----------
    n : cupy.ndarray
        The wigner distribution function n, shape (nq, nat3, nat3).
    velocity_operator : cupy.ndarray
        The velocity operator for the phonon modes, shape (nq, nat3, nat3).
    phonon_freq : cupy.ndarray
        The phonon frequencies [rad/s], shape (nq, nat3).
    volume : float
        The volume of the cell [m^3].

    Returns
    -------
    cupy.ndarray
        The thermal flux calculated from the Wigner distribution function n, shape (nq, nat3, nat3).

    See Also
    --------
    flux_from_n : Wrapper function that extracts material properties from a Material instance.

    """
    freq_sum = phonon_freq[:, :, None] + phonon_freq[:, None, :]
    flux = freq_sum * velocity_operator * xp.transpose(n, axes=(0, 2, 1))
    flux *= hbar / 2 / volume / n.shape[0]
    return flux


def dT_bte_prb(omg_ft, k_ft, phonon_freq, linewidth, group_velocity, heat_capacity, weight, heat):  # pragma: no cover
    """Eq (9) from Phys Rev. B 104, 245424 [https://doi.org/10.1103/PhysRevB.104.245424]."""
    nq = phonon_freq.shape[0]
    nat3 = phonon_freq.shape[1]

    phonon_freq = phonon_freq.flatten()
    linewidth = linewidth.flatten()
    heat_capacity = heat_capacity.flatten()
    group_velocity = group_velocity.reshape(nq * nat3)
    heat = heat.flatten()
    weight = xp.repeat(weight, nat3).flatten()
    p = heat

    okv = omg_ft + (group_velocity * k_ft)
    A_inv = 1 / (linewidth + 1j * okv)
    Dc = okv * heat_capacity
    Ap = xp.dot(A_inv, p)
    ADc = xp.dot(A_inv, Dc * weight)
    num = Ap
    den = 1j * ADc
    dT = xp.sum(num) / xp.sum(den)
    return dT


def dT_bte_from_wte(
    omg_ft, k_ft, phonon_freq, linewidth, group_velocity, heat_capacity, weight, heat, volume
):  # pragma: no cover
    """Eq (9) from Phys Rev. B 104, 245424 [https://doi.org/10.1103/PhysRevB.104.245424] for WTE result.

    Here we are using the correct normalization factors to match the BTE.
    """
    nq = phonon_freq.shape[0]
    nat3 = phonon_freq.shape[1]

    phonon_freq = phonon_freq.flatten()
    linewidth = linewidth.flatten()
    heat_capacity = heat_capacity.flatten()
    group_velocity = group_velocity.reshape(nq * nat3)
    heat = heat.flatten()
    weight = xp.repeat(weight, nat3).flatten()
    p = heat

    okv = omg_ft + (group_velocity * k_ft)
    giwkv = linewidth - 1j * okv

    num = xp.sum(hbar * phonon_freq / volume / nq * p / giwkv)
    den = xp.sum(heat_capacity) - xp.sum(linewidth * heat_capacity / giwkv)
    dT = num / den
    return dT


def kappa_eff_prb(k_ft, linewidth, group_velocity, heat_capacity):  # pragma: no cover
    """Calculate the effective thermal conductivity that would follow from the BTE.

    This equation is taken from
    a draft of Phys. Rev. B 104, 245424 [https://doi.org/10.1103/PhysRevB.104.245424]. It was replaced by Eq. (12)
    in the published version. The version here gives better results.

    Parameters
    ----------
    k_ft : float
        The thermal grating wavevector in rad/m.
    linewidth, group_velocity, heat_capacity : array_like
        The linewidth in rad/s, group velocity in m/s, and heat capacity in J/m^3/K of the phonon modes. 2D arrays with
        shape (nq, nat3). Note that the group velocity is expected to be along the grating wavevector direction.

    Returns
    -------
    float
        The effective thermal conductivity in W/m/K.

    """
    linewidth = linewidth.flatten()
    heat_capacity = heat_capacity.flatten()
    group_velocity = group_velocity.flatten()
    heat_capacity = heat_capacity.flatten()

    ctot = np.sum(heat_capacity)
    okv = group_velocity * k_ft
    A_inv = 1 / (linewidth + 1j * okv)
    p = heat_capacity / ctot
    num = np.real(np.dot(group_velocity.T, A_inv * heat_capacity * group_velocity))
    den = 1 - np.real(np.dot(okv, A_inv * p))
    return num / den
