"""Base classes and methods for the greenWTE package."""

import time
import warnings
from abc import ABC, abstractmethod
from argparse import Namespace

import cupy as cp
import numpy as np
import nvtx
from cupyx.scipy.interpolate import PchipInterpolator
from cupyx.scipy.sparse.linalg import gmres
from scipy.constants import hbar
from scipy.optimize import root


class Material:
    """Material specific container class for physical properties.

    This class can either be populated by passing arrays directly or by loading information from the output of a
    phono3py calculation.
    """

    _degeneracy_mask = None
    _degeneracy_mask_offdiag = None
    _degeneracy_cache_key = None  # (tol, nq, nat3, id(phonon_freq))

    def __init__(
        self,
        temperature,
        velocity_operator,
        phonon_freq,
        linewidth,
        heat_capacity,
        volume,
        name=None,
        degeneracy_tol=1e-7,
    ):
        """Initialize the Material class with physical properties."""
        self.temperature = temperature
        self.velocity_operator = velocity_operator
        self.phonon_freq = phonon_freq
        self.linewidth = linewidth
        self.heat_capacity = heat_capacity
        self.volume = volume
        self.dtypec = self.velocity_operator.dtype
        self.dtyper = self.phonon_freq.dtype
        self.name = name
        self.nq = self.phonon_freq.shape[0]
        self.nat3 = self.phonon_freq.shape[1]
        self._degeneracy_tol = degeneracy_tol

    @classmethod
    def from_phono3py(
        cls, filename, temperature, dir_idx=0, dtyper=cp.float64, dtypec=cp.complex128, degeneracy_tol=1e-7
    ):
        """Load material properties from a phono3py output file.

        Parameters
        ----------
        filename : str
            The path to the phono3py output file.
        temperature : float
            The temperature at which to load the data.
        dir_idx : int, optional
            The index of the directory containing the output files (default is 0).
        dtyper : cupy.dtype, optional
            The data type for the real parts of the matrices, default is cp.float64.
        dtypec : cupy.dtype, optional
            The data type for the complex matrices, default is cp.complex128.
        degeneracy_tol : float, optional
            The degeneracy tolerance for the material, default is 1e-7.

        Returns
        -------
        Material
            An instance of the Material class with the loaded properties.

        """
        from .solve import load_phono3py_data

        velocity_operator, phonon_freq, linewidth, heat_capacity, volume, _ = load_phono3py_data(
            filename, temperature=temperature, dir_idx=dir_idx, dtyper=dtyper, dtypec=dtypec
        )
        return cls(
            temperature,
            velocity_operator,
            phonon_freq,
            linewidth,
            heat_capacity,
            volume,
            name=filename,
            degeneracy_tol=degeneracy_tol,
        )

    def __repr__(self):
        """Return a string representation of the Material."""
        return f"{self.name}@{self.temperature}K with {self.nq} qpoints and {self.nat3} modes"

    @property
    def degeneracy_tol(self):
        """Get the degeneracy tolerance.

        This value is a relative to the maximum phonon frequency per q-point.
        """
        return self._degeneracy_tol

    @degeneracy_tol.setter
    def degeneracy_tol(self, value):
        new_tol = value
        if new_tol != self.degeneracy_tol:
            self._degeneracy_tol = new_tol
            self._degeneracy_mask = None
            self._degeneracy_mask_offdiag = None

    def _compute_degeneracy_masks(self):
        """Compute the degeneracy masks for the materials phonon system."""
        key = (self._degeneracy_tol, self.nq, self.nat3, int(self.phonon_freq.data.ptr))
        if key == self._degeneracy_cache_key and self._degeneracy_mask is not None:
            return  # up to date
        deg, off = _degeneracy_mask(self.phonon_freq, self._degeneracy_tol)
        self._degeneracy_mask = deg
        self._degeneracy_mask_offdiag = off
        self._degeneracy_cache_key = key

    @property
    def degeneracy_mask(self) -> cp.ndarray:
        """Boolean mask of degenerate pairs per q-point.

        Returns
        -------
        cupy.ndarray
            Array of shape ``(nq, nat3, nat3)`` with ``True`` where modes
            ``i`` and ``j`` are considered degenerate at that ``q`` (includes the diagonal).

        """
        self._compute_degeneracy_masks()
        return self._degeneracy_mask

    @property
    def degeneracy_mask_offdiag(self) -> cp.ndarray:
        """Boolean mask of non-degenerate off-diagonal pairs per q-point.

        Returns
        -------
        cupy.ndarray
            Array of shape ``(nq, nat3, nat3)`` with ``True`` where ``i != j`` and
            the pair is **not** considered degenerate at that ``q``.

        """
        self._compute_degeneracy_masks()
        return self._degeneracy_mask_offdiag


class AitkenAccelerator:
    """Aitken's delta-squared process for accelerating convergence of sequences. `Wiki <https://en.wikipedia.org/wiki/Aitken%27s_delta-squared_process>`_.

    Attributes
    ----------
    history : list
        A list to store the history of temperature changes dT. The last three entries are used for the acceleration.

    """

    def __init__(self):
        """Initialize the AitkenAccelerator."""
        self.history = []

    def reset(self):
        """Reset the AitkenAccelerator history."""
        self.history.clear()

    def update(self, dT_new):
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


def estimate_initial_dT(omg_ft, history, dtyper=cp.float64, dtypec=cp.complex128):
    """Estimate the initial temperature change dT for a given temporal Fourier variable omg_ft.

    We try to interpolate
    the dT values from the history of previous omg_ft values. If no history is available, we return a default of 1e-6.

    Parameters
    ----------
    omg_ft : float
        The temporal Fourier variable [Hz].
    history : list of tuples
        A list of (omg_ft, dT) tuples representing the history of previous temperature changes.
    dtyper : cupy.dtype, optional
        The data type for the real parts of the matrices, default is cp.float64.
    dtypec : cupy.dtype, optional
        The data type for the complex matrices, default is cp.complex128.

    Returns
    -------
    dT_guess : cupy.ndarray
        The estimated initial temperature change dT [K] for the given omg_ft. If no history is available, returns 1e-6.

    """
    if not history:
        return cp.asarray(1 + 1j)

    omg_fts, dTs = zip(*sorted(history))
    omg_fts = cp.asarray(omg_fts, dtype=dtyper)
    dTs = cp.asarray(dTs, dtype=dtypec)

    if len(omg_fts) == 1:
        return dTs[0]

    interp_func = PchipInterpolator(omg_fts, dTs, extrapolate=True)
    dT_guess = interp_func(omg_ft)

    return dT_guess


class SolverBase(ABC):
    r"""Abstract base class for Wigner Transport Equation (WTE) solvers.

    This class defines the common interface, data structures, and solver
    control logic for computing the Wigner distribution function `n` and
    derived transport quantities such as flux and thermal conductivity
    from a given material model, source term, and temporal Fourier
    frequency grid.

    Concrete subclasses must implement the :meth:`_dT_to_N` method to
    perform the actual mapping from a temperature change ``dT`` to the
    Wigner distribution function ``n`` for a specific solver
    implementation.

    The class supports multiple outer-solver strategies:
    - **plain**: fixed-point iteration
    - **aitken**: Aitken's Δ² acceleration of fixed-point iteration
    - **root**: root finding in real/imag space

    Parameters
    ----------
    omg_ft_array : cupy.ndarray
        1D array of temporal Fourier variables :math:`\omega` [Hz] for which
        the WTE will be solved.
    k_ft : float
        Magnitude of the spatial Fourier variable :math:`k` [m⁻¹].
    material : Material
        Material object containing the necessary physical properties
        (phonon frequencies, linewidths, heat capacities, etc.).
    source : cupy.ndarray
        Source term of the WTE, with shape ``(nq, nat3, nat3)``.
    max_iter : int, optional
        Maximum number of iterations for the outer solver (default is 100).
    conv_thr : float, optional
        Convergence threshold for the outer solver. The exact interpretation
        depends on the outer solver method (default is 1e-12).
    outer_solver : {'root', 'aitken', 'plain'}, optional
        Outer solver strategy to use. One of:
        - ``'root'``: nonlinear root finding in 2D space (real & imaginary parts)
        - ``'aitken'``: Aitken Δ² acceleration applied to fixed-point iteration
        - ``'plain'``: plain fixed-point iteration without acceleration
        Default is ``'root'``.
    command_line_args : argparse.Namespace, optional
        Optional namespace of parsed command-line arguments for controlling
        solver behavior or I/O (default is an empty Namespace).
    residual_weights : tuple[float, float], optional
        Weights for the real and imaginary parts of the residual (default is (1.0, 0.0)).

    Attributes
    ----------
    dT : cupy.ndarray
        Temperature change [K] for each frequency, shape ``(n_omg_ft,)``.
    dT_init : cupy.ndarray
        Initial guess for `dT` for each frequency, shape ``(n_omg_ft,)``.
    n : cupy.ndarray
        Wigner distribution function for each frequency, shape
        ``(n_omg_ft, nq, nat3, nat3)``.
    niter : cupy.ndarray
        Number of outer-solver iterations taken for each frequency,
        shape ``(n_omg_ft,)``.
    iter_time_list : list of list of float
        Iteration times (seconds) for each frequency.
    dT_convergence_list : list of list of complex
        Sequence of `dT` values over iterations for each frequency.
    n_convergence_list : list of list of float
        Sequence of relative changes in `n` over iterations for each frequency.
    gmres_residual_list : list
        GMRES residuals from the inner solver (if applicable).
    history : list of tuple
        List of ``(omega_ft, dT)`` pairs from previous runs (used for initial guesses).
    progress : bool
        If ``True``, prints progress for single-frequency solves.

    Notes
    -----
    - The solver stores intermediate convergence data in lists during the run.
      After solving all frequencies, :meth:`_solution_lists_to_arrays` can
      be used to convert them into CuPy arrays with NaN padding for easier
      post-processing.
    - The actual numerical strategy for mapping ``dT`` to ``n`` is deferred
      to subclasses via the :meth:`_dT_to_N` method.

    """

    _flux = None
    _kappa = None
    _kappa_p = None
    _kappa_c = None

    def __init__(
        self,
        omg_ft_array: cp.ndarray,
        k_ft: float,
        material: Material,
        source: cp.ndarray,
        *,
        max_iter: int = 100,
        conv_thr: float = 1e-12,
        outer_solver: str = "root",
        command_line_args=Namespace(),
        residual_weights: tuple[float, float] = (1.0, 0.0),
    ):
        """Initialize SolverBase."""
        self.omg_ft_array = omg_ft_array
        self.k_ft = k_ft
        self.material = material
        self.source = source
        self.max_iter = max_iter
        self.conv_thr = conv_thr
        self.outer_solver = outer_solver
        self.command_line_args = command_line_args
        self.dtyper = material.dtyper
        self.dtypec = material.dtypec
        self.nq = material.nq
        self.nat3 = material.nat3

        self.history = []
        self.dT = cp.zeros_like(self.omg_ft_array, dtype=self.dtypec)
        self.dT_init = cp.zeros_like(self.omg_ft_array, dtype=self.dtypec)
        self.n = cp.zeros((self.omg_ft_array.shape[0], self.nq, self.nat3, self.nat3), dtype=self.dtypec)
        self.niter = cp.zeros(self.omg_ft_array.shape[0], dtype=cp.int32)
        self.iter_time_list = []
        self.dT_convergence_list = []
        self.n_convergence_list = []
        self.gmres_residual_list = []
        self.progress = self.omg_ft_array.shape[0] == 1
        self._wR = residual_weights[0]
        self._wI = residual_weights[1]

    @abstractmethod
    def _dT_to_N(
        self,
        dT: complex,
        omg_ft: float,
        *,
        omg_idx: int,
        sol_guess: cp.ndarray | None = None,
    ) -> tuple[cp.ndarray, list]:
        """Implement by subclasses to solve for the Wigner distribution function n from dT."""
        pass

    def run(self):
        """Run the WTE solver for each temporal Fourier variable in omg_ft_array.

        This method uses the specified outer solver (Aitken, plain, or root) to solve the WTE for each omg_ft in
        omg_ft_array. After running, the results are stored in the class attributes dT, dT_init, n, niter,
        n_convergence, iter_time_list, dT_convergence_list, n_convergence_list, and gmres_residual_list.
        """
        if self.outer_solver == "aitken":
            run_func = self._run_solver_aitken
        elif self.outer_solver == "plain":
            run_func = self._run_solver_plain
        elif self.outer_solver == "root":
            run_func = self._run_solver_root
        else:
            raise ValueError(f"Unknown outer solver: {self.outer_solver}")

        for i, omg_ft in enumerate(self.omg_ft_array):
            ret = run_func(i, omg_ft)
            self.dT[i] = ret[1]
            self.dT_init[i] = ret[2]
            self.n[i] = ret[3]
            self.niter[i] = ret[4]
            self.iter_time_list.append(ret[5])
            self.dT_convergence_list.append(ret[6])
            self.n_convergence_list.append(ret[7])
            self.gmres_residual_list.append(ret[8])

            # check self-consistency by checking norm(n(dT) - n)
            last_n = ret[3]
            theoretical_next_n, _ = self._dT_to_N(
                dT=ret[1],
                omg_ft=omg_ft,
                omg_idx=i,
                sol_guess=None,
            )
            n_step_norm = cp.linalg.norm(theoretical_next_n - last_n) / cp.linalg.norm(last_n)
            self.n_convergence_list[i].append(n_step_norm)

            if self.progress:
                print("")
            width = len(str(len(self.omg_ft_array)))
            print(
                f"[{i + 1:{width}d}/{len(self.omg_ft_array)}] "
                f"k={self.k_ft:.2e} "
                f"w={omg_ft:.2e} "
                f"dT={self.dT[i]: .2e} "
                f"nit={self.niter[i]: 4d} "
                f"it_time={cp.mean(cp.array(self.iter_time_list[-1])):.2f} "
                f"n_conv={n_step_norm:.1e} "
            )

        self._solution_lists_to_arrays()

    def _run_solver_aitken(self, omg_idx, omg_ft):
        """Run the WTE solver using Aitken's delta-squared process for acceleration.

        Parameters
        ----------
        omg_idx : int
            The index of the temporal Fourier variable for which the WTE will be solved.
        omg_ft : float
            The temporal Fourier variable [Hz] for which the WTE will be solved.

        Returns
        -------
        omg_ft : float
            The input temporal Fourier variable [Hz].
        dT : complex
            The calculated temperature change dT [K] for the given omg_ft.
        dT_init : complex
            The initial temperature change dT [K] estimated for the given omg_ft.
        n : cupy.ndarray
            The wigner distribution function n for the given omg_ft, shape (nq, nat3, nat3).
        niter : int
            The number of iterations taken for the outer solver to converge for the given omg_ft.
        iter_times : list
            A list of iteration times for the outer solver for the given omg_ft.
        dT_convergence : list
            A list of convergence values for the temperature changes dT for the given omg_ft.
        n_convergence : list
            A list of convergence values for the wigner distribution function n for the given omg_ft.
        gmres_residual : list
            A list of GMRES residuals for the given omg_ft. Empty if the inner solver is not GMRES.

        """
        accelerator = AitkenAccelerator()
        dT_init = estimate_initial_dT(
            omg_ft=omg_ft, history=self.history, dtyper=self.material.dtyper, dtypec=self.material.dtypec
        )
        dT = dT_init
        n = None
        iter_times = []
        dT_convergence = []
        n_convergence = []
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
            r = dT - dT_new
            r_norm = float(np.hypot(self._wR * cp.real(r).item(), self._wI * cp.imag(r).item()))
            dT_convergence.append(dT_new)
            dT = accelerator.update(dT_new)

            iter_times.append(time.time() - iter_start)

            if n_prev is not None:
                n_step_norm = cp.linalg.norm(n - n_prev) / cp.linalg.norm(n_prev)
                n_convergence.append(n_step_norm)

            if np.abs(r_norm) < self.conv_thr:
                self.history.append((omg_ft, dT_new))
                break

        return omg_ft, dT, dT_init, n, iterations, iter_times, dT_convergence, n_convergence, gmres_residual

    def _run_solver_plain(self, omg_idx, omg_ft):
        """Run the WTE solver without acceleration, iterating until convergence.

        Parameters
        ----------
        omg_idx : int
            The index of the temporal Fourier variable for which the WTE will be solved.
        omg_ft : float
            The temporal Fourier variable [Hz] for which the WTE will be solved.

        Returns
        -------
        omg_ft : float
            The input temporal Fourier variable [Hz].
        dT : complex
            The calculated temperature change dT [K] for the given omg_ft.
        dT_init : complex
            The initial temperature change dT [K] estimated for the given omg_ft.
        n : cupy.ndarray
            The wigner distribution function n for the given omg_ft, shape (nq, nat3, nat3).
        niter : int
            The number of iterations taken for the outer solver to converge for the given omg_ft.
        iter_times : list
            A list of iteration times for the outer solver for the given omg_ft.
        dT_convergence : list
            A list of convergence values for the temperature changes dT for the given omg_ft.
        n_convergence : list
            A list of convergence values for the wigner distribution function n for the given omg_ft.
        gmres_residual : list
            A list of GMRES residuals for the given omg_ft. Empty if the inner solver is not GMRES.

        """
        dT_init = estimate_initial_dT(
            omg_ft=omg_ft, history=self.history, dtyper=self.material.dtyper, dtypec=self.material.dtypec
        )
        dT = dT_init
        n = None
        iter_times = []
        dT_convergence = []
        n_convergence = []
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
            r = dT - dT_new
            r_norm = float(np.hypot(self._wR * cp.real(r).item(), self._wI * cp.imag(r).item()))
            dT_convergence.append(dT_new)
            dT = dT_new

            iter_times.append(time.time() - iter_start)

            if n_prev is not None:
                n_step_norm = cp.linalg.norm(n - n_prev) / cp.linalg.norm(n_prev)
                n_convergence.append(n_step_norm)

            if r_norm < self.conv_thr:
                self.history.append((omg_ft, dT))
                break

        return omg_ft, dT, dT_init, n, iterations, iter_times, dT_convergence, n_convergence, gmres_residual

    def _run_solver_root(self, omg_idx, omg_ft):
        """Run the WTE solver using root finding to solve for the temperature change dT.

        Usually converges MUCH faster than Aitken or plain methods.

        Parameters
        ----------
        omg_idx : int
            The index of the temporal Fourier variable for which the WTE will be solved.
        omg_ft : float
            The temporal Fourier variable [Hz] for which the WTE will be solved.

        Returns
        -------
        omg_ft : float
            The input temporal Fourier variable [Hz].
        dT : complex
            The calculated temperature change dT [K] for the given omg_ft.
        dT_init : complex
            The initial temperature change dT [K] estimated for the given omg_ft.
        n : cupy.ndarray
            The wigner distribution function n for the given omg_ft, shape (nq, nat3, nat3).
        niter : int
            The number of iterations taken for the outer solver to converge for the given omg_ft.
        iter_times : list
            A list of iteration times for the outer solver for the given omg_ft.
        dT_convergence : list
            A list of convergence values for the temperature changes dT for the given omg_ft.
        n_convergence : list
            A list of convergence values for the wigner distribution function n for the given omg_ft.
        gmres_residual : list
            A list of GMRES residuals for the given omg_ft. Empty if the inner solver is not GMRES.

        """
        dT_init = estimate_initial_dT(
            omg_ft=omg_ft, history=self.history, dtyper=self.material.dtyper, dtypec=self.material.dtypec
        )

        iter_times = []
        dT_convergence = []
        n_convergence = []
        gmres_residual = []
        n = None
        n_old = None

        def residual_vec(x):
            # x is np.array([Re(dT), Im(dT)]) on CPU (neccessary for scipy)
            nonlocal n, n_old, dT_convergence, gmres_residual, iter_times
            dT = cp.asarray(x[0] + 1j * x[1], dtype=self.material.dtypec)
            iter_start = time.time()
            n_old = n

            n, resid = self._dT_to_N(
                dT=dT,
                omg_ft=omg_ft,
                omg_idx=omg_idx,
                sol_guess=None,
            )
            gmres_residual.append(resid)

            dT_new = N_to_dT(n, self.material)
            dT_convergence.append(dT_new)

            if n_old is not None:
                n_step_norm = cp.linalg.norm(n - n_old) / cp.linalg.norm(n_old)
                n_convergence.append(n_step_norm)

            iter_times.append(time.time() - iter_start)

            rR = float(cp.real(dT - dT_new).item())
            rI = float(cp.imag(dT - dT_new).item())
            return np.array([self._wR * rR, self._wI * rI], dtype=self.material.dtyper)

        x0 = np.array([cp.real(dT_init).item(), cp.imag(dT_init).item()], dtype=self.material.dtyper)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            sol = root(
                residual_vec,
                x0=x0,
                method="hybr",
                tol=self.conv_thr,
                options={"xtol": self.conv_thr, "maxfev": self.max_iter + 2},
            )

        fun_norm = np.linalg.norm(sol.fun) if hasattr(sol, "fun") else np.inf
        converged = bool(sol.success) or (fun_norm <= self.conv_thr)

        if not converged:
            print(RuntimeWarning(f"Root finding failed to converge: {sol}"))

        dT = cp.asarray(sol.x[0] + 1j * sol.x[1], dtype=self.material.dtypec)
        self.history.append((omg_ft, dT))
        niter = len(iter_times)
        return omg_ft, dT, dT_init, n, niter, iter_times, dT_convergence, n_convergence, gmres_residual

    @property
    def residual_weights(self):
        """Return the weights for the real and imaginary parts of the residual."""
        return self._wR, self._wI
    
    @residual_weights.setter
    def residual_weights(self, weights: tuple[float, float]):
        """Set the weights for the real and imaginary parts of the residual."""
        self._wR = weights[0]
        self._wI = weights[1]

    def _solution_lists_to_arrays(self):
        """Convert the lists of iteration times, dT convergence, n convergence, and GMRES residuals into cupy arrays.

        We can only do that after all omg_ft have been solved, since the lengths of the lists can vary. All entries in
        the lists are padded with NaNs.
        """
        max_len = max(len(times) for times in self.iter_time_list)
        iter_times_array = cp.full((len(self.iter_time_list), max_len), np.nan, dtype=self.material.dtyper)
        for i, times in enumerate(self.iter_time_list):
            times = cp.asarray(times, dtype=self.material.dtyper)
            iter_times_array[i, : len(times)] = times
        self.iter_time = iter_times_array

        max_len = max(len(dTs) for dTs in self.dT_convergence_list)
        dT_convergence_array = cp.full((len(self.dT_convergence_list), max_len), np.nan, dtype=self.material.dtypec)
        for i, dTs in enumerate(self.dT_convergence_list):
            dTs = cp.asarray(dTs, dtype=self.material.dtypec)
            dT_convergence_array[i, : len(dTs)] = dTs
        self.dT_convergence = dT_convergence_array

        max_len = max(len(n_convs) for n_convs in self.n_convergence_list)
        n_convergence_array = cp.full((len(self.n_convergence_list), max_len), np.nan, dtype=self.material.dtyper)
        for i, n_convs in enumerate(self.n_convergence_list):
            n_convs = cp.asarray(n_convs, dtype=self.material.dtyper)
            n_convergence_array[i, : len(n_convs)] = n_convs
        self.n_convergence = n_convergence_array

        n_omega = len(self.omg_ft_array)
        max_outer = max(len(outer) for outer in self.gmres_residual_list)
        max_q = max(len(q_list) for outer in self.gmres_residual_list for q_list in outer)
        max_gmres = max(len(res_list) for outer in self.gmres_residual_list for q_list in outer for res_list in q_list)
        gmres_residual_array = cp.full((n_omega, max_outer, max_q, max_gmres), np.nan, dtype=self.material.dtyper)
        for i, outer in enumerate(self.gmres_residual_list):
            for j, q_list in enumerate(outer):
                for k, res_list in enumerate(q_list):
                    res_list = cp.asarray(res_list, dtype=self.material.dtyper)
                    gmres_residual_array[i, j, k, : len(res_list)] = res_list
        self.gmres_residual = gmres_residual_array

    @property
    def flux(self, recompute=False):
        """Calculate the thermal flux.

        Calculate J from the Wigner distribution function n and the velocity operator.

        Parameters
        ----------
        recompute : bool, optional
            If True, recomputes the flux even if it has already been calculated. Default is False.

        Returns
        -------
        flux : cupy.ndarray
            The thermal flux [W/m^2] for each omg_ft, shape (n_omg_ft, nq, nat3, nat3).

        """
        if self._flux is not None and not recompute:
            return self._flux
        if not self.iter_time_list:
            raise ValueError("Solver has not been run yet. Please run the solver first.")

        self._flux = cp.zeros_like(self.n, dtype=self.material.dtypec)
        for i, _ in enumerate(self.omg_ft_array):
            self._flux[i] = flux_from_n(self.n[i], self.material)
        return self._flux

    @flux.setter
    def flux(self, value):
        self._flux = value

    @property
    def kappa(self):
        """Calculate the total thermal conductivity kappa.

        Compute kappa from the Wigner distribution function n and the temperature change dT.

        Returns
        -------
        kappa : cupy.ndarray
            The thermal conductivity [W/m/K] for each omg_ft, shape (n_omg_ft,).

        """
        if self._kappa is not None:
            return self._kappa
        if not self.iter_time_list:
            raise ValueError("Solver has not been run yet. Please run the solver first.")

        num = cp.einsum("wqij->w", self.flux)
        den = 1j * self.k_ft * self.dT
        self._kappa = _safe_divide(num, den)
        return self._kappa

    @property
    def kappa_p(self):
        """Calculate the thermal conductivity contribution from the populations.

        Calculate kappa_p from the Wigner distribution function n and the temperature change dT.

        Returns
        -------
        kappa_p : cupy.ndarray
            The thermal conductivity [W/m/K] contribution from the populations for each omg_ft, shape (n_omg_ft,).

        """
        if self._kappa_p is not None:
            return self._kappa_p
        if not self.iter_time_list:
            raise ValueError("Solver has not been run yet. Please run the solver first.")

        flux_diag = cp.einsum("wqii->w", self.flux)
        den = 1j * self.k_ft * self.dT
        self._kappa_p = _safe_divide(flux_diag, den)
        return self._kappa_p

    @property
    def kappa_c(self):
        """Calculate the thermal conductivity contribution from the coherences.

        Calculate kappa_c from the Wigner distribution function n and the temperature change dT.

        Returns
        -------
        kappa_c : cupy.ndarray
            The thermal conductivity [W/m/K] contribution from the coherences for each omg_ft, shape (n_omg_ft,).

        """
        if self._kappa_c is not None:
            return self._kappa_c
        if not self.iter_time_list:
            raise ValueError("Solver has not been run yet. Please run the solver first.")

        flux_total = cp.einsum("wqij->w", self.flux)
        flux_diag = cp.einsum("wqii->w", self.flux)
        flux_offdiag = flux_total - flux_diag
        den = 1j * self.k_ft * self.dT
        self._kappa_c = _safe_divide(flux_offdiag, den)
        return self._kappa_c


def _safe_divide(num: cp.ndarray, den: cp.ndarray, eps: float = 1e-300) -> cp.ndarray:
    """Elementwise num/den with 0 weher |den|<=eps (avoinds 0/0 -> NaN).

    Parameters
    ----------
    num : cupy.ndarray
        The numerator array.
    den : cupy.ndarray
        The denominator array.
    eps : float, optional
        A small value to avoid division by zero. Default is 1e-300.

    Returns
    -------
    cupy.ndarray
        The elementwise division result.

    """
    out = cp.zeros_like(num, dtype=num.dtype)
    mask = cp.abs(den) > eps
    if mask.any():
        out[mask] = num[mask] / den[mask]
    return out


def _degeneracy_mask(phonon_freq: cp.ndarray, tol: float = 1 - 10):
    """Build mask for degenerate vs non-degenerate pairs i,j per q-point.

    Parameters
    ----------
    phonon_freq : cupy.ndarray
        The phonon frequencies [Hz], shape (nq, nat3).
    tol : float, optional
        The tolerance for considering pairs as degenerate. The tolerance is relative to the maximum phonon frequency at
        each q-point. Default is 1e-10.

    Returns
    -------
    deg_mask : cupy.ndarray
        The mask for degenerate pairs [nq, nat3, nat3].
    offblock_mask : cupy.ndarray
        The mask for non-degenerate off-diagonal pairs [nq, nat3, nat3].

    """
    phonon_freq_max = cp.max(cp.abs(phonon_freq), axis=1)
    # |w_i - w_j| <= tol * w_max(q)
    diff = cp.abs(phonon_freq[:, :, None] - phonon_freq[:, None, :])
    thresh = (tol * phonon_freq_max)[:, None, None]
    deg_mask = diff <= thresh

    eye = cp.eye(phonon_freq.shape[1], dtype=cp.bool_)[None, :, :]
    offblock_mask = (~deg_mask) & (~eye)  # non-degenerate off-diagonals

    return deg_mask, offblock_mask


def N_to_dT(n: cp.ndarray, material: Material) -> complex:
    """Calculate the temperature change dT for a wigner distribution n.

    Parameters
    ----------
    n : cupy.ndarray
        The wigner distribution function n, shape (nq, nat3, nat3).
    material : Material
        The material object containing phonon frequencies, heat capacity, and volume.

    Returns
    -------
    complex
        The temperature change dT.

    """
    return _N_to_dT(n, material.phonon_freq, material.heat_capacity, material.volume)


def _N_to_dT(n: cp.ndarray, phonon_freq: cp.ndarray, heat_capacity: cp.ndarray, volume: float) -> complex:
    """Calculate the temperature change dT for a wigner distribution n.

    Parameters
    ----------
    n : cupy.ndarray
        The wigner distribution function n, shape (nq, nat3, nat3).
    phonon_freq : cupy.ndarray
        The phonon frequencies [Hz], shape (nq, nat3).
    heat_capacity : cupy.ndarray
        The heat capacity [J/m^3/K] of the phonon modes, shape (nq, nat3).
    volume : float
        The volume of the cell [m^3].

    Returns
    -------
    complex
        The temperature change dT.

    """
    nq = n.shape[0]
    dT = 0
    for i in range(nq):
        # delta_n = cp.diag(n[i]) - nbar(phonon_freq[i], temperature)
        # dT += cp.sum(phonon_freq[i] * delta_n)
        dT += cp.sum(phonon_freq[i] * cp.diag(n[i]))
    dT *= hbar / volume / nq
    dT /= cp.sum(heat_capacity)
    return dT


def dT_to_N_iterative(
    dT: complex,
    omg_ft: float,
    k_ft: float,
    material: Material,
    source: cp.ndarray,
    sol_guess=None,
    solver="gmres",
    conv_thr=1e-12,
    progress=False,
) -> tuple[cp.ndarray, list]:
    """Calculate the wigner distribution function n from the temperature change dT.

    This function solves the linear equation system that arises from the WTE for the wigner distribution function n
    for a given temperature change dT.

    Parameters
    ----------
    dT : complex
        The temperature change dT [K].
    omg_ft : float
        The temporal Fourier variable [Hz].
    k_ft : float
        The thermal grating wavevector [1/m].
    material : Material
        The material object containing phonon frequencies, heat capacity, and volume.
    source : cupy.ndarray
        The source term of the WTE, shape (nq, nat3, nat3).
    sol_guess : cupy.ndarray, optional
        The initial guess for the solution, shape (nq, nat3, nat3).
    solver : {"gmres", "cgesv"}
        The solver to use for the linear system. gmres uses :func:`cupyx.scipy.sparse.linalg.gmres`, while cgesv uses
        :func:`cupy.linalg.solve`.
    conv_thr : float, optional
        The convergence threshold for the solver, default is 1e-12.
    progress : bool, optional
        If True, a `.` is printed after each iteration to indicate progress. Default is False.

    Returns
    -------
    n : cupy.ndarray
        The Wigner distribution function n, shape (nq, nat3, nat3).
    outer_residuals : list
        A list of residuals for each iteration of the solver.

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
        sol_guess=sol_guess,
        dtyper=material.dtyper,
        dtypec=material.dtypec,
        solver=solver,
        conv_thr=conv_thr,
        progress=progress,
    )


def _dT_to_N_iterative(
    dT: complex,
    omg_ft: float,
    k_ft: float,
    phonon_freq: cp.ndarray,
    linewidth: cp.ndarray,
    velocity_operator: cp.ndarray,
    heat_capacity: cp.ndarray,
    volume: float,
    source: cp.ndarray,
    sol_guess=None,
    dtyper=cp.float64,
    dtypec=cp.complex128,
    solver="gmres",
    conv_thr=1e-12,
    progress=False,
) -> tuple[cp.ndarray, list]:
    """Calculate the wigner distribution function n from the temperature change dT.

    This function solves the linear equation system that arises from the WTE for the wigner distribution function n
    for a given temperature change dT in an interative manner. This is much slower than the direct method, but does not
    require knowledge of the Green's function.

    Parameters
    ----------
    dT : complex
        The temperature change dT [K].
    omg_ft : float
        The temporal Fourier variable [Hz].
    k_ft : float
        The thermal grating wavevector [1/m].
    phonon_freq : cupy.ndarray
        The phonon frequencies [Hz], shape (nq, nat3).
    linewidth : cupy.ndarray
        The linewidths [Hz], shape (nq, nat3).
    velocity_operator : cupy.ndarray
        The velocity operator for the phonon modes, shape (nq, nat3, nat3).
    heat_capacity : cupy.ndarray
        The heat capacity [J/m^3/K] of the phonon modes, shape (nq, nat3).
    volume : float
        The volume of the cell [m^3].
    source : cupy.ndarray
        The source term of the WTE, shape (nq, nat3, nat3).
    sol_guess : cupy.ndarray, optional
        The initial guess for the solution, shape (nq, nat3, nat3).
    dtyper : cupy.dtype, optional
        The data type for the real parts of the matrices, default is cp.float64.
    dtypec : cupy.dtype, optional
        The data type for the complex matrices, default is cp.complex128.
    solver : {"gmres", "cgesv"}
        The solver to use for the linear system. gmres uses :func:`cupyx.scipy.sparse.linalg.gmres`, while cgesv uses
        :func:`cupy.linalg.solve`.
    conv_thr : float, optional
        The convergence threshold for the solver, default is 1e-12.
    progress : bool, optional
        If True, a `.` is printed after each iteration to indicate progress. Default is False.

    Returns
    -------
    n : cupy.ndarray
        The Wigner distribution function n, shape (nq, nat3, nat3).
    outer_residuals : list
        A list of residuals for each iteration of the solver.

    """
    with nvtx.annotate("init dT_to_N", color="blue"):
        nq = phonon_freq.shape[0]
        nat3 = phonon_freq.shape[1]
        n = cp.zeros((nq, nat3, nat3), dtype=dtypec)

        I_small = cp.eye(nat3, dtype=dtyper)
        I_big = cp.eye(nat3**2, dtype=dtypec)
        OMG = cp.zeros((nat3, nat3), dtype=dtyper)
        GAM = cp.zeros((nat3, nat3), dtype=dtyper)

        outer_residuals = []

    for ii in range(nq):
        with nvtx.annotate("init q", color="purple"):
            cp.fill_diagonal(OMG, phonon_freq[ii])
            cp.fill_diagonal(GAM, linewidth[ii])

            gv_op = velocity_operator[ii]

            # term1 = cp.kron(I, OMG) - cp.kron(OMG, I) - (omg_ft * I_big)
            term1 = (
                cp.einsum("ij,kl->ikjl", I_small, OMG).reshape(nat3**2, nat3**2)
                - cp.einsum("ij,kl->ikjl", OMG, I_small).reshape(nat3**2, nat3**2)
                - omg_ft * I_big
            )
            # term2 = (k_ft / 2) * (cp.kron(I, gv_op) + cp.kron(gv_op.T, I))
            term2 = (k_ft / 2) * (
                cp.einsum("ij,kl->ikjl", I_small, gv_op).reshape(nat3**2, nat3**2)
                + cp.einsum("ij,kl->ikjl", gv_op.T, I_small).reshape(nat3**2, nat3**2)
            )
            # term3 = 0.5 * (cp.kron(I, GAM) + cp.kron(GAM, I))
            term3 = 0.5 * (
                cp.einsum("ij,kl->ikjl", I_small, GAM).reshape(nat3**2, nat3**2)
                + cp.einsum("ij,kl->ikjl", GAM, I_small).reshape(nat3**2, nat3**2)
            )

            lhs = (1j * (term1 - term2)) + term3

            nbar_deltat = volume * nq / hbar / phonon_freq[ii] * heat_capacity[ii] * cp.array(dT, dtype=dtypec)
            rhs = cp.copy(source[ii])
            cp.fill_diagonal(rhs, cp.diag(source[ii]) + linewidth[ii] * nbar_deltat)
            rhs = rhs.flatten()

            residuals = []

            def gmres_callback(residual):
                residuals.append(residual)

        if solver == "gmres":
            with nvtx.annotate("gmres", color="green"):
                guess = sol_guess[ii].flatten() if sol_guess is not None else cp.zeros_like(rhs, dtype=dtypec)
                sol, info = gmres(
                    lhs, rhs, x0=guess, callback=gmres_callback, tol=conv_thr, M=cp.diag(1 / lhs.diagonal())
                )
            if info != 0:
                print(f"GMRES failed to converge: {info}")
        elif solver == "cgesv":
            sol = cp.linalg.solve(lhs, rhs)
        else:
            raise ValueError(f"Unknown inner solver: {solver}")
        outer_residuals.append(residuals)
        sol = sol.reshape(nat3, nat3)
        n[ii] = sol
    if progress:
        print(".", end="")
    return n, outer_residuals


def dT_to_N_matmul(
    dT: complex,
    material: Material,
    green: cp.ndarray,
    source: cp.ndarray,
) -> cp.ndarray:
    """Calculate the wigner distribution function n from the temperature change dT.

    With knowledge of the Green's function one can obtain the Wigner distribution function n direcly via matrix
    multiplication of the Green's function with the source term. No need to iteratively solve a linear system of
    equations.

    Parameters
    ----------
    dT : complex
        The temperature change dT [K].
    material: Material
        The material object containing all relevant properties.
    source : cupy.ndarray
        The source term of the WTE, shape (nq, nat3, nat3).
    green : cupy.ndarray
        The Green's function, shape (nq, nat3, nat3).

    Returns
    -------
    n : cupy.ndarray
        The Wigner distribution function n, shape (nq, nat3, nat3).

    Notes
    -----
    This function is a vectorized equivalent of the following more legible loop-based implementation:

    .. code-block:: python

        nq, nat3 = material.nq, material.nat3
        n = cp.zeros((nq, nat3, nat3), dtype=material.dtypec)
        for iq in range(nq):
            nbar_deltat = (
                material.volume * nq / hbar
                / material.phonon_freq[iq]
                * material.heat_capacity[iq]
                * cp.array(dT, dtype=material.dtypec)
            )
            rhs = cp.copy(source[iq])
            cp.fill_diagonal(
                rhs,
                cp.diag(source[iq]) + material.linewidth[iq] * nbar_deltat
            )
            n[iq] = (green[iq] @ rhs.flatten()).reshape(nat3, nat3)
        return n

    The vectorized form avoids explicit Python loops and uses batched
    matrix multiplication for improved performance on the GPU.

    """
    nq, nat3 = material.nq, material.nat3
    m = nat3**2

    green = cp.asarray(green)
    green = cp.ascontiguousarray(green).reshape(nq, m, m)
    rhs = cp.ascontiguousarray(source).reshape(nq, nat3, nat3).copy()
    prefac = material.volume * nq / hbar * material.heat_capacity / material.phonon_freq * dT

    i = cp.arange(nat3)
    rhs[:, i, i] += material.linewidth * prefac

    rhs_flat = rhs.reshape(nq, m)
    n_flat = (green @ rhs_flat[..., None]).squeeze(-1)

    return n_flat.reshape(nq, nat3, nat3)


def flux_from_n(n: cp.ndarray, material: Material) -> cp.ndarray:
    """Evaluate the thermal flux from the Wigner distribution function n.

    It corresponds to Equation (42) in Phys. Rev. X 12, 041011 [https://doi.org/10.1103/PhysRevX.12.041011].

    Parameters
    ----------
    n : cupy.ndarray
        The wigner distribution function n, shape (nq, nat3, nat3).
    material : Material
        The material object containing velocity operator, phonon frequencies, and volume.

    Returns
    -------
    flux : cupy.ndarray
        The thermal flux calculated from the Wigner distribution function n, shape (nq, nat3, nat3).

    """
    return _flux_from_n(n, material.velocity_operator, material.phonon_freq, material.volume)


def _flux_from_n(n, velocity_operator, phonon_freq, volume):
    """Evaluate the thermal flux from the Wigner distribution function n.

    It corresponds to Equation (42) in Phys. Rev. X 12, 041011 [https://doi.org/10.1103/PhysRevX.12.041011].

    Parameters
    ----------
    n : cupy.ndarray
        The wigner distribution function n, shape (nq, nat3, nat3).
    velocity_operator : cupy.ndarray
        The velocity operator for the phonon modes, shape (nq, nat3, nat3).
    phonon_freq : cupy.ndarray
        The phonon frequencies [Hz], shape (nq, nat3).
    volume : float
        The volume of the cell [m^3].

    Returns
    -------
    flux : cupy.ndarray
        The thermal flux calculated from the Wigner distribution function n, shape (nq, nat3, nat3).

    """
    freq_sum = phonon_freq[:, :, None] + phonon_freq[:, None, :]
    flux = freq_sum * velocity_operator * cp.transpose(n, axes=(0, 2, 1))
    flux *= hbar / 2 / volume / n.shape[0]
    return flux


def dT_bte_prb(omg_ft, k_ft, phonon_freq, linewidth, group_velocity, heat_capacity, weight, heat):
    """Eq (9) from Phys Rev. B 104, 245424 [https://doi.org/10.1103/PhysRevB.104.245424]."""
    nq = phonon_freq.shape[0]
    nat3 = phonon_freq.shape[1]

    phonon_freq = phonon_freq.flatten()
    linewidth = linewidth.flatten()
    heat_capacity = heat_capacity.flatten()
    group_velocity = group_velocity.reshape(nq * nat3)
    heat = heat.flatten()
    weight = cp.repeat(weight, nat3).flatten()
    p = heat

    okv = omg_ft + (group_velocity * k_ft)
    A_inv = 1 / (linewidth + 1j * okv)
    Dc = okv * heat_capacity
    Ap = cp.dot(A_inv, p)
    ADc = cp.dot(A_inv, Dc * weight)
    num = Ap
    den = 1j * ADc
    dT = cp.sum(num) / cp.sum(den)
    return dT


def dT_bte_from_wte(omg_ft, k_ft, phonon_freq, linewidth, group_velocity, heat_capacity, weight, heat, volume):
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
    weight = cp.repeat(weight, nat3).flatten()
    p = heat

    okv = omg_ft + (group_velocity * k_ft)
    giwkv = linewidth - 1j * okv

    num = cp.sum(hbar * phonon_freq / volume / nq * p / giwkv)
    den = cp.sum(heat_capacity) - cp.sum(linewidth * heat_capacity / giwkv)
    dT = num / den
    return dT


def kappa_eff_prb(k_ft, linewidth, group_velocity, heat_capacity):
    """Calculate the effective thermal conductivity that would follow from the BTE.

    This equation is taken from
    a draft of Phys. Rev. B 104, 245424 [https://doi.org/10.1103/PhysRevB.104.245424]. It was replaced by Eq. (12)
    in the published version. The version here gives better results.

    Parameters
    ----------
    k_ft : float
        The thermal grating wavevector in 1/m.
    linewidth, group_velocity, heat_capacity : array_like
        The linewidth [Hz], group velocity [m/s], and heat capacity [J/m^3/K] of the phonon modes. 2D arrays with
        shape (nq, nat3). Note that the group velocity is expected to be along the grating wavevector direction.

    Returns
    -------
    float
        The effective thermal conductivity [W/m/K].

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
