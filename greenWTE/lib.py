"""Library module for solving the Wigner Transport Equation (WTE) with a source term."""

import time
import warnings
from argparse import Namespace

import cupy as cp
import numpy as np
import nvtx
from cupyx.scipy.interpolate import PchipInterpolator
from cupyx.scipy.sparse.linalg import gmres
from scipy.constants import hbar
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import root_scalar
from scipy.signal import savgol_filter

# triggered when cupyx.scipy.sparse.linalg.gmres
# invokes np.linalg.lstsq under the hood
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=r".*`rcond` parameter will change to the default of machine precision.*"
)


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


@nvtx.annotate("N_to_dT", color="blue")
def N_to_dT(n: cp.ndarray, phonon_freq: cp.ndarray, heat_capacity: cp.ndarray, volume: float) -> complex:
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


def dT_to_N(
    dT: complex,
    omg_ft: float,
    k_ft: float,
    phonon_freq: cp.ndarray,
    linewidth: cp.ndarray,
    velocity_operator: cp.ndarray,
    heat_capacity: cp.ndarray,
    volume: float,
    heat: cp.ndarray,
    sol_guess=None,
    dtyper=cp.float64,
    dtypec=cp.complex128,
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
    heat : cupy.ndarray
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
            rhs = cp.copy(heat[ii])
            cp.fill_diagonal(rhs, cp.diag(heat[ii]) + linewidth[ii] * nbar_deltat)
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
        return cp.asarray(1e-6)

    omg_fts, dTs = zip(*sorted(history))
    omg_fts = cp.asarray(omg_fts, dtype=dtyper)
    dTs = cp.asarray(dTs, dtype=dtypec)

    if len(omg_fts) == 1:
        return dTs[0]

    interp_func = PchipInterpolator(omg_fts, dTs, extrapolate=True)
    dT_guess = interp_func(omg_ft)

    return dT_guess


@nvtx.annotate("AitkenAccelerator", color="orange")
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


class Solver:
    """Class to solve the Wigner Transport Equation (WTE).

    Solver for the WTE for the wigner distribution function n and the temperature change dT with a source term. This
    class provides methods to run the solver using different outer solvers (aitken, plain, root) and an inner solver
    (GMRES or CGESV).

    Parameters
    ----------
    omg_ft_array : cupy.ndarray
        An array of temporal Fourier variables [Hz] for which the WTE will be solved.
    k_ft : cupy.ndarray
        The single thermal grating wavevector [1/m] the system will be solved for.
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
    heat : cupy.ndarray
        The source term of the WTE, shape (nq, nat3, nat3).
    max_iter : int, optional
        The maximum number of iterations for the outer solver. Default is 100.
    conv_thr : float, optional
        The convergence threshold for the outer and inner solver. Default is 1e-12.
    dtyper : cupy.dtype, optional
        The data type for the real parts of the matrices. Default is cp.float64.
    dtypec : cupy.dtype, optional
        The data type for the complex matrices. Default is cp.complex128.
    outer_solver : {"aitken", "plain", "root"}
        The outer solver to use for the WTE. Options are "aitken", "plain", or "root". Default is "aitken".
    inner_solver : {"gmres", "cgesv"}
        The inner solver to use for the linear system. Options are "gmres" or "cgesv". Default is "gmres".
    command_line_args : argparse.Namespace, optional
        Command line arguments for used when calling the solver from the command line. Used to save additional
        information to the HDF5 file.

    Attributes
    ----------
    omg_ft_array : cupy.ndarray
        An array of temporal Fourier variables [Hz] for which the WTE will be solved.
    k_ft : cupy.ndarray
        The single thermal grating wavevector [1/m] the system will be solved for.
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
    heat : cupy.ndarray
        The source term of the WTE, shape (nq, nat3, nat3).
    max_iter : int
        The maximum number of iterations for the outer solver.
    conv_thr : float
        The convergence threshold for the outer and inner solver, default is 1e-12.
    dtyper : cupy.dtype
        The data type for the real parts of the matrices, default is cp.float64.
    dtypec : cupy.dtype
        The data type for the complex matrices, default is cp.complex128.
    outer_solver : {"aitken", "plain", "root"}
        The outer solver to use for the WTE. Options are "aitken", "plain", or "root".
    inner_solver : {"gmres", "cgesv"}
        The inner solver to use for the linear system. Options are "gmres" or "cgesv".
    history : list
        A list to store the history of temperature changes dT within the outer solver.
    dT : cupy.ndarray
        The calculated temperature changes dT for each omg_ft, shape (n_omg_ft,).
    dT_init : cupy.ndarray
        The initial temperature changes dT for each omg_ft, shape (n_omg_ft,).
    n : cupy.ndarray
        The wigner distribution function n for each omg_ft, shape (n_omg_ft, nq, nat3, nat3).
    niter : cupy.ndarray
        The number of iterations taken for the outer solver to converge for each omg_ft, shape (n_omg_ft,).
    iter_time_list : list
        A list of iteration times for each omg_ft.
    dT_convergence_list : list
        A list of convergence values for the temperature changes dT for each omg_ft.
    n_convergence_list : list
        A list of convergence values for the wigner distribution function n for each omg_ft.
    gmres_residual_list : list
        A list of GMRES residuals for each omg_ft.
    iter_time : cupy.ndarray
        All iteration times for the outer solver, shape (n_omg_ft, max_iters).
    dT_convergence : cupy.ndarray
        The convergence of the temperature changes dT for each omg_ft, shape (n_omg_ft, max_iters).
    n_convergence : cupy.ndarray
        The convergence of the wigner distribution function n for each omg_ft, shape (n_omg_ft, max_iters).
    gmres_residual : cupy.ndarray
        The GMRES residuals for each omg_ft, shape (n_omg_ft, max_iters).
    progress : bool
        If True, prints progress information during the solver run. Default is False for multiple omg_ft, True for
        single omg_ft.

    """

    __outer_solver_options = ["aitken", "plain", "root"]
    _kappa = None
    _kappa_p = None
    _kappa_c = None
    _flux = None

    def __init__(
        self,
        omg_ft_array: cp.ndarray,
        k_ft: cp.ndarray,
        phonon_freq: cp.ndarray,
        linewidth: cp.ndarray,
        velocity_operator: cp.ndarray,
        heat_capacity: cp.ndarray,
        volume: float,
        source: cp.ndarray,
        max_iter=100,
        conv_thr=1e-12,
        dtyper=cp.float64,
        dtypec=cp.complex128,
        outer_solver="aitken",
        inner_solver="gmres",
        command_line_args=Namespace(),
    ) -> None:
        """Initialize the Solver with the given parameters."""
        self.omg_ft_array = omg_ft_array
        self.k_ft = k_ft
        self.phonon_freq = phonon_freq
        self.linewidth = linewidth
        self.velocity_operator = velocity_operator
        self.heat_capacity = heat_capacity
        self.volume = volume
        self.source = source
        self.max_iter = max_iter
        self.conv_thr = conv_thr
        self.dtyper = dtyper
        self.dtypec = dtypec
        self.history = []
        if outer_solver not in self.__outer_solver_options:
            raise ValueError(f"Unknown outer solver: {outer_solver}")
        self.outer_solver = outer_solver
        self.inner_solver = inner_solver
        self.command_line_args = command_line_args

        # result arrays
        self.dT = cp.zeros_like(omg_ft_array, dtype=dtypec)
        self.dT_init = cp.zeros_like(omg_ft_array, dtype=dtypec)
        self.n = cp.zeros(
            (len(omg_ft_array), phonon_freq.shape[0], phonon_freq.shape[1], phonon_freq.shape[1]), dtype=dtypec
        )
        self.niter = cp.zeros_like(omg_ft_array, dtype=cp.int32)
        self.n_convergence = cp.zeros_like(omg_ft_array, dtype=dtyper)
        self.iter_time_list = []
        self.dT_convergence_list = []
        self.n_convergence_list = []
        self.gmres_residual_list = []
        self.iter_time = None
        self.dT_convergence = None
        self.n_convergence = None
        self.gmres_residual = None
        if self.omg_ft_array.shape[0] == 1:
            self.progress = True
        else:
            self.progress = False

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
            ret = run_func(omg_ft)
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
            theoretical_next_n, _ = dT_to_N(
                dT=ret[1],
                omg_ft=omg_ft,
                k_ft=self.k_ft,
                phonon_freq=self.phonon_freq,
                linewidth=self.linewidth,
                velocity_operator=self.velocity_operator,
                heat_capacity=self.heat_capacity,
                volume=self.volume,
                heat=self.source,
                sol_guess=None,
                dtyper=self.dtyper,
                dtypec=self.dtypec,
                solver=self.inner_solver,
                conv_thr=self.conv_thr,
                progress=self.progress,
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

    def _run_solver_aitken(self, omg_ft):
        """Run the WTE solver using Aitken's delta-squared process for acceleration.

        Parameters
        ----------
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
        dT_init = estimate_initial_dT(omg_ft=omg_ft, history=self.history, dtyper=self.dtyper, dtypec=self.dtypec)
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
            dT_prev = dT
            n_prev = n

            n, resid = dT_to_N(
                dT=dT,
                omg_ft=omg_ft,
                k_ft=self.k_ft,
                phonon_freq=self.phonon_freq,
                linewidth=self.linewidth,
                velocity_operator=self.velocity_operator,
                heat_capacity=self.heat_capacity,
                volume=self.volume,
                heat=self.source,
                sol_guess=n_prev,
                dtyper=self.dtyper,
                dtypec=self.dtypec,
                solver=self.inner_solver,
                conv_thr=self.conv_thr,
            )
            gmres_residual.append(resid)
            dT_new = N_to_dT(n, self.phonon_freq, self.heat_capacity, self.volume)
            dT = accelerator.update(dT_new)
            dT_convergence.append(dT_new)

            rel_diff = np.abs(np.imag(dT - dT_prev)) / np.abs(np.imag(dT_prev))
            iter_times.append(time.time() - iter_start)

            if n_prev is not None:
                n_step_norm = cp.linalg.norm(n - n_prev) / cp.linalg.norm(n_prev)
                n_convergence.append(n_step_norm)

            if np.abs(rel_diff) < self.conv_thr:
                self.history.append((omg_ft, dT_new))
                break

        return omg_ft, dT, dT_init, n, iterations, iter_times, dT_convergence, n_convergence, gmres_residual

    def _run_solver_plain(self, omg_ft):
        """Run the WTE solver without acceleration, iterating until convergence.

        Parameters
        ----------
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
        dT_init = estimate_initial_dT(omg_ft=omg_ft, history=self.history, dtyper=self.dtyper, dtypec=self.dtypec)
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
            dT_prev = dT
            n_prev = n

            n, resid = dT_to_N(
                dT=dT,
                omg_ft=omg_ft,
                k_ft=self.k_ft,
                phonon_freq=self.phonon_freq,
                linewidth=self.linewidth,
                velocity_operator=self.velocity_operator,
                heat_capacity=self.heat_capacity,
                volume=self.volume,
                heat=self.source,
                sol_guess=n_prev,
                dtyper=self.dtyper,
                dtypec=self.dtypec,
                solver=self.inner_solver,
                conv_thr=self.conv_thr,
            )
            gmres_residual.append(resid)
            dT = N_to_dT(n, self.phonon_freq, self.heat_capacity, self.volume)
            dT_convergence.append(dT)

            rel_diff = np.abs(np.imag(dT - dT_prev)) / np.abs(np.imag(dT_prev))
            iter_times.append(time.time() - iter_start)

            if n_prev is not None:
                n_step_norm = cp.linalg.norm(n - n_prev) / cp.linalg.norm(n_prev)
                n_convergence.append(n_step_norm)

            if np.abs(rel_diff) < self.conv_thr:
                self.history.append((omg_ft, dT))
                break

        return omg_ft, dT, dT_init, n, iterations, iter_times, dT_convergence, n_convergence, gmres_residual

    def _run_solver_root(self, omg_ft):
        """Run the WTE solver using root finding to solve for the temperature change dT.

        Usually converges MUCH faster than Aitken or plain methods.

        Parameters
        ----------
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
        dT_init = estimate_initial_dT(omg_ft=omg_ft, history=self.history, dtyper=self.dtyper, dtypec=self.dtypec)

        iter_times = []
        dT_convergence = []
        n_convergence = []
        gmres_residual = []
        n = None
        n_old = None

        def residual_func(dT):
            nonlocal n, n_old, dT_convergence, gmres_residual, iter_times
            dT = cp.asarray(dT, dtype=self.dtypec)
            iter_start = time.time()
            n_old = n

            n, resid = dT_to_N(
                dT=dT,
                omg_ft=omg_ft,
                k_ft=self.k_ft,
                phonon_freq=self.phonon_freq,
                linewidth=self.linewidth,
                velocity_operator=self.velocity_operator,
                heat_capacity=self.heat_capacity,
                volume=self.volume,
                heat=self.source,
                sol_guess=None,
                dtyper=self.dtyper,
                dtypec=self.dtypec,
                solver=self.inner_solver,
                conv_thr=self.conv_thr,
                progress=self.progress,
            )
            gmres_residual.append(resid)

            dT_new = N_to_dT(n, self.phonon_freq, self.heat_capacity, self.volume)
            dT_convergence.append(dT_new)
            if n_old is not None:
                n_step_norm = cp.linalg.norm(n - n_old) / cp.linalg.norm(n_old)
                n_convergence.append(n_step_norm)

            iter_times.append(time.time() - iter_start)
            ret = cp.imag(dT - dT_new)
            return ret

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            sol = root_scalar(
                residual_func,
                x0=dT_init.get(),
                method="secant",
                xtol=self.conv_thr,
                maxiter=self.max_iter,
                rtol=self.conv_thr,
            )

        if not sol.converged:
            print(RuntimeWarning(f"Root finding failed to converge: {sol}"))

        dT = cp.asarray(sol.root, dtype=self.dtypec)
        self.history.append((omg_ft, dT))
        return omg_ft, dT, dT_init, n, sol.iterations, iter_times, dT_convergence, n_convergence, gmres_residual

    def _solution_lists_to_arrays(self):
        """Convert the lists of iteration times, dT convergence, n convergence, and GMRES residuals into cupy arrays.

        We can only do that after all omg_ft have been solved, since the lengths of the lists can vary. All entries in
        the lists are padded with NaNs.
        """
        max_len = max(len(times) for times in self.iter_time_list)
        iter_times_array = cp.full((len(self.iter_time_list), max_len), np.nan, dtype=self.dtyper)
        for i, times in enumerate(self.iter_time_list):
            times = cp.asarray(times, dtype=self.dtyper)
            iter_times_array[i, : len(times)] = times
        self.iter_time = iter_times_array

        max_len = max(len(dTs) for dTs in self.dT_convergence_list)
        dT_convergence_array = cp.full((len(self.dT_convergence_list), max_len), np.nan, dtype=self.dtypec)
        for i, dTs in enumerate(self.dT_convergence_list):
            dTs = cp.asarray(dTs, dtype=self.dtypec)
            dT_convergence_array[i, : len(dTs)] = dTs
        self.dT_convergence = dT_convergence_array

        max_len = max(len(n_convs) for n_convs in self.n_convergence_list)
        n_convergence_array = cp.full((len(self.n_convergence_list), max_len), np.nan, dtype=self.dtyper)
        for i, n_convs in enumerate(self.n_convergence_list):
            n_convs = cp.asarray(n_convs, dtype=self.dtyper)
            n_convergence_array[i, : len(n_convs)] = n_convs
        self.n_convergence = n_convergence_array

        n_omega = len(self.omg_ft_array)
        max_outer = max(len(outer) for outer in self.gmres_residual_list)
        max_q = max(len(q_list) for outer in self.gmres_residual_list for q_list in outer)
        max_gmres = max(len(res_list) for outer in self.gmres_residual_list for q_list in outer for res_list in q_list)
        gmres_residual_array = cp.full((n_omega, max_outer, max_q, max_gmres), np.nan, dtype=self.dtyper)
        for i, outer in enumerate(self.gmres_residual_list):
            for j, q_list in enumerate(outer):
                for k, res_list in enumerate(q_list):
                    res_list = cp.asarray(res_list, dtype=self.dtyper)
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

        self._flux = cp.zeros_like(self.n, dtype=self.dtypec)
        for i, _ in enumerate(self.omg_ft_array):
            self._flux[i] = flux_from_n(
                n=self.n[i],
                velocity_operator=self.velocity_operator,
                phonon_freq=self.phonon_freq,
                volume=self.volume,
            )
        return self._flux

    @flux.setter
    def flux(self, value):
        self._flux = value

    @property
    def kappa(self, recompute=False):
        """Calculate the total thermal conductivity kappa.

        Compute kappa from the Wigner distribution function n and the temperature change dT.

        Returns
        -------
        kappa : cupy.ndarray
            The thermal conductivity [W/m/K] for each omg_ft, shape (n_omg_ft,).

        """
        if self._kappa is not None and not recompute:
            return self._kappa
        if not self.iter_time_list:
            raise ValueError("Solver has not been run yet. Please run the solver first.")

        self._kappa = cp.sum(self.flux) / (1j * self.k_ft * self.dT)
        return self._kappa

    @kappa.setter
    def kappa(self, value):
        self._kappa = value

    @property
    def kappa_p(self, recompute=False):
        """Calculate the thermal conductivity contribution from the populations.

        Calculate kappa_p from the Wigner distribution function n and the temperature change dT.

        Parameters
        ----------
        recompute : bool, optional
            If True, recomputes the thermal conductivity contribution from the populations even if it has already been
            calculated. Default is False.

        Returns
        -------
        kappa_p : cupy.ndarray
            The thermal conductivity [W/m/K] contribution from the populations for each omg_ft, shape (n_omg_ft,).

        """
        if self._kappa_p is not None and not recompute:
            return self._kappa_p
        if not self.iter_time_list:
            raise ValueError("Solver has not been run yet. Please run the solver first.")

        flux_diag = cp.sum(cp.einsum("wqii->wqi", self.flux), axis=(1, 2))
        self._kappa_p = flux_diag / (1j * self.k_ft * self.dT)
        return self._kappa_p

    @kappa_p.setter
    def kappa_p(self, value):
        self._kappa_p = value

    @property
    def kappa_c(self, recompute=False):
        """Calculate the thermal conductivity contribution from the coherences.

        Calculate kappa_c from the Wigner distribution function n and the temperature change dT.

        Parameters
        ----------
        recompute : bool, optional
            If True, recomputes the thermal conductivity contribution from the coherences even if it has already been
            calculated. Default is False.

        Returns
        -------
        kappa_c : cupy.ndarray
            The thermal conductivity [W/m/K] contribution from the coherences for each omg_ft, shape (n_omg_ft,).

        """
        if self._kappa_c is not None and not recompute:
            return self._kappa_c
        if not self.iter_time_list:
            raise ValueError("Solver has not been run yet. Please run the solver first.")

        flux_offdiag = cp.sum(self.flux, axis=(1, 2, 3)) - cp.sum(cp.einsum("wqii->wqi", self.flux), axis=(1, 2))
        self._kappa_c = flux_offdiag / (1j * self.k_ft * self.dT)
        return self._kappa_c


def flux_from_n(n, velocity_operator, phonon_freq, volume):
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


##################
# Plotting utils #
##################
def interpolate_onto_path(qpath, qmesh, data, n=1000, smooth=False, smooth_kwargs=None):
    """Interpolate data along a specified path in a multidimensional space.

    Given a path defined by a sequence of points (`qpath`), this function interpolates
    the provided data (`data`) defined on a mesh (`qmesh`) along the path, optionally
    applying smoothing to the interpolated data.

    Parameters
    ----------
    qpath : array_like, shape (N, D)
        Sequence of points defining the path along which to interpolate, where N is the
        number of path points and D is the dimensionality.
    qmesh : array_like, shape (M, D)
        Coordinates of the mesh points where the data is originally defined, where M is
        the number of mesh points.
    data : array_like, shape (M, K)
        Data values defined on the mesh points, where K is the number of data components
        (e.g., different physical quantities).
    n : int, optional
        Total number of points to interpolate along the entire path. Default is 1000.
    smooth : bool, optional
        If True, applies a Savitzky-Golay filter to smooth the interpolated data.
        Default is False.
    smooth_kwargs : dict, optional
        Keyword arguments to pass to `scipy.signal.savgol_filter` for smoothing.
        Default is {"window_length": 21, "polyorder": 1}.

    Returns
    -------
    qpath_interp : ndarray, shape (n, D)
        Interpolated points along the path.
    interp_data : ndarray, shape (n, K)
        Interpolated (and optionally smoothed) data values at each point along the path.
    qpath_idxs : ndarray, shape (N + 1,)
        Indices indicating the start of each segment in the interpolated path.

    Notes
    -----
    - Uses `scipy.interpolate.LinearNDInterpolator` for interpolation.
    - If smoothing is enabled, uses `scipy.signal.savgol_filter` on each data component.
    - The number of points per segment is proportional to the segment length.

    """
    if smooth_kwargs is None:
        smooth_kwargs = {"window_length": 21, "polyorder": 1}
    n_segs = np.zeros(len(qpath) - 1)
    for i in range(len(qpath) - 1):
        n_segs[i] = int(np.linalg.norm(qpath[i + 1] - qpath[i]) * n)
    n_segs /= np.sum(n_segs) / n
    n_segs = np.round(n_segs).astype(int)
    n_segs[n_segs < 1] = 1
    qpath_idxs = np.insert(np.cumsum(n_segs), 0, 0)

    qpath_interp = []
    for i in range(len(qpath) - 1):
        seg = np.linspace(qpath[i], qpath[i + 1], n_segs[i] + 1)
        if i < len(qpath) - 2:
            seg = seg[:-1]
        qpath_interp.append(seg)
    qpath_interp = np.vstack(qpath_interp)

    interp_data = np.zeros((len(qpath_interp), data.shape[1]), dtype=data.dtype)
    for i in range(data.shape[1]):
        interp_func = LinearNDInterpolator(qmesh, data[:, i], fill_value=np.NaN)
        interp_data[:, i] = interp_func(qpath_interp)

    if smooth:
        for i in range(interp_data.shape[1]):
            interp_data[:, i] = savgol_filter(interp_data[:, i], **smooth_kwargs)

    return qpath_interp, interp_data, qpath_idxs
