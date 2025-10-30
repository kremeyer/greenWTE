"""Green-operator utilities for WTE in the relaxation-time approximation (RTA)."""

from abc import ABC, abstractmethod
from argparse import Namespace

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from . import nvtx_utils, xp
from .base import Material, N_to_dT, SolverBase, dT_to_N_matmul
from .io import GreenContainer


class RTAWignerOperator:
    r"""Wigner operator in the relaxation-time approximation (RTA).

    Builds the block-diagonal Wigner operator :math:`\mathcal{L}(q; \omega, k)` for a single temporal frequency
    :math:`\omega` and spatial wavevector magnitude :math:`k`.

    Parameters
    ----------
    omg_ft : float
        Temporal Fourier variable in rad/s.
    k_ft : float
        Magnitude of the spatial Fourier variable in rad/m.
    material : :py:class:`~greenWTE.base.Material`
        Material object containing the necessary material properties.

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

    """

    def __init__(
        self,
        omg_ft: xp.ndarray,
        k_ft: xp.ndarray,
        material: Material,
    ) -> None:
        """Initialize the factory with the given physical parameters."""
        self.omg_ft = omg_ft
        self.k_ft = k_ft
        self.material = material
        self.nq = material.nq
        self.nat3 = material.nat3
        self._op = None
        self.dtyper = material.dtyper
        self.dtypec = material.dtypec

    def __getitem__(self, iq: int) -> xp.ndarray:
        """Return a single-q-point operator block.

        Parameters
        ----------
        iq : int
            The index of the q-point to retrieve.

        Returns
        -------
        cupy.ndarray
            The Wigner operator block for the specified q-point, with shape (nat3^2, nat3^2).

        """
        if self._op is None:
            raise RuntimeError("Wigner operator not computed yet.")
        return self._op[iq]

    def __len__(self) -> int:
        """Return the number of q-points in the Wigner operator."""
        return self.nq

    def __iter__(self) -> Self:
        """Allow iteration over the Wigner operators q-points."""
        if self._op is None:
            raise RuntimeError("Wigner operator not computed yet.")
        return iter(self._op)

    def compute(self, recompute: bool = False) -> None:
        """Assemble the Wigner operator blocks on the GPU.

        Parameters
        ----------
        recompute : bool, optional
            If ``False`` and the operator is already available, do nothing. If ``True``, rebuild all blocks.

        """
        if self._op is not None and not recompute:
            return

        with nvtx_utils.annotate("build WignerOperator", color="orange"):
            self._op = xp.zeros((self.nq, self.nat3**2, self.nat3**2), dtype=self.dtypec)
            I_small = xp.eye(self.nat3, dtype=self.dtyper)
            I_big = xp.eye(self.nat3**2, dtype=self.dtypec)
            OMG = xp.zeros((self.nat3, self.nat3), dtype=self.dtyper)
            GAM = xp.zeros((self.nat3, self.nat3), dtype=self.dtyper)

            for ii in range(self.nq):
                xp.fill_diagonal(OMG, self.material.phonon_freq[ii])
                xp.fill_diagonal(GAM, self.material.linewidth[ii])

                gv_op = self.material.velocity_operator[ii]

                # term1 = xp.kron(I_small, OMG) - xp.kron(OMG, I_small) - (omg_ft * I_big)
                term1 = (
                    xp.einsum("ij,kl->ikjl", I_small, OMG).reshape(self.nat3**2, self.nat3**2)
                    - xp.einsum("ij,kl->ikjl", OMG, I_small).reshape(self.nat3**2, self.nat3**2)
                    - self.omg_ft * I_big
                )
                # term2 = (k_ft / 2) * (xp.kron(I_small, gv_op) + xp.kron(gv_op.T, I_small))
                term2 = (self.k_ft / 2) * (
                    xp.einsum("ij,kl->ikjl", I_small, gv_op).reshape(self.nat3**2, self.nat3**2)
                    + xp.einsum("ij,kl->ikjl", gv_op.T, I_small).reshape(self.nat3**2, self.nat3**2)
                )
                # term3 = 0.5 * (xp.kron(I_small, GAM) + xp.kron(GAM, I_small))
                term3 = 0.5 * (
                    xp.einsum("ij,kl->ikjl", I_small, GAM).reshape(self.nat3**2, self.nat3**2)
                    + xp.einsum("ij,kl->ikjl", GAM, I_small).reshape(self.nat3**2, self.nat3**2)
                )

                self._op[ii] = (1j * (term1 - term2)) + term3


class GreenOperatorBase(ABC):
    r"""Abstract base class for Green operators :math:`\mathcal{G} = \mathcal{L}^{-1}`.

    Provides a common interface for Green's operators, whether computed or loaded from disk.

    See Also
    --------
    :py:class:`~RTAGreenOperator` : Green's operator computed from the RTA Wigner operator.
    :py:class:`~DiskGreenOperator` : Disk-based Green's operator that loads.

    """

    __array_priority__ = 1000

    def _require_ready(self) -> None:
        """Ensure the Green's operator is computed before accessing it."""
        if self._green is None:
            raise RuntimeError("Green's operator not computed yet.")

    def __getitem__(self, iq: int) -> xp.ndarray:
        """Allow indexing to access the Green's function for a specific q-point."""
        self._require_ready()
        return self._green[iq]

    def __matmul__(self, other: xp.ndarray) -> xp.ndarray:
        """Allow matrix multiplication with the Green's function."""
        self._require_ready()
        return self._green @ other

    def __len__(self) -> int:
        """Return the number of q-points in the Green's function."""
        self._require_ready()
        return len(self._green)

    def __iter__(self) -> Self:
        """Allow iteration over the Green's function q-points."""
        self._require_ready()
        return iter(self._green)

    @property
    def __cuda_array_interface__(self) -> dict:  # pragma: no cover
        """Return the CUDA array interface for the Green's function."""
        self._require_ready()
        return self._green.__cuda_array_interface__

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the shape of the Green's function."""
        self._require_ready()
        return self._green.shape

    @property
    def dtype(self) -> xp.dtype:
        """Return the dtype of the Green's function."""
        self._require_ready()
        return self._green.dtype

    def squeeze(self) -> xp.ndarray:
        """Return the Green's function with singleton dimensions removed."""
        self._require_ready()
        return self._green.squeeze()

    def __repr__(self) -> str:
        """Return a string representation of the Green's operator."""
        return f"<RTAGreenOperator: {len(self)} q-points at w={self.omg_ft:.2e} with dtype={self.dtypec}>"

    @abstractmethod
    def compute(self, recompute: bool = False, **kwargs) -> None:
        """Compute or load the Green's function from disk."""

    def free(self) -> None:
        """Free the memory used by the Green's operator."""
        if self._green is not None:
            self._green = None
            if hasattr(self, "wigner_operator"):
                self.wigner_operator._op = None


class RTAGreenOperator(GreenOperatorBase):
    r"""Green operator in RTA computed by inverting a :py:class:`~greenWTE.green.RTAWignerOperator`.

    Parameters
    ----------
    wigner_operator : :py:class:`~greenWTE.green.RTAWignerOperator`
        Wigner operator to be inverted to obtain the Green's operator.

    Attributes
    ----------
    omg_ft : float
        Temporal Fourier variable in rad/s.
    k_ft : float
        Magnitude of the spatial Fourier variable in rad/m.
    material : :py:class:`~greenWTE.base.Material`
        Material associated with the Green's operator.
    nq : int
        Number of q-points.
    nat3 : int
        Number of phonon modes (3 times the number of atoms in the unit cell).
    dtyper : cupy.dtype
        Data type for the real-valued arrays.
    dtypec : cupy.dtype
        Data type for the complex-valued arrays.

    """

    def __init__(self, wigner_operator: RTAWignerOperator) -> None:
        """Initialize the Green's operator with the Wigner operator."""
        self.wigner_operator = wigner_operator
        self.omg_ft = wigner_operator.omg_ft
        self.k_ft = wigner_operator.k_ft
        self._green = None
        self.material = wigner_operator.material
        self.nq = wigner_operator.nq
        self.nat3 = wigner_operator.nat3
        self.dtyper = wigner_operator.dtyper
        self.dtypec = wigner_operator.dtypec

    def compute(self, recompute: bool = False, clear_wigner: bool = True) -> None:
        """Invert the Wigner operator to obtain the Green operator on the GPU.

        Parameters
        ----------
        recompute : bool, optional
            If ``False`` and the operator is already available, do nothing. If ``True``, rebuild all blocks.
        clear_wigner : bool, optional
            If ``True``, free the memory used by the Wigner operator after computing the Green's function.

        """
        if self._green is not None and not recompute:
            return

        self.wigner_operator.compute()  # will not recompute if already done

        self._green = xp.zeros_like(self.wigner_operator._op, dtype=self.dtypec)

        with nvtx_utils.annotate("invert WignerOperator", color="red"):
            self._green = xp.linalg.inv(self.wigner_operator._op)

        if clear_wigner:
            self.wigner_operator._op = None


class DiskGreenOperator(GreenOperatorBase):
    r"""Disk-backed Green operator loaded from a :py:class:`~greenWTE.io.GreenContainer`.

    Loads a single slab ``(nq, m, m)`` for the requested :math:`\omega` and :math:`k` into GPU memory on demand,
    allowing the solver to operate without recomputing or storing all Green operators simultaneously.

    Parameters
    ----------
    container : :py:class:`~greenWTE.io.GreenContainer`
        Container object that manages the disk storage of Green's operators.
    omg_ft : float
        Temporal Fourier variable in rad/s.
    k_ft : float
        Spatial Fourier variable in rad/m.
    material : :py:class:`~greenWTE.base.Material`
        Material object containing the necessary material properties.
    atol : float, optional
        Absolute tolerance for matching stored :math:`\omega` and :math:`k` values.

    Attributes
    ----------
    material : :py:class:`~greenWTE.base.Material`
        Material associated with the Green's operator.
    nq : int
        Number of q-points.
    nat3 : int
        Number of phonon modes (3 times the number of atoms in the unit cell).
    dtyper : cupy.dtype
        Data type for the real-valued arrays.
    dtypec : cupy.dtype
        Data type for the complex-valued arrays.

    """

    def __init__(
        self,
        container: GreenContainer,
        omg_ft: float,
        k_ft: float,
        material: Material,
        atol: float = 0.0,
    ) -> None:
        """Initialize the disk-based Green's operator. No I/O is performed here."""
        self.omg_ft = omg_ft
        self.k_ft = k_ft
        self.material = material
        self.nq = material.nq
        self.nat3 = material.nat3
        self.dtyper = material.dtyper
        self.dtypec = material.dtypec

        self._gc = container
        self._atol = atol
        self._green = None

    def compute(self, recompute: bool = False) -> None:
        """Load the Green's operator from disk.

        Parameters
        ----------
        recompute : bool, optional
            If ``False`` and the data is already present, do nothing. If ``True``, reload from disk.

        """
        if self._green is not None and not recompute:
            return
        # pull the Green's operator from disk
        arr = self._gc.get_bz_block(self.omg_ft, self.k_ft, atol=self._atol)
        self._green = xp.ascontiguousarray(xp.asarray(arr, dtype=self.dtypec))


class GreenWTESolver(SolverBase):
    r"""Wigner Transport Equation solver using precomputed Green's operators.

    This solver implements the mapping from a temperature change ``dT`` to the Wigner distribution function ``n`` via
    direct matrix multiplication with precomputed Green's function operators. The approach bypasses the need for
    iterative inner solvers such as GMRES and can be significantly faster when the Green's functions are available.

    Parameters
    ----------
    omg_ft_array : cupy.ndarray
        1D array of temporal Fourier variables in rad/s for which the WTE will be solved.
    k_ft : float
        Magnitude of the spatial Fourier variable in rad/m.
    material : :py:class:`~greenWTE.base.Material`
        Material object containing the necessary material properties.
    greens : list of RTAGreenOperator
        List of precomputed Green's function operators, one for each ``omg_ft`` value in `omg_ft_array`. Each operator
        must implement matrix multiplication to map the source term to the Wigner distribution function.
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

    Raises
    ------
    ValueError
        If the number of supplied Green's operators does not match the length of `omg_ft_array`.

    Notes
    -----
    - This implementation calls :func:`dT_to_N_matmul` to apply the Green's operator for the given frequency index. No
      residuals are generated, and the second return value of :meth:`_dT_to_N` is always a list of empty lists
      (one per ``nq``).
    - Since there is no iterative inner solver, performance is largely determined by the cost of the Green's operator
      matrix multiplication.

    See Also
    --------
    :py:class:`~greenWTE.base.SolverBase` : Parent class that provides the outer-solver infrastructure.
    :py:class:`~greenWTE.iterative.IterativeWTESolver` : WTE solver using iterative methods.

    """

    inner_solver = "None"  # no need for inner solver; string because we want to write that to hdf5

    def __init__(
        self,
        omg_ft_array: xp.ndarray,
        k_ft: xp.ndarray,
        material: Material,
        greens: list[RTAGreenOperator],
        source: xp.ndarray,
        source_type: str = "energy",
        dT_init: complex = 1.0 + 1.0j,
        max_iter: int = 100,
        conv_thr_rel: float = 1e-12,
        conv_thr_abs: float = 0,
        outer_solver: str = "root",
        command_line_args: Namespace = Namespace(),
        print_progress: bool = False,
    ) -> None:
        """Initialize GreenWTESolver."""
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
        if not len(greens) == len(omg_ft_array):
            raise ValueError("Number of Green's operators must match the number of omg_ft values.")
        self.greens = greens

    def _dT_to_N(
        self,
        dT: complex,
        omg_ft: float,
        omg_idx: int,
        sol_guess: xp.ndarray | None = None,
    ) -> tuple[xp.ndarray, list]:
        n = dT_to_N_matmul(
            dT=dT,
            material=self.material,
            green=self.greens[omg_idx],
            source=self.source,
            source_type=self.source_type,
        )
        return n, [[] for _ in range(self.material.nq)]  # no residuals from matrix multiplication

    def run(self, free: bool = True) -> None:
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
            self.max_iter = 1  # just for print formatting
        else:
            raise ValueError(f"Unknown outer solver: {self.outer_solver}")

        for i, omg_ft in enumerate(self.omg_ft_array):
            g = self.greens[i]
            g.compute()  # ensure the Green's operator is computed or loaded from disk
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

            if free:
                g.free()  # free memory allocated to the Green's operator

        self._solution_lists_to_arrays()
