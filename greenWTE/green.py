"""Module for explicitly inverting the Linear Operator of the WTE to obtain the Green's function."""

from abc import ABC, abstractmethod
from argparse import Namespace

import cupy as cp

from .base import Material, SolverBase, dT_to_N_matmul
from .io import GreenContainer


class RTAWignerOperator:
    """Class to represent the Wigner operator in the relaxation time approximation."""

    def __init__(
        self,
        omg_ft: cp.ndarray,
        k_ft: cp.ndarray,
        material: Material,
    ):
        """Initialize the factory with the given physical parameters."""
        self.omg_ft = omg_ft
        self.k_ft = k_ft
        self.material = material
        self.nq = material.nq
        self.nat3 = material.nat3
        self._op = None
        self.dtyper = material.dtyper
        self.dtypec = material.dtypec

    def __getitem__(self, iq):
        """Allow indexing to access the Wigner operator for a specific q-point."""
        if self._op is None:
            raise RuntimeError("Wigner operator not computed yet.")
        return self._op[iq]

    def __len__(self):
        """Return the number of q-points in the Wigner operator."""
        if self._op is None:
            raise RuntimeError("Wigner operator not computed yet.")
        return len(self._op)

    def __iter__(self):
        """Allow iteration over the Wigner operators q-points."""
        if self._op is None:
            raise RuntimeError("Wigner operator not computed yet.")
        return iter(self._op)

    def compute(self, recompute=False):
        """Compute the Wigner operator."""
        if self._op is not None and not recompute:
            return

        self._op = cp.zeros((self.nq, self.nat3**2, self.nat3**2), dtype=self.dtypec)
        I_small = cp.eye(self.nat3, dtype=self.dtyper)
        I_big = cp.eye(self.nat3**2, dtype=self.dtypec)
        OMG = cp.zeros((self.nat3, self.nat3), dtype=self.dtyper)
        GAM = cp.zeros((self.nat3, self.nat3), dtype=self.dtyper)

        for ii in range(self.nq):
            cp.fill_diagonal(OMG, self.material.phonon_freq[ii])
            cp.fill_diagonal(GAM, self.material.linewidth[ii])

            gv_op = self.material.velocity_operator[ii]

            # term1 = cp.kron(I_small, OMG) - cp.kron(OMG, I_small) - (omg_ft * I_big)
            term1 = (
                cp.einsum("ij,kl->ikjl", I_small, OMG).reshape(self.nat3**2, self.nat3**2)
                - cp.einsum("ij,kl->ikjl", OMG, I_small).reshape(self.nat3**2, self.nat3**2)
                - self.omg_ft * I_big
            )
            # term2 = (k_ft / 2) * (cp.kron(I_small, gv_op) + cp.kron(gv_op.T, I_small))
            term2 = (self.k_ft / 2) * (
                cp.einsum("ij,kl->ikjl", I_small, gv_op).reshape(self.nat3**2, self.nat3**2)
                + cp.einsum("ij,kl->ikjl", gv_op.conj().T, I_small).reshape(self.nat3**2, self.nat3**2)
            )
            # term3 = 0.5 * (cp.kron(I_small, GAM) + cp.kron(GAM, I_small))
            term3 = 0.5 * (
                cp.einsum("ij,kl->ikjl", I_small, GAM).reshape(self.nat3**2, self.nat3**2)
                + cp.einsum("ij,kl->ikjl", GAM, I_small).reshape(self.nat3**2, self.nat3**2)
            )

            self._op[ii] = (1j * (term1 - term2)) + term3


class GreenOperatorBase(ABC):
    """Base class for Green's operators.

    Provides a common interface for Green's operators, whether computed or loaded from disk.
    """

    __array_priority__ = 1000

    def _require_ready(self):
        """Ensure the Green's operator is computed before accessing it."""
        if self._green is None:
            raise RuntimeError("Green's operator not computed yet. Call compute() first.")

    def __getitem__(self, iq):
        """Allow indexing to access the Green's function for a specific q-point."""
        self._require_ready()
        return self._green[iq]

    def __matmul__(self, other):
        """Allow matrix multiplication with the Green's function."""
        self._require_ready()
        return self._green @ other

    def __len__(self):
        """Return the number of q-points in the Green's function."""
        self._require_ready()
        return len(self._green)

    def __iter__(self):
        """Allow iteration over the Green's function q-points."""
        self._require_ready()
        return iter(self._green)

    @property
    def __cuda_array_interface__(self):
        """Return the CUDA array interface for the Green's function."""
        self._require_ready()
        return self._green.__cuda_array_interface__

    @property
    def shape(self):
        """Return the shape of the Green's function."""
        self._require_ready()
        return self._green.shape

    @property
    def dtype(self):
        """Return the dtype of the Green's function."""
        self._require_ready()
        return self._green.dtype

    def squeeze(self):
        """Return the Green's function with singleton dimensions removed."""
        self._require_ready()
        return self._green.squeeze()

    def __repr__(self):
        """Return a string representation of the Green's operator."""
        return f"<RTAGreenOperator: {len(self)} q-points at w={self.omg_ft:.2e} with dtype={self.dtypec.__name__}>"

    @abstractmethod
    def compute(self, recompute=False, **kwargs):
        """Compute or load the Green's function from disk."""

    def free(self):
        """Free the memory used by the Green's operator."""
        if self._green is not None:
            self._green = None
            if hasattr(self, "wigner_operator"):
                self.wigner_operator._op = None


class RTAGreenOperator(GreenOperatorBase):
    """Class to compute the Green's function in the relaxation time approximation from the Wigner operator."""

    def __init__(self, wigner_operator: RTAWignerOperator):
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

    def compute(self, recompute=False, clear_wigner=True):
        """Compute the Green's function from the Wigner operator.

        Parameters
        ----------
        recompute : bool
            Whether to recompute the Green's function if it has already been computed.
        clear_wigner : bool
            Whether to clear the Wigner operator after computing the Green's function.

        """
        if self._green is not None and not recompute:
            return

        self.wigner_operator.compute()  # will not recompute if already done

        self._green = cp.zeros_like(self.wigner_operator._op, dtype=self.dtypec)

        for ii in range(len(self.wigner_operator)):
            self._green[ii] = cp.linalg.inv(self.wigner_operator[ii])

        if clear_wigner:
            self.wigner_operator._op = None


class DiskGreenOperator(GreenOperatorBase):
    """Disk-based Green's operator that loads precomputed Green's functions from disk.

    Loads one (nq, m, m) slab to GPU on demand.
    """

    def __init__(
        self,
        container: GreenContainer,
        omg_ft: float,
        k_ft: float,
        material: Material,
        atol: float = 1e-6,
        as_gpu: bool = True,
    ):
        """Initialize the disk-based Green's operator."""
        self.omg_ft = omg_ft
        self.k_ft = k_ft
        self.material = material
        self.nq = material.nq
        self.nat3 = material.nat3
        self.dtyper = material.dtyper
        self.dtypec = material.dtypec

        self._gc = container
        self._as_gpu = as_gpu
        self._atol = atol
        self._green = None

    def compute(self, recompute=False):
        """Load the Green's operator from disk or recompute it if necessary."""
        if self._green is not None and not recompute:
            return
        # pull the Green's operator from disk
        arr = self._gc.get_bz_block(self.omg_ft, self.k_ft, as_gpu=self._as_gpu, atol=self._atol)
        self._green = cp.ascontiguousarray(cp.asarray(arr, dtype=self.dtypec))


class GreenWTESolver(SolverBase):
    r"""Wigner Transport Equation solver using precomputed Green's operators.

    This solver implements the mapping from a temperature change ``dT`` to
    the Wigner distribution function ``n`` via direct matrix multiplication
    with precomputed Green's function operators. The approach bypasses the
    need for iterative inner solvers such as GMRES and can be significantly
    faster when the Green's functions are available.

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
    greens : list of RTAGreenOperator
        List of precomputed Green's function operators, one for each
        ``omg_ft`` value in `omg_ft_array`. Each operator must implement
        matrix multiplication to map the source term to the Wigner
        distribution function.
    max_iter : int, optional
        Maximum number of iterations for the outer solver (default is 100).
    conv_thr : float, optional
        Convergence threshold for the outer solver (default is 1e-12).
    outer_solver : {'root', 'aitken', 'plain'}, optional
        Outer solver strategy to use. One of:
        - ``'root'``: nonlinear root finding in 2D space (real & imaginary parts)
        - ``'aitken'``: Aitken Δ² acceleration applied to fixed-point iteration
        - ``'plain'``: plain fixed-point iteration without acceleration
        Default is ``'root'``.
    command_line_args : argparse.Namespace, optional
        Optional namespace of parsed command-line arguments for controlling
        solver behavior or I/O (default is an empty Namespace).

    Raises
    ------
    ValueError
        If the number of supplied Green's operators does not match the length
        of `omg_ft_array`.

    Attributes
    ----------
    greens : list of RTAGreenOperator
        The precomputed Green's function operators, indexed by frequency.

    Notes
    -----
    - This implementation calls :func:`dT_to_N_matmul` to apply the Green's
      operator for the given frequency index. No residuals are generated,
      and the second return value of :meth:`_dT_to_N` is always a list of
      empty lists (one per ``nq``).
    - Since there is no iterative inner solver, performance is largely
      determined by the cost of the Green's operator matrix multiplication.

    See Also
    --------
    SolverBase : Parent class that provides the outer-solver infrastructure.
    IterativeWTESolver : Alternative solver using iterative linear solvers
                         (not shown here).

    """

    def __init__(
        self,
        omg_ft_array: cp.ndarray,
        k_ft: cp.ndarray,
        material: Material,
        source: cp.ndarray,
        greens: list[RTAGreenOperator],
        max_iter=100,
        conv_thr=1e-12,
        outer_solver="root",
        command_line_args=Namespace(),
        residual_weights: tuple[float, float] = (1.0, 0.0),
    ) -> None:
        """Initialize GreenWTESolver."""
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
        if not len(greens) == len(omg_ft_array):
            raise ValueError("Number of Green's operators must match the number of omg_ft values.")
        self.greens = greens

    def _dT_to_N(
        self,
        dT: complex,
        omg_ft: float,
        omg_idx: int,
        sol_guess: cp.ndarray | None = None,
    ) -> tuple[cp.ndarray, list]:
        n = dT_to_N_matmul(
            dT=dT,
            material=self.material,
            green=self.greens[omg_idx],
            source=self.source,
        )
        return n, [[] for _ in range(self.material.nq)]  # no residuals from matrix multiplication

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
            g = self.greens[i]
            g.compute()  # ensure the Green's operator is computed or loaded from disk
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

            g.free()  # free memory used by the Green's operator

        self._solution_lists_to_arrays()
