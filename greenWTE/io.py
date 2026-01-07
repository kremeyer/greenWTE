"""IO module for Green's function transport equation (WTE) calculations.

This module provides a high-level interface for storing and retrieving precomputed Green's operators in an HDF5
container. It supports lazy dataset creation, dynamic resizing, and optional GPU (CuPy) compatibility for data I/O.
"""

import contextlib
import json
from os import PathLike
from types import TracebackType
from typing import Protocol, runtime_checkable

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import h5py
import numpy as np
from scipy.constants import elementary_charge

from greenWTE.base import SolverBase

from . import hdf5plugin, to_cpu, xp

SCHEMA = "rta-greens/2"


def _ensure(
    f: h5py.File,
    name: str,
    *,
    shape: tuple[int, ...],
    maxshape: tuple[int, ...] | None,
    chunks: tuple[int, ...],
    dtype: np.dtype | str,
    **kwargs,
) -> h5py.Dataset:
    """Ensure that a dataset exists in an HDF5 file.

    If the dataset named ``name`` exists in the file ``f``, it is returned. Otherwise, the dataset is created with the
    given shape, dtype, chunking, compression, and unlimited dimensions.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file object.
    name : str
        Dataset path.
    shape : tuple of int
        Initial dataset shape.
    maxshape : tuple of int or None
        Maximum shape; use ``None`` for dimensions that may grow.
    chunks : tuple of int
        Chunk sizes per dimension.
    dtype : dtype or str
        On-disk dtype.
    **kwargs : dict
        Additional keyword arguments forwarded to :py:meth:`h5py.File.create_dataset`.

    Returns
    -------
    h5py.Dataset
        The existing or newly created dataset.

    """
    if name in f:
        return f[name]
    return f.create_dataset(name, shape=shape, maxshape=maxshape, chunks=chunks, dtype=np.dtype(dtype), **kwargs)


def _find_or_append_1d(dset, value: float, atol: float = 0.0):
    """Find the index of ``value`` in a 1-D dataset, appending if absent.

    Parameters
    ----------
    dset : h5py.Dataset
        One-dimensional dataset.
    value : float
        Scalar value to find or append.
    atol : float, optional
        Absolute tolerance for equality checks.

    Returns
    -------
    int
        The index of the matching or newly appended value.

    """
    val = _scalar_to_float(value)
    data = dset[...]
    for i, v in enumerate(data):
        if np.allclose(v, val, atol=atol):
            return i
    i = len(data)
    dset.resize((i + 1,))
    dset[i] = val
    return i


def _find_index_1d(dset, value: float, atol: float = 0.0):
    """Find the index of ``value`` in a 1-D dataset.

    Parameters
    ----------
    dset : h5py.Dataset
        One-dimensional dataset.
    value : float
        Scalar value to find or append.
    atol : float, optional
        Absolute tolerance for equality checks.

    Returns
    -------
    int
        Index if found; ``-1`` otherwise.

    """
    val = _scalar_to_float(value)
    data = dset[...]
    for i, v in enumerate(data):
        if np.allclose(v, val, atol=atol):
            return i
    return -1  # Not found


def _scalar_to_float(x) -> float:
    """Convert Python/NumPy/CuPy scalar or size-1 array to float."""
    if isinstance(x, (int, float, np.generic)):
        return float(x)
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except (TypeError, ValueError):
            pass
    if hasattr(x, "get"):
        x = x.get()
    a = np.asarray(x)
    if a.size != 1:
        raise TypeError("Input is not a scalar.")
    return float(a.item())


@runtime_checkable
class GreenOperatorLike(Protocol):
    """Used for type checking Green's operator-like objects."""

    _green: xp.ndarray


class GreenContainer:
    r"""Container for HDF5-stored Green's operators.

    This class manages the storage of precomputed Green's operators indexed by frequency ``omega``, wavevector magnitude
    ``k``, and a q-point index ``q``. Data is stored in a 5D array with shape (Nw, Nk, nq, m, m), where
    :math:`m = (3*n_\mathrm{at})^2`.

    Parameters
    ----------
    path : os.PathLike
        Path to the HDF5 file. Created if it does not exist.
    nat3 : int, optional
        Number of atoms times 3; defines the block size of the operator. Only required
        if this is a new Green's function file. Read from file if None.
    nq : int, optional
        Number of q-points. Only required if this is a new Green's function file. Read from file if None.
    dtype : dtype or str, optional
        Complex or real data type for storage.
    meta : dict, optional
        Arbitrary metadata to persist as root attributes.
    tile_B : int, optional
        Block size for chunking the matrix dimensions.
    read_only : bool, optional
        If True, open the file in read-only mode. Default is False. This will allow multiple readers to access the file
        simultaneously.

    Attributes
    ----------
    f : h5py.File
        Open HDF5 handle.
    ds_w, ds_k : h5py.Dataset
        The 1-D frequency and wavevector datasets.
    ds_mask : h5py.Dataset
        Boolean mask indicating which ``(w, k, q)`` blocks exist.
    ds_tens : h5py.Dataset
        Tensor operator blocks with shape ``(Nw, Nk, nq, m, m)``.
    m : int
        Matrix block dimension, `nat3**2`.

    """

    def __init__(
        self,
        path: PathLike,
        nat3: int | None = None,
        nq: int | None = None,
        dtype: xp.dtype = xp.complex128,
        meta: dict | None = None,
        tile_B: int = 512,
        read_only: bool = False,
    ) -> None:
        """Initialize the GreenContainer."""
        self.path = path
        if nat3 is None:
            self.nat3 = int(h5py.File(path, "r").attrs["nat3"])
        else:
            self.nat3 = nat3
        if nq is None:
            self.nq = int(h5py.File(path, "r").attrs["nq"])
        else:
            self.nq = nq
        self.m = self.nat3**2
        self.dtype = np.dtype(dtype)
        self.meta = meta or {}
        self.B = min(int(tile_B), self.m)

        mode = "r" if read_only else "a"
        self.f = h5py.File(path, mode, libver="latest", rdcc_nbytes=512 * 1024 * 1024, rdcc_w0=0.9)

        # Root attrs
        if "schema" not in self.f.attrs:
            self.f.attrs["schema"] = SCHEMA
            self.f.attrs["nat3"] = self.nat3
            self.f.attrs["nq"] = self.nq
            self.f.attrs["m"] = self.m
            self.f.attrs["dtype"] = str(self.dtype)
            if self.meta:
                self.f.attrs["meta"] = json.dumps(self.meta)
        else:
            on_disk_dtype = xp.dtype(self.f.attrs["dtype"])
            if on_disk_dtype != self.dtype:
                raise TypeError(f"Data type mismatch: {on_disk_dtype} (on disk) != {self.dtype} (requested)")

        # Always create 1D index datasets up front
        self.ds_w = _ensure(self.f, "omega", shape=(0,), maxshape=(None,), chunks=(1024,), dtype=np.float64)
        self.ds_k = _ensure(self.f, "k", shape=(0,), maxshape=(None,), chunks=(1024,), dtype=np.float64)

        # Defer /green and /mask creation until the first (w,k) is added
        self._have_main = ("green" in self.f) and ("mask" in self.f)
        if self._have_main:
            self.ds_tens = self.f["green"]
            self.ds_mask = self.f["mask"]

    def __enter__(self) -> Self:
        """Enter the context manager, returning self."""
        return self

    def __exit__(
        self, type_: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        """Exit the context manager, closing the file."""
        self.close()

    def _ensure_main(self) -> None:
        """Create /green and /mask if missing, with current Nw,Nk (>0)."""
        if self._have_main:
            return
        Nw, Nk = len(self.ds_w), len(self.ds_k)
        if Nw == 0 or Nk == 0:
            # safeguard; should not happen
            raise RuntimeError("internal: _ensure_main called with zero Nw or Nk")
        self.ds_tens = _ensure(
            self.f,
            "green",
            shape=(Nw, Nk, self.nq, self.m, self.m),
            maxshape=(None, None, self.nq, self.m, self.m),
            chunks=(1, 1, 1, self.B, self.B),
            dtype=self.dtype,
            compression=hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
        )
        self.ds_mask = _ensure(
            self.f,
            "mask",
            shape=(Nw, Nk, self.nq),
            maxshape=(None, None, self.nq),
            chunks=(4, 4, min(256, self.nq)),
            dtype=np.uint8,
        )
        self._have_main = True

    def close(self) -> None:
        """Close the underlying HDF5 file."""
        with contextlib.suppress(Exception):
            self.f.close()

    def indices(self, w: float, k: float, atol: float = 0.0) -> tuple[int, int]:
        """Return ``(iw, ik)`` indices for the given ``(w, k)``. Append if missing.

        Parameters
        ----------
        w : float
            Angular frequency.
        k : float
            Wavevector magnitude.
        atol : float, optional
            Absolute tolerance for comparing existing values.

        Returns
        -------
        tuple of int
            ``(iw, ik)`` indices into the ``omega`` and ``k`` datasets.

        """
        iw = _find_or_append_1d(self.ds_w, w, atol=atol)
        ik = _find_or_append_1d(self.ds_k, k, atol=atol)

        # If this append made Nw or Nk go from 0 -> 1, create main datasets now
        if not self._have_main and (len(self.ds_w) > 0 and len(self.ds_k) > 0):
            self._ensure_main()

        # If main exists and indices exceed current shape, resize
        if self._have_main and (iw >= self.ds_tens.shape[0] or ik >= self.ds_tens.shape[1]):
            new_Nw = max(self.ds_tens.shape[0], iw + 1)
            new_Nk = max(self.ds_tens.shape[1], ik + 1)
            self.ds_tens.resize((new_Nw, new_Nk, self.nq, self.m, self.m))
            self.ds_mask.resize((new_Nw, new_Nk, self.nq))

        return iw, ik

    def find_indices(self, w: float, k: float, atol: float = 0.0) -> tuple[int, int]:
        """Return ``(iw, ik)`` indices for the given ``(w, k)``.

        Parameters
        ----------
        w : float
            Angular frequency.
        k : float
            Wavevector magnitude.
        atol : float, optional
            Absolute tolerance for comparing existing values.

        Returns
        -------
        tuple of int
            ``(iw, ik)`` indices into the ``omega`` and ``k`` datasets.

        Raises
        ------
        KeyError
            If the omega or k value is not found within the specified tolerance.

        """
        iw = _find_index_1d(self.ds_w, w, atol=atol)
        ik = _find_index_1d(self.ds_k, k, atol=atol)
        if iw < 0 or ik < 0:
            raise KeyError(f"Indices for w={w}, k={k} not found.")
        return iw, ik

    def has(self, w: float, k: float, q: int, atol: float = 0.0) -> bool:
        """Check whether a Green's operator block exists for given indices.

        Parameters
        ----------
        w : float
            Angular frequency.
        k : float
            Wavevector magnitude.
        q : int
            q-point index.
        atol : float, optional
            Absolute tolerance for comparing existing values.

        Returns
        -------
        bool
            ``True`` if the block exists, ``False`` otherwise.

        """
        try:
            iw, ik = self.find_indices(w, k, atol=atol)
        except KeyError:
            return False
        return bool(self.ds_mask[iw, ik, int(q)])

    def has_bz_block(self, w: float, k: float, atol: float = 0.0) -> bool:
        """Check whether a Green's operator block exists for the full Brillouin zone.

        Parameters
        ----------
        w : float
            Angular frequency.
        k : float
            Wavevector magnitude.
        atol : float, optional
            Absolute tolerance for comparing existing values.

        Returns
        -------
        bool
            ``True`` if the block exists, ``False`` otherwise.

        """
        try:
            iw, ik = self.find_indices(w, k, atol=atol)
        except KeyError:
            return False
        if not self._have_main:
            return False
        return bool(np.all(self.ds_mask[iw, ik, :]))  # cast from np._bool to python bool

    def get(self, w: float, k: float, q: int, atol: float = 0.0) -> xp.ndarray:
        """Load a single Green’s operator block ``G[w, k, q]``.

        Parameters
        ----------
        w : float
            Angular frequency.
        k : float
            Wavevector magnitude.
        q : int
            q-point index.
        atol : float, optional
            Absolute tolerance for comparing existing values.

        Returns
        -------
        cupy.ndarray
            A GPU array with shape ``(m, m)`` and complex dtype matching the on-disk dataset.

        Raises
        ------
        KeyError
            If the requested block is missing.

        """
        iw, ik = self.find_indices(w, k, atol=atol)
        if not self._have_main or not bool(self.ds_mask[iw, ik, int(q)]):
            raise KeyError(f"Requested block w={w}, k={k}, q={q} not found.")
        out = self.ds_tens[iw, ik, int(q), :, :][...]  # NumPy array
        return xp.asarray(out)

    def get_bz_block(self, w: float, k: float, atol: float = 0.0) -> xp.ndarray:
        """Load all q-blocks for a given pair ``(w, k)``.

        Parameters
        ----------
        w : float
            Angular frequency.
        k : float
            Wavevector magnitude.
        atol : float, optional
            Absolute tolerance for comparing existing values.

        Returns
        -------
        cupy.ndarray
            The `(nq, m, m)` array of operator blocks for the specified (w, k).

        Raises
        ------
        KeyError
            If the requested block is missing.

        """
        iw, ik = self.find_indices(w, k, atol=atol)
        tens = self.ds_tens[iw, ik, :, :, :]
        mask = self.ds_mask[iw, ik, :]
        if not np.all(mask):
            raise KeyError(f"Missing q-point blocks for w={w}, k={k}.")
        return xp.asarray(tens)

    def put(self, w: float, k: float, q: int, data: xp.ndarray, atol: float = 0.0, flush: bool = True) -> None:
        """Store a Green's operator block ``G[w, k, q]``.

        Parameters
        ----------
        w : float
            Angular frequency.
        k : float
            Wavevector magnitude.
        q : int
            q-point index.
        data : cupy.ndarray
            Operator block to store; must have shape `(m, m)`.
        atol : float, optional
            Absolute tolerance for comparing existing values.
        flush : bool, optional
            If True, flush the file to disk after writing.

        Raises
        ------
        ValueError
            If the shape of `data` does not match `(m, m)`.

        """
        iw, ik = self.indices(w, k, atol=atol)
        # ensure main exists (it will after indices())
        if not self._have_main:
            self._ensure_main()
        arr = data.get() if hasattr(data, "get") else data
        raw = data._green if isinstance(data, GreenOperatorLike) else data
        arr = raw.get() if hasattr(raw, "get") else raw
        if arr.shape != (self.m, self.m):
            raise ValueError(f"Data shape {arr.shape} != {(self.m, self.m)}.")
        if arr.dtype != self.ds_tens.dtype:
            raise TypeError(f"dtype mismatch: data {arr.dtype} != {self.ds_tens.dtype}")
        self.ds_tens[iw, ik, int(q), :, :] = arr
        self.ds_mask[iw, ik, int(q)] = 1
        if flush:
            self.f.flush()

    def put_bz_block(self, w: float, k: float, data: xp.ndarray, atol: float = 0.0, flush: bool = True) -> None:
        """Store the full Brillouin-zone block for a pair ``(w, k)``.

        Parameters
        ----------
        w : float
            Angular frequency.
        k : float
            Wavevector magnitude.
        data : cupy.ndarray or greenWTE.green.RTAGreenOperator
            Array of operator blocks to store; must have shape `(nq, m, m)`.
        atol : float, optional
            Absolute tolerance for comparing existing values.
        flush : bool, optional
            If True, flush the file to disk after writing.

        Raises
        ------
        ValueError
            If the shape of `data` does not match `(nq, m, m)`.

        """
        iw, ik = self.indices(w, k, atol=atol)
        # ensure main exists (it will after indices())
        if not self._have_main:
            self._ensure_main()
        arr = data._green if isinstance(data, GreenOperatorLike) else data
        arr = arr.get() if hasattr(arr, "get") else arr
        if arr.shape != (self.nq, self.m, self.m):
            raise ValueError(f"Data shape {arr.shape} != {(self.nq, self.m, self.m)}.")
        if arr.dtype != self.ds_tens.dtype:
            raise TypeError(f"dtype mismatch: data {arr.dtype} != {self.ds_tens.dtype}")
        self.ds_tens[iw, ik, :, :, :] = arr
        self.ds_mask[iw, ik, :] = 1
        if flush:
            self.f.flush()

    def omegas(self, k: float | None = None) -> np.ndarray | None:
        r"""Return all stored :math:`\omega` values.

        Parameters
        ----------
        k : float, optional
            If provided, return only the frequencies for which **any** q-block exists at this ``k`` (according to the
            mask).

        Returns
        -------
        numpy.ndarray | None
            One-dimensional array of frequencies in rad/s. If ``k`` is provided but not found, returns ``None``.

        """
        if k is None:
            return self.ds_w[...]
        ik = _find_index_1d(self.ds_k, k)
        if ik < 0:
            return
        return self.ds_w[...][self.ds_mask[:, ik, :].any(axis=1)]

    def ks(self, w: float | None = None) -> np.ndarray | None:
        r"""Return all stored :math:`k` values.

        Parameters
        ----------
        w : float, optional
            If provided, return only the wavevector magnitudes for which **any** q-block exists at this ``omega``
            (according to the mask).

        Returns
        -------
        numpy.ndarray | None
            One-dimensional array of wavevector magnitudes in 1/m. If ``w`` is provided but not found, returns ``None``.

        """
        if w is None:
            return self.ds_k[...]
        iw = _find_index_1d(self.ds_w, w)
        if iw < 0:
            return
        return self.ds_k[...][self.ds_mask[iw, :, :].any(axis=1)]


def load_phono3py_data(
    filename: PathLike,
    temperature: float,
    dir_idx: int,
    exclude_gamma: bool = True,
    dtyper: xp.dtype = xp.float64,
    dtypec: xp.dtype = xp.complex128,
) -> tuple[xp.ndarray, xp.ndarray, xp.ndarray, xp.ndarray, xp.ndarray, float, xp.ndarray]:
    r"""Load data from a phono3py-generated HDF5 file.

    Parameters
    ----------
    filename : os.PathLike
        Path to the :mod:phono3py HDF5 file.
    temperature : float
        Temperature in Kelvin at which to read linewidths and heat capacities.
    dir_idx : int
        Cartesian direction index of the velocity operator (0=x, 1=y, 2=z).
    exclude_gamma : bool, optional
        If ``True``, drop the :math:`\Gamma` point from the beginning of each q-point-indexed dataset.
    dtyper : dtype or str, optional
        Real data type for loading real-valued arrays.
    dtypec : dtype or str, optional
        Complex data type for loading complex-valued arrays.

    Returns
    -------
    qpoint : cupy.ndarray
        q-points of, shape ``(nq, 3)``.
    velocity_operator : cupy.ndarray
        Velocity operator in m/s, shape ``(nq, nat3, nat3)``.
    phonon_freq : cupy.ndarray
        Phonon frequencies in rad/s, shape ``(nq, nat3)``.
    linewidth : cupy.ndarray
        Linewidths of each mode in rad/s, shape ``(nq, nat3)``.
    heat_capacity : cupy.ndarray
        Heat capacity of each mode in J/m^3/K, shape ``(nq, nat3)``.
    volume : float
        Volume of the system in m^3.
    weight : cupy.ndarray
        Normalized q-point weights, shape ``(nq,)``.

    Raises
    ------
    ValueError
        If the specified temperature is not found in the input file.

    """
    with h5py.File(filename, "r") as h5f:
        available_temperatures = list(h5f["temperature"][()])
        if temperature in available_temperatures:
            temperature_index = available_temperatures.index(temperature)
        else:
            raise ValueError(
                f"Temperature {temperature} not found in the input file."
                f"Available temperatures: {available_temperatures}"
            )
        q_idx = int(exclude_gamma)
        qpoint = xp.array(h5f["qpoint"][q_idx:, :])  # (nq, 3) | 1
        velocity_operator = (
            xp.array(h5f["velocity_operator_sym"][q_idx:, ..., dir_idx], dtype=dtypec) * 1e2
        )  # (nq, nat3, nat3) | m/s
        phonon_freq = xp.array(h5f["frequency"][q_idx:], dtype=dtyper) * 1e12 * 2 * xp.pi  # (nq, nat3) | [rad/s]
        linewidth = xp.array(h5f["gamma"][temperature_index, q_idx:], dtype=dtyper)  # (nT, nq, nat3)
        linewidth += xp.array(h5f["gamma_isotope"][q_idx:], dtype=dtyper)  # (nq, nat3)
        linewidth += xp.array(h5f["gamma_boundary"][q_idx:], dtype=dtyper)  # (nq, nat3)
        linewidth *= 1e12 * 2 * xp.pi  # [rad/s] | ordinal to angular frequency
        linewidth *= 2  # HWHM -> FWHM
        volume = xp.array(h5f["volume"][()], dtype=dtyper) * 1e-30  # m^3
        weight = xp.array(h5f["weight"][q_idx:], dtype=dtyper)  # nq | 1
        weight /= xp.sum(weight)
        heat_capacity = (
            xp.array(h5f["heat_capacity"][temperature_index, q_idx:], dtype=dtyper) * elementary_charge
        )  # (nT, nq, nat3) | J/K
        heat_capacity *= weight[:, None] / volume  # (nq, nat3) | J/(K m^3)

    return (
        qpoint,
        velocity_operator,
        phonon_freq,
        linewidth,
        heat_capacity,
        volume,
        weight,
    )


def save_solver_result(filename: PathLike, solver: SolverBase, **kwargs) -> None:
    r"""Save a solver run to an HDF5 file.

    This is a light-weight, analysis-friendly snapshot of a full frequency-domain solve, including the temperature
    response, Wigner distribution, and iteration metadata.

    Parameters
    ----------
    filename : os.PathLike
        Path to the output HDF5 file.
    solver : greenWTE.solver.SolverBase
        A solver instance (e.g., :py:class:`iterative.IterativeWTESolver` or :py:class:`green.GreenWTESolver`) that
        exposes the recorded arrays as attributes.
    **kwargs : dict
        Additional key-value pairs to store as datasets in the root group. Scalars, lists, and arrays are supported.

    Notes
    -----
    The function records (if available):

    - ``dT``: complex temperature response over frequencies ``(Nw,)``.
    - ``dT_init``: initial guess ``(Nw,)``.
    - ``n``: Wigner distribution ``(Nw, nq, m)`` or similar aggregate.
    - ``niter``: number of outer iterations per frequency.
    - ``iter_time``: wall-clock time per frequency in seconds.
    - ``gmres_residual``: GMRES residual history (ragged) concatenated.
    - ``dT_iterates``: stored iterates of ``dT`` (stacked).
    - ``n_norms``: norms of ``n`` per outer iteration.
    - ``source``: source term used in the solve.
    - ``omega``: angular frequencies in rad/s.
    - ``k``: wavevector magnitudes in rad/m.

    In addition, command-line arguments found in ``solver.command_line_args`` are
    persisted as datasets if present.

    """
    with h5py.File(filename, "w") as h5f:
        h5f.create_dataset("dT", data=to_cpu(solver.dT))
        h5f["dT"].attrs["units"] = "Kelvin"
        h5f.create_dataset("dT_init", data=to_cpu(solver.dT_init))
        h5f["dT_init"].attrs["units"] = "Kelvin"
        h5f.create_dataset("n", data=to_cpu(solver.n))
        h5f.create_dataset("niter", data=to_cpu(solver.niter))
        h5f.create_dataset("iter_time", data=to_cpu(solver.iter_time))
        h5f["iter_time"].attrs["units"] = "seconds"
        h5f.create_dataset("gmres_residual", data=to_cpu(solver.gmres_residual))
        h5f.create_dataset("dT_iterates", data=to_cpu(solver.dT_iterates))
        h5f["dT_iterates"].attrs["units"] = "Kelvin"
        h5f.create_dataset("n_norms", data=to_cpu(solver.n_norms))
        h5f.create_dataset("source", data=to_cpu(solver.source))

        h5f.create_dataset("omega", data=to_cpu(solver.omg_ft_array))
        h5f["omega"].attrs["units"] = "radians/second"
        h5f.create_dataset("k", data=to_cpu(solver.k_ft))
        h5f["k"].attrs["units"] = "radians/meter"
        h5f.create_dataset("max_iter", data=to_cpu(solver.max_iter))
        h5f.create_dataset("conv_thr_rel", data=to_cpu(solver.conv_thr_rel))
        h5f.create_dataset("conv_thr_abs", data=to_cpu(solver.conv_thr_abs))
        h5f["conv_thr_abs"].attrs["units"] = "Kelvin"
        h5f.create_dataset("dtype_real", data=str(solver.material.dtyper))
        h5f.create_dataset("dtype_complex", data=str(solver.material.dtypec))
        h5f.create_dataset("outer_solver", data=solver.outer_solver)
        h5f.create_dataset("inner_solver", data=solver.inner_solver)

        h5f.create_dataset("kappa", data=to_cpu(solver.kappa))
        h5f["kappa"].attrs["units"] = "Watts/meter/Kelvin"
        h5f.create_dataset("kappa_P", data=to_cpu(solver.kappa_p))
        h5f["kappa_P"].attrs["units"] = "Watts/meter/Kelvin"
        h5f.create_dataset("kappa_C", data=to_cpu(solver.kappa_c))
        h5f["kappa_C"].attrs["units"] = "Watts/meter/Kelvin"

        for key, value in vars(solver.command_line_args).items():
            if key in h5f:
                continue
            v = to_cpu(value)
            if isinstance(v, (list, tuple)):
                v = np.asarray(v)
            h5f.create_dataset(key, data=v)

        for key, value in kwargs.items():
            v = to_cpu(value)
            if isinstance(v, (list, tuple)):
                v = np.asarray(v)
            h5f.create_dataset(key, data=v)
