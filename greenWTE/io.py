"""IO module for Green's function transport equation (WTE) calculations.

This module provides a high-level interface for storing and retrieving
precomputed Green's operators in an HDF5 container. It supports lazy dataset
creation, dynamic resizing, and optional GPU (CuPy) compatibility for data
I/O.
"""

import contextlib
import json
from typing import Protocol, runtime_checkable

import cupy as cp
import h5py
import numpy as np

SCHEMA = "rta-greens/1"


def _ensure(f, name, *, shape, maxshape, chunks, dtype, **kwargs):
    """Ensure that a dataset exists in an HDF5 file.

    If the dataset `name` exists in the file `f`, it is returned.
    Otherwise, it is created with the provided parameters.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file object.
    name : str
        Name of the dataset.
    shape : tuple of int
        Initial dataset shape.
    maxshape : tuple of int or None
        Maximum shape of the dataset; use None for unlimited dimensions.
    chunks : tuple of int
        Chunk sizes for each dimension.
    dtype : dtype or str
        Data type of the dataset; converted to a NumPy dtype.
    **kwargs : dict
        Additional keyword arguments passed to `h5py.File.create_dataset`.

    Returns
    -------
    h5py.Dataset
        The existing or newly created dataset.

    """
    if name in f:
        return f[name]
    return f.create_dataset(name, shape=shape, maxshape=maxshape, chunks=chunks, dtype=np.dtype(dtype), **kwargs)


def _find_or_append_1d(dset, value, atol=0.0):
    """Find or append a scalar value to a 1D dataset.

    Parameters
    ----------
    dset : h5py.Dataset
        A 1D dataset.
    value : float
        Scalar value to find or append.
    atol : float, optional
        Absolute tolerance for comparison when matching existing values.

    Returns
    -------
    int
        The index of the matching or newly appended value.

    """
    val = float(np.asarray(value))
    data = dset[...]
    for i, v in enumerate(data):
        if np.allclose(v, val, atol=atol):
            return i
    i = len(data)
    dset.resize((i + 1,))
    dset[i] = val
    return i


def _find_index_1d(dset, value, atol=0.0):
    """Fint index of `value` in a 1D dataset."""
    val = float(np.asarray(value))
    data = dset[...]
    for i, v in enumerate(data):
        if np.allclose(v, val, atol=atol):
            return i
    return -1  # Not found


@runtime_checkable
class GreenOperatorLike(Protocol):
    """Used for type checking Green's operator-like objects."""

    _green: cp.ndarray


class GreenContainer:
    """Container for storing Green's operators in an HDF5 file.

    This class manages the storage of precomputed Green's operators indexed
    by frequency (`omega`), wavevector magnitude (`k`), and a q-point index.
    Data is stored in a 5D array with shape `(Nw, Nk, nq, m, m)`, where
    `m = nat3**2`.

    Parameters
    ----------
    path : str
        Path to the HDF5 file. Created if it does not exist.
    nat3 : int
        Number of atoms times 3; defines the block size of the operator.
    nq : int
        Number of q-points.
    dtype : dtype or str, optional
        Complex or real data type for storage. Default is `cp.complex128`.
    meta : dict, optional
        Additional metadata to store in the file root attributes.
    tile_B : int, optional
        Block size for chunking the matrix dimensions. Default is 512.

    Attributes
    ----------
    nat3 : int
        Number of atoms times 3.
    nq : int
        Number of q-points.
    m : int
        Matrix block dimension, `nat3**2`.
    dtype : numpy.dtype
        Data type of stored arrays.
    omegas : numpy.ndarray
        Array of stored omega values.
    ks : numpy.ndarray
        Array of stored k values.

    """

    def __init__(self, path, nat3, nq, dtype=cp.complex128, meta=None, tile_B=512):
        """Initialize the GreenContainer."""
        self.path = path
        self.nat3 = int(nat3)
        self.nq = int(nq)
        self.m = self.nat3**2
        self.dtype = np.dtype(dtype)
        self.meta = meta or {}
        self.B = min(int(tile_B), self.m)

        self.f = h5py.File(path, "a", libver="latest", rdcc_nbytes=512 * 1024 * 1024, rdcc_w0=0.9)

        # Root attrs
        if "schema" not in self.f.attrs:
            self.f.attrs["schema"] = SCHEMA
            self.f.attrs["nat3"] = self.nat3
            self.f.attrs["nq"] = self.nq
            self.f.attrs["m"] = self.m
            self.f.attrs["dtype"] = str(self.dtype)
            if self.meta:
                self.f.attrs["meta"] = json.dumps(self.meta)

        # Always create 1D index datasets up front
        self.ds_w = _ensure(self.f, "omega", shape=(0,), maxshape=(None,), chunks=(1024,), dtype=np.float64)
        self.ds_k = _ensure(self.f, "k", shape=(0,), maxshape=(None,), chunks=(1024,), dtype=np.float64)

        # Defer /green and /mask creation until the first (w,k) is added
        self._have_main = ("green" in self.f) and ("mask" in self.f)
        if self._have_main:
            self.ds_tens = self.f["green"]
            self.ds_mask = self.f["mask"]

    def __enter__(self):
        """Enter the context manager, returning self."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager, closing the file."""
        self.close()

    def _ensure_main(self):
        """Create /green and /mask if missing, with current Nw,Nk (>0)."""
        if self._have_main:
            return
        Nw, Nk = len(self.ds_w), len(self.ds_k)
        if Nw == 0 or Nk == 0:
            # Should never happen; we call this only after appending w/k
            raise RuntimeError("internal: _ensure_main called with zero Nw or Nk")
        self.ds_tens = _ensure(
            self.f,
            "green",
            shape=(Nw, Nk, self.nq, self.m, self.m),
            maxshape=(None, None, self.nq, self.m, self.m),
            chunks=(1, 1, 1, self.B, self.B),
            dtype=self.dtype,
            compression="gzip",
            shuffle=True,
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

    def close(self):
        """Close the underlying HDF5 file."""
        with contextlib.suppress(Exception):
            self.f.close()

    def indices(self, w, k, atol=0.0):
        """Find or append indices for given omega and k values.

        Parameters
        ----------
        w : float
            Frequency value (omega).
        k : float
            Wavevector magnitude (k).
        atol : float, optional
            Absolute tolerance for comparing existing values.

        Returns
        -------
        (int, int)
            Tuple `(iw, ik)` of indices in the omega and k datasets.

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

    def find_indices(self, w, k, atol=0.0):
        """Find indices for given omega and k values; raise KeyError if not missing.

        Parameters
        ----------
        w : float
            Frequency value (omega).
        k : float
            Wavevector magnitude (k).
        atol : float, optional
            Absolute tolerance for comparing existing values.

        Returns
        -------
        (int, int)
            Tuple `(iw, ik)` of indices in the omega and k datasets.

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

    def has(self, w, k, q, atol=0.0) -> bool:
        """Check whether a Green's operator block exists for given indices.

        Parameters
        ----------
        w : float
            Frequency value.
        k : float
            Wavevector magnitude.
        q : int
            Q-point index.
        atol : float, optional
            Absolute tolerance for matching omega/k.

        Returns
        -------
        bool
            True if the block exists, False otherwise.

        """
        iw, ik = self.indices(w, k, atol=atol)
        if not self._have_main:
            return False
        return bool(self.ds_mask[iw, ik, int(q)])

    def get(self, w, k, q, as_gpu=True, atol=0.0):
        """Retrieve a stored Green's operator block.

        Parameters
        ----------
        w : float
            Frequency value.
        k : float
            Wavevector magnitude.
        q : int
            Q-point index.
        as_gpu : bool, optional
            If True, return a CuPy array; otherwise return a NumPy array. Default is True.
        atol : float, optional
            Absolute tolerance for matching omega/k. Default is 0.0.

        Returns
        -------
        numpy.ndarray or cupy.ndarray
            The `(m, m)` operator block.

        """
        iw, ik = self.indices(w, k, atol=atol)
        if not self._have_main or not bool(self.ds_mask[iw, ik, int(q)]):
            raise KeyError(f"Requested block w={w}, k={k}, q={q} not found.")
        out = self.ds_tens[iw, ik, int(q), :, :][...]  # NumPy array
        return cp.asarray(out) if as_gpu else out

    def get_bz_block(self, w, k, as_gpu=True, atol=0.0):
        """Retrieve all stored blocks of the full Brillouin zone for a specific (omega, k) pair.

        Parameters
        ----------
        w : float
            Frequency value.
        k : float
            Wavevector magnitude.
        as_gpu : bool, optional
            If True, return a CuPy array; otherwise return a NumPy array. Default is True.
        atol : float, optional
            Absolute tolerance for matching omega/k. Default is 0.0.

        Returns
        -------
        numpy.ndarray or cupy.ndarray
            The `(nq, m, m)` array of operator blocks for the specified (w, k).

        """
        iw, ik = self.find_indices(w, k, atol=atol)
        tens = self.ds_tens[iw, ik, :, :, :]
        mask = self.ds_mask[iw, ik, :]
        if not np.all(mask):
            raise KeyError(f"Missing q-point blocks for w={w}, k={k}.")
        return cp.asarray(tens) if as_gpu else tens

    def put(self, w, k, q, data, atol=0.0, flush=True):
        """Store a Green's operator block.

        Parameters
        ----------
        w : float
            Frequency value.
        k : float
            Wavevector magnitude.
        q : int
            Q-point index.
        data : numpy.ndarray or cupy.ndarray
            Operator block to store; must have shape `(m, m)`.
        atol : float, optional
            Absolute tolerance for matching omega/k. Default is 0.0.
        flush : bool, optional
            If True, flush the file to disk after writing. Default is True.

        Raises
        ------
        ValueError
            If the shape of `data` does not match `(m, m)`.

        """
        iw, ik = self.indices(w, k, atol=atol)
        # ensure main exists (it will after indices())
        if not self._have_main:
            self._ensure_main()
        arr = data.get() if isinstance(data, cp.ndarray) else data
        if arr.shape != (self.m, self.m):
            raise ValueError(f"Data shape {arr.shape} != {(self.m, self.m)}.")
        if arr.dtype != self.ds_tens.dtype:
            arr = arr.astype(self.ds_tens.dtype, copy=False)
        self.ds_tens[iw, ik, int(q), :, :] = arr
        self.ds_mask[iw, ik, int(q)] = 1
        if flush:
            self.f.flush()

    def put_bz_block(self, w, k, data, atol=0.0, flush=True):
        """Store all clocks of the Brillouin zone for a specific (omega, k) pair.

        Parameters
        ----------
        w : float
            Frequency value.
        k : float
            Wavevector magnitude.
        data : numpy.ndarray or cupy.ndarray or greenWTE.green.RTAGreenOperator
            Array of operator blocks to store; must have shape `(nq, m, m)`.
        atol : float, optional
            Absolute tolerance for matching omega/k. Default is 0.0.
        flush : bool, optional
            If True, flush the file to disk after writing. Default is True.

        Raises
        ------
        ValueError
            If the shape of `data` does not match `(nq, m, m)`.

        """
        iw, ik = self.indices(w, k, atol=atol)
        # ensure main exists (it will after indices())
        if not self._have_main:
            self._ensure_main()
        if isinstance(data, GreenOperatorLike):
            arr = data._green
        arr = arr.get() if isinstance(arr, cp.ndarray) else arr
        if arr.shape != (self.nq, self.m, self.m):
            raise ValueError(f"Data shape {arr.shape} != {(self.nq, self.m, self.m)}.")
        if arr.dtype != self.ds_tens.dtype:
            arr = arr.astype(self.ds_tens.dtype, copy=False)
        self.ds_tens[iw, ik, :, :, :] = arr
        self.ds_mask[iw, ik, :] = 1
        if flush:
            self.f.flush()

    @property
    def omegas(self):
        """Return all stored omega values."""
        return self.ds_w[...]

    @property
    def ks(self):
        """Return all stored k values."""
        return self.ds_k[...]
