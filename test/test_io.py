"""Test cases for IO functionality of the greenWTE package."""

import json
from os.path import join as pj

import cupy as cp
import greenWTE.io as io_mod
import h5py
import numpy as np
import pytest
from greenWTE.base import Material
from greenWTE.green import RTAGreenOperator, RTAWignerOperator
from greenWTE.io import GreenContainer

from .defaults import DEFAULT_TEMPERATURE, DEFAULT_TEMPORAL_FREQUENCY, DEFAULT_THERMAL_GRATING, SI_INPUT_PATH


class _FakeCupyArray:
    """Minimal object that behaves like a CuPy array for .get() paths."""

    def __init__(self, arr):
        self._arr = np.asarray(arr)
    def get(self):
        return self._arr


@pytest.fixture(autouse=True)
def fake_cupy_and_no_bitshuffle(monkeypatch):
    """Replace cupy with a lightweight shim and disable bitshuffle usage inside _ensure_main."""
    class FakeCP:
        class ndarray:  # distinct from np.ndarray, so isinstance(np.array, cp.ndarray) is False
            pass
        complex128 = np.complex128
        float64 = np.float64
        pi = np.pi

        @staticmethod
        def array(x, dtype=None):
            return np.array(x, dtype=dtype)

        @staticmethod
        def asarray(x):
            return np.asarray(x)

        @staticmethod
        def dtype(x):
            return np.dtype(x)

        @staticmethod
        def sum(x, axis=None):
            return np.sum(x, axis=axis)

    monkeypatch.setattr(io_mod, "cp", FakeCP, raising=True)

    # patch _ensure_main to avoid bitshuffle filter
    def _ensure_main_no_bitshuffle(self):
        if self._have_main:
            return
        Nw, Nk = len(self.ds_w), len(self.ds_k)
        if Nw == 0 or Nk == 0:
            raise RuntimeError("internal: _ensure_main called with zero Nw or Nk")
        self.ds_tens = io_mod._ensure(
            self.f,
            "green",
            shape=(Nw, Nk, self.nq, self.m, self.m),
            maxshape=(None, None, self.nq, self.m, self.m),
            chunks=(1, 1, 1, self.B, self.B),
            dtype=self.dtype,
        )
        self.ds_mask = io_mod._ensure(
            self.f,
            "mask",
            shape=(Nw, Nk, self.nq),
            maxshape=(None, None, self.nq),
            chunks=(4, 4, min(256, self.nq)),
            dtype=np.uint8,
        )
        self._have_main = True

    monkeypatch.setattr(io_mod.GreenContainer, "_ensure_main", _ensure_main_no_bitshuffle, raising=True)


@pytest.fixture
def _empty_h5(tmp_path):
    path = tmp_path / "greens.h5"
    with h5py.File(path, "w"):
        pass
    return path


@pytest.fixture
def seeded_h5(tmp_path):
    """HDF5 file with nat3/nq attrs and matching dtype set on-disk."""
    path = tmp_path / "seeded.h5"
    with h5py.File(path, "w") as f:
        f.attrs["nat3"] = 2
        f.attrs["nq"] = 3
        f.attrs["m"] = 4
        f.attrs["schema"] = io_mod.SCHEMA
        f.attrs["dtype"] = "complex128"
    return path

def test__ensure_existing_dataset(tmp_path):
    """Test that _ensure does not overwrite existing datasets."""
    p = tmp_path / "t.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("x", shape=(1,), dtype=np.float64)
    with h5py.File(p, "a") as f:
        d1 = io_mod._ensure(f, "x", shape=(0,), maxshape=(None,), chunks=(1,), dtype=np.float64)
        assert d1.name.endswith("/x")


def test__find_or_append_1d_found_and_append(tmp_path):
    """Test that _find_or_append_1d finds or appends values correctly."""
    p = tmp_path / "t.h5"
    with h5py.File(p, "w") as f:
        d = f.create_dataset("a", shape=(0,), maxshape=(None,), dtype=np.float64)
        # append when not present
        i0 = io_mod._find_or_append_1d(d, 1.0)
        assert i0 == 0 and d.shape == (1,)
        # find when present (branch where allclose is True)
        i1 = io_mod._find_or_append_1d(d, 1.0)
        assert i1 == 0 and d.shape == (1,)


def test__find_index_1d_found_and_notfound(tmp_path):
    """Test that _find_index_1d finds or not finds values correctly."""
    p = tmp_path / "t.h5"
    with h5py.File(p, "w") as f:
        d = f.create_dataset("b", data=np.array([1.0, 2.0], dtype=np.float64))
        assert io_mod._find_index_1d(d, 2.0) == 1
        assert io_mod._find_index_1d(d, 3.14) == -1


def test_container_init_new_file_sets_attrs_and_meta(_empty_h5):
    """Test that the GreenContainer initializes correctly with a new file."""
    meta = {"a": 1}
    gc = io_mod.GreenContainer(str(_empty_h5), nat3=2, nq=3, dtype=io_mod.cp.complex128, meta=meta)
    try:
        assert gc.m == 4
        # root attrs exist and meta is JSON
        assert gc.f.attrs["schema"] == io_mod.SCHEMA
        assert json.loads(gc.f.attrs["meta"]) == meta
    finally:
        gc.close()


def test_container_init_reads_nat3_nq_from_file(seeded_h5):
    """Test that the GreenContainer initializes correctly with an existing file."""
    gc = io_mod.GreenContainer(str(seeded_h5), nat3=None, nq=None, dtype=io_mod.cp.complex128, read_only=False)
    try:
        assert gc.nat3 == 2 and gc.nq == 3 and gc.m == 4
    finally:
        gc.close()


def test_container_init_dtype_mismatch_raises(tmp_path):
    """Test that the GreenContainer raises a TypeError on dtype mismatch."""
    p = tmp_path / "dtype_mismatch.h5"
    with h5py.File(p, "w") as f:
        f.attrs["schema"] = io_mod.SCHEMA
        f.attrs["nat3"] = 2
        f.attrs["nq"] = 3
        f.attrs["m"] = 4
        f.attrs["dtype"] = "complex64"  # on-disk different
    with pytest.raises(TypeError):
        io_mod.GreenContainer(str(p), nat3=None, nq=None, dtype=io_mod.cp.complex128)


def test__ensure_main_early_return_and_zero_error(_empty_h5):
    """Test _ensure_main early return and zero-length error path."""
    gc = io_mod.GreenContainer(str(_empty_h5), nat3=2, nq=3)
    try:
        # Make it think main exists -> early return branch
        gc._have_main = True
        gc._ensure_main()  # should just return
        # Force the zero-length error path
        gc._have_main = False
        gc.ds_w.resize((0,))
        gc.ds_k.resize((0,))
        with pytest.raises(RuntimeError):
            gc._ensure_main()
    finally:
        gc.close()


def test_indices_resize_and_find_indices_paths(_empty_h5):
    """Test index resizing and finding indices."""
    gc = io_mod.GreenContainer(str(_empty_h5), nat3=2, nq=3)
    try:
        iw0, ik0 = gc.indices(1.0, 2.0)  # creates main
        assert (iw0, ik0) == (0, 0)
        # add new omega -> triggers resize
        iw1, ik1 = gc.indices(1.23, 2.0)
        assert iw1 == 1 and ik1 == 0
        # find_indices not found -> KeyError
        with pytest.raises(KeyError):
            gc.find_indices(9.9, 2.0)
    finally:
        gc.close()


def test_has_and_has_bz_block_variants(_empty_h5):
    """Test has and has_bz_block variants."""
    gc = io_mod.GreenContainer(str(_empty_h5), nat3=2, nq=3)
    try:
        # nothing stored yet
        assert gc.has(1.0, 2.0, q=0) is False
        # store one block
        mat = np.zeros((gc.m, gc.m), dtype=np.complex128)
        gc.put(1.0, 2.0, q=1, data=mat, flush=False)
        assert gc.has(1.0, 2.0, q=1) is True
        assert gc.has_bz_block(1.0, 2.0) is False  # not all q present
        # store full BZ
        full = np.zeros((gc.nq, gc.m, gc.m), dtype=np.complex128)
        gc.put_bz_block(1.0, 3.0, data=full, flush=False)
        assert gc.has_bz_block(1.0, 3.0) is True
        # keyerror branch in has_bz_block when w/k missing
        assert gc.has_bz_block(9.9, 9.9) is False
    finally:
        gc.close()


def test_get_and_get_bz_block_paths(_empty_h5):
    """Test get and get_bz_block paths."""
    gc = io_mod.GreenContainer(str(_empty_h5), nat3=2, nq=2)
    try:
        # get missing -> KeyError
        with pytest.raises(KeyError):
            gc.get(1.0, 2.0, q=0)
        # put one q, get as numpy and "gpu"
        mat = np.eye(gc.m, dtype=np.complex128)
        gc.put(1.0, 2.0, q=0, data=mat, flush=True)
        np_out = gc.get(1.0, 2.0, q=0, as_gpu=False)
        gpu_out = gc.get(1.0, 2.0, q=0, as_gpu=True)  # via cp.asarray shim
        assert np.allclose(np_out, gpu_out)
        # get_bz_block: first missing some q -> KeyError
        with pytest.raises(KeyError):
            gc.get_bz_block(1.0, 2.0)
        # now fill all q and get
        full = np.zeros((gc.nq, gc.m, gc.m), dtype=np.complex128)
        gc.put_bz_block(1.0, 2.0, data=full, flush=True)
        bz = gc.get_bz_block(1.0, 2.0, as_gpu=False)
        assert bz.shape == (gc.nq, gc.m, gc.m)
    finally:
        gc.close()


def test_put_and_put_bz_block_shape_dtype_and_flush(_empty_h5):
    """Test put and put_bz_block shape, dtype, and flush behavior."""
    gc = io_mod.GreenContainer(str(_empty_h5), nat3=2, nq=2)
    try:
        good = np.zeros((gc.m, gc.m), dtype=np.complex128)
        gc.put(0.1, 0.2, q=0, data=good, flush=False)  # hit flush=False branch
        # bad shape
        with pytest.raises(ValueError):
            gc.put(0.1, 0.2, q=1, data=np.zeros((gc.m, gc.m + 1), dtype=np.complex128))
        # bad dtype
        with pytest.raises(TypeError):
            gc.put(0.1, 0.2, q=1, data=np.zeros((gc.m, gc.m), dtype=np.float64))
        # put_bz_block: ensure flush=False branch and type/shape checks
        full = np.zeros((gc.nq, gc.m, gc.m), dtype=np.complex128)
        gc.put_bz_block(0.3, 0.4, data=full, flush=False)
        with pytest.raises(ValueError):
            gc.put_bz_block(0.5, 0.6, data=np.zeros((gc.nq + 1, gc.m, gc.m), dtype=np.complex128))
        with pytest.raises(TypeError):
            gc.put_bz_block(0.5, 0.6, data=np.zeros((gc.nq, gc.m, gc.m), dtype=np.float64))
        # Also exercise the GreenOperatorLike path
        class Wrapper:
            def __init__(self, arr):
                self._green = arr
        gc.put_bz_block(0.7, 0.8, data=Wrapper(full), flush=False)
    finally:
        gc.close()


def test_omegas_and_ks_filters(_empty_h5):
    """Test omegas and ks filtering."""
    gc = io_mod.GreenContainer(str(_empty_h5), nat3=2, nq=2)
    try:
        # store blocks at (w,k) = (1,10) and (2,20)
        base = np.zeros((gc.m, gc.m), dtype=np.complex128)
        gc.put(1.0, 10.0, q=0, data=base, flush=False)
        gc.put(2.0, 20.0, q=1, data=base, flush=False)
        # omegas: no arg -> all stored values
        ow_all = gc.omegas()
        assert set(np.asarray(ow_all)) == {1.0, 2.0}
        # omegas filtered by k
        assert gc.omegas(k=999.0) is None  # unknown k
        ow_k10 = gc.omegas(k=10.0)
        assert np.allclose(ow_k10, [1.0])
        # ks symmetric behavior
        kk_all = gc.ks()
        assert set(np.asarray(kk_all)) == {10.0, 20.0}
        assert gc.ks(w=999.0) is None
        kk_w2 = gc.ks(w=2.0)
        assert np.allclose(kk_w2, [20.0])
    finally:
        gc.close()


def test_load_phono3py_data_success_and_missing(tmp_path):
    """Test loading of Phono3py data with success and missing cases."""
    p = tmp_path / "ph3.h5"
    # craft minimal, consistent arrays
    temperature = [300.0]
    nq = 3
    nat3 = 2
    qpoint = np.arange(nq * 3).reshape(nq, 3)
    vel = np.zeros((nq, nat3, nat3, 1), dtype=np.complex128)  # last axis for dir_idx
    freq = np.ones((nq, nat3))
    gamma = np.zeros((len(temperature), nq, nat3))
    gamma_iso = np.zeros((nq, nat3))
    gamma_b = np.zeros((nq, nat3))
    volume = np.array(1.0)
    weight = np.ones((nq,))
    heat_cap = np.ones((len(temperature), nq, nat3))

    with h5py.File(p, "w") as f:
        f.create_dataset("temperature", data=np.array(temperature))
        f.create_dataset("qpoint", data=qpoint)
        f.create_dataset("velocity_operator_sym", data=vel)
        f.create_dataset("frequency", data=freq)
        f.create_dataset("gamma", data=gamma)
        f.create_dataset("gamma_isotope", data=gamma_iso)
        f.create_dataset("gamma_boundary", data=gamma_b)
        f.create_dataset("volume", data=volume)
        f.create_dataset("weight", data=weight)
        f.create_dataset("heat_capacity", data=heat_cap)

    # success path (exclude_gamma=True -> skip first q-point)
    qpt, vop, *_ = io_mod.load_phono3py_data(str(p), temperature=300.0, dir_idx=0)
    assert qpt.shape[0] == nq - 1 and vop.shape[-1] == nat3  # basic sanity

    # missing temperature -> ValueError
    with pytest.raises(ValueError):
        io_mod.load_phono3py_data(str(p), temperature=123.0, dir_idx=0)


def test_save_solver_result_branches(tmp_path):
    """Test saving of solver results with various branches."""
    p = tmp_path / "solver_out.h5"

    # fake "solver" object with required attributes
    class Mat:
        def __init__(self, x):
            self._x = np.asarray(x)
        def get(self):
            return self._x

    class Material:
        dtyper = np.float64
        dtypec = np.complex128

    class Args:
        # include 'k' to collide with an existing dataset name -> exercises the "continue" path
        def __init__(self, extra_cp, extra_py):
            self.k = 123  # will be skipped
            self.extra_array = extra_cp     # cp.ndarray-like -> .get() path
            self.extra_value = extra_py     # plain value

    cp_like = _FakeCupyArray(np.array([1, 2, 3]))

    class Solver:
        # arrays saved via .get()
        dT           = Mat([0.])
        dT_init      = Mat([1.])
        n            = Mat([2.])
        niter        = Mat([3])
        iter_time    = Mat([4.])
        gmres_residual = Mat([5.])
        dT_iterates  = Mat([6.])
        n_norms      = Mat([7.])
        source       = Mat([8.])
        # meta & params
        omg_ft_array = Mat([0.1, 0.2])
        k_ft         = 0.5
        max_iter     = 10
        conv_thr_rel = 1e-6
        conv_thr_abs = 1e-8
        material     = Material()
        outer_solver = "gmres"
        inner_solver = "ilu"
        kappa        = Mat([9.])
        kappa_p      = Mat([10.])
        kappa_c      = Mat([11.])
        command_line_args = Args(extra_cp=cp_like, extra_py=42)

    io_mod.save_solver_result(str(p), Solver(), kw_cp=cp_like, kw_plain=99)

    with h5py.File(p, "r") as f:
        # existing datasets
        assert "dT" in f and "kappa_C" in f and "omega" in f and "k" in f
        # command_line_args:
        #  - 'k' should have been skipped (dataset already present)
        assert "k" in f and f["k"][()].shape == ()  # original scalar dataset
        #  - extra_array stored via .get()
        assert "extra_array" in f and np.allclose(f["extra_array"][...], [1,2,3])
        #  - extra_value stored as plain scalar
        assert "extra_value" in f and f["extra_value"][()] == 42
        # kwargs handling for cp.ndarray-like and plain values
        assert "kw_cp" in f and np.allclose(f["kw_cp"][...], [1,2,3])
        assert "kw_plain" in f and f["kw_plain"][()] == 99


def test_green_container_io(tmp_path):
    """Test the I/O functionality of the GreenContainer."""
    material = Material.from_phono3py(SI_INPUT_PATH, DEFAULT_TEMPERATURE)

    rwo = RTAWignerOperator(omg_ft=DEFAULT_TEMPORAL_FREQUENCY, k_ft=DEFAULT_THERMAL_GRATING, material=material)
    rwo.compute()

    rgo = RTAGreenOperator(rwo)
    rgo.compute(clear_wigner=True)

    # check writing as full bz block
    with GreenContainer(path=pj(tmp_path, "test-green.hdf5"), nat3=material.nat3, nq=material.nq) as gc:
        gc.put_bz_block(DEFAULT_TEMPORAL_FREQUENCY, DEFAULT_THERMAL_GRATING, rgo)

    with GreenContainer(path=pj(tmp_path, "test-green.hdf5"), nat3=material.nat3, nq=material.nq) as gc:
        retrieved = gc.get_bz_block(DEFAULT_TEMPORAL_FREQUENCY, DEFAULT_THERMAL_GRATING)

    assert cp.allclose(retrieved, rgo.squeeze(), atol=1e-12, rtol=1e-12)
    assert retrieved.dtype == rgo.squeeze().dtype

    # check writing as individual q-point entries
    with GreenContainer(path=pj(tmp_path, "test-green.hdf5"), nat3=material.nat3, nq=material.nq) as gc:
        for iq in range(material.nq):
            gc.put(DEFAULT_TEMPORAL_FREQUENCY, DEFAULT_THERMAL_GRATING, iq, rgo[iq])

    with GreenContainer(path=pj(tmp_path, "test-green.hdf5"), nat3=material.nat3, nq=material.nq) as gc:
        retrieved = gc.get_bz_block(DEFAULT_TEMPORAL_FREQUENCY, DEFAULT_THERMAL_GRATING)

    assert cp.allclose(retrieved, rgo.squeeze(), atol=1e-12, rtol=1e-12)
    assert retrieved.dtype == rgo.squeeze().dtype