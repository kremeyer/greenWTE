"""Test cases for the full Green's function part of the greenWTE package."""

from os.path import join as pj

import pytest

from greenWTE import HAVE_GPU, xp
from greenWTE.base import Material
from greenWTE.green import DiskGreenOperator, GreenWTESolver, RTAGreenOperator, RTAWignerOperator
from greenWTE.io import GreenContainer
from greenWTE.iterative import IterativeWTESolver
from greenWTE.sources import source_term_full, source_term_gradT

from .defaults import (
    CSPBBR3_INPUT_PATH,
    DEFAULT_TEMPERATURE,
    DEFAULT_TEMPORAL_FREQUENCY,
    DEFAULT_THERMAL_GRATING,
    SI_INPUT_PATH,
)


def test_rtawigneroperator_basic():
    """Test the basic functionality of the RTAWignerOperator."""
    material = Material.from_phono3py(SI_INPUT_PATH, DEFAULT_TEMPERATURE)

    rwo = RTAWignerOperator(omg_ft=DEFAULT_TEMPORAL_FREQUENCY, k_ft=DEFAULT_THERMAL_GRATING, material=material)

    # test error before computing
    with pytest.raises(RuntimeError, match="Wigner operator not computed yet."):
        _ = rwo[0]
    with pytest.raises(RuntimeError, match="Wigner operator not computed yet."):
        for _ in rwo:
            pass
    # test len
    assert len(rwo) == material.nq

    rwo.compute()
    # test iteration and indexing
    _ = rwo[0]
    for _ in rwo:
        pass


def test_rtagreenoperator_basic():
    """Test the basic functionality of the RTAGreenOperator."""
    material = Material.from_phono3py(SI_INPUT_PATH, DEFAULT_TEMPERATURE)

    rwo = RTAWignerOperator(omg_ft=DEFAULT_TEMPORAL_FREQUENCY, k_ft=DEFAULT_THERMAL_GRATING, material=material)
    rwo.compute()
    rgo = RTAGreenOperator(rwo)

    # test error before computing
    with pytest.raises(RuntimeError, match="Green's operator not computed yet."):
        _ = rgo[0]

    rgo.compute()
    # test matmul
    _ = rgo @ xp.empty((rgo.nq, rgo.nat3**2, rgo.nat3**2))

    # test iter
    for _ in rgo:
        pass

    # test shape
    assert rgo.shape == (rgo.nq, rgo.nat3**2, rgo.nat3**2)

    # test dtype
    assert rwo.dtypec == rgo.dtype

    # test free
    rgo.free()
    assert rgo._green is None
    rgo.free()


def test_rta_green_operator_consistency():
    """Test the consistency of the RTA Green's operator with the Wigner operator.

    L * G = I.
    """
    material = Material.from_phono3py(SI_INPUT_PATH, DEFAULT_TEMPERATURE)

    rwo = RTAWignerOperator(
        omg_ft=xp.array([DEFAULT_TEMPORAL_FREQUENCY]), k_ft=DEFAULT_THERMAL_GRATING, material=material
    )
    rwo.compute()

    rgo = RTAGreenOperator(rwo)
    rgo.compute(clear_wigner=False)

    identity = xp.eye(rgo.nat3**2)
    for iq in range(len(rgo)):
        assert xp.allclose(rwo[iq] @ rgo[iq], identity, atol=1e-12, rtol=1e-12)


def test_solver_with_green_container(tmp_path):
    """Test the GreenWTESolver with a GreenContainer."""
    material = Material.from_phono3py(SI_INPUT_PATH, DEFAULT_TEMPERATURE)

    source = source_term_gradT(
        DEFAULT_THERMAL_GRATING,
        material.velocity_operator,
        material.phonon_freq,
        material.linewidth,
        material.heat_capacity,
        material.volume,
    )

    rwo = RTAWignerOperator(
        omg_ft=xp.array([DEFAULT_TEMPORAL_FREQUENCY]), k_ft=DEFAULT_THERMAL_GRATING, material=material
    )
    rwo.compute()

    rgo = RTAGreenOperator(rwo)
    rgo.compute()

    with GreenContainer(path=pj(tmp_path, "test-green.hdf5"), nat3=material.nat3, nq=material.nq) as gc:
        gc.put_bz_block(DEFAULT_TEMPORAL_FREQUENCY, DEFAULT_THERMAL_GRATING, rgo)

    with GreenContainer(path=pj(tmp_path, "test-green.hdf5"), nat3=material.nat3, nq=material.nq) as gc:
        dgo = DiskGreenOperator(gc, DEFAULT_TEMPORAL_FREQUENCY, DEFAULT_THERMAL_GRATING, material)

        green_solver = GreenWTESolver(
            omg_ft_array=xp.array([DEFAULT_TEMPORAL_FREQUENCY]),
            k_ft=DEFAULT_THERMAL_GRATING,
            material=material,
            source=source,
            outer_solver="root",
            greens=[dgo],
        )
        green_solver.run()


@pytest.mark.parametrize(
    "material_path",
    [
        pytest.param(SI_INPUT_PATH),
        pytest.param(CSPBBR3_INPUT_PATH, marks=pytest.mark.skipif(not HAVE_GPU, reason="CSPbBr3 test requires GPU")),
    ],
)
def test_green_vs_iterative_solver_gradient(material_path):
    """Test the Green's operator against an iterative solver and ensure that the Wigner distribution functions match."""
    material = Material.from_phono3py(
        material_path, DEFAULT_TEMPERATURE, dir_idx=0, dtyper=xp.float32, dtypec=xp.complex64
    )

    source = source_term_gradT(
        DEFAULT_THERMAL_GRATING,
        material.velocity_operator,
        material.phonon_freq,
        material.linewidth,
        material.heat_capacity,
        material.volume,
    )

    iterative_solver = IterativeWTESolver(
        omg_ft_array=xp.array([DEFAULT_TEMPORAL_FREQUENCY]),
        k_ft=DEFAULT_THERMAL_GRATING,
        material=material,
        source=source,
        source_type="gradient",
        outer_solver="none",
        inner_solver="gmres",
    )

    iterative_solver.run()

    rwo = RTAWignerOperator(
        omg_ft=xp.array([DEFAULT_TEMPORAL_FREQUENCY]), k_ft=DEFAULT_THERMAL_GRATING, material=material
    )
    rwo.compute()

    rgo = RTAGreenOperator(rwo)
    rgo.compute(clear_wigner=True)

    green_solver = GreenWTESolver(
        omg_ft_array=xp.array([DEFAULT_TEMPORAL_FREQUENCY]),
        k_ft=DEFAULT_THERMAL_GRATING,
        material=material,
        source=source,
        source_type="gradient",
        outer_solver="none",
        greens=[rgo],
    )
    green_solver.run()

    xp.testing.assert_allclose(iterative_solver.dT, green_solver.dT, atol=2e-7, rtol=2e-7)
    xp.testing.assert_allclose(iterative_solver.n[0], green_solver.n[0], atol=2e-7, rtol=2e-7)


@pytest.mark.parametrize("outer_solver", ["root", "aitken", "plain"])
@pytest.mark.parametrize(
    "material_path",
    [
        pytest.param(SI_INPUT_PATH),
        pytest.param(CSPBBR3_INPUT_PATH, marks=pytest.mark.skipif(not HAVE_GPU, reason="CSPbBr3 test requires GPU")),
    ],
)
def test_green_vs_iterative_energy(outer_solver, material_path):
    """Test the Green's operator against an iterative solver and ensure that the Wigner distribution functions match."""
    conv_thr_rel = 1e-4
    conv_thr_abs = 0

    material = Material.from_phono3py(
        material_path, DEFAULT_TEMPERATURE, dir_idx=0, dtyper=xp.float32, dtypec=xp.complex64
    )

    source = source_term_full(material.heat_capacity)

    iterative_solver = IterativeWTESolver(
        omg_ft_array=xp.array([DEFAULT_TEMPORAL_FREQUENCY]),
        k_ft=DEFAULT_THERMAL_GRATING,
        material=material,
        source=source,
        source_type="energy",
        outer_solver=outer_solver,
        inner_solver="gmres",
        conv_thr_rel=conv_thr_rel,
        conv_thr_abs=conv_thr_abs,
    )

    iterative_solver.run()

    rwo = RTAWignerOperator(
        omg_ft=xp.array([DEFAULT_TEMPORAL_FREQUENCY]), k_ft=DEFAULT_THERMAL_GRATING, material=material
    )
    rwo.compute()

    rgo = RTAGreenOperator(rwo)
    rgo.compute(clear_wigner=True)

    green_solver = GreenWTESolver(
        omg_ft_array=xp.array([DEFAULT_TEMPORAL_FREQUENCY]),
        k_ft=DEFAULT_THERMAL_GRATING,
        material=material,
        source=source,
        source_type="energy",
        outer_solver=outer_solver,
        greens=[rgo],
        conv_thr_rel=conv_thr_rel,
        conv_thr_abs=conv_thr_abs,
    )
    green_solver.run()

    # add small atol here, because near-zero elements can have large relative errors
    xp.testing.assert_allclose(iterative_solver.dT, green_solver.dT, rtol=conv_thr_rel, atol=1e-7)
    xp.testing.assert_allclose(iterative_solver.n[0], green_solver.n[0], rtol=conv_thr_rel, atol=1e-7)


def test_diskgreenoperator_attribute_caching():
    """Test that attributes are cached properly."""
    # bypass __init__
    dgo = object.__new__(DiskGreenOperator)

    arb_val = xp.asarray([1.0, 2.0, 3.0])
    dgo._green = arb_val
    dgo.compute()  # make sure this does not touch the set value
    assert dgo._green is arb_val
