"""Test cases for the full Green's function part of the greenWTE package."""

from os.path import join as pj

import cupy as cp
import pytest
from greenWTE.base import Material
from greenWTE.green import DiskGreenOperator, GreenWTESolver, RTAGreenOperator, RTAWignerOperator
from greenWTE.io import GreenContainer
from greenWTE.iterative import IterativeWTESolver
from greenWTE.sources import source_term_gradT

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


def test_rta_green_operator_consistency():
    """Test the consistency of the RTA Green's operator with the Wigner operator.

    L * G = I.
    """
    material = Material.from_phono3py(SI_INPUT_PATH, DEFAULT_TEMPERATURE)

    rwo = RTAWignerOperator(
        omg_ft=cp.array([DEFAULT_TEMPORAL_FREQUENCY]), k_ft=DEFAULT_THERMAL_GRATING, material=material
    )
    rwo.compute()

    rgo = RTAGreenOperator(rwo)
    rgo.compute(clear_wigner=False)

    identity = cp.eye(rgo.nat3**2)
    for iq in range(len(rgo)):
        assert cp.allclose(rwo[iq] @ rgo[iq], identity, atol=1e-12, rtol=1e-12)


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
        omg_ft=cp.array([DEFAULT_TEMPORAL_FREQUENCY]), k_ft=DEFAULT_THERMAL_GRATING, material=material
    )
    rwo.compute()

    rgo = RTAGreenOperator(rwo)
    rgo.compute()

    with GreenContainer(path=pj(tmp_path, "test-green.hdf5"), nat3=material.nat3, nq=material.nq) as gc:
        gc.put_bz_block(DEFAULT_TEMPORAL_FREQUENCY, DEFAULT_THERMAL_GRATING, rgo)

    with GreenContainer(path=pj(tmp_path, "test-green.hdf5"), nat3=material.nat3, nq=material.nq) as gc:
        dgo = DiskGreenOperator(gc, DEFAULT_TEMPORAL_FREQUENCY, DEFAULT_THERMAL_GRATING, material)

        green_solver = GreenWTESolver(
            omg_ft_array=cp.array([DEFAULT_TEMPORAL_FREQUENCY]),
            k_ft=DEFAULT_THERMAL_GRATING,
            material=material,
            source=source,
            outer_solver="root",
            greens=[dgo],
        )
        green_solver.run()


@pytest.mark.parametrize("material_path", [SI_INPUT_PATH, CSPBBR3_INPUT_PATH])
def test_green_vs_iterative_solver(material_path):
    """Test the Green's operator against an iterative solver and ensure that the Wigner distribution functions match."""
    material = Material.from_phono3py(
        material_path, DEFAULT_TEMPERATURE, dir_idx=0, dtyper=cp.float32, dtypec=cp.complex64
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
        omg_ft_array=cp.array([DEFAULT_TEMPORAL_FREQUENCY]),
        k_ft=DEFAULT_THERMAL_GRATING,
        material=material,
        source=source,
        outer_solver="none",
        inner_solver="gmres",
    )

    iterative_solver.run()

    rwo = RTAWignerOperator(
        omg_ft=cp.array([DEFAULT_TEMPORAL_FREQUENCY]), k_ft=DEFAULT_THERMAL_GRATING, material=material
    )
    rwo.compute()

    rgo = RTAGreenOperator(rwo)
    rgo.compute(clear_wigner=True)

    green_solver = GreenWTESolver(
        omg_ft_array=cp.array([DEFAULT_TEMPORAL_FREQUENCY]),
        k_ft=DEFAULT_THERMAL_GRATING,
        material=material,
        source=source,
        outer_solver="none",
        greens=[rgo],
    )
    green_solver.run()

    cp.testing.assert_allclose(iterative_solver.dT, green_solver.dT, atol=2e-7, rtol=2e-7)
    cp.testing.assert_allclose(iterative_solver.n[0], green_solver.n[0], atol=2e-7, rtol=2e-7)
