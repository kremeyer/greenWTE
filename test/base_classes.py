"""Test base functionality of classes."""

import cupy as cp
import pytest
from greenWTE.base import AitkenAccelerator, Material, estimate_initial_dT
from greenWTE.iterative import IterativeWTESolver
from greenWTE.sources import source_term_gradT

from .defaults import DEFAULT_TEMPERATURE, DEFAULT_TEMPORAL_FREQUENCY, DEFAULT_THERMAL_GRATING, SI_INPUT_PATH


def test_material():
    """Test helpers. Actual functionality is inherently tested in actual simulations."""
    m = Material.from_phono3py(SI_INPUT_PATH, DEFAULT_TEMPERATURE)

    # check that all info is printed
    representation = m.__repr__()
    assert f"{DEFAULT_TEMPERATURE}K" in representation
    assert SI_INPUT_PATH in representation
    assert f"{m.nq} qpoints" in representation
    assert f"{m.nat3} modes" in representation

    # check that we can index the class
    m_indexed = m[0]
    assert m_indexed.nq == 1
    assert m_indexed.nat3 == m.nat3
    assert m_indexed.temperature == m.temperature
    assert m_indexed.velocity_operator.shape[1:] == m.velocity_operator.shape[1:]
    assert m_indexed.phonon_freq.shape[1:] == m.phonon_freq.shape[1:]
    assert m_indexed.linewidth.shape[1:] == m.linewidth.shape[1:]
    assert m_indexed.heat_capacity.shape[1:] == m.heat_capacity.shape[1:]
    assert m_indexed.volume == m.volume
    assert m_indexed.name == m.name


def test_aitken_accelerator():
    """Test AitkenAccelerator class."""
    a = AitkenAccelerator()

    a.update(1 + 1j)
    a.update(2 + 2j)
    a.update(3 + 3j)

    a.reset()
    assert not a.history

    # tests edge case of small denominator dT2 - 2 * dT1 + dT0
    a.update(0)
    a.update(1)
    dT_accel = a.update(2)
    assert dT_accel == 2


def test_initial_dT_estimation():
    """Test the initial dT estimation function."""
    history = []
    # default value
    assert estimate_initial_dT(0, history) == cp.asarray((1.0 + 1.0j))

    history.append((1, 3.0 + 3.0j))
    history.append((0, 1.0 + 1.0j))
    # interpolation
    assert estimate_initial_dT(0.5, history) == cp.asarray((2.0 + 2.0j))

    # extrapolation
    assert estimate_initial_dT(1.5, history) == cp.asarray((4.0 + 4.0j))


def test_wrong_outer_solver_error():
    """Test that an error is raised when using an invalid outer solver."""
    material = Material.from_phono3py(SI_INPUT_PATH, DEFAULT_TEMPERATURE)

    source = cp.empty((material.nq, material.nat3, material.nat3))

    solver = IterativeWTESolver(
        omg_ft_array=cp.array([DEFAULT_TEMPORAL_FREQUENCY]),
        k_ft=DEFAULT_THERMAL_GRATING,
        material=material,
        source=source,
        outer_solver="invalid",
        inner_solver="cgesv",
    )

    with pytest.raises(ValueError, match="Unknown outer solver: invalid"):
        solver.run()


def test_wrong_inner_solver_error():
    """Test that an error is raised when using an invalid inner solver."""
    material = Material.from_phono3py(SI_INPUT_PATH, DEFAULT_TEMPERATURE)

    source = cp.empty((material.nq, material.nat3, material.nat3))

    solver = IterativeWTESolver(
        omg_ft_array=cp.array([DEFAULT_TEMPORAL_FREQUENCY]),
        k_ft=DEFAULT_THERMAL_GRATING,
        material=material,
        source=source,
        outer_solver="root",
        inner_solver="invalid",
    )

    with pytest.raises(ValueError, match="Unknown inner solver: invalid"):
        solver.run()


def test_wrong_source_type_error():
    """Test that an error is raised when the source is not a valid type."""
    material = Material.from_phono3py(SI_INPUT_PATH, DEFAULT_TEMPERATURE)

    source = cp.empty((material.nq, material.nat3, material.nat3))

    with pytest.raises(ValueError, match="Unknown source type: invalid"):
        IterativeWTESolver(
            omg_ft_array=cp.array([DEFAULT_TEMPORAL_FREQUENCY]),
            k_ft=DEFAULT_THERMAL_GRATING,
            material=material,
            source=source,
            source_type="invalid",
        )


def test_solver_not_run_error():
    """Test that an error is raised when some properties are accessed before the solver has been run."""
    material = Material.from_phono3py(SI_INPUT_PATH, DEFAULT_TEMPERATURE)

    source = cp.empty((material.nq, material.nat3, material.nat3))

    solver = IterativeWTESolver(
        omg_ft_array=cp.array([DEFAULT_TEMPORAL_FREQUENCY]),
        k_ft=DEFAULT_THERMAL_GRATING,
        material=material,
        source=source,
        outer_solver="root",
        inner_solver="invalid",
    )

    with pytest.raises(RuntimeError, match="Solver has not been run yet. Please run the solver first."):
        _ = solver.flux
    with pytest.raises(RuntimeError, match="Solver has not been run yet. Please run the solver first."):
        _ = solver.kappa
    with pytest.raises(RuntimeError, match="Solver has not been run yet. Please run the solver first."):
        _ = solver.kappa_c
    with pytest.raises(RuntimeError, match="Solver has not been run yet. Please run the solver first."):
        _ = solver.kappa_p


def test_printing_options(capfd):
    """Test that printing can be turned on and off."""
    material = Material.from_phono3py(SI_INPUT_PATH, DEFAULT_TEMPERATURE, dir_idx=0)

    source = source_term_gradT(
        DEFAULT_THERMAL_GRATING,
        material.velocity_operator,
        material.phonon_freq,
        material.linewidth,
        material.heat_capacity,
        material.volume,
    )

    # one omega point will print progress of iterations
    solver = IterativeWTESolver(
        omg_ft_array=cp.array([DEFAULT_TEMPORAL_FREQUENCY]),
        k_ft=DEFAULT_THERMAL_GRATING,
        material=material,
        source=source,
        source_type="gradient",
        outer_solver="none",
        inner_solver="cgesv",
        print_progress=True,
    )

    solver.run()
    out, err = capfd.readouterr()
    assert out.startswith("..")
    assert "[1/1] k=1.00e+04 w=0.00e+00 dT= 1.00e+00+1.00e+00j n_it=1" in out
    assert err == ""

    # two omega points will print not progress of iterations
    solver = IterativeWTESolver(
        omg_ft_array=cp.array([DEFAULT_TEMPORAL_FREQUENCY, DEFAULT_TEMPORAL_FREQUENCY * 2]),
        k_ft=DEFAULT_THERMAL_GRATING,
        material=material,
        source=source,
        source_type="gradient",
        outer_solver="none",
        inner_solver="cgesv",
        print_progress=True,
    )

    solver.run()
    out, err = capfd.readouterr()
    assert out.startswith("[1/2] k=1.00e+04 w=0.00e+00 dT= 1.00e+00+1.00e+00j n_it=1")
    assert err == ""

    # no printing
    solver = IterativeWTESolver(
        omg_ft_array=cp.array([DEFAULT_TEMPORAL_FREQUENCY, DEFAULT_TEMPORAL_FREQUENCY * 2]),
        k_ft=DEFAULT_THERMAL_GRATING,
        material=material,
        source=source,
        source_type="gradient",
        outer_solver="none",
        inner_solver="cgesv",
        print_progress=False,
    )

    solver.run()
    out, err = capfd.readouterr()
    assert out == ""
    assert err == ""
