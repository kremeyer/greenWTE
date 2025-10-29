"""Test base functionality of classes."""

import numpy as np
import pytest

from greenWTE import xp
from greenWTE.base import AitkenAccelerator, Material, dT_to_N_iterative, dT_to_N_matmul, estimate_initial_dT
from greenWTE.green import GreenWTESolver, RTAGreenOperator, RTAWignerOperator
from greenWTE.iterative import IterativeWTESolver
from greenWTE.sources import source_term_gradT

from . import _final_residual_and_scale
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

    # check that we can iterate
    for _ in m:
        pass


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
    assert estimate_initial_dT(0, history) == xp.asarray((1.0 + 1.0j))

    history.append((1, 3.0 + 3.0j))
    history.append((0, 1.0 + 1.0j))
    # interpolation
    assert estimate_initial_dT(0.5, history) == xp.asarray((2.0 + 2.0j))

    # extrapolation
    assert estimate_initial_dT(1.5, history) == xp.asarray((4.0 + 4.0j))


def test_wrong_outer_solver_error():
    """Test that an error is raised when using an invalid outer solver."""
    material = Material.from_phono3py(SI_INPUT_PATH, DEFAULT_TEMPERATURE)

    source = xp.empty((material.nq, material.nat3, material.nat3))

    solver = IterativeWTESolver(
        omg_ft_array=xp.array([DEFAULT_TEMPORAL_FREQUENCY]),
        k_ft=DEFAULT_THERMAL_GRATING,
        material=material,
        source=source,
        outer_solver="invalid",
        inner_solver="cgesv",
    )

    with pytest.raises(ValueError, match="Unknown outer solver: invalid"):
        solver.run()

    rwo = RTAWignerOperator(
        omg_ft=xp.array([DEFAULT_TEMPORAL_FREQUENCY]), k_ft=DEFAULT_THERMAL_GRATING, material=material
    )
    rwo.compute()

    rgo = RTAGreenOperator(rwo)
    rgo.compute(clear_wigner=True)

    with pytest.raises(ValueError, match="Unknown outer solver: invalid"):
        solver = GreenWTESolver(
            omg_ft_array=xp.array([DEFAULT_TEMPORAL_FREQUENCY]),
            k_ft=DEFAULT_THERMAL_GRATING,
            material=material,
            source=source,
            outer_solver="invalid",
            greens=[rgo],
        )
        solver.run()


def test_wrong_inner_solver_error():
    """Test that an error is raised when using an invalid inner solver."""
    material = Material.from_phono3py(SI_INPUT_PATH, DEFAULT_TEMPERATURE)

    source = xp.empty((material.nq, material.nat3, material.nat3))

    solver = IterativeWTESolver(
        omg_ft_array=xp.array([DEFAULT_TEMPORAL_FREQUENCY]),
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

    source = xp.empty((material.nq, material.nat3, material.nat3))

    with pytest.raises(ValueError, match="Unknown source type: invalid"):
        IterativeWTESolver(
            omg_ft_array=xp.array([DEFAULT_TEMPORAL_FREQUENCY]),
            k_ft=DEFAULT_THERMAL_GRATING,
            material=material,
            source=source,
            source_type="invalid",
        )

    with pytest.raises(ValueError, match="Unknown source type: invalid"):
        dT_to_N_iterative(0 + 0j, 0, 0, material, source, "invalid")

    rwo = RTAWignerOperator(
        omg_ft=xp.array([DEFAULT_TEMPORAL_FREQUENCY]), k_ft=DEFAULT_THERMAL_GRATING, material=material
    )
    rwo.compute()

    rgo = RTAGreenOperator(rwo)
    rgo.compute(clear_wigner=True)

    with pytest.raises(ValueError, match="Unknown source type: invalid"):
        GreenWTESolver(
            omg_ft_array=xp.array([DEFAULT_TEMPORAL_FREQUENCY]),
            k_ft=DEFAULT_THERMAL_GRATING,
            material=material,
            source=source,
            source_type="invalid",
            outer_solver="none",
            greens=[rgo],
        )

    with pytest.raises(ValueError, match="Unknown source type: invalid"):
        dT_to_N_matmul(0 + 0j, material, rgo, source, "invalid")


def test_root_not_converged():
    """Test that the convergence check at the end of the root routine is working."""
    conv_thr_rel = 1e-20  # beyond machine precision
    conv_thr_abs = 0.0

    material = Material.from_phono3py(
        SI_INPUT_PATH, DEFAULT_TEMPERATURE, dir_idx=0, dtyper=xp.float32, dtypec=xp.complex64
    )
    source = source_term_gradT(
        DEFAULT_THERMAL_GRATING,
        material.velocity_operator,
        material.phonon_freq,
        material.linewidth,
        material.heat_capacity,
        material.volume,
    )

    solver = IterativeWTESolver(
        omg_ft_array=xp.array([DEFAULT_TEMPORAL_FREQUENCY]),
        k_ft=DEFAULT_THERMAL_GRATING,
        material=material,
        source=source,
        outer_solver="root",
        inner_solver="cgesv",
        conv_thr_abs=conv_thr_abs,
        conv_thr_rel=conv_thr_rel,
        max_iter=25,
    )
    solver.run()
    niter_root = solver.niter
    print(niter_root)

    r_abs, scale, dT_final, _ = _final_residual_and_scale(solver, material)
    thresh = conv_thr_abs + conv_thr_rel * scale
    assert r_abs >= thresh, f"|F|={r_abs:.3e} > rtol*scale={thresh:.3e} (|dT|~{abs(dT_final):.3e})"


def test_solver_not_run_error():
    """Test that an error is raised when some properties are accessed before the solver has been run."""
    material = Material.from_phono3py(SI_INPUT_PATH, DEFAULT_TEMPERATURE)

    source = xp.empty((material.nq, material.nat3, material.nat3))

    solver = IterativeWTESolver(
        omg_ft_array=xp.array([DEFAULT_TEMPORAL_FREQUENCY]),
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


def test_printing_options_iterative(capfd):
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
        omg_ft_array=xp.array([DEFAULT_TEMPORAL_FREQUENCY]),
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
    assert out.startswith(".")
    assert "[1/1] k=1.00e+04 w=0.00e+00 dT= 1.00e+00+1.00e+00j n_it=1" in out
    assert err == ""

    # two omega points will print not progress of iterations
    solver = IterativeWTESolver(
        omg_ft_array=xp.array([DEFAULT_TEMPORAL_FREQUENCY, DEFAULT_TEMPORAL_FREQUENCY * 2]),
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
        omg_ft_array=xp.array([DEFAULT_TEMPORAL_FREQUENCY, DEFAULT_TEMPORAL_FREQUENCY * 2]),
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


def test_printing_options_green(capfd):
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

    rwo1 = RTAWignerOperator(
        omg_ft=xp.array([DEFAULT_TEMPORAL_FREQUENCY]), k_ft=DEFAULT_THERMAL_GRATING, material=material
    )
    rwo1.compute()
    rgo1 = RTAGreenOperator(rwo1)
    rgo1.compute(clear_wigner=True)

    rwo2 = RTAWignerOperator(
        omg_ft=xp.array([DEFAULT_TEMPORAL_FREQUENCY * 2]), k_ft=DEFAULT_THERMAL_GRATING, material=material
    )
    rwo2.compute()
    rgo2 = RTAGreenOperator(rwo1)
    rgo2.compute(clear_wigner=True)

    # one omega point will print progress of iterations
    solver = GreenWTESolver(
        omg_ft_array=xp.array([DEFAULT_TEMPORAL_FREQUENCY]),
        k_ft=DEFAULT_THERMAL_GRATING,
        material=material,
        greens=[rgo1],
        source=source,
        source_type="gradient",
        outer_solver="none",
        print_progress=True,
    )

    solver.run()
    out, err = capfd.readouterr()
    assert "[1/1] k=1.00e+04 w=0.00e+00 dT= 1.00e+00+1.00e+00j n_it=1" in out
    assert err == ""

    # two omega points will print not progress of iterations
    solver = GreenWTESolver(
        omg_ft_array=xp.array([DEFAULT_TEMPORAL_FREQUENCY, DEFAULT_TEMPORAL_FREQUENCY * 2]),
        k_ft=DEFAULT_THERMAL_GRATING,
        material=material,
        greens=[rgo1, rgo2],
        source=source,
        source_type="gradient",
        outer_solver="none",
        print_progress=True,
    )

    solver.run()
    out, err = capfd.readouterr()
    assert "[1/2] k=1.00e+04 w=0.00e+00 dT= 1.00e+00+1.00e+00j n_it=1" in out
    assert err == ""

    # no printing
    solver = GreenWTESolver(
        omg_ft_array=xp.array([DEFAULT_TEMPORAL_FREQUENCY, DEFAULT_TEMPORAL_FREQUENCY * 2]),
        k_ft=DEFAULT_THERMAL_GRATING,
        material=material,
        greens=[rgo1, rgo2],
        source=source,
        source_type="gradient",
        outer_solver="none",
        print_progress=False,
    )

    solver.run(free=False)  # just to have this covered somewhere
    out, err = capfd.readouterr()
    assert out == ""
    assert err == ""


def test_omega_green_length_mismatch():
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

    rwo = RTAWignerOperator(
        omg_ft=xp.array([DEFAULT_TEMPORAL_FREQUENCY]), k_ft=DEFAULT_THERMAL_GRATING, material=material
    )
    rwo.compute()
    rgo = RTAGreenOperator(rwo)
    rgo.compute(clear_wigner=True)

    with pytest.raises(ValueError, match="Number of Green's operators must match the number of omg_ft values."):
        GreenWTESolver(
            omg_ft_array=xp.array([DEFAULT_TEMPORAL_FREQUENCY, DEFAULT_TEMPORAL_FREQUENCY * 2]),
            k_ft=DEFAULT_THERMAL_GRATING,
            material=material,
            greens=[rgo],
            source=source,
            source_type="gradient",
            outer_solver="none",
            print_progress=False,
        )


def test_basesolver_attribute_caching():
    """Test that attributes are cached properly."""
    # bypass __init__
    solver = object.__new__(IterativeWTESolver)

    cached = xp.asarray([1.0, 2.0, 3.0])
    solver._kappa = cached
    out = solver.kappa
    assert out is cached

    cached = xp.asarray([4.0, 5.0, 6.0])
    solver._kappa_p = cached
    out = solver.kappa_p
    assert out is cached

    cached = xp.asarray([7.0, 8.0, 9.0])
    solver._kappa_c = cached
    out = solver.kappa_c
    assert out is cached


def test_safe_divide():
    """Test that _safe_divide can handle numpy and cupy arrays and zero-divisions are caught."""
    from greenWTE.base import _safe_divide

    # numpy arrays
    num = np.array([1.0, 2.0, 3.0])
    den = np.array([1.0, 0.0, 3.0])
    out = _safe_divide(num, den)
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, np.array([1.0, 0.0, 1.0]))

    # cupy arrays
    num = xp.array([1.0, 2.0, 3.0])
    den = xp.array([1.0, 0.0, 3.0])
    out = _safe_divide(num, den)
    assert isinstance(out, xp.ndarray)
    assert xp.allclose(out, xp.array([1.0, 0.0, 1.0]))

    # mixed arrays
    num = np.array([1.0, 2.0, 3.0])
    den = xp.array([1.0, 0.0, 3.0])
    out = _safe_divide(num, den)
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, np.array([1.0, 0.0, 1.0]))

    num = xp.array([1.0, 2.0, 3.0])
    den = np.array([1.0, 0.0, 3.0])
    out = _safe_divide(num, den)
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, np.array([1.0, 0.0, 1.0]))
