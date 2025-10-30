"""Test suite for the greenWTE package."""

import warnings
from os.path import join as pj

import h5py
import pytest

from greenWTE import HAVE_GPU, xp
from greenWTE.base import Material
from greenWTE.io import save_solver_result
from greenWTE.iterative import IterativeWTESolver
from greenWTE.sources import source_term_full, source_term_gradT

from . import _final_residual_and_scale
from .defaults import (
    CSPBBR3_INPUT_PATH,
    DEFAULT_TEMPERATURE,
    DEFAULT_TEMPORAL_FREQUENCY,
    DEFAULT_THERMAL_GRATING,
    SI_INPUT_PATH,
)

# triggered when cupyx.scipy.sparse.linalg.gmres
# invokes np.linalg.lstsq under the hood
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=r".*`rcond` parameter will change to the default of machine precision.*"
)


def test_silicon_isotropy():
    """Test the isotropy of the thermal conductivity in silicon."""
    kappas = []
    for direction in range(3):
        material = Material.from_phono3py(SI_INPUT_PATH, DEFAULT_TEMPERATURE, dir_idx=direction)

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
            source_type="gradient",
            outer_solver="none",
            inner_solver="cgesv",
            conv_thr_rel=1e-14,
        )

        solver.run()
        kappas.append((xp.real(solver.kappa_p), xp.real(solver.kappa_c)))

    kappas = xp.array(kappas)
    xp.testing.assert_allclose(kappas[:, 0], xp.mean(kappas[:, 0]), rtol=0.01)


def test_high_frequency_roll_off():
    """Test that dT drops to zero at high frequencies."""
    omegas = xp.array([0, 1e16, 1e300])
    material = Material.from_phono3py(SI_INPUT_PATH, DEFAULT_TEMPERATURE, dir_idx=0)

    source = 5e8 * source_term_full(material.heat_capacity)

    solver = IterativeWTESolver(
        omg_ft_array=omegas,
        k_ft=DEFAULT_THERMAL_GRATING,
        material=material,
        source=source,
        outer_solver="aitken",
        inner_solver="cgesv",
        conv_thr_rel=1e-14,
    )

    solver.run()
    print(xp.abs(solver.dT), solver.dT)
    assert xp.abs(solver.dT[0]) > 1
    assert xp.abs(solver.dT[1]) < 1e-9
    assert xp.abs(solver.dT[2]) < 1e-15


@pytest.mark.skipif(not HAVE_GPU, reason="CSPbBr3 test requires GPU")
def test_cspbbr3_anisotropy():
    """Test the anisotropy of the thermal conductivity in CsPbBr3."""
    kappas = []
    for direction in range(3):
        material = Material.from_phono3py(CSPBBR3_INPUT_PATH, DEFAULT_TEMPERATURE, dir_idx=direction)

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
            source_type="gradient",
            outer_solver="none",
            inner_solver="gmres",
            conv_thr_rel=1e-14,
        )

        solver.run()
        kappas.append((xp.real(solver.kappa_p), xp.real(solver.kappa_c)))

    kappas = xp.array(kappas)
    assert not xp.allclose(kappas[:, 0], xp.mean(kappas[:, 0]), rtol=0.2)
    assert not xp.allclose(kappas[:, 1], xp.mean(kappas[:, 1]), rtol=0.2)


@pytest.mark.parametrize("outer_solver", ["aitken", "plain", "root"])
@pytest.mark.parametrize("inner_solver", ["cgesv", "gmres"])
def test_dT_convergence_absolute(outer_solver, inner_solver):
    """Converges under absolute tolerance only: |F(dT)| <= conv_thr_a."""
    conv_thr_abs = 1e-7 if outer_solver != "plain" else 1e-3  # plain is slow
    conv_thr_rel = 0.0  # disable relative tolerance

    material = Material.from_phono3py(SI_INPUT_PATH, DEFAULT_TEMPERATURE, dir_idx=0)
    source = 5e8 * source_term_full(material.heat_capacity)

    solver = IterativeWTESolver(
        omg_ft_array=xp.array([DEFAULT_TEMPORAL_FREQUENCY]),
        k_ft=DEFAULT_THERMAL_GRATING,
        material=material,
        source=source,
        outer_solver=outer_solver,
        inner_solver=inner_solver,
        conv_thr_abs=conv_thr_abs,
        conv_thr_rel=conv_thr_rel,
        max_iter=1000,
    )
    solver.run()

    r_abs, scale, *_ = _final_residual_and_scale(solver, material)
    assert r_abs <= conv_thr_abs, f"|F|={r_abs:.3e} > atol={conv_thr_abs:.3e} (scale={scale:.3e})"


@pytest.mark.parametrize("outer_solver", ["aitken", "plain", "root"])
@pytest.mark.parametrize("inner_solver", ["cgesv", "gmres"])
def test_dT_convergence_relative(outer_solver, inner_solver):
    """Converges under relative tolerance only: |F(dT)| <= conv_thr_r * max(|dT|,|dT_next|,1)."""
    conv_thr_rel = 1e-7 if outer_solver != "plain" else 1e-3  # plain is slow
    conv_thr_abs = 0.0  # disable absolute tolerance

    material = Material.from_phono3py(SI_INPUT_PATH, DEFAULT_TEMPERATURE, dir_idx=0)
    source = 5e8 * source_term_full(material.heat_capacity)

    solver = IterativeWTESolver(
        omg_ft_array=xp.array([DEFAULT_TEMPORAL_FREQUENCY]),
        k_ft=DEFAULT_THERMAL_GRATING,
        material=material,
        source=source,
        outer_solver=outer_solver,
        inner_solver=inner_solver,
        conv_thr_abs=conv_thr_abs,
        conv_thr_rel=conv_thr_rel,
        max_iter=1000,
    )
    solver.run()

    r_abs, scale, dT_final, _ = _final_residual_and_scale(solver, material)
    thresh = conv_thr_abs + conv_thr_rel * scale
    assert r_abs <= thresh, f"|F|={r_abs:.3e} > rtol*scale={thresh:.3e} (|dT|~{abs(dT_final):.3e})"


def test_diag_velocity_operator():
    """Test that there is no thermal conductivity from the coherences if the velocity operator is diagonal."""
    material = Material.from_phono3py(SI_INPUT_PATH, DEFAULT_TEMPERATURE, dir_idx=0)
    offdiag_mask = ~xp.eye(material.velocity_operator.shape[1], dtype=xp.bool_)
    material.velocity_operator[:, offdiag_mask] = 0

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
        source_type="gradient",
        outer_solver="root",
        inner_solver="cgesv",
    )

    solver.run()

    xp.testing.assert_array_less(xp.real(solver.kappa_c), 1e-12)


def test_output_file_dimensions(tmp_path):
    """Test the dimensions of the output file created by the solver."""
    output_filename = pj(tmp_path, "test_output_file_dimensions.h5")

    material = Material.from_phono3py(SI_INPUT_PATH, DEFAULT_TEMPERATURE, dir_idx=0)

    source = source_term_gradT(
        DEFAULT_THERMAL_GRATING,
        material.velocity_operator,
        material.phonon_freq,
        material.linewidth,
        material.heat_capacity,
        material.volume,
    )

    max_iter = 2  # we can't have the solver converge to be able to test the output file dimensions
    nw = 3
    nq, nat3 = material.phonon_freq.shape

    solver = IterativeWTESolver(
        omg_ft_array=xp.array([0, 1e3, 1e6]),
        k_ft=DEFAULT_THERMAL_GRATING,
        material=material,
        source=source,
        outer_solver="plain",
        inner_solver="cgesv",
        max_iter=max_iter,
    )

    solver.run()

    save_solver_result(output_filename, solver, temperature=DEFAULT_TEMPERATURE)

    with h5py.File(output_filename, "r") as h5f:
        assert "conv_thr_abs" in h5f
        assert "conv_thr_rel" in h5f
        assert "dT" in h5f
        assert h5f["dT"].shape == (nw,)
        assert "dT_iterates" in h5f
        assert h5f["dT_iterates"].shape == (nw, max_iter)
        assert "dtype_complex" in h5f
        assert "dtype_real" in h5f
        assert "gmres_residual" in h5f
        assert h5f["gmres_residual"].shape[:-1] == (nw, max_iter, nq)
        assert "inner_solver" in h5f
        assert "outer_solver" in h5f
        assert "iter_time" in h5f
        assert h5f["iter_time"].shape == (nw, max_iter)
        assert "k" in h5f
        assert "max_iter" in h5f
        assert "n" in h5f
        assert h5f["n"].shape == (nw, nq, nat3, nat3)
        assert "n_norms" in h5f
        assert h5f["n_norms"].shape == (nw, max_iter)
        assert "niter" in h5f
        assert h5f["niter"].shape == (nw,)
        assert "omega" in h5f
        assert h5f["omega"].shape == (nw,)
        assert "outer_solver" in h5f
        assert "source" in h5f
        assert h5f["source"].shape == (nq, nat3, nat3)
        assert "temperature" in h5f
