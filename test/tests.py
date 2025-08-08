"""Test suite for the greenWTE package."""

import warnings
from os.path import join as pj

import cupy as cp
import h5py
import pytest
from greenWTE.base import Material
from greenWTE.iterative import IterativeWTESolver
from greenWTE.solve import save_solver_result
from greenWTE.sources import source_term_gradT

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
            material.heat_capacity,
            material.volume,
        )

        solver = IterativeWTESolver(
            omg_ft_array=cp.array([DEFAULT_TEMPORAL_FREQUENCY]),
            k_ft=DEFAULT_THERMAL_GRATING,
            material=material,
            source=source,
            outer_solver="root",
            inner_solver="gmres",
            conv_thr=1e-14,
        )

        solver.run()
        solver.dT = cp.real(solver.dT)  # use the real part of the solution to compute thermal conductivities
        solver.n[0] = solver._dT_to_N(solver.dT, DEFAULT_TEMPORAL_FREQUENCY, 0)[0]
        kappas.append((cp.real(solver.kappa_p), cp.real(solver.kappa_c)))

    kappas = cp.array(kappas)
    cp.testing.assert_allclose(kappas[:, 0], cp.mean(kappas[:, 0]), atol=2)
    cp.testing.assert_allclose(kappas[:, 1], cp.mean(kappas[:, 1]), atol=2)


def test_cspbbr3_anisotropy():
    """Test the anisotropy of the thermal conductivity in CsPbBr3."""
    kappas = []
    for direction in range(3):
        material = Material.from_phono3py(CSPBBR3_INPUT_PATH, DEFAULT_TEMPERATURE, dir_idx=direction)

        source = source_term_gradT(
            DEFAULT_THERMAL_GRATING,
            material.velocity_operator,
            material.phonon_freq,
            material.heat_capacity,
            material.volume,
        )

        solver = IterativeWTESolver(
            omg_ft_array=cp.array([DEFAULT_TEMPORAL_FREQUENCY]),
            k_ft=DEFAULT_THERMAL_GRATING,
            material=material,
            source=source,
            outer_solver="root",
            inner_solver="gmres",
            conv_thr=1e-14,
        )

        solver.run()
        solver.dT = cp.real(solver.dT)  # use the real part of the solution to compute thermal conductivities
        solver.n[0] = solver._dT_to_N(solver.dT, DEFAULT_TEMPORAL_FREQUENCY, 0)[0]
        kappas.append((cp.real(solver.kappa_p), cp.real(solver.kappa_c)))

    kappas = cp.array(kappas)
    assert not cp.allclose(kappas[:, 0], kappas[:, 1], atol=0.1)
    assert not cp.allclose(kappas[:, 0], kappas[:, 1], atol=0.1)


@pytest.mark.parametrize("outer_solver", ["aitken", "plain", "root"])
def test_dT_convergence_cgesv(outer_solver):
    """Test the convergence of the dT solution with the CGESV solver."""
    tolerance = 1e-5

    material = Material.from_phono3py(SI_INPUT_PATH, DEFAULT_TEMPERATURE, dir_idx=0)

    source = source_term_gradT(
        DEFAULT_THERMAL_GRATING,
        material.velocity_operator,
        material.phonon_freq,
        material.heat_capacity,
        material.volume,
    )

    solver = IterativeWTESolver(
        omg_ft_array=cp.array([DEFAULT_TEMPORAL_FREQUENCY]),
        k_ft=DEFAULT_THERMAL_GRATING,
        material=material,
        source=source,
        outer_solver=outer_solver,
        inner_solver="cgesv",
        conv_thr=tolerance,
        max_iter=1000,
    )
    solver.run()

    last = float(cp.abs(cp.imag(solver.dT_convergence[0, -1])))
    prev = float(cp.abs(cp.imag(solver.dT_convergence[0, -2])))

    assert (last - prev) / prev < tolerance


def test_diag_velocity_operator():
    """Test that there is no thermal conductivity from the coherences if the velocity operator is diagonal."""
    material = Material.from_phono3py(SI_INPUT_PATH, DEFAULT_TEMPERATURE, dir_idx=0)
    offdiag_mask = ~cp.eye(material.velocity_operator.shape[1], dtype=cp.bool_)
    material.velocity_operator[:, offdiag_mask] = 0

    source = source_term_gradT(
        DEFAULT_THERMAL_GRATING,
        material.velocity_operator,
        material.phonon_freq,
        material.heat_capacity,
        material.volume,
    )

    solver = IterativeWTESolver(
        omg_ft_array=cp.array([DEFAULT_TEMPORAL_FREQUENCY]),
        k_ft=DEFAULT_THERMAL_GRATING,
        material=material,
        source=source,
        outer_solver="root",
        inner_solver="cgesv",
    )

    solver.run()

    cp.testing.assert_array_less(cp.real(solver.kappa_c), 1e-4)


def test_output_file_dimensions(tmp_path):
    """Test the dimensions of the output file created by the solver."""
    output_filename = pj(tmp_path, "test_output_file_dimensions.h5")

    material = Material.from_phono3py(SI_INPUT_PATH, DEFAULT_TEMPERATURE, dir_idx=0)

    source = source_term_gradT(
        DEFAULT_THERMAL_GRATING,
        material.velocity_operator,
        material.phonon_freq,
        material.heat_capacity,
        material.volume,
    )

    max_iter = 2  # we can't have the solver converge to be able to test the output file dimensions
    nw = 3
    nq, nat3 = material.phonon_freq.shape

    solver = IterativeWTESolver(
        omg_ft_array=cp.array([0, 1e3, 1e6]),
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
        assert "conv_thr" in h5f
        assert "dT" in h5f
        assert h5f["dT"].shape == (nw,)
        assert "dT_convergence" in h5f
        assert h5f["dT_convergence"].shape == (nw, max_iter)
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
        assert "n_convergence" in h5f
        assert h5f["n_convergence"].shape == (nw, max_iter)
        assert "niter" in h5f
        assert h5f["niter"].shape == (nw,)
        assert "omega" in h5f
        assert h5f["omega"].shape == (nw,)
        assert "outer_solver" in h5f
        assert "source" in h5f
        assert h5f["source"].shape == (nq, nat3, nat3)
        assert "temperature" in h5f
