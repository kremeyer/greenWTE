"""Tests for the closed-form 'affine' outer solver."""

import numpy as np
import pytest

from greenWTE import to_cpu, xp
from greenWTE.base import (
    Material,
    N_to_dT,
    SolverBase,
    affine_remainder_rhs,
    dT_to_N_iterative,
    equilibrium_n_per_kelvin,
)
from greenWTE.green import GreenWTESolver, RTAGreenOperator, RTAWignerOperator
from greenWTE.iterative import IterativeWTESolver
from greenWTE.sources import source_term_diag, source_term_full, source_term_gradT

from .defaults import SI_INPUT_PATH

OMEGAS = [1e3, 1e9]


@pytest.fixture(scope="module")
def si_material():
    """Silicon at 300 K in double precision."""
    return Material.from_phono3py(SI_INPUT_PATH, temperature=300)


def _self_consistency_error(solver, material):
    """Largest relative deviation |T(dT) - dT| / |dT| over all frequencies."""
    errs = []
    for i, w in enumerate(solver.omg_ft_array):
        n_next, _ = solver._dT_to_N(dT=solver.dT[i], omg_ft=float(w), omg_idx=i)
        dT_next = complex(N_to_dT(n_next, material))
        errs.append(abs(dT_next - complex(solver.dT[i])) / abs(complex(solver.dT[i])))
    return max(errs)


def test_equilibrium_n_maps_to_one_kelvin(si_material):
    """N_to_dT(diag(nbar)) must be exactly 1 K; the affine solver relies on it."""
    nbar = equilibrium_n_per_kelvin(si_material)
    n = xp.zeros((si_material.nq, si_material.nat3, si_material.nat3), dtype=si_material.dtypec)
    idx = xp.arange(si_material.nat3)
    n[:, idx, idx] = nbar
    assert abs(complex(N_to_dT(n, si_material)) - 1) < 1e-13


@pytest.mark.parametrize("omg_ft", OMEGAS)
@pytest.mark.parametrize("k_ft", [1e4, 1e7])
def test_remainder_identity(si_material, omg_ft, k_ft):
    """G(Gamma nbar) = nbar + G r, i.e. the cancellation-free 1 - b equals the naive one."""
    source = source_term_diag(si_material.heat_capacity)
    nbar = equilibrium_n_per_kelvin(si_material)
    r = affine_remainder_rhs(omg_ft, k_ft, si_material, nbar)

    def T(dT, src):
        n, _ = dT_to_N_iterative(dT, omg_ft, k_ft, si_material, src, "energy", solver="cgesv")
        return complex(N_to_dT(n, si_material))

    b_naive = T(1.0, source) - T(0.0, source)
    one_minus_b = -T(0.0, r)
    # b ~ 1, so the naive 1 - b carries an absolute round-off error of ~1e-13 regardless of how small 1 - b is
    assert abs(one_minus_b - (1 - b_naive)) < 1e-11


@pytest.mark.parametrize("source_fn", [source_term_diag, source_term_full])
@pytest.mark.parametrize("k_ft", [1e4, 1e7])
def test_affine_self_consistent_and_solvers_agree(si_material, source_fn, k_ft):
    """The closed-form dT is a fixed point, and iterative (gmres, cgesv) and Green paths agree."""
    source = source_fn(si_material.heat_capacity)
    omegas = xp.asarray(OMEGAS)

    results = {}
    for inner in ["gmres", "cgesv"]:
        s = IterativeWTESolver(
            omegas, k_ft, si_material, source, source_type="energy", outer_solver="affine", inner_solver=inner
        )
        s.run()
        assert _self_consistency_error(s, si_material) < 1e-10
        results[inner] = (to_cpu(s.dT), to_cpu(s.n))

    greens = []
    for w in OMEGAS:
        g = RTAGreenOperator(RTAWignerOperator(omg_ft=w, k_ft=k_ft, material=si_material))
        g.compute()
        greens.append(g)
    sg = GreenWTESolver(omegas, k_ft, si_material, greens, source, source_type="energy", outer_solver="affine")
    sg.run(free=False)
    assert _self_consistency_error(sg, si_material) < 1e-10
    results["green"] = (to_cpu(sg.dT), to_cpu(sg.n))

    ref_dT, ref_n = results["cgesv"]
    for name in ["gmres", "green"]:
        dT, n = results[name]
        np.testing.assert_allclose(dT, ref_dT, rtol=1e-9, err_msg=name)
        np.testing.assert_allclose(n, ref_n, rtol=1e-8, atol=1e-10 * np.max(np.abs(ref_n)), err_msg=name)

    # bookkeeping: two Green applications, a single "iterate"
    assert np.all(to_cpu(sg.niter) == 2)
    assert sg.dT_iterates.shape == (len(OMEGAS), 1)


@pytest.mark.parametrize(
    "outer_solver, k_ft, rtol",
    [
        ("aitken", 1e8, 1e-9),
        ("root", 1e8, 1e-9),
        # diffusive regime: 1/|1-b| ~ 1e7 amplifies the ~1e-13 fixed-point residual into the dT error
        ("root", 1e4, 1e-4),
    ],
)
def test_iterative_outer_solvers_reach_fixed_point(si_material, outer_solver, k_ft, rtol):
    """Iterative outer solvers must converge to the true fixed point for small (normalized-source) dT.

    Regression test: a 1 K floor in the convergence scale made the relative tolerance an absolute one far larger
    than dT ~ 1e-14 K, so these solvers stopped with fixed-point errors of 1-30% while reporting success. Aitken is
    not tested in the diffusive regime: its second differences ~ (1-b)^2 drown in round-off there, see
    :func:`test_non_converged_outer_solver_warns`.
    """
    source = source_term_full(si_material.heat_capacity)
    omegas = xp.asarray(OMEGAS)

    ref = IterativeWTESolver(
        omegas, k_ft, si_material, source, source_type="energy", outer_solver="affine", inner_solver="cgesv"
    )
    ref.run()
    s = IterativeWTESolver(
        omegas, k_ft, si_material, source, source_type="energy", outer_solver=outer_solver, inner_solver="cgesv"
    )
    s.run()

    assert np.max(np.abs(to_cpu(ref.dT))) < 1e-6  # |dT| << 1 K: the regime that used to break
    assert _self_consistency_error(s, si_material) < 1e-10
    np.testing.assert_allclose(to_cpu(s.dT), to_cpu(ref.dT), rtol=rtol)


@pytest.mark.parametrize("outer_solver", ["aitken", "plain"])
def test_non_converged_outer_solver_warns(si_material, outer_solver, capfd):
    """Hitting max_iter must be reported instead of silently returning an unconverged dT."""
    source = source_term_full(si_material.heat_capacity)
    s = IterativeWTESolver([1e3], 1e4, si_material, source, source_type="energy", outer_solver=outer_solver, max_iter=5)
    s.run()
    out = capfd.readouterr().out
    assert f"{outer_solver} outer solver did not converge within max_iter=5" in out


def test_affine_rejects_gradient_source(si_material):
    """With a gradient source dT is imposed, so the affine solve is meaningless."""
    k_ft = 1e6
    source = source_term_gradT(
        k_ft,
        si_material.velocity_operator,
        si_material.phonon_freq,
        si_material.linewidth,
        si_material.heat_capacity,
        si_material.volume,
    )
    s = IterativeWTESolver([1e9], k_ft, si_material, source, source_type="gradient", outer_solver="affine")
    with pytest.raises(ValueError, match="requires an energy source"):
        s.run()


def test_apply_green_not_implemented(si_material):
    """Solvers that do not implement _apply_green cannot use the affine outer solver."""

    class NoGreen(IterativeWTESolver):
        _apply_green = SolverBase._apply_green

    source = source_term_diag(si_material.heat_capacity)
    s = NoGreen([1e9], 1e6, si_material, source, source_type="energy", outer_solver="affine")
    with pytest.raises(NotImplementedError, match="does not support the 'affine' outer solver"):
        s.run()
