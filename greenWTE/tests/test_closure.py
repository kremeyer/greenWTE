"""Tests for the energy-conserving closure of the RTA relaxation term."""

import numpy as np
import pytest

from greenWTE import to_cpu, xp
from greenWTE.base import Material, N_to_dT, N_to_dT_closure, equilibrium_n_per_kelvin
from greenWTE.iterative import IterativeWTESolver
from greenWTE.sources import source_term_diag, source_term_full, source_term_gradT

from .defaults import SI_INPUT_PATH


@pytest.fixture(scope="module")
def si_material():
    """Silicon at 300 K in double precision."""
    return Material.from_phono3py(SI_INPUT_PATH, temperature=300)


def _kappa_gradient(material, k_ft):
    """Compute the static thermal conductivity from the temperature-gradient source (no closure involved)."""
    source = source_term_gradT(
        k_ft,
        material.velocity_operator,
        material.phonon_freq,
        material.linewidth,
        material.heat_capacity,
        material.volume,
    )
    s = IterativeWTESolver([0.0], k_ft, material, source, source_type="gradient", outer_solver="none")
    s.run()
    return float(np.real(s.kappa[0]))


def test_closure_equals_energy_temperature_for_uniform_linewidth(si_material):
    """With a single relaxation time both temperature definitions coincide for any n."""
    m = si_material
    uniform = Material(
        temperature=m.temperature,
        velocity_operator=m.velocity_operator,
        phonon_freq=m.phonon_freq,
        linewidth=xp.full_like(m.linewidth, 1e12),
        heat_capacity=m.heat_capacity,
        volume=m.volume,
    )
    rng = np.random.default_rng(0)
    n = xp.asarray(rng.normal(size=(m.nq, m.nat3, m.nat3)) + 1j * rng.normal(size=(m.nq, m.nat3, m.nat3)))
    assert abs(complex(N_to_dT_closure(n, uniform)) / complex(N_to_dT(n, uniform)) - 1) < 1e-12
    # ... but not in general
    assert abs(complex(N_to_dT_closure(n, m)) / complex(N_to_dT(n, m)) - 1) > 1e-3


@pytest.mark.parametrize("outer_solver", ["affine", "root"])
@pytest.mark.parametrize("k_ft", [1e4, 1e8])
def test_collisions_conserve_energy(si_material, outer_solver, k_ft):
    """The relaxation term -Gamma (n - nbar dT) of the converged solution must not create or destroy energy."""
    m = si_material
    s = IterativeWTESolver(
        [0.0, 1e9],
        k_ft,
        m,
        source_term_full(m.heat_capacity),
        source_type="energy",
        outer_solver=outer_solver,
        inner_solver="cgesv",
    )
    s.run()
    nbar = equilibrium_n_per_kelvin(m)
    for i in range(2):
        populations = xp.diagonal(s.n[i], axis1=1, axis2=2)
        relaxed = m.phonon_freq * m.linewidth * populations
        rate = complex(xp.sum(relaxed - m.phonon_freq * m.linewidth * nbar * s.dT_closure[i]))
        assert abs(rate) < 1e-9 * abs(complex(xp.sum(relaxed)))


def test_reported_dT(si_material):
    """Energy sources report the energy temperature of the solution, gradient sources the imposed dT."""
    m = si_material
    s = IterativeWTESolver(
        [0.0, 1e9], 1e6, m, source_term_full(m.heat_capacity), source_type="energy", outer_solver="affine"
    )
    s.run()
    for i in range(2):
        assert abs(complex(s.dT[i]) / complex(N_to_dT(s.n[i], m)) - 1) < 1e-12
        assert abs(complex(s.dT_closure[i]) / complex(N_to_dT_closure(s.n[i], m)) - 1) < 1e-10
        # the two differ away from the diffusive limit
        assert abs(complex(s.dT[i]) / complex(s.dT_closure[i]) - 1) > 1e-5

    k_ft = 1e6
    source = source_term_gradT(k_ft, m.velocity_operator, m.phonon_freq, m.linewidth, m.heat_capacity, m.volume)
    g = IterativeWTESolver([0.0], k_ft, m, source, source_type="gradient", outer_solver="none", dT_init=2.0 + 0.5j)
    g.run()
    assert complex(g.dT[0]) == 2.0 + 0.5j
    assert complex(g.dT_closure[0]) == 2.0 + 0.5j


def test_diffusive_limit_diffusivity(si_material):
    """A heat source decays with D = kappa / C in the diffusive limit, kappa from the gradient source.

    With an energy-weighted temperature in the relaxation term the decay rate of Si comes out ~2.6x too fast.
    """
    m = si_material
    k_ft = 1e4  # grating period 0.6 mm >> phonon mean free paths
    d_ref = _kappa_gradient(m, k_ft) / float(xp.sum(m.heat_capacity))

    omegas = np.logspace(1, 3.3, 6)  # below the decay rate D k^2 ~ 4e3 1/s
    s = IterativeWTESolver(
        omegas,
        k_ft,
        m,
        source_term_diag(m.heat_capacity),
        source_type="energy",
        outer_solver="affine",
        inner_solver="cgesv",
    )
    s.run()

    # diffusion: dT = A / (D k^2 - i w)  <=>  1/dT = (D k^2)/A - i w/A
    inv = 1 / to_cpu(s.dT)
    design = np.block([[np.ones((6, 1)), np.zeros((6, 1))], [np.zeros((6, 1)), -omegas[:, None]]])
    (rate_over_A, one_over_A), *_ = np.linalg.lstsq(design, np.concatenate([inv.real, inv.imag]), rcond=None)
    d_fit = rate_over_A / one_over_A / k_ft**2

    assert d_fit / d_ref == pytest.approx(1, abs=1e-3)
