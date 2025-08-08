"""Test cases for the full Green's function part of the greenWTE package."""

import cupy as cp
from greenWTE.base import Material
from greenWTE.green import RTAGreenOperator, RTAWignerOperator

from .defaults import DEFAULT_TEMPERATURE, DEFAULT_TEMPORAL_FREQUENCY, DEFAULT_THERMAL_GRATING, SI_INPUT_PATH


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
