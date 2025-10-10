"""Default values and constants for testing the greenWTE package."""

from os.path import join
from pathlib import Path

DEFAULT_TEMPERATURE = 100  # K
DEFAULT_THERMAL_GRATING = 10**4  # rad/m
DEFAULT_TEMPORAL_FREQUENCY = 0  # rad/s

_base = Path(__file__).parent
SI_INPUT_PATH = join(_base, "Si-kappa-m555.hdf5")
CSPBBR3_INPUT_PATH = join(_base, "CsPbBr3-kappa-m443.hdf5")
