"""Test cases for the command line interface of the greenWTE package."""

import subprocess
import sys
from os.path import join as pj

import numpy as np
import pytest

from .defaults import DEFAULT_TEMPERATURE, DEFAULT_TEMPORAL_FREQUENCY, DEFAULT_THERMAL_GRATING, SI_INPUT_PATH


def test_run_as_module():
    """Test that the greenWTE.solve module can be run as a module."""
    cmd = [sys.executable, "-m", "greenWTE.solve", "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Command failed with error: {result.stderr}"
    assert "usage:" in result.stdout, "Help message not found in output"


@pytest.mark.parametrize(
    "cli_arg",
    [
        ("-t", str(DEFAULT_TEMPERATURE)),
        ("--temperature", str(DEFAULT_TEMPERATURE)),
        ("-k", str(np.log10(DEFAULT_THERMAL_GRATING))),
        ("--spacial_frequency", str(np.log10(DEFAULT_THERMAL_GRATING))),
        ("-w", str(DEFAULT_TEMPORAL_FREQUENCY)),
        ("-w", "-1", "1", "3"),
        ("--omega_range", str(DEFAULT_TEMPORAL_FREQUENCY)),
        ("-xg", "True"),
        ("-xg", "False"),
        ("--exclude_gamma", "True"),
        ("-m", "2"),
        ("--max_iter", "2"),
        ("-c", "1e-3"),
        ("--conv_thr", "1e-3"),
        ("-sp",),
        ("--single_precision",),
        ("-is", "cgesv"),
        ("-is", "gmres"),
        ("--inner_solver", "cgesv"),
        ("-os", "plain"),
        ("-os", "aitken"),
        ("-os", "root"),
        ("--outer_solver", "plain"),
        ("--diag-velocity-operator",),
        ("-s", "gradT"),
        ("-s", "diag"),
        ("-s", "diagonal"),
        ("-s", "full"),
        ("-s", "offdiagonal"),
        ("-s", "anticommutator"),
        ("--source-type", "gradT"),
        ("-d", "x"),
        ("-d", "y"),
        ("-d", "z"),
        ("--direction", "x"),
    ],
)
def test_cli_options(tmp_path, cli_arg):
    """Test various command line interface options for the greenWTE.solve module."""
    output_file = pj(tmp_path, "test_cli_options.h5")
    cmd = [sys.executable, "-m", "greenWTE.solve", SI_INPUT_PATH, output_file, *cli_arg, "--dry-run"]
    print(cmd)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Command failed with error: {result.stderr}"
