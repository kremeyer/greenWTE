"""Test cases for the command line interface of the greenWTE package."""

import subprocess
import sys
from os.path import join as pj

import numpy as np
import pytest

from .defaults import DEFAULT_TEMPERATURE, DEFAULT_TEMPORAL_FREQUENCY, DEFAULT_THERMAL_GRATING, SI_INPUT_PATH


@pytest.mark.parametrize("module_name", ["greenWTE.solve_iter", "greenWTE.precompute_green", "greenWTE.solve_green"])
def test_run_as_module(module_name):
    """Test that some greenWTE modules can be run as such."""
    cmd = [sys.executable, "-m", module_name, "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Command failed with error: {result.stderr}"
    assert "usage:" in result.stdout, "Help message not found in output"


@pytest.mark.parametrize(
    "cli_arg",
    [
        ("-t", str(DEFAULT_TEMPERATURE)),
        ("--temperature", str(DEFAULT_TEMPERATURE)),
        ("-k", str(np.log10(DEFAULT_THERMAL_GRATING))),
        ("--spatial-frequency", str(np.log10(DEFAULT_THERMAL_GRATING))),
        ("-w", str(DEFAULT_TEMPORAL_FREQUENCY)),
        ("-w", "-1", "1", "3"),
        ("--omega-range", str(DEFAULT_TEMPORAL_FREQUENCY)),
        ("-xg", "True"),
        ("-xg", "False"),
        ("--exclude-gamma", "True"),
        ("-m", "2"),
        ("--max-iter", "2"),
        ("-cr", "1e-3"),
        ("--conv-thr-rel", "1e-3"),
        ("-ca", "1e-3"),
        ("--conv-thr-abs", "1e-3"),
        ("-sp",),
        ("--single-precision",),
        ("-is", "cgesv"),
        ("-is", "gmres"),
        ("--inner-solver", "cgesv"),
        ("-os", "plain"),
        ("-os", "aitken"),
        ("-os", "root"),
        ("-os", "none"),
        ("--outer-solver", "plain"),
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
def test_cli_options_solve_iter(tmp_path, cli_arg):
    """Test various command line interface options for the greenWTE.solve_iter module."""
    output_file = pj(tmp_path, "test_cli_options.h5")
    cmd = [sys.executable, "-m", "greenWTE.solve_iter", SI_INPUT_PATH, output_file, *cli_arg, "--dry-run"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Command failed with error: {result.stderr}"


@pytest.mark.parametrize(
    "cli_arg",
    [
        ("-t", str(DEFAULT_TEMPERATURE)),
        ("-t", "50", "60", "70"),
        ("--temperature-range", str(DEFAULT_TEMPERATURE)),
        ("-k", str(np.log10(DEFAULT_THERMAL_GRATING))),
        ("-k", "2", "3", "4"),
        ("--spatial-frequency-range", str(np.log10(DEFAULT_THERMAL_GRATING))),
        ("-w", str(DEFAULT_TEMPORAL_FREQUENCY)),
        ("-w", "-1", "1", "3"),
        ("--omega-range", str(DEFAULT_TEMPORAL_FREQUENCY)),
        ("-d", "x"),
        ("-d", "y"),
        ("-d", "z"),
        ("--direction", "x"),
        ("--tqdm",),
        ("--batch",),
    ],
)
def test_cli_options_precompute_green(tmp_path, cli_arg):
    """Test various command line interface options for the greenWTE.precompute_green module."""
    output_file = pj(tmp_path, "test_cli_options.h5")
    cmd = [sys.executable, "-m", "greenWTE.precompute_green", SI_INPUT_PATH, output_file, *cli_arg, "--dry-run"]
    print(cmd)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Command failed with error: {result.stderr}"


@pytest.mark.parametrize(
    "cli_arg",
    [
        ("-t", "50", "70"),
        ("-t", "50", "60", "70", "80"),
        ("-k", "2", "3"),
        ("-k", "2", "3", "4", "5"),
        ("-w", "-1", "1"),
        ("-w", "-1", "1", "3", "5"),
        ("-d", "a"),
    ],
)
def test_cli_errors_precompute_green(tmp_path, cli_arg):
    """Test various command line interface options for the greenWTE.precompute_green module."""
    output_file = pj(tmp_path, "test_cli_options.h5")
    cmd = [sys.executable, "-m", "greenWTE.precompute_green", SI_INPUT_PATH, output_file, *cli_arg, "--dry-run"]
    print(cmd)
    # with pytest.raises(subprocess.CalledProcessError):
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode != 0, f"Command succeeded unexpectedly: {result.stdout}"
    assert "ValueError" in result.stderr or "invalid choice" in result.stderr
