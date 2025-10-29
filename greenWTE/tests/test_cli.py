"""Test cases for the command line interface of the greenWTE package."""

import contextlib
import os
import signal
import subprocess
import sys
import time
from os.path import join as pj
from pathlib import Path

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
        ("-w", "-1", "1"),
        ("-w", "-1", "1", "3", "5"),
    ],
)
def test_cli_errors_solve_iter(tmp_path, cli_arg):
    """Test various command line interface options for the greenWTE.solve_iter module."""
    output_file = pj(tmp_path, "test_cli_options.h5")
    cmd = [sys.executable, "-m", "greenWTE.solve_iter", SI_INPUT_PATH, output_file, *cli_arg]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode != 0, f"Command succeeded unexpectedly: {result.stdout}"
    assert "ValueError" in result.stderr or "invalid choice" in result.stderr


def test_normal_solve_iter_runs(tmp_path):
    """Test that the normal execution of greenWTE.solve_iter works as intended."""
    tmp_file = pj(tmp_path, "test_solve_iter.h5")
    cmd = [
        sys.executable,
        "-m",
        "greenWTE.solve_iter",
        SI_INPUT_PATH,
        tmp_file,
        "-t",
        "50",
        "-k",
        "0",
        "-w",
        "0",
        "-os",
        "none",
        "-is",
        "cgesv",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("RETURN CODE:", result.returncode)

    assert result.returncode == 0, f"Command failed with error: {result.stderr}"


@pytest.fixture(scope="session")
def precomputed_green(tmp_path_factory):
    """Fixture to provide a precomputed Green's function file for testing."""
    base = tmp_path_factory.mktemp("greens_cache")  # shared across all params/tests
    base = Path(base)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "greenWTE.precompute_green",
            str(SI_INPUT_PATH),
            str(base),
            "-t",
            "350",
            "-k",
            str(np.log10(DEFAULT_THERMAL_GRATING)),
            "-w",
            "5",
            "15",
            "25",  # make this cover what solve_green expects
        ],
        check=True,
    )

    green_name = Path(SI_INPUT_PATH).name.replace(".hdf5", "-T350.hdf5").replace("kappa", "green")
    green_file = base / green_name
    return green_file


@pytest.mark.parametrize(
    "cli_arg",
    [
        ("-t", "350"),
        ("--temperature", "350"),
        ("-k", str(np.log10(DEFAULT_THERMAL_GRATING))),
        ("--spatial-frequency", str(np.log10(DEFAULT_THERMAL_GRATING))),
        ("-w", str(DEFAULT_TEMPORAL_FREQUENCY)),
        ("-w", "-1", "1", "3"),
        ("--omega-range", str(DEFAULT_TEMPORAL_FREQUENCY)),
        ("-m", "2"),
        ("--max-iter", "2"),
        ("-cr", "1e-3"),
        ("--conv-thr-rel", "1e-3"),
        ("-ca", "1e-3"),
        ("--conv-thr-abs", "1e-3"),
        ("-dp",),
        ("--double-precision",),
        ("-os", "plain"),
        ("-os", "aitken"),
        ("-os", "root"),
        ("-os", "none"),
        ("--outer-solver", "plain"),
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
def test_cli_options_solve_green(tmp_path, cli_arg, precomputed_green):
    """Test various command line interface options for the greenWTE.solve_green module."""
    output_file = pj(tmp_path, "test_cli_options.h5")
    cmd = [
        sys.executable,
        "-m",
        "greenWTE.solve_green",
        SI_INPUT_PATH,
        str(precomputed_green),
        output_file,
        *cli_arg,
        "--dry-run",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Command failed with error: {result.stderr}"


def test_normal_solve_green_runs(tmp_path, precomputed_green):
    """Test that the normal execution of greenWTE.solve_green works as intended."""
    tmp_file = pj(tmp_path, "test_solve_green.h5")
    cmd = [
        sys.executable,
        "-m",
        "greenWTE.solve_green",
        SI_INPUT_PATH,
        str(precomputed_green),
        tmp_file,
        "-t",
        "350",
        "-k",
        str(np.log10(DEFAULT_THERMAL_GRATING)),
        "-w",
        "5",
        "-os",
        "none",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("RETURN CODE:", result.returncode)

    assert result.returncode == 0, f"Command failed with error: {result.stderr}"


@pytest.mark.parametrize(
    "cli_arg",
    [
        ("-w", "-1", "1"),
        ("-w", "-1", "1", "3", "5"),
    ],
)
def test_cli_errors_solve_green(tmp_path, cli_arg):
    """Test various command line interface options for the greenWTE.solve_green module."""
    output_file = pj(tmp_path, "test_cli_options.h5")
    cmd = [sys.executable, "-m", "greenWTE.solve_green", SI_INPUT_PATH, SI_INPUT_PATH, output_file, *cli_arg]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode != 0, f"Command succeeded unexpectedly: {result.stdout}"
    assert "ValueError" in result.stderr or "invalid choice" in result.stderr


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
        ("--batch",),
        ("-dp",),
        ("--double-precision",),
    ],
)
def test_cli_options_precompute_green(tmp_path, cli_arg):
    """Test various command line interface options for the greenWTE.precompute_green module."""
    output_file = pj(tmp_path, "test_cli_options.h5")
    cmd = [sys.executable, "-m", "greenWTE.precompute_green", SI_INPUT_PATH, output_file, *cli_arg, "--dry-run"]
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
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode != 0, f"Command succeeded unexpectedly: {result.stdout}"
    assert "ValueError" in result.stderr or "invalid choice" in result.stderr


def test_precompute_green_signal_handler(tmp_path):
    """Test that the signal handler to cancel a computation in greenWTE.precompute_green works as intended."""
    cmd = [
        sys.executable,
        "-m",
        "greenWTE.precompute_green",
        SI_INPUT_PATH,
        tmp_path,
        "-t 50",
        "-k 0",
        "-w",
        "0",
        "15",
        "100",
    ]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        time.sleep(1.5)
        os.kill(proc.pid, signal.SIGINT)
        out, err = proc.communicate(timeout=10)
    finally:
        # Make sure it's not left running even if assertions fail
        with contextlib.suppress(Exception):
            proc.kill()
    print("STDOUT:", out)
    print("STDERR:", err)
    print("RETURN CODE:", proc.returncode)

    assert proc.returncode == -2, f"Expected exit -2 after second SIGINT.\nSTDOUT:\n{out}\nSTDERR:\n{err}"
    assert "Signal 2 received - finishing current write, flushing, and exiting..." in out


@pytest.mark.parametrize("batch", [True, False])
def test_normal_precompute_green_runs(tmp_path, batch):
    """Test that the normal execution of greenWTE.precompute_green works as intended."""
    cmd = [
        sys.executable,
        "-m",
        "greenWTE.precompute_green",
        SI_INPUT_PATH,
        tmp_path,
        "-t 50",
        "-k 0",
    ]
    if batch:
        cmd.append("--batch")

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("RETURN CODE:", result.returncode)

    assert result.returncode == 0, f"Command failed with error: {result.stderr}"

    # run twice to check that existing files are detected properly
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("RETURN CODE:", result.returncode)

    # check that all the greens functions from the first run were found in the second run
    for line in result.stdout.splitlines()[-16:]:
        assert "0." in line or "found" in line
    assert result.returncode == 0, f"Command failed with error: {result.stderr}"
