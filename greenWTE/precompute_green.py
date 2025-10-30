r"""CLI to precompute Green's operators and store them in an on-disk HDF5 container.

This script builds relaxation-time approximation (RTA) **Wigner Green's operators** :math:`\mathcal G(q;\omega,k)`
for a given material, across a grid of temporal frequencies :math:`\omega` in rad/s, spatial frequencies
:math:`k` in rad/m, and temperatures :math:`T` in K. Results are written to an HDF5 file per temperature using the
:py:class:`~greenWTE.io.GreenContainer` layout. Each file stores the 5-D dataset ``green[Nw, Nk, nq, m, m]`` with
:py:class:`hdf5plugin.Bitshuffle` compression.

Typical use (log-spaced frequency grids)::

    python -m greenWTE.precompute_green input.hdf5 out_dir -t 300 600 2 -k 3 9 7 -w 0 15 16 -d x

Notes
-----
- Frequencies provided via ``-k`` and ``-w`` are interpreted in **log10 units** when three values are supplied,
  i.e. ``-k k0 k1 n`` produces ``np.logspace(k0, k1, n)`` in rad/m; likewise for ``-w`` in rad/s. If a single
  value is given, it is treated as :math:`10^x` in the corresponding units.
- When ``--batch`` is enabled, the full Brillouin-zone block (all ``q``) is computed and written in one call. This is
  fastest but can be memory-intensive. Without ``--batch``, each ``q`` is processed independently to reduce peak memory.
- The script installs signal handlers for ``SIGINT``, ``SIGTERM``, and ``SIGUSR1``. Upon the first signal, it flips a
  global ``STOP`` flag, letting the current compute/write finish, then exits gracefully after the current block.
  A second signal forces an immediate exit.

"""

import signal
from argparse import ArgumentParser, Namespace
from typing import Iterable

import numpy as np

from . import xp


def get_parser() -> ArgumentParser:
    """Get the argument parser for the CLI."""
    parser = ArgumentParser(description="greenWTE Green's function precomputation")
    parser.add_argument("input", type=str, help="HDF5 input file from phono3py")
    parser.add_argument("output", type=str, help="output directory for HDF5 file(s)")
    parser.add_argument(
        "--batch", action="store_true", help="Enable batch mode; process full (w, k) pair; can be memory-heavy"
    )

    parser.add_argument(
        "-t", "--temperature-range", type=float, nargs="+", default=[50, 400, 8], help="temperature range in Kelvin"
    )
    parser.add_argument(
        "-k",
        "--spatial-frequency-range",
        type=float,
        nargs="+",
        default=[3, 9, 7],
        help="spatial frequency range in 10^(rad/m)",
    )
    parser.add_argument(
        "-w", "--omega-range", type=float, nargs="+", default=[0, 15, 16], help="temporal frequency range in 10^(rad/s)"
    )
    parser.add_argument(
        "-d",
        "--direction",
        type=str,
        choices=["x", "y", "z"],
        default="x",
        help="direction of the temperature grating vector",
    )
    parser.add_argument("-dp", "--double-precision", action="store_true", help="use double precision")
    parser.add_argument(
        "--dry-run", action="store_true", help="initialize but do not run the calculation; for testing purposes"
    )

    return parser


def parse_arguments(argv: Iterable[str] | None = None) -> Namespace:
    """Parse and validate command-line arguments for Green's precomputation.

    The frequency grids accept either a **single** number (interpreted as a base-10 exponent) or **three** numbers
    ``start stop num`` meaning ``np.logspace(start, stop, num)``. Temperatures accept either a single integer or three
    integers ``start stop num`` meaning ``np.linspace(start, stop, num)`` (rounded to whole kelvin).

    Parameters
    ----------
    argv : Iterable[str] or None, optional
        CLI arguments to parse (defaults to ``sys.argv[1:]`` when ``None``).

    Returns
    -------
    argparse.Namespace
        Parsed arguments.

    Raises
    ------
    ValueError
        If any of the ``*_range`` options are not specified as either 1 value or 3 values.

    """
    parser = get_parser()
    a = parser.parse_args()

    if len(a.omega_range) == 1:
        a.omega_range = np.array([10 ** (float(a.omega_range[0]))])
    elif len(a.omega_range) == 3:
        a.omega_range[-1] = int(a.omega_range[-1])
        a.omega_range = np.logspace(*a.omega_range)
    else:
        raise ValueError("omega_range must be a single value or 3 values (start, stop, num)")

    if len(a.spatial_frequency_range) == 1:
        a.spatial_frequency_range = np.array([10 ** (float(a.spatial_frequency_range[0]))])
    elif len(a.spatial_frequency_range) == 3:
        a.spatial_frequency_range[-1] = int(a.spatial_frequency_range[-1])
        a.spatial_frequency_range = np.logspace(*a.spatial_frequency_range)
    else:
        raise ValueError("spatial_frequency_range must be a single value or 3 values (start, stop, num)")

    if len(a.temperature_range) == 1:
        a.temperature_range = np.array([int(a.temperature_range[0])])
    elif len(a.temperature_range) == 3:
        a.temperature_range[-1] = int(a.temperature_range[-1])
        a.temperature_range = np.linspace(*a.temperature_range)
        a.temperature_range = np.round(a.temperature_range, 0).astype(np.int32)
    else:
        raise ValueError("temperature_range must be a single value or 3 values (start, stop, num)")

    return a


global STOP
STOP = False


def request_stop(signal: int, frame) -> None:
    """Handle termination signals (SIGINT, SIGTERM, SIGUSR1) with graceful shutdown.

    On first signal, set a global ``STOP`` flag so the main loop finishes the current compute/write and flushes
    the HDF5 file, then exits. On a second signal, exit immediately with non-zero status.

    Parameters
    ----------
    signal : int
        The signal that was received.
    frame : frame
        The current stack frame.

    Notes
    -----
    - ``SIGINT`` usually corresponds to a manual ``KeyboardInterrupt`` (Ctrl-C).
    - ``SIGTERM`` is commonly used by schedulers (e.g., Slurm) on cancellation or timeout.
    - ``SIGUSR1`` can be set as an early warning signal in Slurm via
      ``#SBATCH --signal=[{R|B}:]SIGUSR1[@sig_time]``. See `Slurm documentation`_ for details.

    .. _Slurm documentation: https://slurm.schedmd.com/sbatch.html#OPT_signal

    """
    global STOP
    if not STOP:  # pragma: no cover
        print(f"Signal {signal} received - finishing current write, flushing, and exiting...")
        STOP = True
    else:  # pragma: no cover
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no branch
    import time
    from os import sep
    from os.path import join as pj

    from .base import Material
    from .green import RTAGreenOperator, RTAWignerOperator
    from .io import GreenContainer

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, request_stop)

    args = parse_arguments()

    if args.double_precision:
        dtyper = xp.float64
        dtypec = xp.complex128
    else:
        dtyper = xp.float32
        dtypec = xp.complex64

    direction_idx = {"x": 0, "y": 1, "z": 2}[args.direction]

    print(
        f"starting computation of Green's function for {args.input} into {args.output}\n"
        f"temperature range: {args.temperature_range}\n"
        f"spatial frequency range: {args.spatial_frequency_range}\n"
        f"temporal frequency range: {args.omega_range}\n"
        f"direction: {args.direction} (index {direction_idx})\n"
        f"batch: {args.batch}\n"
        f"double precision: {args.double_precision}\n"
    )

    if args.dry_run:
        print("exiting dry run...")
        import sys

        sys.exit(0)

    for temperature in args.temperature_range:
        material = Material.from_phono3py(args.input, temperature, dir_idx=direction_idx, dtyper=dtyper, dtypec=dtypec)
        filename = pj(args.output, f"{material.name.split(sep)[-1]}")
        filename = filename.replace(".hdf5", f"-T{temperature:03d}.hdf5")
        filename = filename.replace("kappa", "green")
        with GreenContainer(path=filename, nat3=material.nat3, nq=material.nq, dtype=material.dtypec) as gc:
            for spatial_frequency in args.spatial_frequency_range:
                for omega in args.omega_range:
                    t0 = time.time()
                    print(f"T={temperature: 4d}K k={spatial_frequency: 5.2e}/m w={omega: 5.2e}rad/s: ", end="")
                    if args.batch:
                        if gc.has_bz_block(omega, spatial_frequency):
                            print("found")
                            continue
                        rwo = RTAWignerOperator(
                            omg_ft=omega,
                            k_ft=spatial_frequency,
                            material=material,
                        )
                        rwo.compute()
                        rgo = RTAGreenOperator(rwo)
                        if STOP:  # pragma: no cover
                            raise KeyboardInterrupt
                        rgo.compute()
                        if STOP:  # pragma: no cover
                            raise KeyboardInterrupt
                        gc.put_bz_block(omega, spatial_frequency, rgo)
                    else:
                        for iq in range(material.nq):
                            if STOP:  # pragma: no cover
                                raise KeyboardInterrupt
                            if gc.has(omega, spatial_frequency, iq):
                                continue
                            rwo = RTAWignerOperator(
                                omg_ft=omega,
                                k_ft=spatial_frequency,
                                material=material[iq],
                            )
                            rwo.compute()
                            if STOP:  # pragma: no cover
                                raise KeyboardInterrupt
                            rgo = RTAGreenOperator(rwo)
                            rgo.compute()
                            if STOP:  # pragma: no cover
                                raise KeyboardInterrupt
                            gc.put(omega, spatial_frequency, iq, rgo.squeeze())
                    print(f"{time.time() - t0:.1f}s")
