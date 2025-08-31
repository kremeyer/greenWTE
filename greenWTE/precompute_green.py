"""User-facing module for precomputing Green's functions and writing them to disk."""

import signal
from argparse import ArgumentParser

import cupy as cp
import numpy as np


def parse_arguments():
    """Parse command-line arguments."""
    parser = ArgumentParser(description="Precompute Green's functions for materials.")
    parser.add_argument("input", type=str, help="HDF5 input file from phono3py")
    parser.add_argument("output", type=str, help="output directory for HDF5 file(s)")
    parser.add_argument("--tqdm", action="store_true", help="Enable progress bar; poor formatting when writing to file")
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
        "-w", "--omega-range", type=float, nargs="+", default=[0, 15, 16], help="temporal frequency range in 10^(Hz)"
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


if __name__ == "__main__":
    import time
    from os import sep
    from os.path import join as pj

    from tqdm import tqdm

    from .base import Material
    from .green import RTAGreenOperator, RTAWignerOperator
    from .io import GreenContainer

    STOP = False

    def request_stop(signal, frame):
        """Handle termination signals.

        Parameters
        ----------
        signal : signal
            The signal that was received.
        frame : frame
            The current stack frame.

        Notes
        -----
        This function is called when a termination signal is received. It sets the global STOP flag to True, indicating
        that the program should stop. The expected signals are:
            - SIGINT  ->  from KeyboardInterrupt triggered manually
            - SIGTERM ->  from slurm when the job is cancelled or times out
            - SIGUSR1 ->  from slurm to warn early before the job times out

        The usage of SIGUSR1 can be configured in the slurm options as `#SBATCH --signal=[{R|B}:]SIGUSR1[@sig_time]`
        ref: https://slurm.schedmd.com/sbatch.html#OPT_signal

        """
        global STOP
        if not STOP:
            print(f"Signal {signal} received - finishing current write, flushing, and exiting...")
            STOP = True
        else:
            raise SystemExit(1)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, request_stop)

    args = parse_arguments()

    if args.double_precision:
        dtyper = cp.float64
        dtypec = cp.complex128
    else:
        dtyper = cp.float32
        dtypec = cp.complex64

    if args.tqdm:
        temperature_progress = args.temperature_range.shape[0] > 1
        spatial_frequency_progress = args.spatial_frequency_range.shape[0] > 1
        omega_progress = args.omega_range.shape[0] > 1
    else:
        temperature_progress = False
        spatial_frequency_progress = False
        omega_progress = False

    direction_idx = {"x": 0, "y": 1, "z": 2}[args.direction]

    print(
        f"starting computation of Green's function for {args.input} into {args.output}\n"
        f"temperature range: {args.temperature_range}\n"
        f"spatial frequency range: {args.spatial_frequency_range}\n"
        f"temporal frequency range: {args.omega_range}\n"
        f"direction: {args.direction} (index {direction_idx})\n"
        f"batch: {args.batch}\n"
        f"tqdm: {args.tqdm}\n"
        f"double precision: {args.double_precision}\n"
    )

    if args.dry_run:
        print("exiting dry run...")
        import sys

        sys.exit(0)

    for temperature in tqdm(args.temperature_range, desc="T", disable=not temperature_progress):
        material = Material.from_phono3py(args.input, temperature, dir_idx=direction_idx, dtyper=dtyper, dtypec=dtypec)
        filename = pj(args.output, f"{material.name.split(sep)[-1]}")
        filename = filename.replace(".hdf5", f"-T{temperature:03d}.hdf5")
        filename = filename.replace("kappa", "green")
        with GreenContainer(path=filename, nat3=material.nat3, nq=material.nq, dtype=material.dtypec) as gc:
            for spatial_frequency in tqdm(
                args.spatial_frequency_range,
                desc="k",
                disable=not spatial_frequency_progress,
                leave=False,
            ):
                for omega in tqdm(
                    args.omega_range,
                    desc="w",
                    disable=not omega_progress,
                    leave=False,
                ):
                    if not args.tqdm:
                        t0 = time.time()
                        print(f"T={temperature: 4d}K k={spatial_frequency: 5.2e}/m w={omega: 5.2e}Hz: ", end="")
                    if args.batch:
                        if gc.has_bz_block(omega, spatial_frequency):
                            if not args.tqdm:
                                print("found")
                            continue
                        rwo = RTAWignerOperator(
                            omg_ft=omega,
                            k_ft=spatial_frequency,
                            material=material,
                        )
                        rwo.compute()
                        rgo = RTAGreenOperator(rwo)
                        if STOP:
                            raise KeyboardInterrupt
                        rgo.compute()
                        if STOP:
                            raise KeyboardInterrupt
                        gc.put_bz_block(omega, spatial_frequency, rgo)
                    else:
                        for iq in range(material.nq):
                            if STOP:
                                raise KeyboardInterrupt
                            if gc.has(omega, spatial_frequency, iq):
                                continue
                            rwo = RTAWignerOperator(
                                omg_ft=omega,
                                k_ft=spatial_frequency,
                                material=material[iq],
                            )
                            rwo.compute()
                            if STOP:
                                raise KeyboardInterrupt
                            rgo = RTAGreenOperator(rwo)
                            rgo.compute()
                            if STOP:
                                raise KeyboardInterrupt
                            gc.put(omega, spatial_frequency, iq, rgo.squeeze())
                    if not args.tqdm:
                        print(f"{time.time() - t0:.1f}s")
