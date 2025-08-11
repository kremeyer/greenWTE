"""User-facing module for precomputing Green's functions and writing them to disk."""

from argparse import ArgumentParser

import cupy as cp
import numpy as np


def parse_arguments():
    """Parse command-line arguments."""
    parser = ArgumentParser(description="Precompute Green's functions for materials.")
    parser.add_argument("input", type=str, help="HDF5 input file from phono3py")
    parser.add_argument("output", type=str, help="output directory for HDF5 file(s)")

    parser.add_argument(
        "-t", "--temperature_range", type=float, default=[50, 400, 8], help="temperature range in Kelvin"
    )
    parser.add_argument(
        "-k",
        "--spatial_frequency_range",
        type=float,
        nargs="+",
        default=[3, 9, 7],
        help="spatial frequency range in 10^(rad/m)",
    )
    parser.add_argument(
        "-w", "--omega_range", type=float, nargs="+", default=[0, 15, 16], help="temporal frequency range in 10^(Hz)"
    )
    parser.add_argument(
        "-d",
        "--direction",
        type=str,
        choices=["x", "y", "z"],
        default="x",
        help="direction of the temperature grating vector",
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
        a.temperature_range = np.array([float(a.temperature_range[0])])
    elif len(a.temperature_range) == 3:
        a.temperature_range[-1] = int(a.temperature_range[-1])
        a.temperature_range = np.linspace(*a.temperature_range)
        a.temperature_range = np.round(a.temperature_range, 0).astype(np.int32)
    else:
        raise ValueError("temperature_range must be a single value or 3 values (start, stop, num)")

    if a.direction not in ["x", "y", "z"]:
        raise ValueError("direction must be one of 'x', 'y', or 'z'")

    return a


if __name__ == "__main__":
    from os import sep
    from os.path import join as pj

    from tqdm import tqdm

    from .base import Material
    from .green import RTAGreenOperator, RTAWignerOperator
    from .io import GreenContainer

    args = parse_arguments()

    temperature_progress = args.temperature_range.shape[0] > 1
    spatial_frequency_progress = args.spatial_frequency_range.shape[0] > 1
    omega_progress = args.omega_range.shape[0] > 1

    direction_idx = {"x": 0, "y": 1, "z": 2}[args.direction]

    for temperature in tqdm(args.temperature_range, desc="T", disable=not temperature_progress):
        material = Material.from_phono3py(
            args.input, temperature, dir_idx=direction_idx, dtyper=cp.float32, dtypec=cp.complex64
        )
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
                    for iq in range(material.nq):
                        try:  # don't recompute if already on disk
                            gc.get(omega, spatial_frequency, iq)
                        except KeyError:
                            rwo = RTAWignerOperator(
                                omg_ft=omega,
                                k_ft=spatial_frequency,
                                material=material[iq],
                            )
                            rwo.compute()
                            rgo = RTAGreenOperator(rwo)
                            rgo.compute()
                            gc.put(omega, spatial_frequency, iq, rgo.squeeze())
