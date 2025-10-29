r"""CLI to solve the phonon Wigner Transport Equation (WTE) with an iterative inner solver.

This command-line interface runs the iterative WTE workflow for a single spatial grating :math:`k` in rad/m and
temperature :math:`T` in K over a set of temporal frequencies :math:`\omega` in rad/s. It builds the requested
source term, constructs a :py:class:`~greenWTE.base.Material` from a :mod:`phono3py` HDF5 file, and then solves
for the complex temperature response using :py:class:`~greenWTE.iterative.IterativeWTESolver`. The script then
prints per-frequency conductivities (total, population, coherence) and writes results/metadata to an HDF5 file.

Notes
-----
- ``--spatial-frequency`` expects a base-10 exponent ``x`` and is converted to :math:`k = 10^x` in rad/m.
- ``--omega-range`` accepts either a **single** exponent (interpreted as :math:`10^x`) or a **triplet**
  ``start stop num`` producing ``xp.logspace(start, stop, num)`` in rad/s, sorted ascending.

"""

from argparse import ArgumentParser, Namespace
from typing import Iterable

from . import sources, xp
from .io import save_solver_result
from .iterative import IterativeWTESolver


def get_parser() -> ArgumentParser:
    """Get the argument parser for the CLI."""
    parser = ArgumentParser(description="greenWTE iterative solver")

    parser.add_argument("input", type=str, help="HDF5 input file from phono3py")
    parser.add_argument("output", type=str, help="HDF5 output file")

    parser.add_argument("-t", "--temperature", type=float, default=350.0)
    parser.add_argument("-k", "--spatial-frequency", type=float, default=7, help="spatial frequency in 10^(rad/m)")
    parser.add_argument(
        "-w", "--omega-range", type=float, nargs="+", default=[5, 15, 25], help="temporal frequency range in 10^(rad/s)"
    )
    parser.add_argument(
        "-xg", "--exclude-gamma", type=bool, default=True, help="exclude gamma point from the calculation"
    )

    parser.add_argument("-m", "--max-iter", type=int, default=100, help="maximum number of iterations")
    parser.add_argument("-cr", "--conv-thr-rel", type=float, default=1e-10, help="relative convergence threshold")
    parser.add_argument("-ca", "--conv-thr-abs", type=float, default=0, help="absolute convergence threshold")
    parser.add_argument("-sp", "--single-precision", action="store_true", help="use single precision")
    parser.add_argument(
        "-is", "--inner-solver", type=str, choices=["gmres", "cgesv"], default="gmres", help="inner solver to use"
    )
    parser.add_argument(
        "-os",
        "--outer-solver",
        type=str,
        choices=["aitken", "plain", "root", "none"],
        default="aitken",
        help="outer solver to use",
    )
    parser.add_argument(
        "--diag-velocity-operator", action="store_true", help="set offdiagonal elements of velocity operator to zero"
    )
    parser.add_argument(
        "-s",
        "--source-type",
        type=str,
        choices=["diagonal", "diag", "full", "offdiagonal", "offdiag", "gradT", "anticommutator"],
        default="gradT",
        help="structure of source term",
    )
    parser.add_argument(
        "-d",
        "--direction",
        type=str,
        choices=["x", "y", "z"],
        default="x",
        help="direction of the temperature grating vector",
    )
    parser.add_argument(
        "-dT",
        "--dT-init",
        type=float,
        nargs=2,
        default=[1.0, 1.0],
        help="real and imaginary parts of the initial guess for temperature change at the first frequency; this value"
        " is used as dT, when outer-solver is chosen as 'none'.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="initialize solver but do not run the calculation; for testing purposes"
    )

    return parser


def parse_arguments(argv: Iterable[str] | None = None) -> Namespace:
    r"""Parse and validate command-line arguments for the Green-operator WTE solver.

    The ``omega_range`` accepts either a **single** number (interpreted as a base-10 exponent) or **three** numbers
    ``start stop num`` meaning ``np.logspace(start, stop, num)``.

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
        If ``omega_range`` options are not specified as either 1 value or 3 values.

    """
    parser = get_parser()
    a = parser.parse_args()

    if len(a.omega_range) == 1:
        a.omega_range = xp.array([10 ** (float(a.omega_range[0]))])
    elif len(a.omega_range) == 3:
        a.omega_range[-1] = int(a.omega_range[-1])
        a.omega_range = xp.logspace(*a.omega_range)
    else:
        raise ValueError("omega_range must be a single value or 3 values (start, stop, num)")
    a.omega_range = xp.sort(a.omega_range)

    a.spatial_frequency = 10 ** float(a.spatial_frequency)

    a.dT_init = tuple(float(x) for x in a.dT_init)

    return a


if __name__ == "__main__":  # pragma: no branch
    from .base import Material

    xp.set_printoptions(
        formatter={
            "complex_kind": lambda z: f"{z: .2e}",
            "float_kind": lambda x: f"{x: .2e}",
        }
    )

    args = parse_arguments()

    if args.single_precision:
        dtyper = xp.float32
        dtypec = xp.complex64
    else:
        dtyper = xp.float64
        dtypec = xp.complex128

    if xp.finfo(dtyper).resolution > args.conv_thr_rel:
        args.conv_thr_rel = 2 * xp.finfo(dtyper).resolution
        print(
            f"Warning: convergence threshold {args.conv_thr_rel} is smaller than machine"
            f" precision {xp.finfo(dtyper).resolution}."
        )
        print(f"Using twice machine precision {args.conv_thr_rel} instead.")

    if args.direction == "x":
        dir_idx = 0
    elif args.direction == "y":
        dir_idx = 1
    elif args.direction == "z":  # pragma: no branch
        dir_idx = 2

    mat = Material.from_phono3py(
        filename=args.input,
        temperature=args.temperature,
        dir_idx=dir_idx,
        exclude_gamma=args.exclude_gamma,
        dtyper=dtyper,
        dtypec=dtypec,
    )

    if args.diag_velocity_operator:
        offdiag_mask = ~xp.eye(mat.velocity_operator.shape[1], dtype=xp.bool_)
        mat.velocity_operator[:, offdiag_mask] = 0

    if args.source_type == "full":
        source = sources.source_term_full(mat.heat_capacity) * 5e8
        source_type_for_solver = "energy"
    elif args.source_type in ["diagonal", "diag"]:
        source = sources.source_term_diag(mat.heat_capacity)
        source_type_for_solver = "energy"
    elif args.source_type in ["offdiagonal", "offdiag"]:
        source = sources.source_term_offdiag(mat.heat_capacity)
        source_type_for_solver = "energy"
    elif args.source_type == "gradT":
        source = sources.source_term_gradT(
            args.spatial_frequency,
            mat.velocity_operator,
            mat.phonon_freq,
            mat.linewidth,
            mat.heat_capacity,
            mat.volume,
        )
        source_type_for_solver = "gradient"
    elif args.source_type == "anticommutator":  # pragma: no branch
        import warnings

        warnings.warn(
            "The anticommutator source term flag is deprecated and will be removed in future versions. "
            "Please use the gradT source term instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        source = sources.source_term_gradT(
            args.spatial_frequency,
            mat.velocity_operator,
            mat.phonon_freq,
            mat.linewidth,
            mat.heat_capacity,
            mat.volume,
        )
        source_type_for_solver = "gradient"

    solver = IterativeWTESolver(
        omg_ft_array=args.omega_range,
        k_ft=args.spatial_frequency,
        material=mat,
        source=source,
        source_type=source_type_for_solver,
        max_iter=args.max_iter,
        conv_thr_rel=args.conv_thr_rel,
        conv_thr_abs=args.conv_thr_abs,
        outer_solver=args.outer_solver,
        inner_solver=args.inner_solver,
        command_line_args=args,
        dT_init=args.dT_init[0] + 1j * args.dT_init[1],
        print_progress=True,
    )

    if args.dry_run:
        import sys

        sys.exit(0)

    solver.run()

    cell_width = 20
    dc = args.direction  # direction character
    header = f"{'frequency':>{cell_width // 2}} "
    header += f"{f'κ_{dc}{dc}':>{cell_width}} "
    header += f"{f'κ_{dc}{dc}_P':>{cell_width}} "
    header += f"{f'κ_{dc}{dc}_C':>{cell_width}} "
    header += f"{f'|κ_{dc}{dc}|':>{cell_width // 2}} "
    header += f"{f'|κ_{dc}{dc}_P|':>{cell_width // 2}} "
    header += f"{f'|κ_{dc}{dc}_C|':>{cell_width // 2}}"
    print("\n", header)
    print(f"{'-' * 107}")
    for i in range(len(solver.omg_ft_array)):
        print(
            f"{solver.omg_ft_array[i]: {(cell_width // 2) + 1}.2e} "
            f"{solver.kappa[i]: {cell_width}.2e} "
            f"{solver.kappa_p[i]: {cell_width}.2e} "
            f"{solver.kappa_c[i]: {cell_width}.2e} "
            f"{xp.abs(solver.kappa[i]): {cell_width // 2}.2e} "
            f"{xp.abs(solver.kappa_p[i]): {cell_width // 2}.2e} "
            f"{xp.abs(solver.kappa_c[i]): {cell_width // 2}.2e}"
        )
    print(f"{'-' * 107}")

    save_solver_result(args.output, solver)
