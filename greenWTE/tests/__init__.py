"""Test suite for the greenWTE package.

This file contains some helper functions and we download the large material data required to run the tests.
"""

import hashlib
import tempfile
import urllib.request
from pathlib import Path

from greenWTE import to_cpu, xp
from greenWTE.base import N_to_dT

ASSETS = {
    # filename: (sha256, [mirrors]).
    # The second and third order forces to calculate force constants and other material properties for Si with phono3py
    # are taken from the phono3py examples here:
    # https://github.com/phonopy/phono3py/tree/648aceac10fb244d25cfa8e8c470794001e0ebe7/example/Si-CRYSTAL
    # Licensed under BSD-3-Clause:
    # https://opensource.org/license/bsd-3-clause
    "Si-kappa-m555.hdf5": (
        "ab9e8ed5560a00319afad5ad291ad94d927ca4a819c26fdacf00763c253778ca",
        [
            "https://kremeyer.eu/Si-kappa-m555.hdf5",
            "https://www.physics.mcgill.ca/~laurenzk/Si-kappa-m555.hdf5",
            "https://nextcloud.computecanada.ca/index.php/s/aMijNxDREjxyQYH/download/Si-kappa-m555.hdf5",
        ],
    ),
    # The second and third order IFCs used to calculate the material properties for CsPbBr3 with phono3py are taken
    # from: M. Simoncelli, N. Marzari, F. Mauri (2021) DOI: 10.24435/materialscloud:g0-yc
    # Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0):
    # https://creativecommons.org/licenses/by/4.0/
    "CsPbBr3-kappa-m443.hdf5": (
        "7d88949a2666584667f3ffef2d3b2aac438ee860751a9a8124d9b963a092d052",
        [
            "https://kremeyer.eu/CsPbBr3-kappa-m443.hdf5",
            "https://www.physics.mcgill.ca/~laurenzk/CsPbBr3-kappa-m443.hdf5",
            "https://nextcloud.computecanada.ca/index.php/s/JbMKp8c8ga655gw/download/CsPbBr3-kappa-m443.hdf5",
        ],
    ),
}

DEST_DIR = Path(__file__).parent


def _sha256(p: Path) -> str:
    """Compute the SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(65536)  # 64kb chunk size
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def _download(url: str, dest: Path) -> None:
    """Download a file from a URL to a destination path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tf:
        temp_path = Path(tf.name)
    try:
        with urllib.request.urlopen(url) as response, temp_path.open("wb") as out_file:
            while True:
                chunk = response.read(65536)  # 64kb chunk size
                if not chunk:
                    break
                out_file.write(chunk)
        temp_path.replace(dest)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _ensure_asset(filename: str, expected_hash: str, urls: list[str]) -> Path:
    """Ensure that the asset file exists and has the correct SHA256 hash."""
    path = DEST_DIR / filename
    if path.exists() and _sha256(path) == expected_hash:
        return path

    last_error = None
    for url in urls:
        try:
            _download(url, path)
            if _sha256(path) != expected_hash:
                path.unlink(missing_ok=True)
                raise ValueError(f"Checksum mismatch for {filename} from {url}")
            return path
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Failed to download {filename} from all mirrors.") from last_error


for _name, (_hash, _urls) in ASSETS.items():
    try:
        _ensure_asset(_name, _hash, _urls)
    except Exception as e:
        print(f"Test data {_name} could not be downloaded.")
        raise e


def _final_residual_and_scale(solver, material):
    """Compute |F(dT)| and the scale used by the solver's combined tolerance."""
    dT_final = solver.dT[0]
    omg_ft = float(to_cpu(solver.omg_ft_array[0]))
    # Evaluate F at the final iterate to avoid Aitken mismatch
    n_next, _ = solver._dT_to_N(dT=dT_final, omg_ft=omg_ft, omg_idx=0, sol_guess=None)
    dT_next = N_to_dT(n_next, material)

    r_abs = float(xp.abs(dT_final - dT_next))
    scale = float(max(xp.abs(dT_final), xp.abs(dT_next), 1.0))
    return r_abs, scale, dT_final, dT_next
