# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [1.1.0] - 2026-11-20

### Added

- `cuda12x-system` and `cuda13x-system` extras for using a CUDA toolkit already present on the system
- CI: check that the CUDA extras install and that the CPU fallback works without a GPU
- CONTRIBUTING.md

### Changed

- `cuda12x` and `cuda13x` extras now require CuPy 14 and install the CUDA libraries from PyPI via `cupy[ctk]`,
  only the NVIDIA driver is needed on the system
- a `RuntimeWarning` is emitted when CuPy is installed but cannot be loaded or no usable GPU is found, instead
  of silently falling back to the CPU backend (or failing later on machines without a GPU)
- docs: updated installation instructions (NumPy 2, choice of CUDA version, system CUDA toolkits)

### Fixed

- GMRES inner solver failing with CuPy 14 (`tol` was renamed to `rtol`)
- `np.NaN` in `greenWTE.post`, which was removed in NumPy 2
- README suggested installing both CUDA extras at the same time

## [1.0.1] - 2026-01-07

### Added

- docs: read the docs integration
- docs: tutorials for the explicit Green's function calculation and CsPbBr3
- CI: automatic releases to Github in addition to PyPI

### Fixed

- `GreenContainer` now accepts CuPy scalars when indexing/appending omega/k datasets, avoiding conversion errors

## [1.0.0] - 2025-12-01

### Added

- CPU support
- CI tests, coverage and linting

### Changed

- moving from bitshuffle to hdf5plugin

## [0.3.1] - 2025-10-26

### Added

- docs: installation, tutorial and CLI args
- recovered python 3.10 compatibility

## [0.3.0] - 2025-10-10

### Added

- Files required for testing are now hosted externally and automatically downloaded when running tests.

### Changed

- dependencies: NVTX is no longer required. A shim to nvtx.annotate was added that mirrors the original signature and does nothing.
- postprocessing: Do tessellation once when interpolating quantity onto a path in the BZ
- tests: Moved tests into greenWTE package, so it's shipped when users install via pip

## [0.2.2] - 2025-09-24

### Added

- Physical units as attributes to the HDF5 datasets.

### Changed

- Moved flux calculation from GPU to CPU to save GPU memory.

## [0.2.1] - 2025-09-17

### Changed

- When using "none" as the outer solver, no n_norm will be calculated, improving speed.

### Removed

- Unused tqdm dependency
