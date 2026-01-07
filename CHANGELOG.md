# Changelog

All notable changes to this project will be documented in this file.

### [Unreleased]

### Added

- docs: read the docs integration
- CI: automatic releases to Github in addition to PyPI

### Fixed

- `GreenContainer` now accepts CuPy scalars when indexing/appending omega/k datasets, avoiding conversion errors

# [1.0.0] - 2025-12-01

### Added

- CPU support
- CI tests, coverage and linting

### Changed

- moving from bitshuffle to hdf5plugin

# [0.3.1] - 2025-10-26

### Added

- docs: installation, tutorial and CLI args
- recovered python 3.10 compatibility

# [0.3.0] - 2025-10-10

### Added

- Files required for testing are now hosted externally and automatically downloaded when running tests.

### Changed

- dependencies: NVTX is not longer required. A shim to nvtx.annotate was added that mirrors the original signature and does nothing.
- postprocessing: Do tessellation once when interpolating quantity onto a path in the BZ
- tests: Moved tests into greenWTE package, so it's shipped when users install via pip

# [0.2.2] - 2025-09-24

### Added

- Physical units as attributes to the HDF5 datasets.

### Changed

- Moved flux calculation from GPU to CPU to save GPU memory.

# [0.2.1] - 2025-09-17

### Changed

- When using "none" as the outer solver, no n_norm will be calculated, improving speed.

### Removed

- Unused tqdm dependency
