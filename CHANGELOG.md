# Changelog

All notable changes to this project will be documented in this file.

# [0.2.3] - 2025

### Added

- Files required for testing are now hosted externally and automatically downloaded when running tests.

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
