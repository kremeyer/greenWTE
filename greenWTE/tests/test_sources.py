"""Test cases for the predefined source terms of the greenWTE package."""

import pytest

from greenWTE import sources, xp


@pytest.mark.filterwarnings("ignore::DeprecationWarning", "ignore::RuntimeWarning:greenWTE.sources")
def test_source_dtypes():
    """Test that the source term functions return the correct dtypes."""
    heat_capacity_f32 = xp.empty((10, 6), dtype=xp.float32)
    heat_capacity_f64 = xp.empty((10, 6), dtype=xp.float64)
    velocity_operator_c64 = xp.empty((10, 6, 6), dtype=xp.complex64)
    velocity_operator_c128 = xp.empty((10, 6, 6), dtype=xp.complex128)
    some_float = 1e7

    for func in [sources.source_term_diag, sources.source_term_full, sources.source_term_offdiag]:
        src_c64 = func(heat_capacity_f32)
        src_c128 = func(heat_capacity_f64)

        assert src_c64.dtype == xp.complex64, f"Expected complex64, got {src_c64.dtype}"
        assert src_c128.dtype == xp.complex128, f"Expected complex128, got {src_c128.dtype}"

        with pytest.raises(ValueError, match="Unsupported dtype"):
            func(xp.empty((10, 6), dtype=xp.int32))

    for func in [sources.source_term_anticommutator, sources.source_term_gradT]:
        src_c64 = func(
            some_float, velocity_operator_c64, heat_capacity_f32, heat_capacity_f32, heat_capacity_f32, some_float
        )
        src_c128 = func(
            some_float, velocity_operator_c128, heat_capacity_f64, heat_capacity_f64, heat_capacity_f64, some_float
        )
        assert src_c64.dtype == xp.complex64, f"Expected complex64, got {src_c64.dtype}"
        assert src_c128.dtype == xp.complex128, f"Expected complex128, got {src_c128.dtype}"


def test_source_structure():
    """Test that the source terms have the right diagonal/off-diagonal structure."""
    heat_capacity = xp.ones((10, 6), dtype=xp.float32)

    src_diag = sources.source_term_diag(heat_capacity)
    for i in range(src_diag.shape[0]):
        assert xp.all(src_diag[i].diagonal() != 0), "Diagonal entries should be non-zero"
        off_diag = src_diag[i] - xp.diag(xp.diag(src_diag[i]))
        assert xp.all(off_diag == 0), "Off-diagonal entries should be zero"

    src_offdiag = sources.source_term_offdiag(heat_capacity)
    for i in range(src_offdiag.shape[0]):
        assert xp.all(src_offdiag[i].diagonal() == 0), "Diagonal entries should be zero"
        off_diag = src_offdiag[i] + xp.diag(xp.diag(xp.ones_like(src_offdiag[i])))
        assert xp.all(off_diag != 0), "Off-diagonal entries should be non-zero"

    src_full = sources.source_term_full(heat_capacity)
    assert xp.all(src_full != 0), "All entries should be non-zero"
