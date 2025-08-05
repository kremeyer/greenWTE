"""Module for setting up predefined source terms for the WTE solver."""

import cupy as cp
from scipy.constants import hbar


def source_term_diag(heat_capacity):
    """Construct the source term for diagonal heating.

    Heating of each mode is proportional to its heat capacity. Off-diagonal entries are zero.

    Parameters
    ----------
    heat_capacity : cupy.ndarray
        Heat capacity of each mode, shape (nq, nat3).

    Returns
    -------
    cupy.ndarray
        Source term for diagonal heating, shape (nq, nat3, nat3).

    """
    nq, nat3 = heat_capacity.shape
    source_term = cp.zeros((nq, nat3, nat3), dtype=heat_capacity.dtype)
    heat_capacity_tot = cp.sum(heat_capacity)
    for i in range(nq):
        cp.fill_diagonal(source_term[i], heat_capacity[i] / heat_capacity_tot)
    if heat_capacity.dtype == cp.float32:
        source_term = source_term.astype(cp.complex64)
    elif heat_capacity.dtype == cp.float64:
        source_term = source_term.astype(cp.complex128)
    return source_term


def source_term_full(heat_capacity):
    """Construct the source term for full heating.

    Heating of each entry is proportional to its heat capacity.

    Parameters
    ----------
    heat_capacity : cupy.ndarray
        Heat capacity of each mode, shape (nq, nat3).

    Returns
    -------
    cupy.ndarray
        Source term for full heating, shape (nq, nat3, nat3).

    """
    heat_capacity_tot = cp.sum(heat_capacity)
    source_term = heat_capacity[:, :, None] * heat_capacity[:, None, :] / (heat_capacity_tot**2)
    if heat_capacity.dtype == cp.float32:
        source_term = source_term.astype(cp.complex64)
    elif heat_capacity.dtype == cp.float64:
        source_term = source_term.astype(cp.complex128)
    return source_term


def source_term_offdiag(heat_capacity):
    """Construct the source term for off-diagonal heating.

    Heating of each entry is proportional to its heat capacity. All diagonal entries are zero.

    Parameters
    ----------
    heat_capacity : cupy.ndarray
        Heat capacity of each mode, shape (nq, nat3).

    Returns
    -------
    cupy.ndarray
        Source term for off-diagonal heating, shape (nq, nat3, nat3).

    """
    source_term = source_term_full(heat_capacity)
    for i in range(source_term.shape[0]):
        cp.fill_diagonal(source_term[i], 0)
    return source_term


def source_term_gradT(k_ft, velocity_operator, phonon_freq, linewidth, heat_capacity, volume):
    """Construct source term to simulate a temperature gradient.

    We drive the system with the drift term of the WTE ik/2 * {V,N} and subtract the inhomogeneous
    contribution from the RTA scattering operator 0.5 * (G,N_bar).

    Parameters
    ----------
    k_ft : float
        The thermal grating wavevector [1/m].
    velocity_operator : cupy.ndarray
        The velocity operator, shape (nq, nat3, nat3).
    phonon_freq : cupy.ndarray
        The phonon frequencies, shape (nq, nat3).
    linewidth : cupy.ndarray
        The linewidths of each mode, shape (nq, nat3).
    heat_capacity : cupy.ndarray
        The heat capacity of each mode, shape (nq, nat3).
    volume : float
        The volume of the system [m^3].

    Returns
    -------
    cupy.ndarray
        Source term for the temperature gradient, shape (nq, nat3, nat3).

    """
    nq, nat3 = heat_capacity.shape
    source_term = cp.zeros((nq, nat3, nat3), dtype=velocity_operator.dtype)
    for i in range(nq):
        v = velocity_operator[i]
        # the dT here is dropped and multiplied to the source in the solver part of the code
        nbar = cp.diag(volume * nq / hbar / phonon_freq[i] * heat_capacity[i])
        G = cp.diag(linewidth[i])
        source_term[i] = 1j * k_ft / 2 * (v @ nbar + nbar @ v) - 0.5 * (G @ nbar + nbar @ G)
    return source_term


def source_term_anticommutator(k_ft, velocity_operator, phonon_freq, linewidth, heat_capacity, volume):
    """Construct the source term using the anticommutator of the velocity operator and the temperature gradient.

    We drive the system with the drift term of the WTE ik/2 * {V,N} and subtract the inhomogeneous
    contribution from the RTA scattering operator 0.5 * (G,N_bar).

    .. deprecated:: 0.2.0
       Please use the :py:func:`source_term_gradT` function instead.

    Parameters
    ----------
    k_ft : float
        The thermal grating wavevector [1/m].
    velocity_operator : cupy.ndarray
        The velocity operator, shape (nq, nat3, nat3).
    phonon_freq : cupy.ndarray
        The phonon frequencies, shape (nq, nat3).
    linewidth : cupy.ndarray
        The linewidths of each mode, shape (nq, nat3).
    heat_capacity : cupy.ndarray
        The heat capacity of each mode, shape (nq, nat3).
    volume : float
        The volume of the system [m^3].

    Returns
    -------
    cupy.ndarray
        Source term for the temperature gradient, shape (nq, nat3, nat3).

    """
    import warnings

    warnings.warn(
        "The anticommutator source term is deprecated and will be removed in future versions. "
        "Please use the gradT source term instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return source_term_gradT(k_ft, velocity_operator, phonon_freq, linewidth, heat_capacity, volume)
