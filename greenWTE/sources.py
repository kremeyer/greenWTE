"""Module for setting up predefined source terms for the WTE solver."""

import cupy as cp
from scipy.constants import hbar


def source_term_diag(heat_capacity):
    """Construct the source term for diagonal heating. Heating of each mode is proportional to its heat capacity.

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
    return source_term


def source_term_full(heat_capacity):
    """Construct the source term for full heating. Heating of each entry is proportional to its heat capacity.

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
    return source_term


def source_term_offdiag(heat_capacity):
    """Construct the source term for off-diagonal heating. Heating of each entry is proportional to its heat capacity.

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


def source_term_gradT(k_ft, velocity_operator, phonon_freq, heat_capacity, volume):
    """Construct the source term to simulate a temperature gradient.

    We define dn/dT as nbar = nq * volume / hbar / phonon_freq * heat_capacity
    and then use the commutator with the velocity operator:
    Q = (i k_ft/2)[V,dn/dT] = -i k_ft/2 (V @ dn/dT - dn/dT @ V)

    Parameters
    ----------
    k_ft : float
        The thermal grating wavevector [1/m].
    velocity_operator : cupy.ndarray
        The velocity operator, shape (nq, nat3, nat3).
    phonon_freq : cupy.ndarray
        The phonon frequencies, shape (nq, nat3).
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
    dndT = nq * volume / hbar / phonon_freq * heat_capacity
    source_term = cp.zeros((nq, nat3, nat3), dtype=cp.complex128)
    prefac = 1j * k_ft / 2
    for i in range(nq):
        v = velocity_operator[i]
        n = cp.diag(dndT[i])
        source_term[i] = -prefac * (v @ n - n @ v)
    return source_term


def source_term_anticommutator(k_ft, velocity_operator, phonon_freq, heat_capacity, volume):
    """Construct the source term using the anticommutator of the velocity operator and the temperature gradient.

    The mathematical definition is Q = (i k_ft/2){V,dn/dT} = -i k_ft/2 (V @ dn/dT + dn/dT @ V) with
    dn/dT = nq * volume / hbar / phonon_freq * heat_capacity.

    Parameters
    ----------
    k_ft : float
        The thermal grating wavevector [1/m].
    velocity_operator : cupy.ndarray
        The velocity operator, shape (nq, nat3, nat3).
    phonon_freq : cupy.ndarray
        The phonon frequencies, shape (nq, nat3).
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
    dndT = nq * volume / hbar / phonon_freq * heat_capacity
    source_term = cp.zeros((nq, nat3, nat3), dtype=cp.complex128)
    prefac = 1j * k_ft / 2
    for i in range(nq):
        v = velocity_operator[i]
        n = cp.diag(dndT[i])
        source_term[i] = -prefac * (v @ n + n @ v)
    return source_term
