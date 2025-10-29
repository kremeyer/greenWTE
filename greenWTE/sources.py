"""Module for setting up predefined source terms for the WTE solver.

Provides energy-injection sources (``diag``, ``full``, ``offdiag``) and a gradient-driven source (``gradT``) used by the
solvers. All functions return a **complex** tensor with shape ``(nq, nat3, nat3)`` and will promote real inputs to
``complex64``/``complex128`` based on the floating dtype of the provided arrays.
"""

from scipy.constants import hbar

from . import xp


def source_term_diag(heat_capacity: xp.ndarray) -> xp.ndarray:
    """Diagonal heating source.

    Construct :math:`P(q)` with diagonal entries proportional to the mode heat capacity at each :math:`q`, with all
    off-diagonal entries set to zero. The diagonal is normalized by the **total** heat capacity summed over all
    ``(q, m)`` so that the sum over all diagonal elements across all ``q`` equals 1.

    Parameters
    ----------
    heat_capacity : cupy.ndarray
        Heat capacity in J/m^3/K of the phonon modes, shape (nq, nat3).

    Returns
    -------
    cupy.ndarray
        Source term for diagonal heating, shape (nq, nat3, nat3).

    """
    nq, nat3 = heat_capacity.shape
    source_term = xp.zeros((nq, nat3, nat3), dtype=heat_capacity.dtype)
    heat_capacity_tot = xp.sum(heat_capacity)
    for i in range(nq):
        xp.fill_diagonal(source_term[i], heat_capacity[i] / heat_capacity_tot)
    if heat_capacity.dtype == xp.float32:
        return source_term.astype(xp.complex64)
    elif heat_capacity.dtype == xp.float64:
        return source_term.astype(xp.complex128)
    else:
        raise ValueError(f"Unsupported dtype {heat_capacity.dtype} for heat_capacity array.")


def source_term_full(heat_capacity: xp.ndarray) -> xp.ndarray:
    """Full heating source.

    Construct :math:`P(q)` with entries proportional to the outer product of per-mode heat capacities at the same
    :math:`q`, normalized by the **square** of the total heat capacity: ``P[q] = (h ⊗ h) / (∑ h)^2``, where
    ``h = heat_capacity[q]`` and the sum is over all ``(q, m)``.

    Parameters
    ----------
    heat_capacity : cupy.ndarray
        Heat capacity in J/m^3/K of the phonon modes, shape (nq, nat3).

    Returns
    -------
    cupy.ndarray
        Source term for full heating, shape (nq, nat3, nat3).

    """
    heat_capacity_tot = xp.sum(heat_capacity)
    source_term = heat_capacity[:, :, None] * heat_capacity[:, None, :] / (heat_capacity_tot**2)
    if heat_capacity.dtype == xp.float32:
        return source_term.astype(xp.complex64)
    elif heat_capacity.dtype == xp.float64:
        return source_term.astype(xp.complex128)
    else:
        raise ValueError(f"Unsupported dtype {heat_capacity.dtype} for heat_capacity array.")


def source_term_offdiag(heat_capacity: xp.ndarray) -> xp.ndarray:
    """Off-diagonal heating source.

    Construct a dense source as in :func:`source_term_full`, then zero the diagonal entries of each ``P[q]``

    Parameters
    ----------
    heat_capacity : cupy.ndarray
        Heat capacity in J/m^3/K of the phonon modes, shape (nq, nat3).

    Returns
    -------
    cupy.ndarray
        Source term for off-diagonal heating, shape (nq, nat3, nat3).

    """
    source_term = source_term_full(heat_capacity)
    for i in range(source_term.shape[0]):
        xp.fill_diagonal(source_term[i], 0)
    return source_term


def source_term_gradT(
    k_ft: float,
    velocity_operator: xp.ndarray,
    phonon_freq: xp.ndarray,
    linewidth: xp.ndarray,
    heat_capacity: xp.ndarray,
    volume: float,
) -> xp.ndarray:
    r"""Temperature-gradient source.

    For each :math:`q`, define :math:`\bar N(q) = \mathrm{diag}\!\left(\frac{V\,n_q}{\hbar\,\omega(q)}\,C(q)\right)` and
    :math:`G(q) = \mathrm{diag}(\Gamma(q))`, where :math:`V` is the cell volume and :math:`n_q` is the number of
    :math:`q`-points. Here :math:`\omega` is the phonon frequency, :math:`\Gamma` the linewidth, and :math:`C` the mode
    heat capacity. The source is

    .. math::

       S(q) = \frac{i\,k}{2}\,\{V(q), \bar N(q)\} \;-\; \frac{1}{2}\,\{G(q), \bar N(q)\},

    with the anticommutator :math:`\{A,B\} = AB + BA`.
    The overall factor of :math:`\Delta T` is intentionally omitted; the outer solver applies it.

    Parameters
    ----------
    k_ft : float
        Thermal grating wavevector in rad/m.
    velocity_operator : cupy.ndarray
        Velocity operator in m/s, shape (nq, nat3, nat3).
    phonon_freq : cupy.ndarray
        Phonon frequencies in rad/s, shape (nq, nat3).
    linewidth : cupy.ndarray
        Linewidths of each mode in rad/s, shape (nq, nat3).
    heat_capacity : cupy.ndarray
        Heat capacity in J/m^3/K of the phonon modes, shape (nq, nat3).
    volume : float
        Volume of the system in m^3.

    Returns
    -------
    cupy.ndarray
        Source term for the temperature gradient, shape (nq, nat3, nat3).

    """
    nq, nat3 = heat_capacity.shape
    source_term = xp.zeros((nq, nat3, nat3), dtype=velocity_operator.dtype)
    for i in range(nq):
        v = velocity_operator[i]
        # the dT here is dropped and multiplied to the source in the solver part of the code
        nbar = xp.diag(volume * nq / hbar / phonon_freq[i] * heat_capacity[i])
        G = xp.diag(linewidth[i])
        source_term[i] = 1j * k_ft / 2 * (v @ nbar + nbar @ v) - 0.5 * (G @ nbar + nbar @ G)
    return source_term


def source_term_anticommutator(
    k_ft: float,
    velocity_operator: xp.ndarray,
    phonon_freq: xp.ndarray,
    linewidth: xp.ndarray,
    heat_capacity: xp.ndarray,
    volume: float,
) -> xp.ndarray:
    """Temperature-gradient source via anticommutator.

    Constructs the same source as :func:`source_term_gradT`. This function is kept for backward
    compatibility and will be removed in a future release.

    .. deprecated:: 0.2.0
       Use :func:`source_term_gradT` instead.

    Parameters
    ----------
    k_ft : float
        Thermal grating wavevector in rad/m.
    velocity_operator : cupy.ndarray
        Velocity operator in m/s, shape (nq, nat3, nat3).
    phonon_freq : cupy.ndarray
        Phonon frequencies in rad/s, shape (nq, nat3).
    linewidth : cupy.ndarray
        Linewidths of each mode in rad/s, shape (nq, nat3).
    heat_capacity : cupy.ndarray
        Heat capacity of each mode in J/m^3/K, shape (nq, nat3).
    volume : float
        Volume of the system in m^3.

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
