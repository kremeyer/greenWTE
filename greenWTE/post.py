"""Post-processing utilities for greenWTE."""

import numpy as np
from numba import njit, prange
from scipy.interpolate import LinearNDInterpolator, PchipInterpolator
from scipy.signal import savgol_filter


def manual_ifft(
    freq: np.ndarray,
    data: np.ndarray,
    n_freq_lin: int = 10000,
    n_t: int | None = None,
    freq_cutoff: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform an inverse FFT manually with interpolation onto a linear frequency grid.

    Parameters
    ----------
    freq : array_like
        Input angular frequency array [rad/s].
    data : array_like
        Input data array corresponding to the frequencies.
    n_freq_lin : int, optional
        Number of points in the linear frequency grid. Default is 10000.
    n_t : int, optional
        Number of time points to compute. If None, it is set to n_freq_lin // 10 + 1.
    freq_cutoff : float, optional
        Frequency cutoff for the linear grid. If None, it is set to the maximum frequency in `freq`.

    Returns
    -------
    t : ndarray
        Time array corresponding to the inverse FFT.
    data_ft : ndarray
        Inverse FFT of the input data.

    Notes
    -----
    The function mirrors the input frequency and data arrays to enforce Hermitian symmetry,
    interpolates the data onto a linear frequency grid using PCHIP interpolation, and then computes
    the inverse FFT using a JIT-compiled function for efficiency.

    """
    # mirror the data and the frequencies onto the negative axis
    # hermitian symmetry for real output of ifft
    freq = np.copy(freq)
    freq = np.tile(freq, 2)
    freq[: len(freq) // 2] = -freq[: len(freq) // 2][::-1]
    data = np.tile(data, 2)
    data[: len(data) // 2] = np.conj(data[: len(data) // 2][::-1])

    # generate linear frequency axis and interpolate data onto it
    if freq_cutoff is None:
        freq_cutoff = freq[-1]
    freq_lin = np.linspace(-freq_cutoff, freq_cutoff, n_freq_lin)
    interpolator_real = PchipInterpolator(freq, data.real)
    interpolator_imag = PchipInterpolator(freq, data.imag)
    data_lin = interpolator_real(freq_lin) + 1j * interpolator_imag(freq_lin)

    # setup time axis
    delta_freq = freq_lin[1] - freq_lin[0]
    t_total = 2 * np.pi / delta_freq
    n_t = n_freq_lin // 10 + 1 if n_t is None else int(n_t) + 1
    t = np.linspace(0, t_total, n_t, endpoint=False)
    # calculate the ifft integral
    data_ft = _ifft_integral_jit(freq_lin, data_lin, t, delta_freq)

    return t[1:], data_ft[1:]


def _ifft_integral(f: np.ndarray, data: np.ndarray, t: np.ndarray, df: float) -> np.ndarray:
    data_ft = np.zeros_like(t, dtype=complex)
    for i, ti in enumerate(t):
        data_ft[i] = np.sum(np.exp(-1j * f * ti) * data) * df
    return data_ft


@njit(parallel=True, fastmath=True)
def _ifft_integral_jit(f: np.ndarray, data: np.ndarray, t: np.ndarray, df: float) -> np.ndarray:
    n_t = t.shape[0]
    n_f = f.shape[0]
    data_ft = np.empty(n_t, dtype=np.complex128)

    for i in prange(n_t):
        ti = t[i]
        acc_real = 0.0
        acc_imag = 0.0
        for j in range(n_f):
            angle = -f[j] * ti
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            d_real = data[j].real
            d_imag = data[j].imag
            acc_real += cos_angle * d_real - sin_angle * d_imag
            acc_imag += cos_angle * d_imag + sin_angle * d_real
        data_ft[i] = complex(acc_real * df, acc_imag * df)

    return data_ft


def interpolate_onto_path(
    qpath: np.ndarray,
    qmesh: np.ndarray,
    data: np.ndarray,
    n: int = 1000,
    smooth: bool = False,
    smooth_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate data along a specified path in a multidimensional space.

    Given a path defined by a sequence of points (`qpath`), this function interpolates
    the provided data (`data`) defined on a mesh (`qmesh`) along the path, optionally
    applying smoothing to the interpolated data.

    Parameters
    ----------
    qpath : array_like, shape (N, D)
        Sequence of points defining the path along which to interpolate, where N is the
        number of path points and D is the dimensionality.
    qmesh : array_like, shape (M, D)
        Coordinates of the mesh points where the data is originally defined, where M is
        the number of mesh points.
    data : array_like, shape (M, K)
        Data values defined on the mesh points, where K is the number of data components
        (e.g., different physical quantities).
    n : int, optional
        Total number of points to interpolate along the entire path. Default is 1000.
    smooth : bool, optional
        If True, applies a Savitzky-Golay filter to smooth the interpolated data.
        Default is False.
    smooth_kwargs : dict, optional
        Keyword arguments to pass to `scipy.signal.savgol_filter` for smoothing.
        Default is {"window_length": 21, "polyorder": 1}.

    Returns
    -------
    qpath_interp : ndarray, shape (n, D)
        Interpolated points along the path.
    interp_data : ndarray, shape (n, K)
        Interpolated (and optionally smoothed) data values at each point along the path.
    qpath_idxs : ndarray, shape (N + 1,)
        Indices indicating the start of each segment in the interpolated path.

    Notes
    -----
    - Uses `scipy.interpolate.LinearNDInterpolator` for interpolation.
    - If smoothing is enabled, uses `scipy.signal.savgol_filter` on each data component.
    - The number of points per segment is proportional to the segment length.

    """
    if smooth_kwargs is None:
        smooth_kwargs = {"window_length": 21, "polyorder": 1}
    n_segs = np.zeros(len(qpath) - 1)
    for i in range(len(qpath) - 1):
        n_segs[i] = np.linalg.norm(qpath[i + 1] - qpath[i]) * n
    n_segs /= np.sum(n_segs) / n
    n_segs = np.round(n_segs).astype(int)
    n_segs[n_segs < 1] = 1
    qpath_idxs = np.insert(np.cumsum(n_segs), 0, 0)
    qpath_interp = []
    for i in range(len(qpath) - 1):
        seg = np.linspace(qpath[i], qpath[i + 1], n_segs[i] + 1)
        if i < len(qpath) - 2:
            seg = seg[:-1]
        qpath_interp.append(seg)
    qpath_interp = np.vstack(qpath_interp)

    interpolator = LinearNDInterpolator(qmesh, data, fill_value=np.NaN)
    interp_data = interpolator(qpath_interp)

    if smooth:
        interp_data = savgol_filter(interp_data, axis=0, **smooth_kwargs)

    return qpath_interp, interp_data, qpath_idxs
