"""
Signal processing utilities for SeismicLens.
Wraps SciPy routines with seismology-specific defaults.
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.fft import rfft, rfftfreq


# ── Tapering ─────────────────────────────────────────────────────────────────

def taper_signal(data: np.ndarray, p: float = 0.05) -> np.ndarray:
    """
    Apply a cosine taper (Tukey window) to both ends of the signal.

    The Tukey window smoothly forces the signal to zero at the edges,
    preventing spectral leakage artifacts in the FFT due to discontinuities
    at the signal boundaries.

    Parameters
    ----------
    data : array_like  – raw signal samples
    p    : float       – fraction of each side to taper (default 5 %)

    Returns
    -------
    np.ndarray – tapered signal
    """
    window = sp_signal.windows.tukey(len(data), alpha=2 * p)
    return data * window


# ── Bandpass filter ──────────────────────────────────────────────────────────

def bandpass_filter(
        data: np.ndarray,
        f_low: float,
        f_high: float,
        fs: float,
        order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter via sosfiltfilt.

    Design
    ------
    The Butterworth filter is designed to have a maximally flat magnitude
    response in the passband. In the Laplace (s) domain its poles lie on
    a circle of radius ω_c:

        |H(jω)|² = 1 / (1 + (ω/ωc)^(2n))

    sosfiltfilt applies the filter twice (forward + backward), yielding
    ZERO phase distortion — critical in seismology to preserve wave arrival
    times. The effective order is doubled to 2n.

    Parameters
    ----------
    data   : input signal
    f_low  : lower corner frequency (Hz)
    f_high : upper corner frequency (Hz)
    fs     : sampling rate (Hz)
    order  : filter order (default 4 → effective 8 with zero-phase)

    Returns
    -------
    np.ndarray – filtered signal
    """
    nyq = fs / 2.0
    if f_high >= nyq:
        raise ValueError(f"High-cut {f_high} Hz >= Nyquist {nyq} Hz")
    if f_low <= 0:
        raise ValueError("Low-cut must be > 0 Hz")
    if f_low >= f_high:
        raise ValueError("Low-cut must be < high-cut")

    sos = sp_signal.butter(
        order, [f_low / nyq, f_high / nyq],
        btype="band", output="sos"
    )
    return sp_signal.sosfiltfilt(sos, data)


# ── FFT ──────────────────────────────────────────────────────────────────────

def compute_fft(data: np.ndarray, fs: float):
    """
    Single-sided amplitude spectrum via real FFT with Hann windowing.

    Mathematical background
    -----------------------
    The Discrete Fourier Transform (DFT) decomposes a discrete signal x[n]
    into N complex exponentials:

        X[k] = Σ_{n=0}^{N-1}  x[n] · e^{-j 2π k n / N}

    where k is the frequency bin index, X[k] ∈ ℂ, and:
        |X[k]|       → amplitude at frequency k·fs/N
        ∠ X[k]       → phase (arg of complex number)
        X[k].real    → cosine coefficient
        X[k].imag    → sine coefficient  (with negative sign convention)

    The rfft variant exploits the Hermitian symmetry of real signals
    (X[N-k] = X[k]*), returning only the N//2+1 unique bins.

    Normalisation: multiply by 2/N to recover the true one-sided amplitude.
    Hann window suppresses spectral leakage from non-periodic signals.

    Returns
    -------
    freqs      : frequency axis (Hz)
    amplitudes : |A(f)| — one-sided amplitude spectrum
    phases_deg : ∠ A(f) in degrees
    """
    n = len(data)
    window = np.hanning(n)
    # Window correction factor (so amplitude is preserved)
    win_corr = n / np.sum(window)
    fft_vals = rfft(data * window) * win_corr
    amplitudes = (2.0 / n) * np.abs(fft_vals)
    phases_deg = np.angle(fft_vals, deg=True)
    freqs = rfftfreq(n, d=1.0 / fs)
    return freqs, amplitudes, phases_deg


# ── Power Spectral Density ────────────────────────────────────────────────────

def compute_psd(data: np.ndarray, fs: float):
    """
    Welch power spectral density estimate.

    PSD: S(f) = |X(f)|² / (fs · N)  [in (counts)²/Hz]

    Returns
    -------
    freqs : frequency axis (Hz)
    psd   : PSD values  (counts²/Hz)
    """
    f, psd = sp_signal.welch(data, fs=fs, nperseg=min(1024, len(data) // 4),
                              window="hann", scaling="density")
    return f, psd


# ── Spectrogram ──────────────────────────────────────────────────────────────

def compute_spectrogram(data: np.ndarray, fs: float, nperseg: int = None):
    """
    Short-Time Fourier Transform (STFT) spectrogram.

    The STFT slides a window w[m] along the signal and computes the DFT
    of each frame:

        STFT(τ, f) = Σ_n  x[n] · w[n-τ] · e^{-j 2π f n / fs}

    This provides time-frequency localisation — at the cost of a
    time-frequency uncertainty trade-off (analogous to Heisenberg's
    uncertainty principle):

        Δt · Δf ≥ 1 / (4π)

    Returns
    -------
    t   : time axis
    f   : frequency axis
    Sxx : power spectral density matrix
    """
    if nperseg is None:
        nperseg = min(256, len(data) // 4)
    f, t, Sxx = sp_signal.spectrogram(
        data, fs=fs, nperseg=nperseg,
        window="hann", noverlap=nperseg // 2
    )
    return t, f, Sxx


# ── STA/LTA ──────────────────────────────────────────────────────────────────

def compute_sta_lta(
        data: np.ndarray,
        fs: float,
        sta_s: float = 1.0,
        lta_s: float = 20.0,
) -> np.ndarray:
    """
    Classic STA/LTA characteristic function for event detection.

    Principle
    ---------
    Short-Term Average (STA) tracks rapid amplitude changes (signal onset).
    Long-Term Average (LTA) represents the background noise level.
    Their ratio spikes sharply when a seismic phase arrives:

        STA(t) = (1/N_sta) Σ x²[t-k]   (k = 0..N_sta-1)
        LTA(t) = (1/N_lta) Σ x²[t-k]   (k = 0..N_lta-1)
        R(t)   = STA(t) / LTA(t)

    Using squared samples makes the detector sensitive to energy, not sign.
    The cumulative-sum (prefix-sum) trick reduces the complexity to O(N).

    Parameters
    ----------
    data  : signal samples
    fs    : sampling rate (Hz)
    sta_s : short-term window (s)
    lta_s : long-term window (s)

    Returns
    -------
    np.ndarray – STA/LTA ratio time series (same length as data)
    """
    sta_n = max(1, int(sta_s * fs))
    lta_n = max(sta_n + 1, int(lta_s * fs))

    data2 = data ** 2
    n = len(data2)
    stalta = np.zeros(n)

    cs = np.cumsum(np.concatenate([[0], data2]))

    for i in range(lta_n, n):
        sta = (cs[i + 1] - cs[i - sta_n + 1]) / sta_n
        lta = (cs[i + 1] - cs[i - lta_n + 1]) / lta_n
        stalta[i] = sta / lta if lta > 0 else 0.0

    return stalta


def detect_p_wave(
        data: np.ndarray,
        fs: float,
        sta_s: float = 1.0,
        lta_s: float = 20.0,
        threshold: float = 3.5,
) -> float | None:
    """
    Return the time (s) of the first STA/LTA trigger above threshold,
    or None if no trigger is found.
    """
    stalta = compute_sta_lta(data, fs, sta_s, lta_s)
    indices = np.where(stalta > threshold)[0]
    if len(indices) == 0:
        return None
    return float(indices[0]) / fs


# ── Dominant frequency & spectral centroid ───────────────────────────────────

def spectral_metrics(freqs: np.ndarray, amplitudes: np.ndarray):
    """
    Returns
    -------
    dominant_f  : frequency of peak amplitude (Hz)
    centroid_f  : spectral centroid — amplitude-weighted mean frequency (Hz)
    bandwidth   : RMS spectral bandwidth (Hz)
    """
    dominant_f = float(freqs[np.argmax(amplitudes)])
    total = np.sum(amplitudes) + 1e-12
    centroid_f = float(np.sum(freqs * amplitudes) / total)
    bandwidth  = float(np.sqrt(np.sum(((freqs - centroid_f) ** 2) * amplitudes) / total))
    return dominant_f, centroid_f, bandwidth

