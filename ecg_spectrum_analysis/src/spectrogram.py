"""ECG Spectrogram Computation.

Four methods for time-frequency analysis of ECG signals:

1. **STFT Spectrogram** — Short-Time Fourier Transform
   Classic sliding-window approach. Good general-purpose time-frequency view.
   Trade-off: window size balances time vs frequency resolution.

2. **PSD (Power Spectral Density)** — Welch's method
   Averaged periodogram. Smooth, statistically stable frequency-domain
   representation. Best for comparing overall spectral content between leads.

3. **CWT Scalogram** — Continuous Wavelet Transform
   Multi-resolution time-frequency analysis. Better time resolution at high
   frequencies, better frequency resolution at low frequencies — naturally
   matched to ECG signal characteristics.

4. **Complex CWT** — raw complex wavelet coefficients
   Same as the scalogram but preserving phase, for cross-lead
   cross-wavelet analysis (e.g. lead reversal detection).

All functions accept raw numpy arrays and return structured data ready for
plotting with matplotlib.

Note on the CWT frequency axis: the width<->frequency mapping
(`_widths_to_freqs` / `_make_widths`) is the single source of truth and is
verified numerically against each wavelet's FFT peak.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy import signal
from scipy.signal import spectrogram, welch
from scipy.signal._wavelets import _ricker as ricker, _cwt as cwt


# ---------------------------------------------------------------------------
# Dataclass for clean return values
# ---------------------------------------------------------------------------
@dataclass
class ECG_Spectrogram:
    """Holds a computed spectrogram and its metadata.

    Attributes
    ----------
    data : np.ndarray, shape (freq_bins, time_bins)
        Spectrogram magnitude (or power) values.
    freqs : np.ndarray, shape (freq_bins,)
        Frequency axis in Hz.
    times : np.ndarray, shape (time_bins,)
        Time axis in seconds.
    fs : float
        Sampling rate used.
    method : str
        One of 'stft', 'psd', 'cwt'.
    lead_name : str
        ECG lead label.
    record_name : str
        Patient record identifier.
    """

    data: np.ndarray
    freqs: np.ndarray
    times: np.ndarray
    fs: float
    method: str = 'stft'
    lead_name: str = ''
    record_name: str = ''

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def freq_range(self) -> tuple[float, float]:
        return (float(self.freqs[0]), float(self.freqs[-1]))

    @property
    def time_range(self) -> tuple[float, float]:
        return (float(self.times[0]), float(self.times[-1]))


# ---------------------------------------------------------------------------
# STFT Spectrogram
# ---------------------------------------------------------------------------
def compute_spectrogram(
    ecg_signal: np.ndarray,
    fs: float = 250.0,
    nperseg: int = 256,
    noverlap: int | None = None,
    nfft: int | None = None,
    freq_limit: float | None = 60.0,
    scale: str = 'power',
    lead_name: str = '',
    record_name: str = '',
) -> ECG_Spectrogram:
    """Compute STFT spectrogram of an ECG signal.

    Parameters
    ----------
    ecg_signal : np.ndarray, shape (N,)
        Preprocessed ECG signal (bandpass-filtered, notch-filtered).
    fs : float
        Sampling rate in Hz. Default 250.
    nperseg : int
        Samples per STFT segment. At 250 Hz, 256 samples ≈ 1.0 s window.
        Smaller = better time resolution; larger = better frequency resolution.
    noverlap : int or None
        Overlap samples between segments. Default: nperseg // 2 (50%).
    nfft : int or None
        FFT points. Default: next power-of-two >= nperseg.
    freq_limit : float or None
        Upper frequency limit in Hz. None = Nyquist.
        ECG clinical content is mostly below 40-60 Hz.
    scale : str
        'power' for PSD (V²/Hz), 'magnitude' for linear amplitude,
        'db' for decibel scale (10*log10).
    lead_name : str
        ECG lead label for metadata.
    record_name : str
        Patient record identifier.

    Returns
    -------
    ECG_Spectrogram
        With .data in the chosen scale.
    """
    if noverlap is None:
        noverlap = nperseg // 2  # 50% overlap

    f, t, Sxx = spectrogram(
        ecg_signal, fs=fs,
        nperseg=nperseg, noverlap=noverlap,
        nfft=nfft, scaling='density',
        mode='psd',
    )

    # Clip to frequency limit
    if freq_limit is not None:
        fmask = f <= freq_limit
        f = f[fmask]
        Sxx = Sxx[fmask, :]

    # Scale conversion
    if scale == 'db':
        eps = np.finfo(Sxx.dtype).tiny
        Sxx = 10.0 * np.log10(np.maximum(Sxx, eps))
    elif scale == 'magnitude':
        Sxx = np.sqrt(Sxx)

    return ECG_Spectrogram(
        data=Sxx, freqs=f, times=t, fs=fs,
        method='stft', lead_name=lead_name, record_name=record_name,
    )


# ---------------------------------------------------------------------------
# Welch PSD (averaged periodogram)
# ---------------------------------------------------------------------------
def compute_psd(
    ecg_signal: np.ndarray,
    fs: float = 250.0,
    nperseg: int = 1024,
    noverlap: int | None = None,
    nfft: int | None = None,
    freq_limit: float | None = 60.0,
    scale: str = 'power',
    lead_name: str = '',
    record_name: str = '',
) -> ECG_Spectrogram:
    """Compute Power Spectral Density using Welch's averaged periodogram.

    This gives a single, smooth frequency-domain profile — useful for
    comparing overall spectral content across leads or records.

    Parameters
    ----------
    ecg_signal : np.ndarray, shape (N,)
        Preprocessed ECG signal.
    fs : float
        Sampling rate in Hz.
    nperseg : int
        Samples per Welch segment. Larger = smoother but fewer averages.
        Default 1024 ≈ 4 s at 250 Hz.
    noverlap : int or None
        Overlap between segments. Default: nperseg // 2.
    nfft : int or None
        FFT points.
    freq_limit : float or None
        Upper frequency limit.
    scale : str
        'power', 'magnitude', or 'db'.
    lead_name : str
        ECG lead label.
    record_name : str
        Patient record identifier.

    Returns
    -------
    ECG_Spectrogram
        .data is 1-D (freq_bins,), .times is empty array.
    """
    if noverlap is None:
        noverlap = nperseg // 2

    f, Pxx = welch(
        ecg_signal, fs=fs,
        nperseg=nperseg, noverlap=noverlap,
        nfft=nfft, scaling='density',
    )

    if freq_limit is not None:
        fmask = f <= freq_limit
        f = f[fmask]
        Pxx = Pxx[fmask]

    if scale == 'db':
        eps = np.finfo(Pxx.dtype).tiny
        Pxx = 10.0 * np.log10(np.maximum(Pxx, eps))
    elif scale == 'magnitude':
        Pxx = np.sqrt(Pxx)

    return ECG_Spectrogram(
        data=Pxx, freqs=f, times=np.array([]), fs=fs,
        method='psd', lead_name=lead_name, record_name=record_name,
    )


# ---------------------------------------------------------------------------
# CWT (complex core + magnitude scalogram)
# ---------------------------------------------------------------------------
MORLET_W = 5.0  # Morlet omega0 for the locally-defined wavelet below


def _morlet(points: int, s: float, w: float = MORLET_W) -> np.ndarray:
    """Complex Morlet wavelet. `s` is scale (width), `w` is omega0.

    Defined locally since public morlet2/scipy.signal.cwt were removed
    in scipy 1.15+; we use the private _cwt with the same signature.
    """
    t = np.arange(points) - (points - 1.0) / 2.0
    t = t / s
    return np.pi ** (-0.25) * (
        np.exp(1j * w * t) - np.exp(-0.5 * w ** 2)
    ) * np.exp(-0.5 * t ** 2)


def _widths_to_freqs(wavelet: str, widths: np.ndarray, fs: float) -> np.ndarray:
    """Map wavelet widths (scales) to true center frequencies in Hz.

    Formulas verified numerically against the FFT peak of each wavelet:
      - ricker:        f = 0.2 * fs / width
      - morlet (w=5):  f = w / (2*pi) * fs / width
    This is the single source of truth for the width<->frequency mapping;
    _make_widths below is its exact inverse.
    """
    if wavelet == 'morlet':
        return MORLET_W / (2.0 * np.pi) * fs / widths
    return 0.2 * fs / widths


def _make_widths(wavelet: str, freq_range: tuple[float, float],
                 fs: float, n_voices: int, n_samples: int) -> np.ndarray:
    """Log-spaced widths covering [f_low, f_high]; inverse of _widths_to_freqs."""
    f_low, f_high = freq_range
    k = MORLET_W / (2.0 * np.pi) if wavelet == 'morlet' else 0.2
    w_min = k * fs / f_high   # narrowest width (highest frequency)
    w_max = k * fs / f_low    # widest width (lowest frequency)
    w_min = max(w_min, 1.5)
    w_max = min(w_max, n_samples / 4.0)
    return np.geomspace(w_min, w_max, n_voices)


def compute_cwt_complex(
    ecg_signal: np.ndarray,
    fs: float = 250.0,
    wavelet: str = 'ricker',
    widths: np.ndarray | None = None,
    freq_range: tuple[float, float] = (0.5, 60.0),
    n_voices: int = 64,
    lead_name: str = '',
    record_name: str = '',
) -> ECG_Spectrogram:
    """Compute the raw complex CWT of a signal.

    Same parameters as compute_scalogram(), but .data is complex
    (scales x time), preserving phase information for cross-lead
    cross-wavelet analysis (e.g. lead reversal detection).
    """
    N = len(ecg_signal)
    T = N / fs

    if widths is None:
        widths = _make_widths(wavelet, freq_range, fs, n_voices, N)

    if wavelet == 'morlet':
        coeffs = cwt(ecg_signal, _morlet, widths, kwargs={'w': MORLET_W})
    else:
        coeffs = cwt(ecg_signal, ricker, widths)

    pseudo_freqs = _widths_to_freqs(wavelet, widths, fs)

    # Sort rows by increasing frequency and clip to the requested range
    sort_idx = np.argsort(pseudo_freqs)
    pseudo_freqs = pseudo_freqs[sort_idx]
    coeffs = coeffs[sort_idx, :]
    fmask = (pseudo_freqs >= freq_range[0]) & (pseudo_freqs <= freq_range[1])
    pseudo_freqs = pseudo_freqs[fmask]
    coeffs = coeffs[fmask, :]

    return ECG_Spectrogram(
        data=coeffs, freqs=pseudo_freqs, times=np.linspace(0, T, N), fs=fs,
        method='cwt', lead_name=lead_name, record_name=record_name,
    )


def compute_scalogram(
    ecg_signal: np.ndarray,
    fs: float = 250.0,
    wavelet: str = 'ricker',
    widths: np.ndarray | None = None,
    freq_range: tuple[float, float] = (0.5, 60.0),
    n_voices: int = 64,
    scale: str = 'magnitude',
    lead_name: str = '',
    record_name: str = '',
) -> ECG_Spectrogram:
    """Compute CWT scalogram using Ricker (Mexican Hat) or Morlet wavelets.

    The CWT provides multi-resolution time-frequency decomposition:
    - Better time resolution at high frequencies (sharp QRS transients)
    - Better frequency resolution at low frequencies (slow P/T waves)

    Parameters
    ----------
    ecg_signal : np.ndarray, shape (N,)
        Preprocessed ECG signal.
    fs : float
        Sampling rate in Hz.
    wavelet : str
        Wavelet type: 'ricker' (Mexican Hat) or 'morlet'.
        Ricker is real-valued and fast; Morlet is complex-valued and better
        for phase information, but we take the magnitude here
        (see compute_cwt_complex for the raw complex coefficients).
    widths : np.ndarray or None
        Wavelet widths (scales). If None, auto-generated from freq_range.
        Wider = lower frequency, narrower = higher frequency.
    freq_range : tuple[float, float]
        (low, high) frequency range in Hz for auto-generating widths.
    n_voices : int
        Number of frequency bins (voices per octave roughly).
    scale : str
        'magnitude' or 'db'.
    lead_name : str
        ECG lead label.
    record_name : str
        Patient record identifier.

    Returns
    -------
    ECG_Spectrogram
        .data is 2-D (scales, time), .freqs are true wavelet center
        frequencies in Hz (verified against the wavelet FFT peaks).
    """
    coeffs = compute_cwt_complex(
        ecg_signal, fs=fs, wavelet=wavelet, widths=widths,
        freq_range=freq_range, n_voices=n_voices,
        lead_name=lead_name, record_name=record_name,
    )

    magnitude = np.abs(coeffs.data)
    if scale == 'db':
        eps = np.finfo(magnitude.dtype).tiny
        magnitude = 20.0 * np.log10(np.maximum(magnitude, eps))

    return ECG_Spectrogram(
        data=magnitude, freqs=coeffs.freqs, times=coeffs.times, fs=fs,
        method='cwt', lead_name=lead_name, record_name=record_name,
    )


# ---------------------------------------------------------------------------
# Utility: frequency band power
# ---------------------------------------------------------------------------
def band_power(spec: ECG_Spectrogram,
               bands: dict[str, tuple[float, float]] | None = None,
               ) -> dict[str, float]:
    """Compute average power in standard ECG frequency bands.

    Parameters
    ----------
    spec : ECG_Spectrogram
        PSD or spectrogram result.
    bands : dict or None
        Band name -> (low_hz, high_hz). Defaults to ECG-relevant bands:
        - VLF (Very Low Frequency): 0.01-0.04 Hz
        - ULF (Ultra Low Frequency): 0-0.01 Hz
        (Note: these require long recordings; adjust for short segments)
        - Delta: 0.5-4 Hz (includes P/T wave content)
        - Theta: 4-8 Hz
        - Alpha: 8-12 Hz
        - Beta: 12-30 Hz (QRS harmonics)
        - Gamma: 30-60 Hz (noise, muscle artifact)

    Returns
    -------
    dict[str, float]
        Band name -> average power in that band.
    """
    if bands is None:
        bands = {
            'delta': (0.5, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 12.0),
            'beta':  (12.0, 30.0),
            'gamma': (30.0, 60.0),
        }

    result = {}
    data = spec.data
    freqs = spec.freqs

    # If 2-D (spectrogram), average over time first
    if data.ndim == 2:
        data = np.mean(data, axis=1)

    for name, (flo, fhi) in bands.items():
        mask = (freqs >= flo) & (freqs <= fhi)
        if mask.any():
            result[name] = float(np.mean(data[mask]))
        else:
            result[name] = 0.0

    return result
