"""ECG signal preprocessing: bandpass filtering, notch filtering, normalization.

All filters use zero-phase (filtfilt) to preserve waveform timing.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, medfilt


class ECGPreprocessor:
    """Preprocess raw ECG signals for feature extraction and HSMM segmentation.

    Pipeline: bandpass (0.5-40 Hz) -> notch (50/60 Hz) -> normalize

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz. Default 250.0.
    notch_freq : float or None
        Power-line frequency. If None, auto-detects from common values (50/60 Hz)
        based on the closest match to typical sampling rates.
    """

    def __init__(self, fs: float = 250.0, notch_freq: float | None = None):
        if fs <= 0:
            raise ValueError(f"Sampling frequency must be positive, got {fs}")

        self.fs = fs
        self.nyquist = fs / 2.0

        # Auto-detect power-line frequency if not specified
        if notch_freq is None:
            self.notch_freq = 50.0 if abs(fs - 250.0) < abs(fs - 360.0) else 60.0
        else:
            self.notch_freq = notch_freq

        self._bandpass_coeffs = None
        self._notch_coeffs = None

    # ------------------------------------------------------------------
    # Bandpass filter (0.5 - 40 Hz, 4th-order Butterworth, zero-phase)
    # ------------------------------------------------------------------
    def _design_bandpass(self, low: float = 0.5, high: float = 40.0, order: int = 4):
        """Design Butterworth bandpass filter coefficients.

        Parameters
        ----------
        low : float
            Low-cut frequency in Hz.
        high : float
            High-cut frequency in Hz.
        order : int
            Filter order.
        """
        nyq = self.nyquist
        low_n = low / nyq
        high_n = high / nyq

        # Sanity: high must be < Nyquist
        high_n = min(high_n, 0.99)

        b, a = butter(order, [low_n, high_n], btype="band")
        self._bandpass_coeffs = (b, a)

    def bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """Apply zero-phase bandpass filter.

        Parameters
        ----------
        signal : np.ndarray, shape (N,)
            Raw ECG signal.

        Returns
        -------
        np.ndarray, shape (N,)
            Bandpass-filtered signal.
        """
        if self._bandpass_coeffs is None:
            self._design_bandpass()

        b, a = self._bandpass_coeffs
        return filtfilt(b, a, signal)

    # ------------------------------------------------------------------
    # Notch filter (power-line interference removal)
    # ------------------------------------------------------------------
    def _design_notch(self, Q: float = 30.0):
        """Design IIR notch filter at the power-line frequency.

        Parameters
        ----------
        Q : float
            Quality factor. Higher = narrower notch.
        """
        b, a = iirnotch(self.notch_freq, Q, self.fs)
        self._notch_coeffs = (b, a)

    def notch_filter(self, signal: np.ndarray) -> np.ndarray:
        """Apply zero-phase notch filter.

        Parameters
        ----------
        signal : np.ndarray, shape (N,)
            Input signal.

        Returns
        -------
        np.ndarray, shape (N,)
            Notch-filtered signal.
        """
        if self._notch_coeffs is None:
            self._design_notch()

        b, a = self._notch_coeffs
        return filtfilt(b, a, signal)

    # ------------------------------------------------------------------
    # Baseline wander removal (median filter)
    # ------------------------------------------------------------------
    def remove_baseline_wander(self, signal: np.ndarray,
                                 window_ms: float = 200.0) -> np.ndarray:
        """Remove baseline wander using median filtering.

        A median filter of ~200ms window estimates the baseline (since QRS is
        ~80-100ms, it is attenuated by the median). Subtracting this from the
        signal removes slow drift while preserving QRS morphology.

        Parameters
        ----------
        signal : np.ndarray, shape (N,)
            Input signal.
        window_ms : float
            Median filter window width in milliseconds.

        Returns
        -------
        np.ndarray, shape (N,)
            De-trended signal.
        """
        window_samples = int(np.round(window_ms / 1000.0 * self.fs))
        # Ensure odd window size for median filter
        if window_samples % 2 == 0:
            window_samples += 1

        baseline = medfilt(signal, kernel_size=window_samples)
        return signal - baseline

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    @staticmethod
    def normalize(signal: np.ndarray) -> np.ndarray:
        """Z-score normalization: (signal - mean) / std.

        Parameters
        ----------
        signal : np.ndarray, shape (N,)
            Input signal.

        Returns
        -------
        np.ndarray, shape (N,)
            Normalized signal.
        """
        mu = np.mean(signal)
        sigma = np.std(signal)
        if sigma < 1e-12:
            return signal - mu  # near-constant signal
        return (signal - mu) / sigma

    # ------------------------------------------------------------------
    # Full preprocessing pipeline
    # ------------------------------------------------------------------
    def preprocess(self, signal: np.ndarray,
                   remove_baseline: bool = True) -> np.ndarray:
        """Run the full preprocessing pipeline.

        Pipeline: median baseline removal -> bandpass -> notch -> normalize

        Parameters
        ----------
        signal : np.ndarray, shape (N,)
            Raw ECG signal.
        remove_baseline : bool
            Whether to apply median-filter baseline removal before bandpass.

        Returns
        -------
        np.ndarray, shape (N,)
            Clean, normalized ECG signal ready for feature extraction.
        """
        sig = signal.copy().astype(np.float64)

        if remove_baseline:
            sig = self.remove_baseline_wander(sig)

        sig = self.bandpass_filter(sig)
        sig = self.notch_filter(sig)
        sig = self.normalize(sig)

        return sig
