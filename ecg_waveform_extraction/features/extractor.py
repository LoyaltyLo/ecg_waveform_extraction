"""Feature extraction: build 3D observation vectors [amplitude, d1, d2] per sample.

Features capture:
- amplitude: raw voltage level (distinguishes peaks from baseline)
- d1 (velocity): slope via Savitzky-Golay 1st derivative (rising vs falling edges)
- d2 (acceleration): curvature via Savitzky-Golay 2nd derivative (sharp QRS vs smooth P/T)
"""

import numpy as np
from scipy.signal import savgol_filter


class FeatureExtractor:
    """Extract 3-dimensional feature vectors from preprocessed ECG signals.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    smooth_window_ms : float
        Savitzky-Golay window width in milliseconds.
        Default 20ms (5 samples at 250 Hz) — wide enough to suppress noise
        but narrow enough to preserve QRS slope details.
    poly_order : int
        Polynomial order for Savitzky-Golay filter.
    """

    def __init__(self, fs: float = 250.0, smooth_window_ms: float = 20.0,
                 poly_order: int = 2):
        self.fs = fs
        self.smooth_window_ms = smooth_window_ms
        self.poly_order = poly_order

        # Compute window length in samples
        self._window_len = int(np.round(smooth_window_ms / 1000.0 * fs))
        # Must be odd
        if self._window_len % 2 == 0:
            self._window_len += 1
        # Must be > poly_order
        if self._window_len <= poly_order:
            self._window_len = poly_order + 1
            if self._window_len % 2 == 0:
                self._window_len += 1

        # Feature means/stds for normalization (set during training)
        self._feat_mean = None
        self._feat_std = None

    @property
    def window_len(self) -> int:
        """Savitzky-Golay window length in samples."""
        return self._window_len

    def compute_derivatives(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1st and 2nd derivatives using Savitzky-Golay smoothing.

        Parameters
        ----------
        signal : np.ndarray, shape (N,)
            Preprocessed ECG signal.

        Returns
        -------
        d1 : np.ndarray, shape (N,)
            First derivative (velocity).
        d2 : np.ndarray, shape (N,)
            Second derivative (acceleration).
        """
        wlen = min(self._window_len, len(signal) - (len(signal) % 2 == 0))
        if wlen < self.poly_order + 1:
            # Signal too short — fall back to numpy gradient
            d1 = np.gradient(signal)
            d2 = np.gradient(d1)
            return d1, d2

        d1 = savgol_filter(signal, window_length=wlen, polyorder=self.poly_order, deriv=1)
        d2 = savgol_filter(signal, window_length=wlen, polyorder=self.poly_order, deriv=2)
        return d1, d2

    def extract(self, signal: np.ndarray) -> np.ndarray:
        """Extract 3D feature matrix from preprocessed ECG signal.

        Parameters
        ----------
        signal : np.ndarray, shape (N,)
            Preprocessed (filtered, normalized) ECG signal.

        Returns
        -------
        features : np.ndarray, shape (N, 3)
            Column 0: amplitude (the signal itself)
            Column 1: first derivative (velocity)
            Column 2: second derivative (acceleration)
        """
        signal = np.asarray(signal, dtype=np.float64)

        if signal.ndim != 1:
            raise ValueError(f"Expected 1-D signal, got shape {signal.shape}")

        d1, d2 = self.compute_derivatives(signal)

        features = np.column_stack([signal, d1, d2])
        return features

    def fit_normalizer(self, features: np.ndarray):
        """Compute per-feature mean and std for z-score normalization.

        Parameters
        ----------
        features : np.ndarray, shape (N, 3)
            Feature matrix (from training data).
        """
        self._feat_mean = np.mean(features, axis=0)
        self._feat_std = np.std(features, axis=0)
        self._feat_std = np.maximum(self._feat_std, 1e-8)

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """Apply z-score normalization using stored parameters.

        Parameters
        ----------
        features : np.ndarray, shape (N, 3)

        Returns
        -------
        np.ndarray, shape (N, 3)
            Normalized features.
        """
        if self._feat_mean is None or self._feat_std is None:
            # Fit on-the-fly if not pre-fitted
            self.fit_normalizer(features)
        return (features - self._feat_mean) / self._feat_std
