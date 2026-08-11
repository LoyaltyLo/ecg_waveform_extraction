"""ECG Spectrum Analysis Package.

Provides tools for converting ECG time-domain signals into frequency-domain
representations: STFT spectrograms, power spectral density (PSD), and
continuous wavelet transform (CWT) scalograms.
"""

from .spectrogram import (
    compute_spectrogram,
    compute_psd,
    compute_scalogram,
    ECG_Spectrogram,
)

__all__ = [
    "compute_spectrogram",
    "compute_psd",
    "compute_scalogram",
    "ECG_Spectrogram",
]
