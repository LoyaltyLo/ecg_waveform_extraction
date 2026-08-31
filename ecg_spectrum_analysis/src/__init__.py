"""All source code for ECG spectrum analysis.

Core modules: spectrogram (STFT / Welch PSD / CWT computation),
plot_spectrogram (matplotlib visualization).
Entry-point script: batch_spectrogram.

Run entry points as modules from the repo root (ECG_engineering), e.g.:
    python -m ecg_spectrum_analysis.src.batch_spectrogram --n 10
output_spectrograms/ lives at the package root, one level up;
scripts anchor it via Path(__file__).
"""
