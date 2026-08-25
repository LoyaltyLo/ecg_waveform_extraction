#!/usr/bin/env python3
"""ECG Baseline Visualization — per-lead baseline wander & correction.

For each record, plots per lead:
  - raw ECG (gray)
  - estimated baseline (red dashed): 0.5 Hz low-pass of the raw signal,
    i.e. the slow drift the pipeline's 0.5-40 Hz bandpass removes
  - baseline-corrected signal (blue): raw - baseline
  - isoelectric zero line (dotted black)

Usage:
    python -m ecg_waveform_extraction.src.plot_ecg_baseline --n 3
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from ecg_waveform_extraction.src.utils.aecg_parser import parse_aecg

AECG_DIR = 'C:/LoyaltyLo/datasets/RA-LA_Reversal/aECG'
OUT_DIR = str(Path(__file__).resolve().parent.parent / 'output/rala_full/_baseline')
MAX_SAMPLES = 4000

LEAD_ORDER = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

C_RAW = '#9e9e9e'
C_BASE = '#f44336'
C_CORR = '#1565c0'


def estimate_baseline(sig: np.ndarray, fs: float, cutoff: float = 0.5,
                      order: int = 4) -> np.ndarray:
    """Low-frequency baseline estimate = what the 0.5 Hz high-pass removes."""
    b, a = butter(order, cutoff / (fs / 2.0), btype='low')
    return filtfilt(b, a, sig)


def plot_record_baseline(aecg: dict, save_path: str, dpi: int = 130):
    """One figure: all available leads, raw + baseline + corrected."""
    signals = aecg['signals']
    fs = aecg['fs']
    leads = [ln for ln in LEAD_ORDER if ln in signals]
    if not leads:
        return False

    ncols = 3
    nrows = (len(leads) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.0 * nrows),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, ln in zip(axes, leads):
        raw = signals[ln][:MAX_SAMPLES].astype(np.float64)
        base = estimate_baseline(raw, fs)
        corr = raw - base
        t = np.arange(len(raw)) / fs

        ax.plot(t, raw, color=C_RAW, lw=0.7, label='raw')
        ax.plot(t, base, color=C_BASE, lw=1.5, ls='--',
                label='baseline (0.5 Hz LP)')
        ax.plot(t, corr, color=C_CORR, lw=0.7, alpha=0.85,
                label='corrected (raw - baseline)')
        ax.axhline(0.0, color='k', lw=0.6, ls=':', alpha=0.6)

        ax.set_title(f'{ln}   wander pp = {np.ptp(base):.3f} mV', fontsize=10)
        ax.set_ylabel('mV', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.15)

    for ax in axes[len(leads):]:
        ax.set_visible(False)

    axes[0].legend(loc='upper right', fontsize=8, framealpha=0.9)
    axes[len(leads) - 1].set_xlabel('Time (s)', fontsize=9)
    fig.suptitle(f"{aecg['filename']} — ECG baseline  (fs = {fs:.0f} Hz)",
                 fontsize=12)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=None,
                        help='process first N records')
    args = parser.parse_args()

    files = sorted(f for f in os.listdir(AECG_DIR) if f.endswith('.aECG'))
    if args.n:
        files = files[:args.n]
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f'ECG baseline plots: {len(files)} records -> {OUT_DIR}')
    for idx, fname in enumerate(files):
        rec = fname.replace('.aECG', '')
        try:
            aecg = parse_aecg(os.path.join(AECG_DIR, fname),
                              max_samples=MAX_SAMPLES)
            out_png = os.path.join(OUT_DIR, f'{rec}_baseline.png')
            ok = plot_record_baseline(aecg, out_png)
            print(f'[{idx + 1:3d}/{len(files)}] {rec}... '
                  f'{"OK" if ok else "SKIP (no leads)"}')
        except Exception as e:
            print(f'[{idx + 1:3d}/{len(files)}] {rec}... ERROR: {e}')

    print('Done.')


if __name__ == '__main__':
    main()
