#!/usr/bin/env python3
"""Segmentation + Baseline combined plot.

One figure per record: 6 limb leads, each showing the HSMM state-colored
segmentation over the baseline-corrected signal (mV), together with the raw
signal, the estimated baseline wander (0.5 Hz low-pass) and R-peak markers.

Reads the batch_limb_leads cache (output/rala_full/_limb_leads/<rec>/);
re-parses raw signals from the aECG dataset (fast, no HSMM re-run).

Usage:
    python -m ecg_waveform_extraction.src.plot_seg_baseline --n 3
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ecg_waveform_extraction.src.utils.aecg_parser import parse_aecg
from ecg_waveform_extraction.src.plot_segmentation import (
    STATE_COLORS, STATE_ORDER, BOUNDARY_STYLES, _fill_states,
)
from ecg_waveform_extraction.src.plot_ecg_baseline import estimate_baseline

AECG_DIR = 'C:/LoyaltyLo/datasets/RA-LA_Reversal/aECG'
CACHE_DIR = str(Path(__file__).resolve().parent.parent / 'output/rala_full/_limb_leads')
OUT_DIR = str(Path(__file__).resolve().parent.parent / 'output/rala_full/_seg_baseline')
MAX_SAMPLES = 4000

LEAD_ORDER = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF']
C_RAW, C_BASE, C_CORR = '#9e9e9e', '#f44336', '#1565c0'


def plot_record(rec: str, rec_dir: str, aecg: dict, save_path: str, dpi: int = 130):
    """6-lead figure: state bands + baseline + corrected signal + R peaks."""
    fs = aecg['fs']
    leads = [ln for ln in LEAD_ORDER
             if os.path.isdir(os.path.join(rec_dir, f'lead_{ln}'))]
    if not leads:
        return False

    fig, axes = plt.subplots(2, 3, figsize=(16, 7.2),
                             sharex=True, constrained_layout=True)
    axes = axes.ravel()

    for ax, ln in zip(axes, leads):
        raw = aecg['signals'].get(ln)
        if raw is None:
            ax.set_visible(False)
            continue
        raw = raw[:MAX_SAMPLES].astype(np.float64)
        base = estimate_baseline(raw, fs)
        corr = raw - base
        t = np.arange(len(raw)) / fs

        # HSMM state bands (from cache), drawn on the corrected signal
        states = np.load(os.path.join(rec_dir, f'lead_{ln}', 'state_labels.npy'))
        state_names = [STATE_ORDER[i] for i in states[:len(corr)]]
        _fill_states(ax, t, corr, state_names, alpha=0.35)

        ax.plot(t, raw, color=C_RAW, lw=0.6, alpha=0.55, label='raw')
        ax.plot(t, base, color=C_BASE, lw=1.5, ls='--',
                label='baseline (0.5 Hz LP)')
        ax.plot(t, corr, color=C_CORR, lw=0.8, label='corrected')
        ax.axhline(0.0, color='k', lw=0.6, ls=':', alpha=0.6)

        # R-peak markers + polarity tally from cached QRS results
        try:
            with open(os.path.join(rec_dir, f'lead_{ln}', 'qrs_polarity.json'),
                      encoding='utf-8') as f:
                beats = json.load(f)
        except Exception:
            beats = []
        for b in beats:
            r = b.get('r_peak')
            if r is not None and 0 <= r < len(t):
                ax.axvline(r / fs, color=BOUNDARY_STYLES['r_peak']['color'],
                           lw=0.9, alpha=0.8)
        n_neg = sum(1 for b in beats if b.get('polarity') == 'negative')
        n_pos = len(beats) - n_neg

        ax.set_title(f'{ln}   wander pp = {np.ptp(base):.3f} µV   '
                     f'QRS: −{n_neg}/+{n_pos}', fontsize=10)
        ax.set_ylabel('µV', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.12)

    for ax in axes[len(leads):]:
        ax.set_visible(False)
    axes[min(len(leads), 6) - 1].set_xlabel('Time (s)', fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color=BOUNDARY_STYLES['r_peak']['color'],
                              lw=1.2))
    labels.append('R peak')
    axes[0].legend(handles, labels, loc='upper right', fontsize=8,
                   framealpha=0.9)

    patches = [Patch(facecolor=STATE_COLORS[s], label=s) for s in STATE_ORDER]
    fig.legend(handles=patches, loc='outside lower center', ncol=9,
               fontsize=8, frameon=False)

    fig.suptitle(f'{rec} — HSMM segmentation + ECG baseline '
                 f'(fs = {fs:.0f} Hz)', fontsize=12)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=None,
                        help='process first N cached records')
    args = parser.parse_args()

    recs = sorted(d for d in os.listdir(CACHE_DIR)
                  if os.path.isfile(os.path.join(CACHE_DIR, d, 'summary.json')))
    if args.n:
        recs = recs[:args.n]
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f'Segmentation + baseline plots: {len(recs)} records -> {OUT_DIR}')
    for idx, rec in enumerate(recs):
        fpath = os.path.join(AECG_DIR, f'{rec}.aECG')
        try:
            aecg = parse_aecg(fpath, max_samples=MAX_SAMPLES)
            ok = plot_record(rec, os.path.join(CACHE_DIR, rec), aecg,
                             os.path.join(OUT_DIR, f'{rec}_seg_baseline.png'))
            print(f'[{idx + 1:3d}/{len(recs)}] {rec}... '
                  f'{"OK" if ok else "SKIP (no lead cache)"}')
        except Exception as e:
            print(f'[{idx + 1:3d}/{len(recs)}] {rec}... ERROR: {e}')

    print('Done.')


if __name__ == '__main__':
    main()
