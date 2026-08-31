#!/usr/bin/env python3
"""Per-lead segmentation + baseline figures — one PNG per (record, lead).

Reads the batch_limb_leads cache (output/rala_full/_limb_leads/<rec>/) and
saves, for every cached record and lead, a single-lead figure with:
  - corrected ECG signal
  - estimated baseline wander (0.5 Hz low-pass, red dashed)
  - three shaded bands only: P (blue), QRS (red, Q+R+S merged), T (green)

Layout: output/rala_full/_seg_baseline_perlead/<lead>/<rec>_<lead>.png
(grouped by lead: I/, II/, III/, AVR/, AVL/, AVF/ — 200 records each)

Usage:
    python -m ecg_waveform_extraction.src.plot_seg_baseline_perlead
    python -m ecg_waveform_extraction.src.plot_seg_baseline_perlead --n 200
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ecg_waveform_extraction.src.utils.aecg_parser import parse_aecg
from ecg_waveform_extraction.src.plot_segmentation import STATE_COLORS, STATE_ORDER
from ecg_waveform_extraction.src.plot_ecg_baseline import estimate_baseline

AECG_DIR = 'C:/LoyaltyLo/datasets/RA-LA_Reversal/aECG'
# NOTE: the aECG XML has no <scale>/<units> tag — raw values are microvolts.
CACHE_DIR = str(Path(__file__).resolve().parent.parent / 'output/rala_full/_limb_leads')
OUT_DIR = str(Path(__file__).resolve().parent.parent / 'output/rala_full/_seg_baseline_perlead')
MAX_SAMPLES = 4000

LEAD_ORDER = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF']
C_BASE, C_CORR = '#f44336', '#1565c0'

# P / QRS / T band colors — reuse the 9-state palette hexes
GROUP_COLORS = {
    'P':   STATE_COLORS['P'],
    'QRS': STATE_COLORS['R'],
    'T':   STATE_COLORS['T'],
}


def _fill_qpt(ax, t, sig, states, alpha: float = 0.30):
    """Fill contiguous P / QRS / T bands (9 HSMM states merged into 3)."""
    states = np.asarray(states)
    group = np.full(len(states), '', dtype=object)
    group[states == STATE_ORDER.index('P')] = 'P'
    for s in ('Q', 'R', 'S'):
        group[states == STATE_ORDER.index(s)] = 'QRS'
    group[states == STATE_ORDER.index('T')] = 'T'

    start = 0
    for i in range(1, len(group) + 1):
        if i == len(group) or group[i] != group[start]:
            if group[start]:
                ax.fill_between(t[start:i], 0.0, sig[start:i],
                                color=GROUP_COLORS[group[start]],
                                alpha=alpha, lw=0)
            start = i


def plot_lead(rec: str, ln: str, raw: np.ndarray, fs: float,
              states: np.ndarray, save_path: str, dpi: int = 130):
    """Single-lead figure: corrected signal + baseline + P/QRS/T bands."""
    raw = raw[:MAX_SAMPLES].astype(np.float64)
    base = estimate_baseline(raw, fs)
    corr = raw - base
    t = np.arange(len(raw)) / fs

    fig, ax = plt.subplots(figsize=(13, 4.2), constrained_layout=True)
    _fill_qpt(ax, t, corr, states[:len(corr)])

    ax.plot(t, base, color=C_BASE, lw=1.5, ls='--',
            label='baseline (0.5 Hz LP)')
    ax.plot(t, corr, color=C_CORR, lw=0.8, label='corrected')

    ax.set_title(f'{rec} — {ln}   wander pp = {np.ptp(base):.3f} µV',
                 fontsize=11)
    ax.set_ylabel('µV', fontsize=9)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.12)

    handles, labels = ax.get_legend_handles_labels()
    handles += [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.9)

    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


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

    print(f'Per-lead segmentation + baseline plots: {len(recs)} records '
          f'-> {OUT_DIR}', flush=True)
    n_saved = 0
    for idx, rec in enumerate(recs):
        rec_dir = os.path.join(CACHE_DIR, rec)
        leads = [ln for ln in LEAD_ORDER
                 if os.path.isfile(os.path.join(rec_dir, f'lead_{ln}',
                                                'state_labels.npy'))]
        if not leads:
            print(f'[{idx + 1:3d}/{len(recs)}] {rec}... SKIP (no lead cache)',
                  flush=True)
            continue
        try:
            aecg = parse_aecg(os.path.join(AECG_DIR, f'{rec}.aECG'),
                              max_samples=MAX_SAMPLES)
        except Exception as e:
            print(f'[{idx + 1:3d}/{len(recs)}] {rec}... ERROR parse: {e}',
                  flush=True)
            continue

        for ln in leads:
            states = np.load(os.path.join(rec_dir, f'lead_{ln}',
                                          'state_labels.npy'))
            out_ln = os.path.join(OUT_DIR, ln)
            os.makedirs(out_ln, exist_ok=True)
            save_path = os.path.join(out_ln, f'{rec}_{ln}.png')
            try:
                plot_lead(rec, ln, aecg['signals'].get(ln), aecg['fs'],
                          states, save_path)
                n_saved += 1
            except Exception as e:
                print(f'[{idx + 1:3d}/{len(recs)}] {rec} {ln}... ERROR: {e}',
                      flush=True)

        print(f'[{idx + 1:3d}/{len(recs)}] {rec}... OK ({len(leads)} leads, '
              f'total {n_saved} PNGs)', flush=True)

    print(f'Done. Saved {n_saved} per-lead figures under {OUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
