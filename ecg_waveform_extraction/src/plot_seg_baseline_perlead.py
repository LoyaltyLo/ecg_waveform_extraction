#!/usr/bin/env python3
"""Per-lead segmentation + baseline figures — one PNG per (record, lead).

Reads the batch_limb_leads cache (output/rala_full/_limb_leads/<rec>/) and
saves, for every cached record and lead, a single-lead figure with:
  - HSMM state-colored bands over the baseline-corrected signal
  - raw signal, estimated baseline wander (0.5 Hz low-pass), zero line
  - R-peak markers and per-lead QRS polarity tally

Layout: output/rala_full/_seg_baseline_perlead/<rec>/<rec>_<lead>.png

Usage:
    python -m ecg_waveform_extraction.src.plot_seg_baseline_perlead
    python -m ecg_waveform_extraction.src.plot_seg_baseline_perlead --n 200
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
# NOTE: the aECG XML has no <scale>/<units> tag — raw values are microvolts.
CACHE_DIR = str(Path(__file__).resolve().parent.parent / 'output/rala_full/_limb_leads')
OUT_DIR = str(Path(__file__).resolve().parent.parent / 'output/rala_full/_seg_baseline_perlead')
MAX_SAMPLES = 4000

LEAD_ORDER = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF']
C_RAW, C_BASE, C_CORR = '#9e9e9e', '#f44336', '#1565c0'


def plot_lead(rec: str, ln: str, raw: np.ndarray, fs: float,
              states: np.ndarray, beats: list, save_path: str,
              dpi: int = 130):
    """Single-lead figure: state bands + baseline + corrected + R peaks."""
    raw = raw[:MAX_SAMPLES].astype(np.float64)
    base = estimate_baseline(raw, fs)
    corr = raw - base
    t = np.arange(len(raw)) / fs

    state_names = [STATE_ORDER[i] for i in states[:len(corr)]]

    fig, ax = plt.subplots(figsize=(13, 4.2), constrained_layout=True)
    _fill_states(ax, t, corr, state_names, alpha=0.35)

    ax.plot(t, raw, color=C_RAW, lw=0.6, alpha=0.55, label='raw')
    ax.plot(t, base, color=C_BASE, lw=1.5, ls='--',
            label='baseline (0.5 Hz LP)')
    ax.plot(t, corr, color=C_CORR, lw=0.8, label='corrected')
    ax.axhline(0.0, color='k', lw=0.6, ls=':', alpha=0.6)

    for b in beats:
        r = b.get('r_peak')
        if r is not None and 0 <= r < len(t):
            ax.axvline(r / fs, color=BOUNDARY_STYLES['r_peak']['color'],
                       lw=0.9, alpha=0.8)

    n_neg = sum(1 for b in beats if b.get('polarity') == 'negative')
    n_pos = len(beats) - n_neg
    ax.set_title(f'{rec} — {ln}   wander pp = {np.ptp(base):.3f} µV   '
                 f'QRS: −{n_neg}/+{n_pos}', fontsize=11)
    ax.set_ylabel('µV', fontsize=9)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.12)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color=BOUNDARY_STYLES['r_peak']['color'],
                              lw=1.2))
    labels.append('R peak')
    ax.legend(handles, labels, loc='upper right', fontsize=8, framealpha=0.9)

    patches = [Patch(facecolor=STATE_COLORS[s], label=s) for s in STATE_ORDER]
    fig.legend(handles=patches, loc='outside lower center', ncol=9,
               fontsize=8, frameon=False)

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
          f'-> {OUT_DIR}')
    n_saved = 0
    for idx, rec in enumerate(recs):
        rec_dir = os.path.join(CACHE_DIR, rec)
        leads = [ln for ln in LEAD_ORDER
                 if os.path.isfile(os.path.join(rec_dir, f'lead_{ln}',
                                                'state_labels.npy'))]
        if not leads:
            print(f'[{idx + 1:3d}/{len(recs)}] {rec}... SKIP (no lead cache)')
            continue
        try:
            aecg = parse_aecg(os.path.join(AECG_DIR, f'{rec}.aECG'),
                              max_samples=MAX_SAMPLES)
        except Exception as e:
            print(f'[{idx + 1:3d}/{len(recs)}] {rec}... ERROR parse: {e}')
            continue

        out_rec = os.path.join(OUT_DIR, rec)
        os.makedirs(out_rec, exist_ok=True)
        for ln in leads:
            states = np.load(os.path.join(rec_dir, f'lead_{ln}',
                                          'state_labels.npy'))
            try:
                with open(os.path.join(rec_dir, f'lead_{ln}',
                                       'qrs_polarity.json'),
                          encoding='utf-8') as f:
                    beats = json.load(f)
            except Exception:
                beats = []
            save_path = os.path.join(out_rec, f'{rec}_{ln}.png')
            try:
                plot_lead(rec, ln, aecg['signals'].get(ln), aecg['fs'],
                          states, beats, save_path)
                n_saved += 1
            except Exception as e:
                print(f'[{idx + 1:3d}/{len(recs)}] {rec} {ln}... ERROR: {e}')

        print(f'[{idx + 1:3d}/{len(recs)}] {rec}... OK ({len(leads)} leads, '
              f'total {n_saved} PNGs)')

    print(f'Done. Saved {n_saved} per-lead figures under {OUT_DIR}')


if __name__ == '__main__':
    main()
