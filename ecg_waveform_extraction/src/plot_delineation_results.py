"""Delineation result plots: HSMM vs prominence-refined P/T boundaries.

Runs the LimbLeadProcessor pipeline twice per record (prominence off/on) and
draws both boundary sets so the refinement can be inspected visually:

  1. Per-lead overviews  — full-record trace with HSMM (blue, dashed edge)
     and prominence-refined (orange, solid edge) P/T windows.
  2. Beat comparison     — top-N beats by boundary shift, aligned at the R
     peak (x in ms), HSMM vs refined side by side.
  3. Synthetic ground truth — 1 kHz synthetic ECG with known boundaries;
     truth (gray dash-dot) vs HSMM vs refined, with median error stats.

Usage:
    /path/to/python -m src.plot_delineation_results
Output: output/delineation_results/ (overviews, beat panels, ground truth).
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.aecg_parser import parse_aecg
from src.utils.data_loader import generate_synthetic_ecg
from src.limb_lead_processor import LimbLeadProcessor, LIMB_LEADS

# ---------------------------------------------------------------------------
AECG_DIR = 'C:/LoyaltyLo/datasets/RA-LA_Reversal/aECG'
OUT_DIR = str(Path(__file__).resolve().parent.parent / 'output/delineation_results')
MAX_SAMPLES = 16000

# Real records to plot: (filename, leads to draw).
RECORDS = [
    ('1805185J6U.aECG', ['I', 'II', 'AVR']),
    ('180518ZG06.aECG', ['II']),
]
N_BEAT_PANELS = 3        # most-shifted beats per comparison figure
SYNTHETIC_FS = 1000.0

# Fixed categorical identity colors (CVD-validated pair, protan deltaE 27.6)
C_HSMM = '#3a63c8'       # blue   — Stage-1 HSMM boundaries
C_PROM = '#d96f1e'       # orange — prominence-refined boundaries
C_TRUTH = '#555555'      # neutral — synthetic ground truth (reference only)
C_QRS = '#888888'        # light gray QRS context band
C_INK = '#222222'        # primary text
C_MUTED = '#666666'      # secondary text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_record(aecg_data):
    """Process one record twice: prominence off (HSMM) and on (refined)."""
    off = LimbLeadProcessor(max_samples=MAX_SAMPLES,
                            use_prominence_delineation=False)
    on = LimbLeadProcessor(max_samples=MAX_SAMPLES,
                           use_prominence_delineation=True)
    _, seg_off = off.process_record(aecg_data)
    _, seg_on = on.process_record(aecg_data)
    return seg_off, seg_on


def _match_beats(beats_off, beats_on, tol):
    """Pair beats across the two runs by r_peak proximity (GMM init is
    nondeterministic, so beat counts can differ slightly)."""
    pairs = []
    used = set()
    for bo in beats_off:
        if bo is None or bo.r_peak <= 0:
            continue
        best, best_d = None, tol
        for j, bn in enumerate(beats_on):
            if bn is None or bn.r_peak <= 0 or j in used:
                continue
            d = abs(bn.r_peak - bo.r_peak)
            if d < best_d:
                best, best_d = (j, bn), d
        if best is not None:
            used.add(best[0])
            pairs.append((bo, best[1]))
    return pairs


def _window(b, on_attr, off_attr):
    """Return (onset, offset) of a beat wave if both valid, else None."""
    on, off = getattr(b, on_attr, -1), getattr(b, off_attr, -1)
    if on >= 0 and off > on:
        return on, off
    return None


def _draw_pt_windows(ax, beats, fs, color, linestyle, alpha, t_max=None):
    """Draw P/T window bands for a list of BeatBoundary objects."""
    for b in beats:
        for attrs in (('p_onset', 'p_offset'), ('t_onset', 't_offset')):
            win = _window(b, *attrs)
            if win is None:
                continue
            if t_max is not None and win[0] > t_max:
                continue
            x0, x1 = win[0] / fs, win[1] / fs
            ax.axvspan(x0, x1, color=color, alpha=alpha, linewidth=0, zorder=2)
            for x in (x0, x1):
                ax.axvline(x, color=color, lw=0.9, ls=linestyle, alpha=0.9,
                           zorder=3)


# ---------------------------------------------------------------------------
# Plot 1: full-record overview
# ---------------------------------------------------------------------------

def plot_overview(rec_name, lead_name, seg_off, seg_on, out_dir):
    sd_off, sd_on = seg_off.get(lead_name), seg_on.get(lead_name)
    if sd_on is None or sd_off is None:
        return None
    fs = sd_on.get('fs', 1000.0)
    ecg = sd_on['filtered_ecg']
    T = len(ecg)
    t = np.arange(T) / fs

    fig, ax = plt.subplots(figsize=(16, 4.2))
    ax.plot(t, ecg, color='#111111', lw=0.5, zorder=1)

    # QRS context (from refined beats)
    for b in sd_on.get('beats', []):
        if b is None or b.q_onset <= 0 or b.s_offset <= 0:
            continue
        ax.axvspan(b.q_onset / fs, b.s_offset / fs, color=C_QRS, alpha=0.10,
                   linewidth=0, zorder=0)

    _draw_pt_windows(ax, sd_off.get('beats', []), fs, C_HSMM, '--', 0.10)
    _draw_pt_windows(ax, sd_on.get('beats', []), fs, C_PROM, '-', 0.20)

    n_prom = sd_on.get('prominence_refined_beats', 0)
    ax.legend(handles=[
        Line2D([0], [0], color='#111111', lw=0.5, label='ECG (filtered)'),
        Patch(facecolor=C_QRS, alpha=0.10, label='QRS (refined)'),
        Patch(facecolor=C_HSMM, alpha=0.10, label='P/T — HSMM'),
        Patch(facecolor=C_PROM, alpha=0.20, label='P/T — prominence'),
    ], loc='upper right', ncol=4, fontsize=8, framealpha=0.7)

    ax.set_xlim(0, T / fs)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (norm)')
    ax.set_title(f'{rec_name} — Lead {lead_name}: HSMM vs prominence '
                 f'(refined {n_prom} beats)', color=C_INK,
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.12)
    fig.tight_layout()

    path = os.path.join(out_dir, f'overview_{lead_name}.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 2: most-shifted beat comparison (R-aligned, ms axis)
# ---------------------------------------------------------------------------

def plot_beat_panels(rec_name, lead_name, seg_off, seg_on, out_dir,
                     n_panels=N_BEAT_PANELS, truth=None):
    sd_off, sd_on = seg_off.get(lead_name), seg_on.get(lead_name)
    if sd_on is None or sd_off is None:
        return None, []
    fs = sd_on.get('fs', 1000.0)
    ecg = sd_on['filtered_ecg']

    pairs = _match_beats(sd_off.get('beats', []), sd_on.get('beats', []),
                         tol=int(0.04 * fs))
    if not pairs:
        return None, []

    # Edge beats (phantom decodes near the record end) have no full plot
    # window — exclude them from the picks.
    T = len(ecg)
    pairs = [(bo, bn) for bo, bn in pairs
             if bn.r_peak - int(0.35 * fs) >= 0
             and bn.r_peak + int(0.55 * fs) < T]
    if not pairs:
        return None, []

    # rank by total boundary shift (ms)
    def _shift(pair):
        bo, bn = pair
        s = 0.0
        for attrs in (('p_onset', 'p_offset'), ('t_onset', 't_offset')):
            wo, wn = _window(bo, *attrs), _window(bn, *attrs)
            if wo and wn:
                s += abs(wn[0] - wo[0]) + abs(wn[1] - wo[1])
        return s / fs * 1000.0

    pairs.sort(key=_shift, reverse=True)
    picks = pairs[:n_panels]

    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 2.6 * len(picks)),
                             sharex=False)
    if len(picks) == 1:
        axes = [axes]

    stats = []
    for ax, (bo, bn) in zip(axes, picks):
        r = bn.r_peak
        ms = lambda idx: (idx - r) / fs * 1000.0
        w0, w1 = ms(max(0, r - int(0.35 * fs))), ms(min(len(ecg) - 1, r + int(0.55 * fs)))
        x = np.arange(max(0, r - int(0.35 * fs)), min(len(ecg), r + int(0.55 * fs)))
        ax.plot((x - r) / fs * 1000.0, ecg[x], color='#111111', lw=0.8, zorder=3)
        ax.axvline(0, color='#333333', lw=0.8, ls=':', zorder=2)

        for b, color, ls, alpha in ((bo, C_HSMM, '--', 0.12),
                                    (bn, C_PROM, '-', 0.22)):
            for attrs in (('p_onset', 'p_offset'), ('t_onset', 't_offset')):
                win = _window(b, *attrs)
                if win is None:
                    continue
                ax.axvspan(ms(win[0]), ms(win[1]), color=color, alpha=alpha,
                           linewidth=0, zorder=1)
                for xb in win:
                    ax.axvline(ms(xb), color=color, lw=1.0, ls=ls, zorder=2)

        if truth is not None:
            tt = min(truth, key=lambda tb: abs(tb['R_peak'] - r))
            if abs(tt['R_peak'] - r) <= int(0.04 * fs):
                for key in ('P_onset', 'P_offset', 'T_onset', 'T_offset'):
                    ax.axvline(ms(tt[key]), color=C_TRUTH, lw=1.0, ls='-.',
                               zorder=2)
                ax.axvline(ms(tt['Q_onset']), color=C_TRUTH, lw=0.8, ls=':', zorder=2)

        d = _shift((bo, bn))
        stats.append({'beat_id': bn.beat_id, 'r_peak': r, 'shift_ms': round(d, 1)})
        src = f"P:{bn.p_source}/T:{bn.t_source}"
        ax.set_title(f'beat {bn.beat_id} (R={r})  |shift|={d:.1f} ms  [{src}]',
                     fontsize=9, color=C_MUTED, loc='left')
        ax.set_xlim(w0, w1)
        ax.set_xlabel('ms relative to R peak')
        ax.grid(True, alpha=0.12)

    handles = [
        Patch(facecolor=C_HSMM, alpha=0.12, label='P/T — HSMM'),
        Patch(facecolor=C_PROM, alpha=0.22, label='P/T — prominence'),
    ]
    if truth is not None:
        handles.append(Line2D([0], [0], color=C_TRUTH, lw=1.0, ls='-.',
                              label='ground truth'))
    axes[0].legend(handles=handles, loc='upper right', fontsize=8,
                   framealpha=0.7, ncol=len(handles))

    fig.suptitle(f'{rec_name} — Lead {lead_name}: top-{len(picks)} shifted '
                 f'beats (R-aligned)', color=C_INK, fontsize=12,
                 fontweight='bold')
    fig.tight_layout()

    path = os.path.join(out_dir, f'beats_{lead_name}.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return path, stats


# ---------------------------------------------------------------------------
# Plot 3: synthetic ground truth
# ---------------------------------------------------------------------------

def plot_synthetic_ground_truth(out_dir):
    data = generate_synthetic_ecg(fs=SYNTHETIC_FS, duration_sec=10.0,
                                  heart_rate=60.0, noise_std=0.02,
                                  random_state=42)
    aecg_like = {
        'filename': 'synthetic_1khz_hr60', 'fs': SYNTHETIC_FS,
        'signals': {'II': data['ecg']},
        'measurements': {}, 'interpretation': '',
    }
    seg_off, seg_on = run_record(aecg_like)

    rec_dir = os.path.join(out_dir, 'synthetic_1khz_hr60')
    path, stats = plot_beat_panels(
        'synthetic_1khz_hr60', 'II',
        {'II': seg_off.get('II')}, {'II': seg_on.get('II')}, rec_dir,
        n_panels=3, truth=data['true_boundaries'])

    # Accuracy summary: HSMM vs refined against truth
    sd_off, sd_on = seg_off['II'], seg_on['II']
    fs = SYNTHETIC_FS
    truth = data['true_boundaries']

    def _errors(beats, attrs, tkeys):
        errs = []
        for b in beats:
            win = _window(b, *attrs)
            if win is None or b.r_peak <= 0:
                continue
            tt = min(truth, key=lambda tb: abs(tb['R_peak'] - b.r_peak))
            if abs(tt['R_peak'] - b.r_peak) > int(0.04 * fs):
                continue
            errs.append(abs(win[0] - tt[tkeys[0]]) / fs * 1000.0)
            errs.append(abs(win[1] - tt[tkeys[1]]) / fs * 1000.0)
        return errs

    rows = []
    for label, beats in (('HSMM', sd_off['beats']), ('prominence', sd_on['beats'])):
        p_err = _errors(beats, ('p_onset', 'p_offset'), ('P_onset', 'P_offset'))
        t_err = _errors(beats, ('t_onset', 't_offset'), ('T_onset', 'T_offset'))
        rows.append((label,
                     float(np.median(p_err)) if p_err else float('nan'),
                     float(np.median(t_err)) if t_err else float('nan')))

    txt = [f"synthetic 1 kHz, HR=60, 10 s — median boundary error vs truth (ms)"]
    for label, pm, tm in rows:
        txt.append(f"  {label:<11} P on/off: {pm:5.1f} ms   T on/off: {tm:5.1f} ms")
    summary = '\n'.join(txt)
    print(summary)
    with open(os.path.join(out_dir, 'synthetic_accuracy.txt'), 'w',
              encoding='utf-8') as f:
        f.write(summary + '\n')
    return path, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'Output: {OUT_DIR}')

    for fname, leads in RECORDS:
        fpath = os.path.join(AECG_DIR, fname)
        if not os.path.exists(fpath):
            print(f'  SKIP {fname} (not found)')
            continue
        rec = os.path.splitext(fname)[0]
        rec_dir = os.path.join(OUT_DIR, rec)
        print(f'  {rec}: processing (2 runs)...')
        aecg = parse_aecg(fpath, max_samples=MAX_SAMPLES)
        seg_off, seg_on = run_record(aecg)
        for lead in leads:
            p = plot_overview(rec, lead, seg_off, seg_on, rec_dir)
            bp, stats = plot_beat_panels(rec, lead, seg_off, seg_on, rec_dir)
            print(f'    lead {lead}: overview={os.path.basename(p) if p else "—"} '
                  f'beats={os.path.basename(bp) if bp else "—"} '
                  f'shifts={[s["shift_ms"] for s in stats]}')

    print('  synthetic ground truth...')
    plot_synthetic_ground_truth(OUT_DIR)
    print('Done.')


if __name__ == '__main__':
    main()
