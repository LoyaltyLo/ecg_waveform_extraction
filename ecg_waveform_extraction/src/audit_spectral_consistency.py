#!/usr/bin/env python3
"""Spectral-consistency audit of HSMM segmentation (200-record cache).

Second opinion on the HSMM state segmentation using band-energy physics:
the P / QRS / T waves occupy distinct frequency bands, so every labeled
segment should show the band mix expected for its wave type, and every
"flat" (ISO/TP) segment should stay quiet relative to the QRS energy.

Method (per lead):
  1. Filterbank: zero-phase Butterworth band energies from filtered_ecg
     (already 0.5-40 Hz, z-scored):
       vlo  0.5-5 Hz   (T wave / residual drift)
       lo   5-10 Hz    (P wave core)
       hi   10-25 Hz   (QRS core)
  2. Parse state_labels.npy into contiguous runs; merge Q+R+S into QRS.
  3. Per-run spectral scores (continuous, higher = more consistent):
       QRS run: hi_share            = E_hi  / E_total
       P   run: p_lo_share          = E_lo / (E_lo + E_hi)
       T   run: t_low_share         = (E_vlo + E_lo) / E_total
       REST run(ISO/TP): rms_ratio  = rms_run / median QRS-run rms
  4. Flags (thresholds below); beat score = min over its runs, beat is
     flagged if any run is flagged.  Runs shorter than MIN_MS are counted
     as "tiny" (fragmentation indicator) instead of being scored.
  5. Rank (record, lead) by beat flag rate -> suspicious list.

Outputs under output/rala_full/_spectral_audit/:
  audit_results.json   full run/beat-level detail
  audit_summary.md     thresholds, metric distributions, ranked list
  <rec>_<lead>.png     top suspicious examples (flagged runs outlined red)

Usage:
    python -m ecg_waveform_extraction.src.audit_spectral_consistency --n 200
    python -m ecg_waveform_extraction.src.audit_spectral_consistency --top 10
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os, json, argparse
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ecg_waveform_extraction.src.plot_segmentation import STATE_ORDER
from ecg_waveform_extraction.src.plot_seg_baseline_perlead import (
    _fill_qpt, GROUP_COLORS,
)

CACHE_DIR = str(Path(__file__).resolve().parent.parent / 'output/rala_full/_limb_leads')
OUT_DIR = str(Path(__file__).resolve().parent.parent / 'output/rala_full/_spectral_audit')

LEAD_ORDER = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF']

# --- filterbank bands (Hz) ------------------------------------------------
BANDS = {'vlo': (0.6, 5.0), 'lo': (5.0, 10.0), 'hi': (10.0, 25.0)}
FILTER_ORDER = 3

# --- flag thresholds (calibrated on 200-record distributions; see summary) -
THRESH = {
    'qrs_hi_min':   0.30,   # QRS run needs >=30% energy in 10-25 Hz (~p8)
    'p_lo_min':     0.50,   # P run: E5-10 >= E10-25 (lo/(lo+hi) >= 0.5, ~p10)
    't_low_min':    0.65,   # T run needs >=65% energy below 10 Hz (~p7)
    'rest_ratio_max': 0.55, # ISO/TP run RMS must stay < 55% of QRS-run RMS (~p90)
    'qrs_snr_min':  1.8,    # QRS RMS / median REST RMS must exceed this
}

# --- minimum run durations (ms) to be spectrally meaningful ----------------
MIN_MS = {'P': 20, 'QRS': 40, 'T': 40, 'REST': 60}
# beat closes on a REST gap longer than this (ms)
BEAT_GAP_MS = 150
# filtfilt edge zone (samples) — runs touching it are scored but not flagged
EDGE_PAD = 64


# ===========================================================================
# Core
# ===========================================================================
def band_energies(ecg: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    """Per-sample energy contribution in each band (squared band signal)."""
    out = {}
    nyq = fs / 2.0
    for name, (lo, hi) in BANDS.items():
        b, a = butter(FILTER_ORDER, [lo / nyq, hi / nyq], btype='band')
        x = filtfilt(b, a, ecg)
        out[name] = x * x
    return out


def parse_runs(states: np.ndarray) -> list[tuple[str, int, int]]:
    """Contiguous state runs, Q+R+S merged into QRS, ISO/TP into REST.

    Returns [(group, i0, i1)] with half-open [i0, i1) samples.
    """
    group_of = {'P': 'P', 'Q': 'QRS', 'R': 'QRS', 'S': 'QRS', 'T': 'T',
                'ISO': 'REST', 'TP': 'REST', 'PR': 'REST', 'ST': 'REST'}
    runs = []
    start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or group_of[STATE_ORDER[states[i]]] != \
                group_of[STATE_ORDER[states[start]]]:
            runs.append((group_of[STATE_ORDER[states[start]]], start, i))
            start = i
    return runs


def score_runs(runs, energies, fs, n_samples):
    """Spectral score + flag per run.  Returns run dicts."""
    # reference levels: median REST rms and median QRS rms (per lead)
    def rms(seg_len, E_tot):
        return float(np.sqrt(E_tot / max(seg_len, 1)))

    rest_rms, qrs_rms = [], []
    pre = []
    for g, i0, i1 in runs:
        dur_ms = (i1 - i0) / fs * 1000.0
        E_tot = float(sum(energies[b][i0:i1].sum() for b in BANDS))
        pre.append((g, i0, i1, dur_ms, E_tot, rms(i1 - i0, E_tot)))
        if g == 'REST' and dur_ms >= MIN_MS['REST']:
            rest_rms.append(pre[-1][5])
        elif g == 'QRS' and dur_ms >= MIN_MS['QRS']:
            qrs_rms.append(pre[-1][5])
    med_rest = float(np.median(rest_rms)) if rest_rms else None
    med_qrs = float(np.median(qrs_rms)) if qrs_rms else None

    out = []
    for g, i0, i1, dur_ms, E_tot, run_rms in pre:
        E = {b: float(energies[b][i0:i1].sum()) for b in BANDS}
        edge = i0 < EDGE_PAD or i1 > n_samples - EDGE_PAD
        r = {'group': g, 'i0': i0, 'i1': i1, 'dur_ms': round(dur_ms, 1),
             'edge': edge}
        if dur_ms < MIN_MS[g]:
            r.update(tiny=True, score=None, flag=None)
            out.append(r)
            continue

        if g == 'REST':
            ratio = run_rms / med_qrs if med_qrs else None
            r.update(score=None if ratio is None else round(1.0 - ratio, 3),
                     metric='rest_ratio', metric_val=None if ratio is None
                     else round(ratio, 3),
                     flag=(ratio is not None and ratio > THRESH['rest_ratio_max']
                           and not edge))
        else:
            E_tot_safe = E_tot if E_tot > 0 else 1e-12
            hi_share = E['hi'] / E_tot_safe
            p_lo = E['lo'] / (E['lo'] + E['hi'] + 1e-12)
            t_low = (E['vlo'] + E['lo']) / E_tot_safe
            snr = run_rms / med_rest if med_rest and med_rest > 0 else None
            if g == 'QRS':
                metric, mval = 'hi_share', hi_share
                thr = THRESH['qrs_hi_min']
            elif g == 'P':
                metric, mval = 'p_lo_share', p_lo
                thr = THRESH['p_lo_min']
            else:  # T
                metric, mval = 't_low_share', t_low
                thr = THRESH['t_low_min']
            bad_shape = mval < thr
            # SNR gate applies to QRS only: P/T waves are naturally
            # low-amplitude (median wave/REST SNR ~1.1) — gating them
            # would flag physiology, not pathology.
            bad_snr = (snr is not None and g == 'QRS'
                       and snr < THRESH['qrs_snr_min'])
            r.update(metric=metric, metric_val=round(mval, 3),
                     snr=None if snr is None else round(snr, 2),
                     score=round(mval, 3),
                     flag=bool((bad_shape or bad_snr) and not edge))
        out.append(r)
    return out


def group_beats(run_list, fs):
    """Group runs into beats: close on REST gap >= BEAT_GAP_MS."""
    beats, cur = [], []
    for r in run_list:
        if r['group'] == 'REST' and r['dur_ms'] >= BEAT_GAP_MS:
            if any(x['group'] == 'QRS' for x in cur):
                beats.append(cur)
            cur = []
        else:
            cur.append(r)
    if any(x['group'] == 'QRS' for x in cur):
        beats.append(cur)
    return beats


def audit_lead(ecg, states, fs):
    """Full audit for one lead.  Returns (lead_summary, run_list, beats)."""
    energies = band_energies(ecg, fs)
    runs = score_runs(parse_runs(states), energies, fs, len(ecg))
    beats = group_beats(runs, fs)

    beat_out = []
    n_flag = 0
    for bi, rs in enumerate(beats):
        # beat spectral score: worst P/QRS/T run (REST runs excluded —
        # their loudness is a separate noise indicator, not wave-shape)
        scored = [r for r in rs
                  if r['group'] != 'REST' and r['score'] is not None]
        flags = [r for r in rs if r['flag']]
        score = min((r['score'] for r in scored), default=None)
        rec = {'beat_id': bi, 'n_runs': len(rs),
               'score': score, 'flag': bool(flags),
               'reasons': [f"{r['group']}@{r['i0']}:{r['metric']}={r['metric_val']}"
                           for r in flags]}
        beat_out.append(rec)
        n_flag += bool(flags)

    tiny = sum(1 for r in runs if r.get('tiny'))
    n_beats = len(beats)
    scored_scores = [r['score'] for r in runs if r['score'] is not None]
    summary = {
        'n_runs': len(runs), 'n_tiny_runs': tiny, 'n_beats': n_beats,
        'n_flagged_beats': n_flag,
        'flag_rate': round(n_flag / n_beats, 3) if n_beats else None,
        'mean_run_score': round(float(np.mean(scored_scores)), 3)
        if scored_scores else None,
    }
    return summary, runs, beat_out


# ===========================================================================
# Reporting
# ===========================================================================
def print_distributions(all_runs):
    """Percentiles per wave metric — threshold calibration record."""
    lines = ['\n--- metric distributions (threshold calibration) ---']
    for grp, metric in [('QRS', 'hi_share'), ('P', 'p_lo_share'),
                        ('T', 't_low_share')]:
        vals = [r['metric_val'] for r in all_runs
                if r['group'] == grp and r.get('metric_val') is not None]
        if not vals:
            continue
        v = np.array(vals)
        lines.append(f"{grp:4s} {metric:12s} n={len(v):5d}  "
                     f"p5={np.percentile(v, 5):.3f} p25={np.percentile(v, 25):.3f} "
                     f"med={np.median(v):.3f} p75={np.percentile(v, 75):.3f} "
                     f"p95={np.percentile(v, 95):.3f}")
    rest = [r['metric_val'] for r in all_runs
            if r['group'] == 'REST' and r.get('metric_val') is not None]
    if rest:
        v = np.array(rest)
        lines.append(f"REST rest_ratio  n={len(v):5d}  "
                     f"p50={np.median(v):.3f} p75={np.percentile(v, 75):.3f} "
                     f"p95={np.percentile(v, 95):.3f} max={v.max():.3f}")
    snr = [r['snr'] for r in all_runs
           if r['group'] == 'QRS' and r.get('snr') is not None]
    if snr:
        v = np.array(snr)
        lines.append(f"QRS  qrs_snr     n={len(v):5d}  "
                     f"p5={np.percentile(v, 5):.2f} p10={np.percentile(v, 10):.2f} "
                     f"med={np.median(v):.2f}")
    print('\n'.join(lines))
    return lines


def plot_flagged(rec, ln, ecg, states, runs, save_path, fs, dpi=130):
    """Signal + P/QRS/T bands; flagged runs outlined in red (status color)."""
    t = np.arange(len(ecg)) / fs
    fig, ax = plt.subplots(figsize=(13, 4.2), constrained_layout=True)
    _fill_qpt(ax, t, ecg, states, alpha=0.30)
    ax.plot(t, ecg, color='#1565c0', lw=0.8)

    from matplotlib.patches import Patch
    for r in runs:
        if r.get('flag'):
            ax.axvspan(r['i0'] / fs, r['i1'] / fs, facecolor='none',
                       edgecolor='#d32f2f', lw=1.8, alpha=0.9, zorder=3)

    n_flag = sum(1 for r in runs if r.get('flag'))
    tiny = sum(1 for r in runs if r.get('tiny'))
    ax.set_title(f'{rec} — {ln}   flagged runs: {n_flag}   tiny runs: {tiny}',
                 fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.12)
    handles = [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()]
    handles.append(Patch(facecolor='none', edgecolor='#d32f2f', lw=1.8,
                         label='flagged'))
    ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.9)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=None,
                        help='audit first N cached records')
    parser.add_argument('--top', type=int, default=20,
                        help='print/save top-N suspicious (record, lead)')
    args = parser.parse_args()

    recs = sorted(d for d in os.listdir(CACHE_DIR)
                  if os.path.isfile(os.path.join(CACHE_DIR, d, 'summary.json')))
    if args.n:
        recs = recs[:args.n]
    os.makedirs(OUT_DIR, exist_ok=True)

    results, all_runs_flat, ranked = {}, [], []
    print(f'Spectral audit: {len(recs)} records')
    for idx, rec in enumerate(recs):
        rec_dir = os.path.join(CACHE_DIR, rec)
        try:
            with open(os.path.join(rec_dir, 'summary.json'),
                      encoding='utf-8') as f:
                fs = json.load(f)['fs']
        except Exception as e:
            print(f'[{idx + 1:3d}/{len(recs)}] {rec}... ERROR summary: {e}')
            continue
        rec_res = {}
        for ln in LEAD_ORDER:
            lead_dir = os.path.join(rec_dir, f'lead_{ln}')
            if not (os.path.isfile(os.path.join(lead_dir, 'filtered_ecg.npy'))
                    and os.path.isfile(os.path.join(lead_dir, 'state_labels.npy'))):
                continue
            ecg = np.load(os.path.join(lead_dir, 'filtered_ecg.npy'))
            states = np.load(os.path.join(lead_dir, 'state_labels.npy'))
            try:
                summary, runs, beats = audit_lead(ecg, states, fs)
            except Exception as e:
                print(f'[{idx + 1:3d}/{len(recs)}] {rec} {ln}... ERROR: {e}')
                continue
            rec_res[ln] = {'summary': summary, 'runs': runs, 'beats': beats}
            all_runs_flat += runs
            if summary['n_beats']:
                ranked.append({'record': rec, 'lead': ln, **summary})
        results[rec] = rec_res
        if (idx + 1) % 20 == 0:
            print(f'  ... {idx + 1}/{len(recs)} records', flush=True)

    dist_lines = print_distributions(all_runs_flat)

    ranked.sort(key=lambda x: (-x['flag_rate'] if x['flag_rate'] is not None
                               else 0, x['mean_run_score'] or 1))
    head = (f"{'record':<14s} {'lead':<5s} {'beats':>5s} {'flag':>5s} "
            f"{'rate':>6s} {'score':>6s} {'tiny':>5s}")
    table = [head]
    for x in ranked[:args.top]:
        table.append(f"{x['record']:<14s} {x['lead']:<5s} {x['n_beats']:>5d} "
                     f"{x['n_flagged_beats']:>5d} {x['flag_rate']:>6.2f} "
                     f"{x['mean_run_score']:>6.3f} {x['n_tiny_runs']:>5d}")
    print(f'\n--- top {min(args.top, len(ranked))} suspicious (record, lead) '
          f'by beat flag rate ---')
    print('\n'.join(table))

    # save full results + summary report
    with open(os.path.join(OUT_DIR, 'audit_results.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'thresholds': THRESH, 'bands': BANDS,
                   'ranked': ranked, 'results': results}, f,
                  ensure_ascii=False, indent=1)

    n_any = sum(1 for x in ranked if x['flag_rate'] and x['flag_rate'] > 0)
    n_half = sum(1 for x in ranked
                 if x['flag_rate'] is not None and x['flag_rate'] >= 0.5)
    with open(os.path.join(OUT_DIR, 'audit_summary.md'), 'w',
              encoding='utf-8') as f:
        f.write('# Spectral-consistency audit of HSMM segmentation\n\n'
                f'{len(recs)} records x 6 limb leads, cache '
                '`output/rala_full/_limb_leads`.\n\n'
                '## Thresholds\n\n```\n' +
                '\n'.join(f'{k:16s} {v}' for k, v in THRESH.items()) +
                '\n```\n\n## Metric distributions\n\n```\n' +
                '\n'.join(dist_lines).strip() + '\n```\n\n'
                f'Leads with >=1 flagged beat: {n_any}/{len(ranked)}; '
                f'with >=50% beats flagged: {n_half}/{len(ranked)}.\n\n'
                f'## Top {min(args.top, len(ranked))} suspicious '
                '(record, lead)\n\n```\n' + '\n'.join(table) + '\n```\n')

    # top-N example figures
    for x in ranked[:min(5, len(ranked))]:
        rec, ln = x['record'], x['lead']
        ecg = np.load(os.path.join(CACHE_DIR, rec, f'lead_{ln}',
                                   'filtered_ecg.npy'))
        states = np.load(os.path.join(CACHE_DIR, rec, f'lead_{ln}',
                                      'state_labels.npy'))
        plot_flagged(rec, ln, ecg, states, results[rec][ln]['runs'],
                     os.path.join(OUT_DIR, f'{rec}_{ln}.png'), fs)

    print(f'\nLeads with >=1 flagged beat: {n_any}/{len(ranked)}; '
          f'>=50% beats flagged: {n_half}/{len(ranked)}')
    print(f'Done. Results -> {OUT_DIR}')


if __name__ == '__main__':
    main()
