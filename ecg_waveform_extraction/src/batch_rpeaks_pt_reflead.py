"""Pan-Tompkins R-peak localization with SQI-based reference-lead selection.

Three stages, all library implementations (no hand-rolled algorithms):

1. **Waveform quality assessment (per lead).** Each of the 12 leads is scored
   with neurokit2's `ecg_quality(method="templatematch")` (Orphanidau et al.,
   2015): the mean correlation between each individual beat morphology and the
   average (template) beat. High score = consistent, clean QRS complexes.
2. **Reference-lead selection.** The lead with the highest quality score is
   chosen (ties -> lead order). Flat / degenerate leads are excluded.
3. **Detection + transfer.** sleepecg's Pan-Tompkins (1985) runs on the
   reference lead only; its R-peak times become the standard for all other
   leads (same cardiac instants marked on every lead).

Selection runs *before* detection, so no detection result can inform it: only
the waveform itself is available. Method choice was made on a 100-record
offline evaluation (the 25 records with |PT HR - device HR| > 5 bpm on lead II
plus 75 random draws), where templatematch reached 90.9% within 5 bpm vs 88.0%
for a Zhao-2018-style composite score, 75.0% for always using lead II, and
98.0% for an oracle picking the best lead with hindsight.

Full-run validation (all 666 records, 2026-09)
----------------------------------------------
Agreement with the device's annotated HR: 98.3% of records within 5 bpm
(median |delta| 0.5, P90 1.4), up from 96.2% when always using lead II.
Selection fixed 17 of the 25 records where lead II was off by more than 5 bpm
and broke 3; records off by more than 20 bpm fell from 9 to 3. The chosen
reference lead never scored below 0.78 (median 0.997), so the quality gate
does not hand detection a degenerate lead.

Reference lead distribution (n=666): V2 178, V4 128, V3 108, V5 75, V6 39,
V1 31, III 30, II 26, AVF 24, I 13, AVL 8, AVR 6 -- precordial in 84% of
records, matching the precordial leads' larger QRS amplitudes and higher
per-lead agreement seen in the independent-lead run.

The 11 records still more than 5 bpm off the device HR are dominated by
genuine arrhythmias and by the limb-lead reversals this dataset is built
around (e.g. 2004226RCY, 180728FK24, where the per-beat count and the device's
representative rate legitimately disagree); they are not detector defects.

Usage:
    python src/batch_rpeaks_pt_reflead.py [--limit N] [--data-dir DIR] [--out-dir DIR]

Outputs
-------
    <out-dir>/plots/<record>.png                  4x3 overview, reference highlighted,
                                                  reference R-peak times on every lead
    <out-dir>/plots_per_lead/<record>/<lead>.png  one figure per (record, lead)
    <out-dir>/sqi_by_lead.csv                     per-(record, lead) quality scores
    <out-dir>/summary_by_record.csv               reference lead, PT stats, device HR
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import neurokit2 as nk
from sleepecg import detect_heartbeats

from ecg_waveform_extraction.src.utils.aecg_parser import parse_aecg
from ecg_waveform_extraction.src.batch_rpeaks_pan_tompkins import (
    DISPLAY_LEADS, ALL_LEADS, read_scale_uv, detect_one_lead, beat_stats,
)

# ---- Config ----
DEFAULT_DATA_DIR = r'C:\LoyaltyLo\datasets\RA-LA_Reversal\aECG'
DEFAULT_OUT_DIR = str(Path(__file__).resolve().parent.parent / 'output' / 'pan_tompkins_rpeaks_ref')
FALLBACK_LEAD = 'II'   # used only if no lead yields a quality score


def lead_quality(sig: np.ndarray, fs: float) -> float:
    """Mean templatematch quality score for one lead (nan if not assessable)."""
    if sig.size == 0 or np.ptp(sig) == 0:
        return np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            q = nk.ecg_quality(sig, sampling_rate=fs, method='templatematch')
        return float(np.mean(q))
    except Exception:
        return np.nan


def save_single_lead_figure(rec_name: str, lead: str, mv: np.ndarray, fs: float,
                            beats: np.ndarray, out_png: str,
                            ref_lead: str, ref_stats: dict, dev_hr=None):
    """One lead with the reference lead's R-peak times marked."""
    fig, ax = plt.subplots(figsize=(12, 3.5))
    t = np.arange(len(mv)) / fs
    ax.plot(t, mv, color='k', linewidth=0.7)
    if len(beats):
        ax.plot(t[beats], mv[beats], 'o', color='#d32f2f', markersize=5, zorder=5)
    tag = '  [reference]' if lead == ref_lead else ''
    title = (f"{rec_name} — lead {lead}{tag}  |  R peaks from reference lead {ref_lead} "
             f"(Pan-Tompkins): {ref_stats['n_beats']} beats"
             + (f", HR {ref_stats['mean_HR_bpm']:.1f} bpm"
                + (f"  (device {dev_hr:.0f})" if dev_hr else '') if ref_stats['n_beats'] >= 2 else ''))
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('mV')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


def process_record(rec: dict, out_png: str, per_lead_dir: str) -> tuple[list[dict], dict]:
    """Score every lead, pick the reference, detect there, plot, return metrics."""
    sigs = rec['signals']
    fs = rec['fs'] or 1000.0
    scale_uv = read_scale_uv(rec['filepath'])
    dev_hr = rec['measurements'].get('HR')

    # ---- 1. quality score per lead ----
    sqis = {lead: lead_quality(sigs[lead], fs) for lead in ALL_LEADS if lead in sigs}

    # ---- 2. reference-lead selection ----
    ranked = sorted(((s, lead) for lead, s in sqis.items() if not np.isnan(s)),
                    key=lambda sl: (-sl[0], ALL_LEADS.index(sl[1])))
    if ranked:
        ref_sqi, ref_lead = ranked[0]
    else:  # no lead assessable: fall back to II / first available
        ref_lead = FALLBACK_LEAD if FALLBACK_LEAD in sigs else next(iter(sigs))
        ref_sqi = np.nan

    # ---- 3. detect on the reference lead ----
    ref_beats, ref_status = detect_one_lead(sigs[ref_lead], fs)
    ref_stats = beat_stats(ref_beats, fs)

    sqi_rows = [{'record': rec['filename'], 'lead': lead, 'sqi_tm': sqis.get(lead, np.nan),
                 'is_reference': lead == ref_lead}
                for lead in ALL_LEADS]

    # ---- combined 4x3 figure: reference beat times marked on every lead ----
    fig, axes = plt.subplots(4, 3, figsize=(16, 9))
    fig.suptitle(f"{rec['filename']}  |  reference lead {ref_lead} (SQI "
                 f"{ref_sqi:.3f})  |  Pan-Tompkins: {ref_stats['n_beats']} R peaks"
                 + (f", HR {ref_stats['mean_HR_bpm']:.1f} bpm" if ref_stats['n_beats'] >= 2 else '')
                 + (f"  (device {dev_hr:.0f} bpm)" if dev_hr else ''),
                 fontsize=13, fontweight='bold', y=0.985)
    for r in range(4):
        for c in range(3):
            ax = axes[r, c]
            lead = DISPLAY_LEADS[r][c]
            if lead not in sigs:
                ax.text(0.5, 0.5, 'lead missing', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='gray')
                ax.set_title(lead, fontsize=10, fontweight='bold', loc='left')
                continue
            mv = sigs[lead] * scale_uv / 1000.0
            t = np.arange(len(mv)) / fs
            ax.plot(t, mv, color='k', linewidth=0.6)
            if len(ref_beats):
                ax.plot(t[ref_beats], mv[ref_beats], 'o', color='#d32f2f',
                        markersize=4, zorder=5)
            s = sqis.get(lead, np.nan)
            star = ' \u2605' if lead == ref_lead else ''
            label = (f"{lead}{star}  SQI {s:.2f}" if not np.isnan(s)
                     else f"{lead}  SQI n/a")
            ax.set_title(label, fontsize=10, fontweight='bold', loc='left')
            ax.grid(True, alpha=0.25)
            if r == 3:
                ax.set_xlabel('Time (s)')
            if c == 0:
                ax.set_ylabel('mV')
            if per_lead_dir:
                save_single_lead_figure(rec['filename'], lead, mv, fs, ref_beats,
                                        os.path.join(per_lead_dir, f'{lead}.png'),
                                        ref_lead, ref_stats, dev_hr)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    row = {
        'record': rec['filename'],
        'ref_lead': ref_lead,
        'ref_sqi': round(ref_sqi, 4) if not np.isnan(ref_sqi) else np.nan,
        'ref_status': ref_status,
        'device_HR_bpm': dev_hr,
        **{k: ref_stats[k] for k in ('n_beats', 'mean_HR_bpm', 'median_RR_ms',
                                     'min_RR_ms', 'max_RR_ms')},
        'interpretation': rec['interpretation'][:120],
    }
    return sqi_rows, row


def main():
    ap = argparse.ArgumentParser(description='SQI reference-lead selection + Pan-Tompkins')
    ap.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    ap.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    ap.add_argument('--limit', type=int, default=None, help='process only first N files (smoke test)')
    args = ap.parse_args()

    plots_dir = os.path.join(args.out_dir, 'plots')
    per_lead_base = os.path.join(args.out_dir, 'plots_per_lead')
    os.makedirs(plots_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(args.data_dir) if f.endswith('.aECG'))
    if args.limit:
        files = files[:args.limit]

    sqi_rows, rec_rows, errors = [], [], []
    t0 = time.time()
    for i, fname in enumerate(files, 1):
        try:
            rec = parse_aecg(os.path.join(args.data_dir, fname))
            if not rec['signals']:
                raise ValueError('no lead signals parsed')
            rec_dir = os.path.join(per_lead_base, rec['filename'])
            os.makedirs(rec_dir, exist_ok=True)
            sr, rr = process_record(rec, os.path.join(plots_dir, f"{rec['filename']}.png"), rec_dir)
            sqi_rows.extend(sr)
            rec_rows.append(rr)
        except Exception as e:
            errors.append((fname, str(e)))
        if i % 50 == 0 or i == len(files):
            print(f'[{i}/{len(files)}] elapsed {time.time() - t0:.0f}s, errors: {len(errors)}', flush=True)

    pd.DataFrame(sqi_rows).to_csv(os.path.join(args.out_dir, 'sqi_by_lead.csv'),
                                  index=False, encoding='utf-8-sig')
    df = pd.DataFrame(rec_rows)
    p_rec = os.path.join(args.out_dir, 'summary_by_record.csv')
    df.to_csv(p_rec, index=False, encoding='utf-8-sig')

    print(f"\nDone: {len(rec_rows)} records, {len(errors)} errors / {len(files)} files "
          f"in {time.time() - t0:.0f}s")
    ok = df.dropna(subset=['mean_HR_bpm', 'device_HR_bpm'])
    if len(ok):
        d = (ok['mean_HR_bpm'] - ok['device_HR_bpm']).abs()
        print(f"Reference-lead PT HR vs device HR: median |d| {d.median():.1f} bpm, "
              f"P90 {d.quantile(0.9):.1f}, within 5 bpm {100 * (d <= 5).mean():.1f}% (n={len(ok)})")
    print('Reference lead distribution:', df['ref_lead'].value_counts().to_dict())
    print(f"CSV: {p_rec}")
    for fname, err in errors[:10]:
        print(f'  ERROR {fname}: {err}')


if __name__ == '__main__':
    main()
