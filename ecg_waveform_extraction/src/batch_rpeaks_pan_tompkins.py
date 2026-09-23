"""R-peak localization on the RA-LA Reversal aECG dataset with Pan-Tompkins (1985).

Uses `sleepecg.detect_heartbeats` — a maintained library implementation of the
Pan & Tompkins (1985) adaptive-threshold beat detector (bandpass 5-30 Hz,
derivative, squaring, centered moving integration, dual adaptive thresholds,
searchback with repeated threshold reduction). No hand-rolled detector code
here: the detector is the library's, called as-is on every lead.

Signals are parsed with the project's unified aECG parser; raw digits are
converted to mV via the per-file <scale> attribute (0.305 uV/unit dataset-wide).

Detection is run independently on EACH of the 12 leads and all 12 result sets
are kept — nothing is selected or merged. For every record a 4x3 overview PNG
is saved in which each lead panel shows that lead's own detected R peaks, and
the per-(record, lead) metrics go into a summary CSV. A second CSV aggregates
the 12 leads per record for auditing.

Validation (12-lead run over all 666 records, 2026-09)
------------------------------------------------------
Agreement between PT mean HR and the device's annotated HR, per lead:
median |delta| = 0.5 bpm, P90 = 1.7 bpm, 7663/7991 lead runs (95.8%) within
5 bpm. Precordial leads are the most reliable (V4 98.0%, V5 97.7%, V2 97.6%
within 5 bpm); limb leads I/AVL the least (~93%), consistent with their
smaller amplitudes and with this dataset's limb-lead reversals.

Cross-lead agreement per record: 79.9% of records have all 12 leads within a
2 bpm HR spread, and 607/666 (91.1%) have at least 11 of 12 leads within 5 bpm
of the device HR. The remaining records are dominated by genuine arrhythmias
(premature beats, AF, AV block, where a per-beat count legitimately differs
from the device's representative rate) and by limb-lead-reversal records --
this dataset's subject matter -- rather than by detector defects.

Known data quirk: record 2112291H3N has a flat V1 lead (reported as status
'flat'; sleepecg refuses constant signals).

Usage:
    python src/batch_rpeaks_pan_tompkins.py [--limit N] [--data-dir DIR] [--out-dir DIR]

Outputs
-------
    <out-dir>/plots/<record>.png     4x3 figure, per-lead detections
    <out-dir>/summary_by_lead.csv    666 x 12 rows, one per (record, lead)
    <out-dir>/summary_by_record.csv  666 rows, descriptive cross-lead stats
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sleepecg import detect_heartbeats

from ecg_waveform_extraction.src.utils.aecg_parser import parse_aecg

# ---- Config ----
DEFAULT_DATA_DIR = r'C:\LoyaltyLo\datasets\RA-LA_Reversal\aECG'
DEFAULT_OUT_DIR = str(Path(__file__).resolve().parent.parent / 'output' / 'pan_tompkins_rpeaks')

# Standard 12-lead display layout (rows of 3), in acquisition order
DISPLAY_LEADS = [
    ['I', 'II', 'III'],
    ['AVR', 'AVL', 'AVF'],
    ['V1', 'V2', 'V3'],
    ['V4', 'V5', 'V6'],
]
ALL_LEADS = [lead for row in DISPLAY_LEADS for lead in row]

# Leads whose annotated HR is treated as the reference for the descriptive audit
# only -- it never influences detection or lead choice.
DEVICE_HR_TOL = 5.0  # bpm


def read_scale_uv(filepath: str) -> float:
    """Read the first <scale> attribute (uV per internal unit) from an aECG file."""
    with open(filepath, 'rb') as f:
        head = f.read(64 * 1024)
    m = re.search(rb'<scale value="([^"]+)" unit="uV"', head)
    return float(m.group(1)) if m else 0.305


def detect_one_lead(sig: np.ndarray, fs: float) -> tuple[np.ndarray, str]:
    """Run Pan-Tompkins on a single lead.

    Returns (beat_indices, status). status is 'ok', 'flat' (sleepecg refuses a
    constant signal), or 'error: ...'.
    """
    if sig.size == 0 or np.ptp(sig) == 0:
        return np.array([], dtype=int), 'flat'
    try:
        return np.asarray(detect_heartbeats(sig, fs), dtype=int), 'ok'
    except ValueError as e:  # sleepecg raises on flat / degenerate signal
        return np.array([], dtype=int), f'error: {e}'


def beat_stats(beats: np.ndarray, fs: float) -> dict:
    """Descriptive metrics for one lead's beat sequence."""
    n = len(beats)
    if n >= 2:
        rr_ms = np.diff(beats) / fs * 1000.0
        return {
            'n_beats': n,
            'mean_HR_bpm': round(60.0 * fs / np.mean(np.diff(beats)), 1),
            'median_RR_ms': round(float(np.median(rr_ms)), 1),
            'min_RR_ms': round(float(rr_ms.min()), 1),
            'max_RR_ms': round(float(rr_ms.max()), 1),
        }
    return {'n_beats': n, 'mean_HR_bpm': np.nan, 'median_RR_ms': np.nan,
            'min_RR_ms': np.nan, 'max_RR_ms': np.nan}


def process_record(rec: dict, out_png: str) -> tuple[list[dict], dict]:
    """Detect on all 12 leads of one record, save its figure, return metrics.

    Returns (per_lead_rows, per_record_row).
    """
    sigs = rec['signals']
    fs = rec['fs'] or 1000.0
    scale_uv = read_scale_uv(rec['filepath'])
    dev_hr = rec['measurements'].get('HR')

    # ---- Detect independently on every lead ----
    results = {}          # lead -> (beats, status)
    per_lead_rows = []
    for lead in ALL_LEADS:
        if lead in sigs:
            beats, status = detect_one_lead(sigs[lead], fs)
        else:
            beats, status = np.array([], dtype=int), 'missing'
        results[lead] = (beats, status)
        per_lead_rows.append({
            'record': rec['filename'],
            'lead': lead,
            'status': status,
            'device_HR_bpm': dev_hr,
            **beat_stats(beats, fs),
        })

    # ---- Figure: 4x3, each panel with that lead's own detections ----
    fig, axes = plt.subplots(4, 3, figsize=(16, 9))
    fig.suptitle(f"{rec['filename']}  |  Pan-Tompkins (sleepecg) run independently on each lead  "
                 f"|  10 s, 12 leads", fontsize=13, fontweight='bold', y=0.985)
    for r in range(4):
        for c in range(3):
            ax = axes[r, c]
            lead = DISPLAY_LEADS[r][c]
            beats, status = results[lead]
            if lead not in sigs:
                ax.text(0.5, 0.5, 'lead missing', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='gray')
                ax.set_title(lead, fontsize=10, fontweight='bold', loc='left')
                continue

            mv = sigs[lead] * scale_uv / 1000.0
            t = np.arange(len(mv)) / fs
            ax.plot(t, mv, color='k', linewidth=0.6)
            if len(beats):
                ax.plot(t[beats], mv[beats], 'o', color='#d32f2f', markersize=4, zorder=5)

            st = beat_stats(beats, fs)
            if status == 'flat':
                label = f'{lead}  flat lead'
            elif status != 'ok':
                label = f'{lead}  {status[:22]}'
            elif st['n_beats'] >= 2:
                label = f"{lead}  n={st['n_beats']}  {st['mean_HR_bpm']:.0f} bpm"
            else:
                label = f"{lead}  n={st['n_beats']}"
            ax.set_title(label, fontsize=10, fontweight='bold', loc='left')
            ax.grid(True, alpha=0.25)
            if r == 3:
                ax.set_xlabel('Time (s)')
            if c == 0:
                ax.set_ylabel('mV')

    # Descriptive header note: how many leads land within tolerance of the
    # device's annotated HR. Reported only -- no lead is picked.
    hrs = [row['mean_HR_bpm'] for row in per_lead_rows
           if row['status'] == 'ok' and not np.isnan(row['mean_HR_bpm'])]
    if dev_hr and hrs:
        agree = sum(1 for h in hrs if abs(h - dev_hr) <= DEVICE_HR_TOL)
        fig.text(0.5, 0.947, f"device HR {dev_hr:.0f} bpm  |  leads within "
                             f"{DEVICE_HR_TOL:.0f} bpm: {agree}/{len(hrs)}",
                 ha='center', va='top', fontsize=10, color='#444444')
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    # ---- Per-record descriptive aggregate (no selection) ----
    ok_hrs = np.array(hrs, dtype=float)
    row = {
        'record': rec['filename'],
        'device_HR_bpm': dev_hr,
        'n_leads_ok': sum(1 for r_ in per_lead_rows if r_['status'] == 'ok'),
        'n_leads_flat': sum(1 for r_ in per_lead_rows if r_['status'] == 'flat'),
        'HR_median_across_leads': round(float(np.median(ok_hrs)), 1) if ok_hrs.size else np.nan,
        'HR_min_across_leads': round(float(ok_hrs.min()), 1) if ok_hrs.size else np.nan,
        'HR_max_across_leads': round(float(ok_hrs.max()), 1) if ok_hrs.size else np.nan,
        'HR_spread_across_leads': round(float(ok_hrs.max() - ok_hrs.min()), 1) if ok_hrs.size else np.nan,
        'n_leads_within_dev_tol': (sum(1 for h in ok_hrs if abs(h - dev_hr) <= DEVICE_HR_TOL)
                                   if dev_hr else np.nan),
        'lead_II_HR_bpm': next((r_['mean_HR_bpm'] for r_ in per_lead_rows if r_['lead'] == 'II'), np.nan),
        'interpretation': rec['interpretation'][:120],
    }
    return per_lead_rows, row


def main():
    ap = argparse.ArgumentParser(description='Pan-Tompkins R-peaks on every lead + per-record plots')
    ap.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    ap.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    ap.add_argument('--limit', type=int, default=None, help='process only first N files (smoke test)')
    args = ap.parse_args()

    plots_dir = os.path.join(args.out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(args.data_dir) if f.endswith('.aECG'))
    if args.limit:
        files = files[:args.limit]

    lead_rows, rec_rows, errors = [], [], []
    t0 = time.time()
    for i, fname in enumerate(files, 1):
        try:
            rec = parse_aecg(os.path.join(args.data_dir, fname))
            if not rec['signals']:
                raise ValueError('no lead signals parsed')
            lr, rr = process_record(rec, os.path.join(plots_dir, f"{rec['filename']}.png"))
            lead_rows.extend(lr)
            rec_rows.append(rr)
        except Exception as e:
            errors.append((fname, str(e)))
        if i % 50 == 0 or i == len(files):
            print(f'[{i}/{len(files)}] elapsed {time.time() - t0:.0f}s, errors: {len(errors)}', flush=True)

    df_lead = pd.DataFrame(lead_rows)
    df_rec = pd.DataFrame(rec_rows)
    p_lead = os.path.join(args.out_dir, 'summary_by_lead.csv')
    p_rec = os.path.join(args.out_dir, 'summary_by_record.csv')
    df_lead.to_csv(p_lead, index=False, encoding='utf-8-sig')
    df_rec.to_csv(p_rec, index=False, encoding='utf-8-sig')

    print(f"\nDone: {len(rec_rows)} records ({len(df_lead)} lead rows), "
          f"{len(errors)} errors / {len(files)} files in {time.time() - t0:.0f}s")
    ok = df_lead[df_lead['status'] == 'ok']
    print(f"Per-lead: {len(ok)} ok, "
          f"{(df_lead['status'] == 'flat').sum()} flat, "
          f"{df_lead['status'].str.startswith('error').sum()} error, "
          f"{(df_lead['status'] == 'missing').sum()} missing")
    dev = ok.dropna(subset=['device_HR_bpm', 'mean_HR_bpm'])
    if len(dev):
        d = (dev['mean_HR_bpm'] - dev['device_HR_bpm']).abs()
        print(f"All leads |PT HR - device HR|: median {d.median():.1f}, P90 {d.quantile(0.9):.1f}, "
              f"within {DEVICE_HR_TOL:.0f} bpm {100 * (d <= DEVICE_HR_TOL).mean():.1f}% (n={len(dev)})")
    li = dev[dev['lead'] == 'II']
    if len(li):
        d2 = (li['mean_HR_bpm'] - li['device_HR_bpm']).abs()
        print(f"Lead II  |PT HR - device HR|: median {d2.median():.1f}, "
              f"within {DEVICE_HR_TOL:.0f} bpm {100 * (d2 <= DEVICE_HR_TOL).mean():.1f}% (n={len(li)})")
    print(f"CSVs: {p_lead}\n      {p_rec}")
    for fname, err in errors[:10]:
        print(f'  ERROR {fname}: {err}')


if __name__ == '__main__':
    main()
