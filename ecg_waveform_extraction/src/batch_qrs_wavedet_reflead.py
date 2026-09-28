"""QRS window delineation (wavedet, Martinez 2004) anchored to reference-lead
Pan-Tompkins R peaks.

Same reference-lead paradigm as batch_rpeaks_pt_reflead.py, one stage further:
after the R peaks are known, determine each beat's QRS range (Q onset to J
point / S offset) and mark that time window on all 12 leads. Library
implementations only, no hand-rolled delineation code:

1. **Reference-lead selection** — identical to batch_rpeaks_pt_reflead.py:
   neurokit2 `ecg_quality(method="templatematch")` on every lead, highest
   score wins (ties -> lead order, fallback II). Selection sees the waveform
   only, never a detection result.
2. **R peaks** — sleepecg Pan-Tompkins on the RAW reference-lead signal
   (`detect_one_lead`).
3. **QRS windows** — wavedet (PyPI port of the ecg-kit wavedet_3D delineator,
   quadratic-spline dyadic wavelet, Martinez et al. 2004) on the CLEANED
   reference-lead signal (`ECGPreprocessor.preprocess`: median baseline
   removal -> 0.5-40 Hz bandpass -> 50 Hz notch -> z-score; all filtfilt,
   zero-phase, so its sample indices address the raw signal 1:1). wavedet
   detects its own R peaks, so each Pan-Tompkins beat is matched to the
   nearest wavedet beat within 50 ms (the same nearest-match pattern as
   `delineation.crosscheck_qrs_boundaries`) and adopts that beat's
   qrs_on/qrs_off. If two PT beats claim one wavedet beat, only the closer
   claimant keeps it.

Plots show the RAW signal in mV (scale attribute), not the z-scored clean
signal: the marks transfer because the preprocessing is zero-phase and
length-preserving.

Physiological sanity gate: windows shorter than 30 ms or longer than 250 ms
are flagged `sane=False` in the CSVs but never dropped.

Caveats found during smoke testing: (a) wavedet's wavelet warm-up and end
retraction systematically cost the FIRST and LAST beat of each record — these
appear as unmatched beats, not as wrong windows; (b) the device's
representative `QRS_on_ms`/`QRS_off_ms` window is on a time base that does not
match the 10 s excerpt start (deltas of hundreds of ms), so `dev_*_delta_ms`
columns are informational only — the honest device validation is the DURATION
comparison (median window duration vs `QRS_dur`).

Validation (all 666 records, 2026-09)
-------------------------------------
0 errors in 1437 s. Reference lead and PT beat counts are identical to the
batch_rpeaks_pt_reflead run (deterministic parity). Of 13,319 PT beats,
12,118 (91.0%) matched a wavedet beat; 936 of the 1,201 unmatched beats are
a record's first or last beat (wavedet wavelet warm-up / end retraction).
Where matched, the two detectors' R positions agree to a median of 1 ms
(P95 2 ms).

Per-record median QRS duration vs the device's `QRS_dur`: median |delta|
10 ms, P90 26 ms, within 20 ms 81.6% and within 40 ms 95.2% of 662 records
(n=662; 4 records without comparable values). No systematic bias: signed
median +1 ms, 51% of records wider than the device, overall median 88 ms vs
device 86 ms.

Sanity gate: 10 unsane windows in 7 records (all wide, 252-436 ms — paced
or arrhythmic morphology; flagged `sane=False`, kept). 4 records (0.6%)
produced no windows, honestly reported rather than force-matched: two where
wavedet marks a different deflection than Pan-Tompkins in AVR/V1 (systematic
+57 ms / -85 ms offsets, outside the protective 50 ms gate), one at 161 bpm
where wavedet's refractory keeps only 5 of 27 beats, one further AVR case.

Usage:
    python src/batch_qrs_wavedet_reflead.py [--limit N] [--records A,B,C]
                                            [--data-dir DIR] [--out-dir DIR]

Outputs
-------
    <out-dir>/plots/<record>.png                  4x3 overview, R markers +
                                                  shaded QRS windows on every lead
    <out-dir>/plots_per_lead/<record>/<lead>.png  one figure per (record, lead)
    <out-dir>/qrs_by_beat.csv                     one row per PT beat
    <out-dir>/summary_by_record.csv               per-record window stats vs device
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ecg_waveform_extraction.src.utils.aecg_parser import parse_aecg
from ecg_waveform_extraction.src.batch_rpeaks_pan_tompkins import (
    DISPLAY_LEADS, ALL_LEADS, read_scale_uv, detect_one_lead, beat_stats,
)
from ecg_waveform_extraction.src.batch_rpeaks_pt_reflead import (
    lead_quality, FALLBACK_LEAD,
)
from ecg_waveform_extraction.src.delineation import WaveletStage
from ecg_waveform_extraction.src.preprocessing import ECGPreprocessor

# ---- Config ----
DEFAULT_DATA_DIR = r'C:\LoyaltyLo\datasets\RA-LA_Reversal\aECG'
DEFAULT_OUT_DIR = str(Path(__file__).resolve().parent.parent / 'output' / 'qrs_wavedet_reflead')
SANE_MIN_MS = 30.0     # below this a "QRS window" is not physiological
SANE_MAX_MS = 250.0    # above this either; flagged, never dropped
SPAN_KW = dict(color='#d32f2f', alpha=0.15, zorder=1)   # matches R-marker red


def match_pt_to_wavedet(pt_beats: np.ndarray, wd_beats: list, fs: float
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Nearest wavedet beat per PT beat within 50 ms (crosscheck pattern).

    Returns (wd_index, abs_delta_samples): wd_index is an index into wd_beats
    or -1 where unmatched. If several PT beats map to the same wavedet beat,
    only the closest one keeps the match.
    """
    tol = int(round(0.05 * fs))
    n = len(pt_beats)
    wd_idx = np.full(n, -1, dtype=int)
    delta = np.full(n, np.inf)
    if n == 0 or not wd_beats:
        return wd_idx, delta
    wd_r = np.array([b.r_peak for b in wd_beats], dtype=np.int64)
    order = np.argsort(wd_r)
    wd_r = wd_r[order]
    pos = np.searchsorted(wd_r, pt_beats)
    lo = np.clip(pos - 1, 0, len(wd_r) - 1)
    hi = np.clip(pos, 0, len(wd_r) - 1)
    best = np.where(np.abs(wd_r[lo] - pt_beats) < np.abs(wd_r[hi] - pt_beats), lo, hi)
    d = np.abs(wd_r[best] - pt_beats)
    ok = d <= tol
    for j in np.unique(best[ok]):            # dedupe shared claims
        rows = np.where((best == j) & ok)[0]
        keep = rows[np.argmin(d[rows])]
        ok[rows] = False
        ok[keep] = True
    wd_idx[ok] = order[best[ok]]
    delta[ok] = d[ok]
    return wd_idx, delta


def _valid_window(wb, n_samples: int):
    """(on, off) sample indices of a beat's QRS window, or None."""
    if wb is None:
        return None
    on, off = int(wb.qrs_onset), int(wb.qrs_offset)
    if on <= 0 or off <= on or off >= n_samples:   # -1 / degenerate / edge
        return None
    return on, off


def save_single_lead_figure(rec_name: str, lead: str, mv: np.ndarray, fs: float,
                            beats: np.ndarray, windows: list, out_png: str,
                            ref_lead: str, n_beats: int, med_dur_ms,
                            dev_qrs=None, dev_hr=None):
    """One lead with R markers and shaded QRS windows (time instants shared)."""
    fig, ax = plt.subplots(figsize=(12, 3.5))
    t = np.arange(len(mv)) / fs
    ax.plot(t, mv, color='k', linewidth=0.7)
    for on, off in windows:
        ax.axvspan(on / fs, off / fs, **SPAN_KW)
    if len(beats):
        ax.plot(t[beats], mv[beats], 'o', color='#d32f2f', markersize=5, zorder=5)
    tag = '  [reference]' if lead == ref_lead else ''
    dur_txt = f", median QRS {med_dur_ms:.0f} ms" if med_dur_ms == med_dur_ms else ''
    dev_txt = f" (device {dev_qrs:.0f} ms)" if dev_qrs and med_dur_ms == med_dur_ms else ''
    hr_txt = f", HR {dev_hr:.0f} bpm" if dev_hr else ''
    ax.set_title(f"{rec_name} — lead {lead}{tag}  |  R peaks + QRS windows from reference "
                 f"lead {ref_lead} (wavedet): {n_beats} beats, {len(windows)} windows"
                 f"{dur_txt}{dev_txt}{hr_txt}", fontsize=10, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('mV')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


def process_record(rec: dict, prep: ECGPreprocessor, stage: WaveletStage,
                   out_png: str, per_lead_dir: str) -> tuple[list[dict], dict]:
    """Full per-record pipeline; returns (beat rows, summary row)."""
    sigs = rec['signals']
    fs = rec['fs'] or 1000.0
    n_samples = len(next(iter(sigs.values())))
    scale_uv = read_scale_uv(rec['filepath'])
    dev_hr = rec['measurements'].get('HR')
    dev_qrs = rec['measurements'].get('QRS_dur')
    dev_win = (rec['annotations'].get('QRS_on_ms'), rec['annotations'].get('QRS_off_ms'))

    # ---- 1-2. reference lead + PT beats (bit-identical to the R-peak script) ----
    sqis = {lead: lead_quality(sigs[lead], fs) for lead in ALL_LEADS if lead in sigs}
    ranked = sorted(((s, lead) for lead, s in sqis.items() if not np.isnan(s)),
                    key=lambda sl: (-sl[0], ALL_LEADS.index(sl[1])))
    if ranked:
        ref_sqi, ref_lead = ranked[0]
    else:
        ref_lead = FALLBACK_LEAD if FALLBACK_LEAD in sigs else next(iter(sigs))
        ref_sqi = np.nan
    pt_beats, ref_status = detect_one_lead(sigs[ref_lead], fs)
    ref_stats = beat_stats(pt_beats, fs)

    # ---- 3. wavedet windows on the cleaned reference lead ----
    clean = prep.preprocess(sigs[ref_lead])
    wd_beats = stage.delineate(clean)
    wd_idx, delta = match_pt_to_wavedet(pt_beats, wd_beats, fs)

    # ---- per-beat rows ----
    beat_rows, windows = [], []
    for i, (r, wi, dl) in enumerate(zip(pt_beats, wd_idx, delta)):
        wb = wd_beats[wi] if wi >= 0 else None
        win = _valid_window(wb, n_samples)
        row = {
            'record': rec['filename'], 'beat_i': i,
            'r_ms': round(r / fs * 1000.0, 1),
            'wd_r_ms': round(wb.r_peak / fs * 1000.0, 1) if wb else np.nan,
            'r_delta_ms': round(dl / fs * 1000.0, 1) if np.isfinite(dl) else np.nan,
            'qrs_on_ms': np.nan, 'qrs_off_ms': np.nan, 'dur_ms': np.nan,
            'matched': wb is not None,
            'sane': np.nan,
            'dev_on_delta_ms': np.nan, 'dev_off_delta_ms': np.nan,
        }
        if win:
            on, off = win
            dur_ms = (off - on) / fs * 1000.0
            windows.append(win)
            row.update(qrs_on_ms=round(on / fs * 1000.0, 1),
                       qrs_off_ms=round(off / fs * 1000.0, 1),
                       dur_ms=round(dur_ms, 1),
                       sane=bool(SANE_MIN_MS <= dur_ms <= SANE_MAX_MS))
        beat_rows.append(row)

    # secondary check: fill dev_* on the window nearest the device's
    # representative QRS window (one row per record at most)
    dev_on_d = dev_off_d = np.nan
    if dev_win[0] is not None and dev_win[1] is not None:
        with_win = [r_ for r_ in beat_rows if r_['qrs_on_ms'] == r_['qrs_on_ms']]
        if with_win:
            dev_c = 0.5 * (dev_win[0] + dev_win[1])
            j = min(with_win, key=lambda r_: abs(0.5 * (r_['qrs_on_ms'] + r_['qrs_off_ms']) - dev_c))
            j['dev_on_delta_ms'] = round(j['qrs_on_ms'] - dev_win[0], 1)
            j['dev_off_delta_ms'] = round(j['qrs_off_ms'] - dev_win[1], 1)
            dev_on_d, dev_off_d = j['dev_on_delta_ms'], j['dev_off_delta_ms']

    # ---- figures: raw mV, R markers + QRS spans on every lead ----
    durs = np.array([r_['dur_ms'] for r_ in beat_rows if r_['dur_ms'] == r_['dur_ms']])
    med_dur = float(np.median(durs)) if durs.size else np.nan
    n_beats = len(pt_beats)
    hr_txt = f", HR {ref_stats['mean_HR_bpm']:.1f} bpm" if ref_stats['n_beats'] >= 2 else ''
    dur_txt = (f", median QRS {med_dur:.0f} ms (device {dev_qrs:.0f} ms)"
               if med_dur == med_dur and dev_qrs else
               (f", median QRS {med_dur:.0f} ms" if med_dur == med_dur else ''))

    fig, axes = plt.subplots(4, 3, figsize=(16, 9))
    fig.suptitle(f"{rec['filename']}  |  reference lead {ref_lead} (SQI "
                 f"{ref_sqi:.3f})  |  PT: {n_beats} R peaks, {len(windows)} QRS "
                 f"windows (wavedet){hr_txt}{dur_txt}",
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
            for on, off in windows:
                ax.axvspan(on / fs, off / fs, **SPAN_KW)
            if len(pt_beats):
                ax.plot(t[pt_beats], mv[pt_beats], 'o', color='#d32f2f',
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
                save_single_lead_figure(rec['filename'], lead, mv, fs, pt_beats,
                                        windows, os.path.join(per_lead_dir, f'{lead}.png'),
                                        ref_lead, n_beats, med_dur, dev_qrs, dev_hr)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    # ---- summary row ----
    n_win = len(windows)
    sane_flags = [r_['sane'] for r_ in beat_rows if r_['sane'] == r_['sane']]
    iqr = float(np.percentile(durs, 75) - np.percentile(durs, 25)) if durs.size else np.nan
    row = {
        'record': rec['filename'],
        'ref_lead': ref_lead,
        'ref_sqi': round(ref_sqi, 4) if not np.isnan(ref_sqi) else np.nan,
        'ref_status': ref_status,
        'device_HR_bpm': dev_hr,
        'device_QRS_dur_ms': dev_qrs,
        'n_wavelet_beats': len(wd_beats),
        'n_beats': n_beats,
        'n_matched': int(sum(r_['matched'] for r_ in beat_rows)),
        'n_windows': n_win,
        'n_sane': int(sum(1 for f_ in sane_flags if f_)),
        'n_unsane': int(sum(1 for f_ in sane_flags if not f_)),
        'dur_median_ms': round(med_dur, 1) if med_dur == med_dur else np.nan,
        'dur_iqr_ms': round(iqr, 1) if iqr == iqr else np.nan,
        'dur_delta_ms': (round(med_dur - dev_qrs, 1)
                         if med_dur == med_dur and dev_qrs else np.nan),
        'abs_dur_delta_ms': (round(abs(med_dur - dev_qrs), 1)
                             if med_dur == med_dur and dev_qrs else np.nan),
        'dev_on_delta_ms': dev_on_d,
        'dev_off_delta_ms': dev_off_d,
        'interpretation': rec['interpretation'][:120],
    }
    return beat_rows, row


def main():
    ap = argparse.ArgumentParser(
        description='wavedet QRS windows anchored to reference-lead PT R peaks')
    ap.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    ap.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    ap.add_argument('--limit', type=int, default=None, help='process only first N files')
    ap.add_argument('--records', default=None,
                    help='comma-separated record names (smoke test, e.g. 2203243H09,...)')
    args = ap.parse_args()

    plots_dir = os.path.join(args.out_dir, 'plots')
    per_lead_base = os.path.join(args.out_dir, 'plots_per_lead')
    os.makedirs(plots_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(args.data_dir) if f.endswith('.aECG'))
    if args.records:
        wanted = {x.strip().removesuffix('.aECG') for x in args.records.split(',')}
        files = [f for f in files if f[:-len('.aECG')] in wanted]
    if args.limit:
        files = files[:args.limit]

    prep = ECGPreprocessor(fs=1000.0)
    stage = WaveletStage(fs=1000.0)

    beat_rows, rec_rows, errors = [], [], []
    t0 = time.time()
    for i, fname in enumerate(files, 1):
        try:
            rec = parse_aecg(os.path.join(args.data_dir, fname))
            if not rec['signals']:
                raise ValueError('no lead signals parsed')
            rec_dir = os.path.join(per_lead_base, rec['filename'])
            os.makedirs(rec_dir, exist_ok=True)
            br, rr = process_record(rec, prep, stage,
                                    os.path.join(plots_dir, f"{rec['filename']}.png"), rec_dir)
            beat_rows.extend(br)
            rec_rows.append(rr)
        except Exception as e:
            errors.append((fname, str(e)))
        if i % 50 == 0 or i == len(files):
            print(f'[{i}/{len(files)}] elapsed {time.time() - t0:.0f}s, errors: {len(errors)}', flush=True)

    df_beat = pd.DataFrame(beat_rows)
    df = pd.DataFrame(rec_rows)
    p_beat = os.path.join(args.out_dir, 'qrs_by_beat.csv')
    p_rec = os.path.join(args.out_dir, 'summary_by_record.csv')
    df_beat.to_csv(p_beat, index=False, encoding='utf-8-sig')
    df.to_csv(p_rec, index=False, encoding='utf-8-sig')

    print(f"\nDone: {len(rec_rows)} records, {len(errors)} errors / {len(files)} files "
          f"in {time.time() - t0:.0f}s")
    n_b = int(df['n_beats'].sum())
    n_m = int(df['n_matched'].sum())
    n_w = int(df['n_windows'].sum())
    if n_b:
        print(f"PT->wavedet match: {n_m}/{n_b} ({100 * n_m / n_b:.1f}%), "
              f"windows {n_w}/{n_b} ({100 * n_w / n_b:.1f}%)")
        # edge effect: how many unmatched beats are a record's first or last beat
        nb = df_beat['record'].map(df.set_index('record')['n_beats'])
        edge = (~df_beat['matched'] & df_beat['beat_i'].eq(0)
                | ~df_beat['matched'] & df_beat['beat_i'].eq(nb.sub(1)))
        n_un = int((~df_beat['matched']).sum())
        if n_un:
            print(f"Unmatched beats: {n_un}, of which first/last-of-record: "
                  f"{int(edge.sum())} (wavedet warm-up / end retraction)")
    ok = df.dropna(subset=['dur_median_ms', 'device_QRS_dur_ms'])
    if len(ok):
        d = ok['abs_dur_delta_ms']
        print(f"Median QRS dur vs device QRS_dur: median |d| {d.median():.1f} ms, "
              f"P90 {d.quantile(0.9):.1f}, within 20 ms {100 * (d <= 20).mean():.1f}%, "
              f"within 40 ms {100 * (d <= 40).mean():.1f}% (n={len(ok)})")
    unsane = df[df['n_unsane'] > 0]
    if len(unsane):
        print(f"Sanity gate: {int(unsane['n_unsane'].sum())} unsane beats in "
              f"{len(unsane)} records: {unsane['record'].head(10).tolist()}")
    zero = df[df['n_windows'] == 0]
    if len(zero):
        print(f"Records with 0 windows: {zero['record'].tolist()}")
    print('Reference lead distribution:', df['ref_lead'].value_counts().to_dict())
    print(f"CSVs: {p_beat}\n      {p_rec}")
    for fname, err in errors[:10]:
        print(f'  ERROR {fname}: {err}')


if __name__ == '__main__':
    main()
