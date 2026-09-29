"""T-wave delineation inside the inter-QRS segment (wavedet, window-gated).

Continues the reference-lead pipeline one stage past QRS ranges: cut the
segment between beat i's QRS offset and beat i+1's QRS onset and judge the
T wave inside it — existence, peak, onset/offset, duration, amplitude and
polarity. Library implementations only:

- Reference lead + R peaks: identical to batch_rpeaks_pt_reflead.py
  (neurokit2 templatematch SQI -> highest-scoring lead, sleepecg
  Pan-Tompkins on the raw signal).
- QRS windows: identical to batch_qrs_wavedet_reflead.py — wavedet
  (Martinez 2004) on the cleaned reference lead, each PT beat matched to
  the nearest wavedet beat within 50 ms.
- T wave: two library sources, pooled and assigned to segments.
  Primary: prominence-delineator (Emrich et al., EUSIPCO 2024) via the
  repo's ProminenceStage wrapper — its T marks are computed on the cleaned
  reference lead ANCHORED ON OUR Pan-Tompkins R peaks (beat-aligned by
  construction), and peak prominence is amplitude-scale robust where
  wavedet's T detection goes sparse on low-amplitude T. Secondary: wavedet's
  T triples, re-anchored — its per-event mark arrays are not positionally
  beat-aligned (NaN slots interleave differently than the QRS arrays'), and
  its wavelet modulus maxima see both signs, so it contributes inverted-T
  marks prominence (a maxima scanner) cannot.

  Each segment takes the candidate triple chosen by, in order: plausible
  T-apex latency after R (100-500 ms), full containment in the segment,
  largest |amplitude| on the cleaned signal:

      seg_start = qrs_off(i)                    (beat i must have a valid QRS window)
      seg_end   = qrs_on(i+1) if present, else next R - 50 ms,
                  else min(r_i + 0.7 * median RR, end of signal)   (last beat)

  Segments are disjoint, so assignment is unambiguous. A beat's T counts
  (`in_window=True`) only if its apex latency is plausible and t_on/t_off
  stay inside the segment; other marks are kept in the CSV but excluded
  from statistics.

Per-beat outputs: T peak/onset/offset (ms), T duration, QT interval
(t_off - qrs_on), T amplitude and polarity on the RAW reference lead in mV
(upright / inverted / flat vs the R peak's polarity; inverted T is common in
this limb-lead-reversal dataset). Sanity gates flag (never drop) T durations
outside 50-450 ms and QT outside 200-600 ms.

Validation (all 666 records, 2026-09)
-------------------------------------
0 errors in 1524 s; reference lead and beat counts identical to the two
previous stages (deterministic parity). Of 12,118 beats with QRS windows,
11,899 (98.2%) got a T wave and 11,896 (99.97% of those) passed the window
gate. Distributions are physiologically plausible: T duration median 157 ms,
|T amplitude| median 0.26 mV, QT median 358 ms (P10-P90: 279-423 ms).
Polarity across all in-window T: 80.5% upright, 12.1% inverted, 7.5% flat —
the inverted share is consistent with this dataset's limb-lead reversals.

Per-record median QT vs the device's `QT_ms` (n=641 records with a
non-zero measurement; the device emits 0 when it did not measure): median
|delta| 21 ms, P90 48 ms, within 30 ms 72.5%, signed median +7 ms (60% of
records wider — prominence T offsets, being right-side prominence bases,
run slightly late, a bias the ProminenceStage wrapper documents).

Honest failures, reported not hidden: 7 records (1.1%) ended with no
in-window T — the 4 records that already had no QRS windows in the previous
stage plus 3 high-rate/arrhythmic records where no candidate triple passed
the apex-latency gate; 75 beats in 33 records carry a `sane_t=False` flag
(T duration outside 50-450 ms, mostly wide-T morphology) and are kept in
the CSVs but visible in every summary.

The single-source design was rejected by measurement: wavedet alone found
T on only 74.6% of beats in a 12-record smoke (its per-event T arrays are
not beat-aligned and its T detection goes sparse on low-amplitude T), and
beat-index lookup of its T marks mispaired whole runs of beats. The pooled
two-source design above was validated on the same 12 records (found 100%,
QT-within-30-ms 83.3%) before the full run.

Usage:
    python src/batch_twave_wavedet_reflead.py [--limit N] [--records A,B,C]
                                              [--data-dir DIR] [--out-dir DIR]

Outputs
-------
    <out-dir>/plots/<record>.png                  4x3 overview, QRS spans (red),
                                                  T windows (blue), R dots,
                                                  T peak triangles on every lead
    <out-dir>/plots_per_lead/<record>/<lead>.png  one figure per (record, lead)
    <out-dir>/t_by_beat.csv                       one row per PT beat
    <out-dir>/summary_by_record.csv               per-record T stats vs device QT
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
from ecg_waveform_extraction.src.batch_qrs_wavedet_reflead import (
    match_pt_to_wavedet, _valid_window,
)
from ecg_waveform_extraction.src.delineation import WaveletStage, ProminenceStage
from ecg_waveform_extraction.src.preprocessing import ECGPreprocessor

# ---- Config ----
DEFAULT_DATA_DIR = r'C:\LoyaltyLo\datasets\RA-LA_Reversal\aECG'
DEFAULT_OUT_DIR = str(Path(__file__).resolve().parent.parent / 'output' / 'twave_wavedet_reflead')
QRS_SPAN = dict(color='#d32f2f', alpha=0.15, zorder=1)
T_SPAN = dict(color='#1976d2', alpha=0.15, zorder=1)
T_COLOR = '#1976d2'
FLAT_T_AMP_MV = 0.05   # |T amplitude| below this -> 'flat', not upright/inverted
T_SANE_MS = (50.0, 450.0)
QT_SANE_MS = (200.0, 600.0)
NEXT_R_GUARD_MS = 50.0  # segment end falls back to next R minus this guard
T_APEX_AFTER_R_MS = (100.0, 500.0)   # plausible T-apex latency from the R peak


def t_polarity(t_amp: float, r_amp: float) -> str:
    """Upright / inverted / flat, judged against the R peak's polarity."""
    if abs(t_amp) < FLAT_T_AMP_MV:
        return 'flat'
    return 'upright' if t_amp * r_amp > 0 else 'inverted'


def _f2i(value) -> int:
    """wavedet mark entry (float, possibly NaN) -> sample index or -1."""
    v = float(value)
    return int(v) if np.isfinite(v) else -1


def t_triples(d_raw) -> list[tuple[int, int, int]]:
    """Pool of valid (t_peak, t_on, t_off) triples from a raw wavedet result."""
    pool = []
    if d_raw is None:
        return pool
    for j in range(len(d_raw.t)):
        tp, ton, toff = _f2i(d_raw.t[j]), _f2i(d_raw.t_on[j]), _f2i(d_raw.t_off[j])
        if tp >= 0 and ton >= 0 and toff >= 0 and ton < toff:
            pool.append((tp, ton, toff))
    return pool


def save_single_lead_figure(rec_name: str, lead: str, mv: np.ndarray, fs: float,
                            beats: np.ndarray, qrs_windows: list, t_windows: list,
                            t_peaks: np.ndarray, out_png: str, ref_lead: str,
                            n_beats: int, med_qt_ms, dev_qt=None, dev_hr=None):
    """One lead with R markers, QRS spans, T windows and T peak triangles."""
    fig, ax = plt.subplots(figsize=(12, 3.5))
    t = np.arange(len(mv)) / fs
    ax.plot(t, mv, color='k', linewidth=0.7)
    for on, off in t_windows:
        ax.axvspan(on / fs, off / fs, **T_SPAN)
    for on, off in qrs_windows:
        ax.axvspan(on / fs, off / fs, **QRS_SPAN)
    if len(t_peaks):
        ax.plot(t[t_peaks], mv[t_peaks], '^', color=T_COLOR, markersize=6, zorder=5)
    if len(beats):
        ax.plot(t[beats], mv[beats], 'o', color='#d32f2f', markersize=5, zorder=5)
    tag = '  [reference]' if lead == ref_lead else ''
    qt_txt = f", median QT {med_qt_ms:.0f} ms" if med_qt_ms == med_qt_ms else ''
    dev_txt = f" (device {dev_qt:.0f} ms)" if dev_qt and med_qt_ms == med_qt_ms else ''
    hr_txt = f", HR {dev_hr:.0f} bpm" if dev_hr else ''
    ax.set_title(f"{rec_name} — lead {lead}{tag}  |  T windows from reference lead "
                 f"{ref_lead} (wavedet, inter-QRS): {len(t_windows)} T waves / {n_beats} beats"
                 f"{qt_txt}{dev_txt}{hr_txt}", fontsize=10, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('mV')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


def process_record(rec: dict, prep: ECGPreprocessor, stage: WaveletStage,
                   prom: ProminenceStage,
                   out_png: str, per_lead_dir: str) -> tuple[list[dict], dict]:
    """Full per-record pipeline; returns (per-beat rows, summary row)."""
    sigs = rec['signals']
    fs = rec['fs'] or 1000.0
    n_samples = len(next(iter(sigs.values())))
    scale_uv = read_scale_uv(rec['filepath'])
    dev_hr = rec['measurements'].get('HR')
    dev_qt = rec['measurements'].get('QT_ms')
    dev_qtc = rec['measurements'].get('QTc_ms')
    # the device emits 0 when it did not measure QT — treat as missing
    if dev_qt is not None and dev_qt <= 0:
        dev_qt = None
    if dev_qtc is not None and dev_qtc <= 0:
        dev_qtc = None

    # ---- 1-3. reference lead, PT beats, wavedet beats, matching (as before) ----
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
    med_rr_ms = ref_stats['median_RR_ms']
    if not (med_rr_ms == med_rr_ms):
        med_rr_ms = 800.0

    clean = prep.preprocess(sigs[ref_lead])
    wd_beats, d_raw = stage.delineate_raw(clean)
    wd_idx, _delta = match_pt_to_wavedet(pt_beats, wd_beats, fs)
    # T candidate pool: wavedet's re-anchored triples + prominence marks
    # computed on the same cleaned lead but anchored on OUR R peaks.
    pool = t_triples(d_raw)
    for pb in prom.delineate(clean, pt_beats):
        if pb.t_peak >= 0 and pb.t_onset >= 0 and pb.t_offset >= 0 \
                and pb.t_onset < pb.t_offset:
            pool.append((pb.t_peak, pb.t_onset, pb.t_offset))

    # raw reference-lead mV: amplitudes and polarities are read here
    mv_ref = sigs[ref_lead] * scale_uv / 1000.0

    # ---- per-beat QRS windows + T gating ----
    beat_rows = []
    qrs_windows = []        # (on, off) samples, valid windows only
    for i, (r, wi) in enumerate(zip(pt_beats, wd_idx)):
        wb = wd_beats[wi] if wi >= 0 else None
        win = _valid_window(wb, n_samples)
        row = {
            'record': rec['filename'], 'beat_i': i,
            'r_ms': round(r / fs * 1000.0, 1),
            'qrs_on_ms': np.nan, 'qrs_off_ms': np.nan, 'seg_end_ms': np.nan,
            't_found': False, 'in_window': False,
            't_peak_ms': np.nan, 't_on_ms': np.nan, 't_off_ms': np.nan,
            't_dur_ms': np.nan, 'qt_ms': np.nan, 't_amp_mv': np.nan,
            't_polarity': '', 'sane_t': np.nan, 'sane_qt': np.nan,
        }
        if not win:                       # no valid QRS window -> no segment start
            beat_rows.append(row)
            continue
        on, off = win
        qrs_windows.append(win)

        # ---- segment end: next beat's qrs_on / next R guard / RR-based tail ----
        if i + 1 < len(pt_beats):
            wb_next = wd_beats[wd_idx[i + 1]] if wd_idx[i + 1] >= 0 else None
            win_next = _valid_window(wb_next, n_samples)
            if win_next:
                seg_end = win_next[0]
            else:
                seg_end = int(pt_beats[i + 1] - NEXT_R_GUARD_MS / 1000.0 * fs)
        else:
            seg_end = int(r + 0.7 * med_rr_ms / 1000.0 * fs)
        seg_end = min(seg_end, n_samples - 1)

        row.update(qrs_on_ms=round(on / fs * 1000.0, 1),
                   qrs_off_ms=round(off / fs * 1000.0, 1),
                   seg_end_ms=round(seg_end / fs * 1000.0, 1))

        # ---- T assignment: pooled triples whose t_peak falls inside this segment ----
        cands = [tr for tr in pool if off <= tr[0] <= seg_end]
        if cands:
            lo_apex, hi_apex = (int(a * fs / 1000.0) for a in T_APEX_AFTER_R_MS)
            tp, ton, toff = max(cands, key=lambda tr: (
                lo_apex <= tr[0] - r <= hi_apex,          # physiological apex latency first
                tr[1] >= off and tr[2] <= seg_end,        # then full containment
                abs(float(clean[tr[0]]))))                # then dominant amplitude
            plausible = lo_apex <= tp - r <= hi_apex
            row['t_found'] = True
            row.update(t_peak_ms=round(tp / fs * 1000.0, 1),
                       t_on_ms=round(ton / fs * 1000.0, 1),
                       t_off_ms=round(toff / fs * 1000.0, 1),
                       t_dur_ms=round((toff - ton) / fs * 1000.0, 1),
                       in_window=bool(plausible and ton >= off and toff <= seg_end))
            if row['in_window']:
                qt_ms = (toff - on) / fs * 1000.0
                row['qt_ms'] = round(qt_ms, 1)
                row['t_amp_mv'] = round(float(mv_ref[tp]), 4)
                row['t_polarity'] = t_polarity(row['t_amp_mv'], float(mv_ref[r]))
                row['sane_t'] = bool(T_SANE_MS[0] <= row['t_dur_ms'] <= T_SANE_MS[1])
                row['sane_qt'] = bool(QT_SANE_MS[0] <= qt_ms <= QT_SANE_MS[1])
        beat_rows.append(row)

    # windows/triangles for the figures (in-window T only)
    t_rows = [r_ for r_ in beat_rows if r_['in_window']]
    t_windows = [(int(r_['t_on_ms'] / 1000.0 * fs), int(r_['t_off_ms'] / 1000.0 * fs))
                 for r_ in t_rows]
    t_peaks = np.array([int(r_['t_peak_ms'] / 1000.0 * fs) for r_ in t_rows], dtype=int)

    qts = np.array([r_['qt_ms'] for r_ in t_rows], dtype=float)
    med_qt = float(np.median(qts)) if qts.size else np.nan
    n_beats = len(pt_beats)
    hr_txt = f", HR {ref_stats['mean_HR_bpm']:.1f} bpm" if ref_stats['n_beats'] >= 2 else ''
    qt_txt = (f", median QT {med_qt:.0f} ms (device {dev_qt:.0f} ms)"
              if med_qt == med_qt and dev_qt else
              (f", median QT {med_qt:.0f} ms" if med_qt == med_qt else ''))

    # ---- figures: raw mV, QRS spans + T spans + markers on every lead ----
    fig, axes = plt.subplots(4, 3, figsize=(16, 9))
    fig.suptitle(f"{rec['filename']}  |  reference lead {ref_lead} (SQI "
                 f"{ref_sqi:.3f})  |  {n_beats} beats, {len(qrs_windows)} QRS windows, "
                 f"{len(t_windows)} T waves in-window{hr_txt}{qt_txt}",
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
            for ton, toff in t_windows:
                ax.axvspan(ton / fs, toff / fs, **T_SPAN)
            for qon, qoff in qrs_windows:
                ax.axvspan(qon / fs, qoff / fs, **QRS_SPAN)
            if len(t_peaks):
                ax.plot(t[t_peaks], mv[t_peaks], '^', color=T_COLOR,
                        markersize=4, zorder=5)
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
                                        qrs_windows, t_windows, t_peaks,
                                        os.path.join(per_lead_dir, f'{lead}.png'),
                                        ref_lead, n_beats, med_qt, dev_qt, dev_hr)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    # ---- summary row ----
    durs = np.array([r_['t_dur_ms'] for r_ in t_rows], dtype=float)
    pol = [r_['t_polarity'] for r_ in t_rows]
    row = {
        'record': rec['filename'],
        'ref_lead': ref_lead,
        'ref_sqi': round(ref_sqi, 4) if not np.isnan(ref_sqi) else np.nan,
        'ref_status': ref_status,
        'device_HR_bpm': dev_hr,
        'device_QT_ms': dev_qt,
        'device_QTc_ms': dev_qtc,
        'n_beats': n_beats,
        'n_qrs_windows': len(qrs_windows),
        'n_t_found': int(sum(r_['t_found'] for r_ in beat_rows)),
        'n_t_in_window': len(t_rows),
        'n_sane_t': int(sum(1 for r_ in t_rows if r_['sane_t'])),
        't_dur_median_ms': round(float(np.median(durs)), 1) if durs.size else np.nan,
        'qt_median_ms': round(med_qt, 1) if med_qt == med_qt else np.nan,
        'qt_delta_ms': (round(med_qt - dev_qt, 1)
                        if med_qt == med_qt and dev_qt else np.nan),
        'abs_qt_delta_ms': (round(abs(med_qt - dev_qt), 1)
                            if med_qt == med_qt and dev_qt else np.nan),
        'n_upright_t': pol.count('upright'),
        'n_inverted_t': pol.count('inverted'),
        'interpretation': rec['interpretation'][:120],
    }
    return beat_rows, row


def main():
    ap = argparse.ArgumentParser(
        description='T-wave delineation in the inter-QRS segment (wavedet, window-gated)')
    ap.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    ap.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    ap.add_argument('--limit', type=int, default=None, help='process only first N files')
    ap.add_argument('--records', default=None,
                    help='comma-separated record names (smoke test)')
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
    prom = ProminenceStage(fs=1000.0)

    beat_rows, rec_rows, errors = [], [], []
    t0 = time.time()
    for i, fname in enumerate(files, 1):
        try:
            rec = parse_aecg(os.path.join(args.data_dir, fname))
            if not rec['signals']:
                raise ValueError('no lead signals parsed')
            rec_dir = os.path.join(per_lead_base, rec['filename'])
            os.makedirs(rec_dir, exist_ok=True)
            br, rr = process_record(rec, prep, stage, prom,
                                    os.path.join(plots_dir, f"{rec['filename']}.png"), rec_dir)
            beat_rows.extend(br)
            rec_rows.append(rr)
        except Exception as e:
            errors.append((fname, str(e)))
        if i % 50 == 0 or i == len(files):
            print(f'[{i}/{len(files)}] elapsed {time.time() - t0:.0f}s, errors: {len(errors)}', flush=True)

    df_beat = pd.DataFrame(beat_rows)
    df = pd.DataFrame(rec_rows)
    p_beat = os.path.join(args.out_dir, 't_by_beat.csv')
    p_rec = os.path.join(args.out_dir, 'summary_by_record.csv')
    df_beat.to_csv(p_beat, index=False, encoding='utf-8-sig')
    df.to_csv(p_rec, index=False, encoding='utf-8-sig')

    print(f"\nDone: {len(rec_rows)} records, {len(errors)} errors / {len(files)} files "
          f"in {time.time() - t0:.0f}s")
    n_q = int(df['n_qrs_windows'].sum())
    n_f = int(df['n_t_found'].sum())
    n_w = int(df['n_t_in_window'].sum())
    if n_q:
        print(f"T found {n_f}/{n_q} ({100 * n_f / n_q:.1f}%), "
              f"in-window {n_w}/{n_f} ({100 * n_w / n_f:.1f}%)" if n_f else
              f"T found {n_f}/{n_q} (0.0%), in-window 0")
    inb = df_beat[df_beat['in_window']]
    if len(inb):
        print('T polarity:', inb['t_polarity'].value_counts().to_dict())
    ok = df[df['device_QT_ms'] > 0].dropna(subset=['qt_median_ms'])
    if len(ok):
        d = ok['abs_qt_delta_ms']
        s = ok['qt_delta_ms']
        print(f"Median QT vs device QT_ms: median |d| {d.median():.1f} ms, P90 {d.quantile(0.9):.1f}, "
              f"within 30 ms {100 * (d <= 30).mean():.1f}% (n={len(ok)}); "
              f"signed median {s.median():+.1f} ms, wider {(s > 0).mean() * 100:.0f}%")
    unsane = df_beat[df_beat['sane_t'] == False]  # noqa: E712  (bool column with NaN)
    if len(unsane):
        print(f"Unsane T durations: {len(unsane)} beats in "
              f"{unsane.record.nunique()} records: {unsane.record.head(10).unique().tolist()}")
    zero = df[df['n_t_in_window'] == 0]
    if len(zero):
        print(f"Records with 0 in-window T: {zero['record'].tolist()}")
    print('Reference lead distribution:', df['ref_lead'].value_counts().to_dict())
    print(f"CSVs: {p_beat}\n      {p_rec}")
    for fname, err in errors[:10]:
        print(f'  ERROR {fname}: {err}')


if __name__ == '__main__':
    main()
