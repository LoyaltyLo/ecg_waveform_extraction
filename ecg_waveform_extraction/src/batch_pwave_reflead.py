"""P-wave delineation in the TP segment (two pooled libraries + SNR gate).

Fourth stage of the reference-lead pipeline: cut the segment between one
beat's T-wave offset and the NEXT beat's QRS onset, and judge the P wave
inside it — existence, peak, onset/offset, duration, amplitude, polarity and
the PR interval. Library implementations only:

- Upstream anchors (R peaks, QRS windows, T offsets) come from the previous
  stages' deterministic outputs: `batch_rpeaks_pt_reflead.py`,
  `batch_qrs_wavedet_reflead.py`, `batch_twave_wavedet_reflead.py`
  (read from their CSVs; three consecutive runs proved bit-identical
  reference-lead choice and beat counts, so the CSVs are exact anchors —
  the SQI/PT/wavedet-QRS chain is not recomputed).
- P candidates from two libraries, pooled and assigned per segment:
  prominence-delineator (Emrich et al., EUSIPCO 2024) via the repo's
  ProminenceStage — run once upright and once on the negated signal, so
  inverted P waves (aVR, retrograde) are covered — anchored on OUR
  Pan-Tompkins R peaks; and wavedet's P triples, re-anchored from
  `WaveletStage.delineate_raw` (its per-event arrays are not positionally
  beat-aligned — the T-wave stage proved this — so marks are pooled, never
  index-trusted).
- Segment and selection (P belongs to the beat whose QRS it precedes):

      seg_start = t_off(i)   (beat i-1's T offset; beat 0 uses signal start;
                              a previous beat without an in-window T -> no
                              segment, honestly skipped and counted)
      seg_end   = qrs_on(i)

  Candidates whose peak falls in the segment compete on (plausible latency
  of the P peak before the QRS onset (40-320 ms), full containment in the
  segment, passing the SNR gate, and being the deflection CLOSEST to the
  QRS onset — the P is the last wave before the QRS; amplitude-max would
  let T-tail/U-wave debris steal the slot, as the T-stage smoke data
  proved).
- **Two-library boundary fusion**: both libraries run P offsets late (bias
  vs device: onset ~-5.5 ms, offset ~+21.5 ms). When the other family
  delineates the same physical P (peak within 40 ms), the reported window
  is the INTERSECTION (on = max of onsets, off = min of offsets) — guarded
  to stay non-degenerate and peak-containing; `p_source` records
  prom / prom_inv / wd / fused pairs.
- False-P control (the core of "judging" a P): the repo's
  `_p_passes_snr` gate — the detrended P amplitude must exceed 2.5x the
  MAD-based noise sigma of the quiet TP window just before the P onset
  (reference window clamped below seg_start). A bump fabricated on a flat
  TP segment (AF, ectopic beats) fails this gate and is not counted.

Per-beat sanity flags (never drops): P duration outside 40-200 ms
(`sane_p`), PR interval outside 80-300 ms (`sane_pr`).

Validation (all 666 records, 2026-10, with boundary fusion)
-------------------------------------
0 errors in 1481 s. The 13,319 beat rows and every ref_lead are bit-identical
to the T-stage CSVs (anchor parity). Of 11,448 TP segments (beats with a QRS
window whose predecessor carries an in-window T): a P candidate was found in
11,391 (99.5%), survived the geometry gate in 11,027 (96.3%) and the SNR gate
in 10,410 (91.0% of segments) — the gate rejected 981 found candidates as
local noise, which is the false-P control working. 6,875 of the 10,410
passing P waves (66%) were boundary-fused across the two library families.

Boundary accuracy vs device (n=642 records with both measured), fusion
effect in brackets vs the unfused first run:
- P duration: median |delta| 10.0 ms [was 27.8], P90 33.4 [47.5], within
  30 ms 85.4% [55.9%]. Signed +6.0 ms, IQR [-1, +16] — onset bias ~-1 ms,
  offset bias ~+5 ms (was -5.5 / +21.5: the intersection window removed
  three quarters of the offset lateness both libraries share).
- PR interval: median |delta| 11.0 ms [13.0], within 30 ms 83.8% [81.0%];
  signed median +1.0 ms [was +5.5] — effectively unbiased, at the device's
  own measurement noise floor.

Two fused-run lessons, kept for the record: (a) the SNR gate must run on
the PICKED LIBRARY'S OWN window, not the trimmed intersection — gating on
the fused window cuts the P flanks and falsely rejects real P waves (first
fused run: gate pass rate fell 91.4% -> 86.8%, 12 extra low-P records;
fixed, pass rate restored to exactly 91.4%); (b) amplitude-max selection
lets T-tail/U-wave debris steal the P slot (smoke-proven: 180522WU7Y PR
error +85 ms wrong, +14 ms after the closest-to-QRS key).

Sanity flags: sane_p 10,909/11,027 (98.9%), sane_pr 10,774/11,027 (97.7%).
Polarity of passing P waves: 4,873 positive (46.8%), 3,321 flat (31.9%),
2,216 negative (21.3%) — the flat/negative share fits the RA-LA reversal
morphology (inverted/near-flat P in V1-V2, negative in AVR).

P presence: median record p_rate 1.00; 19 records below 0.5, honestly
reported (e.g. 2112155IE6: 0/7 on V1 despite device PR 173 — pool/SNR
failure, not forced). Records whose interpretation mentions AF
(房颤/心房颤动/AF/atrial) have median p_rate 0.78 vs 1.00 for the rest
(informational cross-check, not a gate).

Usage:
    python src/batch_pwave_reflead.py [--limit N] [--records A,B,C]
                                      [--data-dir DIR] [--out-dir DIR]
                                      [--t-stage-dir DIR]

Outputs
-------
    <out-dir>/plots/<record>.png                  4x3 overview, P (green) +
                                                  QRS (red) + T (blue) spans,
                                                  R/T/P peak markers
    <out-dir>/plots_per_lead/<record>/<lead>.png  one figure per (record, lead)
    <out-dir>/p_by_beat.csv                       one row per upstream beat
    <out-dir>/summary_by_record.csv               per-record P stats vs device
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
    DISPLAY_LEADS, ALL_LEADS, read_scale_uv,
)
from ecg_waveform_extraction.src.delineation import WaveletStage, ProminenceStage
from ecg_waveform_extraction.src.delineation.prominence_stage import _p_passes_snr
from ecg_waveform_extraction.src.preprocessing import ECGPreprocessor

# ---- Config ----
DEFAULT_DATA_DIR = r'C:\LoyaltyLo\datasets\RA-LA_Reversal\aECG'
DEFAULT_OUT_DIR = str(Path(__file__).resolve().parent.parent / 'output' / 'pwave_reflead')
DEFAULT_T_DIR = str(Path(__file__).resolve().parent.parent / 'output' / 'twave_wavedet_reflead')
QRS_SPAN = dict(color='#d32f2f', alpha=0.15, zorder=1)
T_SPAN = dict(color='#1976d2', alpha=0.15, zorder=1)
P_SPAN = dict(color='#2e7d32', alpha=0.18, zorder=1)
P_COLOR = '#2e7d32'
FLAT_P_AMP_MV = 0.03
P_SANE_MS = (40.0, 200.0)
PR_SANE_MS = (80.0, 300.0)
P_BEFORE_QRS_MS = (40.0, 320.0)   # plausible P-peak latency before qrs_on
FUSE_PEAK_MS = 40.0               # cross-library peak distance = same physical P


def _f2i(value) -> int:
    """wavedet mark entry (float, possibly NaN) -> sample index or -1."""
    v = float(value)
    return int(v) if np.isfinite(v) else -1


def _ms2idx(ms: float, fs: float) -> int:
    """T-stage CSV millisecond value -> sample index."""
    return int(round(ms / 1000.0 * fs))


def p_polarity(p_amp: float) -> str:
    """Positive / negative / flat by absolute amplitude (P has no R to mirror)."""
    if abs(p_amp) < FLAT_P_AMP_MV:
        return 'flat'
    return 'positive' if p_amp > 0 else 'negative'


def save_single_lead_figure(rec_name: str, lead: str, mv: np.ndarray, fs: float,
                            r_peaks: np.ndarray, qrs_windows: list, t_windows: list,
                            t_peaks: np.ndarray, p_windows: list, p_peaks: np.ndarray,
                            out_png: str, ref_lead: str, med_pr_ms, dev_pr=None):
    """One lead with the full P/QRS/T annotation set."""
    fig, ax = plt.subplots(figsize=(12, 3.5))
    t = np.arange(len(mv)) / fs
    ax.plot(t, mv, color='k', linewidth=0.7)
    for on, off in p_windows:
        ax.axvspan(on / fs, off / fs, **P_SPAN)
    for on, off in t_windows:
        ax.axvspan(on / fs, off / fs, **T_SPAN)
    for on, off in qrs_windows:
        ax.axvspan(on / fs, off / fs, **QRS_SPAN)
    if len(p_peaks):
        ax.plot(t[p_peaks], mv[p_peaks], 'v', color=P_COLOR, markersize=6, zorder=6)
    if len(t_peaks):
        ax.plot(t[t_peaks], mv[t_peaks], '^', color='#1976d2', markersize=6, zorder=5)
    if len(r_peaks):
        ax.plot(t[r_peaks], mv[r_peaks], 'o', color='#d32f2f', markersize=5, zorder=5)
    tag = '  [reference]' if lead == ref_lead else ''
    pr_txt = f", median PR {med_pr_ms:.0f} ms" if med_pr_ms == med_pr_ms else ''
    dev_txt = f" (device {dev_pr:.0f} ms)" if dev_pr and med_pr_ms == med_pr_ms else ''
    ax.set_title(f"{rec_name} — lead {lead}{tag}  |  P waves from reference lead "
                 f"{ref_lead} (TP segment): {len(p_windows)} P waves{pr_txt}{dev_txt}",
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('mV')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


def process_record(rec: dict, t_rows: pd.DataFrame, prep: ECGPreprocessor,
                   stage: WaveletStage, prom: ProminenceStage,
                   out_png: str, per_lead_dir: str) -> tuple[list[dict], dict]:
    """P-wave judgement for one record; returns (per-beat rows, summary row)."""
    sigs = rec['signals']
    fs = rec['fs'] or 1000.0
    n_samples = len(next(iter(sigs.values())))
    scale_uv = read_scale_uv(rec['filepath'])
    ref_lead = str(t_rows['ref_lead'].iloc[0])
    dev_hr = rec['measurements'].get('HR')
    dev_pdur = rec['measurements'].get('P_dur')
    dev_pr = rec['measurements'].get('PR_ms')
    for name, v in (('P_dur', dev_pdur), ('PR_ms', dev_pr)):
        if v is not None and v <= 0:      # device emits 0 when unmeasured
            if name == 'P_dur':
                dev_pdur = None
            else:
                dev_pr = None

    # upstream anchors (samples) from the T-stage rows
    r_idx = np.array([_ms2idx(m, fs) for m in t_rows['r_ms']], dtype=int)
    qrs_on = [_ms2idx(m, fs) if m == m else -1 for m in t_rows['qrs_on_ms']]
    qrs_off = [_ms2idx(m, fs) if m == m else -1 for m in t_rows['qrs_off_ms']]
    t_on = [_ms2idx(m, fs) if m == m else -1 for m in t_rows['t_on_ms']]
    t_off = [_ms2idx(m, fs) if m == m else -1 for m in t_rows['t_off_ms']]
    in_win_t = [bool(x) for x in t_rows['in_window']]

    mv_ref = sigs[ref_lead] * scale_uv / 1000.0
    clean = prep.preprocess(sigs[ref_lead])

    # ---- P candidate pool: prominence (upright + inverted) + wavedet ----
    # entries are (peak, onset, offset, kind); kind family: prominence
    # ('prom'/'prom_inv') vs wavedet ('wd') — used by the boundary fusion.
    pool = []
    for pb in prom.delineate(clean, r_idx):
        for kind, pk, on, off in (('prom', pb.p_peak, pb.p_onset, pb.p_offset),
                                  ('prom_inv', pb.p_inv_peak, pb.p_inv_onset,
                                   pb.p_inv_offset)):
            if pk >= 0 and on >= 0 and off >= 0 and on < off and on <= pk <= off:
                pool.append((pk, on, off, kind))
    _wd, d_raw = stage.delineate_raw(clean)
    if d_raw is not None:
        for j in range(len(d_raw.p)):
            pk, on, off = _f2i(d_raw.p[j]), _f2i(d_raw.p_on[j]), _f2i(d_raw.p_off[j])
            if pk >= 0 and on >= 0 and off >= 0 and on < off:
                pool.append((pk, on, off, 'wd'))

    lo_lat, hi_lat = (int(a * fs / 1000.0) for a in P_BEFORE_QRS_MS)

    # ---- per-beat: segment [prev t_off, this qrs_on] -> P assignment ----
    beat_rows = []
    for j in range(len(r_idx)):
        row = {
            'record': rec['filename'], 'beat_i': j,
            'r_ms': round(r_idx[j] / fs * 1000.0, 1),
            'qrs_on_ms': round(qrs_on[j] / fs * 1000.0, 1) if qrs_on[j] > 0 else np.nan,
            'seg_start_ms': np.nan, 'seg_end_ms': np.nan,
            'has_segment': False,
            'p_found': False, 'in_window': False, 'passes_snr': np.nan,
            'p_source': '',
            'p_peak_ms': np.nan, 'p_on_ms': np.nan, 'p_off_ms': np.nan,
            'p_dur_ms': np.nan, 'p_amp_mv': np.nan, 'p_polarity': '',
            'pr_ms': np.nan, 'sane_p': np.nan, 'sane_pr': np.nan,
        }
        if qrs_on[j] <= 0:
            beat_rows.append(row)         # no QRS window -> no segment end
            continue
        seg_end = qrs_on[j]
        if j == 0:
            seg_start = 0
        elif in_win_t[j - 1] and t_off[j - 1] > 0:
            seg_start = t_off[j - 1]
        else:
            beat_rows.append(row)         # previous beat has no T end -> skip
            continue
        if seg_end - seg_start < int(0.05 * fs):
            beat_rows.append(row)         # degenerate segment
            continue
        row['has_segment'] = True
        row.update(seg_start_ms=round(seg_start / fs * 1000.0, 1),
                   seg_end_ms=round(seg_end / fs * 1000.0, 1))

        cands = [tr for tr in pool if seg_start <= tr[0] <= seg_end]
        if cands:
            # P = the LAST clean deflection before the QRS. Unlike the T
            # stage (dominant wave of its segment -> amplitude-max), the TP
            # segment holds T-tail/U-wave debris that is often AMPLITUDE-
            # dominant while the true P is the deflection nearest the QRS
            # (smoke-proven: 180522WU7Y device PR 145 / P_dur 108 -> true
            # p_peak ~90 ms, amplitude-max picked a 156 ms bump). Final
            # key therefore: plausible -> contained -> passes SNR ->
            # closest peak to the QRS onset.
            pk, on, off, kind = max(cands, key=lambda tr: (
                lo_lat <= seg_end - tr[0] <= hi_lat,        # plausible latency
                tr[1] >= seg_start and tr[2] <= seg_end,    # full containment
                _p_passes_snr(clean, tr[1], tr[2], fs, ref_floor=seg_start),
                -(seg_end - tr[0])))                        # nearest the QRS
            # Existence judgement (SNR gate) runs on the PICKED LIBRARY'S
            # OWN window: fusion below only refines the reported boundaries
            # and must not alter whether a P counts — computing the gate on
            # the trimmed intersection cuts the P flanks and falsely
            # rejects real P waves (first fused full run lost exactly this:
            # gate pass rate 91.4% -> 86.8%).
            gate = _p_passes_snr(clean, on, off, fs, ref_floor=seg_start)
            # ---- two-library boundary fusion (intersection window) ----
            # Both libraries run their P offsets late (full-run bias: onset
            # ~-5.5 ms, offset ~+21.5 ms vs device). When the OTHER family
            # delineates the same physical P (peak within FUSE_PEAK_MS),
            # take the intersection: on = max(on), off = min(off) — the
            # earlier-offset library pulls the window back in. Guarded so
            # the fused window stays non-degenerate and still contains the
            # peak; otherwise the picked library's window stands.
            fam = 0 if kind.startswith('prom') else 1
            partners = [tr for tr in cands
                        if (0 if tr[3].startswith('prom') else 1) != fam]
            if partners:
                q = min(partners, key=lambda tr: abs(tr[0] - pk))
                if abs(q[0] - pk) <= int(FUSE_PEAK_MS / 1000.0 * fs):
                    fon, foff = max(on, q[1]), min(off, q[2])
                    if fon < foff and fon <= pk <= foff:
                        on, off = fon, foff
                        kind = f'{kind}+{q[3]}'
            plausible = lo_lat <= seg_end - pk <= hi_lat
            contained = on >= seg_start and off <= seg_end
            row['p_found'] = True
            row['p_source'] = kind
            row.update(p_peak_ms=round(pk / fs * 1000.0, 1),
                       p_on_ms=round(on / fs * 1000.0, 1),
                       p_off_ms=round(off / fs * 1000.0, 1),
                       p_dur_ms=round((off - on) / fs * 1000.0, 1),
                       in_window=bool(plausible and contained))
            if row['in_window']:
                row['passes_snr'] = bool(gate)   # library window, pre-fusion
                row['pr_ms'] = round((seg_end - on) / fs * 1000.0, 1)
                row['p_amp_mv'] = round(float(mv_ref[pk]), 4)
                row['p_polarity'] = p_polarity(row['p_amp_mv'])
                row['sane_p'] = bool(P_SANE_MS[0] <= row['p_dur_ms'] <= P_SANE_MS[1])
                row['sane_pr'] = bool(PR_SANE_MS[0] <= row['pr_ms'] <= PR_SANE_MS[1])
        beat_rows.append(row)

    # ---- figure data: windows/peaks for plotting (QRS/T from upstream) ----
    qrs_windows = [(qrs_on[j], qrs_off[j]) for j in range(len(r_idx)) if qrs_on[j] > 0]
    t_windows = [(t_on[j], t_off[j]) for j in range(len(r_idx)) if in_win_t[j]]
    t_peaks = np.array([_ms2idx(m, fs) for m in t_rows['t_peak_ms'] if m == m], dtype=int)
    p_ok = [r_ for r_ in beat_rows if r_['p_found'] and r_['in_window'] and r_['passes_snr']]
    p_windows = [(_ms2idx(r_['p_on_ms'], fs), _ms2idx(r_['p_off_ms'], fs)) for r_ in p_ok]
    p_peaks = np.array([_ms2idx(r_['p_peak_ms'], fs) for r_ in p_ok], dtype=int)

    prs = np.array([r_['pr_ms'] for r_ in p_ok], dtype=float)
    med_pr = float(np.median(prs)) if prs.size else np.nan
    n_seg = int(sum(1 for r_ in beat_rows if r_['has_segment']))

    hr_txt = f", HR {dev_hr:.0f} bpm" if dev_hr else ''
    pr_txt = (f", median PR {med_pr:.0f} ms (device {dev_pr:.0f} ms)"
              if med_pr == med_pr and dev_pr else
              (f", median PR {med_pr:.0f} ms" if med_pr == med_pr else ''))

    fig, axes = plt.subplots(4, 3, figsize=(16, 9))
    fig.suptitle(f"{rec['filename']}  |  reference lead {ref_lead}  |  "
                 f"{len(r_idx)} beats, P waves {len(p_ok)}/{n_seg} in TP segments"
                 f"{hr_txt}{pr_txt}", fontsize=13, fontweight='bold', y=0.985)
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
            for pon, poff in p_windows:
                ax.axvspan(pon / fs, poff / fs, **P_SPAN)
            for ton, toff in t_windows:
                ax.axvspan(ton / fs, toff / fs, **T_SPAN)
            for qon, qoff in qrs_windows:
                ax.axvspan(qon / fs, qoff / fs, **QRS_SPAN)
            if len(p_peaks):
                ax.plot(t[p_peaks], mv[p_peaks], 'v', color=P_COLOR,
                        markersize=4, zorder=6)
            if len(t_peaks):
                ax.plot(t[t_peaks], mv[t_peaks], '^', color='#1976d2',
                        markersize=4, zorder=5)
            if len(r_idx):
                ax.plot(t[r_idx], mv[r_idx], 'o', color='#d32f2f',
                        markersize=4, zorder=5)
            star = ' \u2605' if lead == ref_lead else ''
            ax.set_title(f"{lead}{star}", fontsize=10, fontweight='bold', loc='left')
            ax.grid(True, alpha=0.25)
            if r == 3:
                ax.set_xlabel('Time (s)')
            if c == 0:
                ax.set_ylabel('mV')
            if per_lead_dir:
                save_single_lead_figure(rec['filename'], lead, mv, fs, r_idx,
                                        qrs_windows, t_windows, t_peaks,
                                        p_windows, p_peaks,
                                        os.path.join(per_lead_dir, f'{lead}.png'),
                                        ref_lead, med_pr, dev_pr)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    # ---- summary row ----
    durs = np.array([r_['p_dur_ms'] for r_ in p_ok], dtype=float)
    pol = [r_['p_polarity'] for r_ in p_ok]
    row = {
        'record': rec['filename'],
        'ref_lead': ref_lead,
        'device_HR_bpm': dev_hr,
        'device_P_dur_ms': dev_pdur,
        'device_PR_ms': dev_pr,
        'n_beats': len(r_idx),
        'n_segments': n_seg,
        'n_p_found': int(sum(r_['p_found'] for r_ in beat_rows)),
        'n_p_pass': len(p_ok),
        'n_fused': sum(1 for r_ in p_ok if '+' in r_['p_source']),
        'p_rate': round(len(p_ok) / n_seg, 3) if n_seg else np.nan,
        'p_dur_median_ms': round(float(np.median(durs)), 1) if durs.size else np.nan,
        'p_dur_delta_ms': (round(float(np.median(durs)) - dev_pdur, 1)
                           if durs.size and dev_pdur else np.nan),
        'pr_median_ms': round(med_pr, 1) if med_pr == med_pr else np.nan,
        'pr_delta_ms': (round(med_pr - dev_pr, 1) if med_pr == med_pr and dev_pr else np.nan),
        'n_positive': pol.count('positive'),
        'n_negative': pol.count('negative'),
        'n_flat': pol.count('flat'),
        'interpretation': rec['interpretation'][:120],
    }
    return beat_rows, row


def main():
    ap = argparse.ArgumentParser(
        description='P-wave delineation in the TP segment (pooled libraries + SNR gate)')
    ap.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    ap.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    ap.add_argument('--t-stage-dir', default=DEFAULT_T_DIR,
                    help='output dir of the T-wave stage (upstream anchors)')
    ap.add_argument('--limit', type=int, default=None, help='process only first N files')
    ap.add_argument('--records', default=None,
                    help='comma-separated record names (smoke test)')
    args = ap.parse_args()

    # upstream anchors: T-stage per-beat table + per-record reference lead
    t_all = pd.read_csv(os.path.join(args.t_stage_dir, 't_by_beat.csv'))
    t_sum = pd.read_csv(os.path.join(args.t_stage_dir, 'summary_by_record.csv'))[
        ['record', 'ref_lead']]
    t_all = t_all.merge(t_sum, on='record', how='left')

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
        rec_name = fname[:-len('.aECG')]
        try:
            t_rows = t_all[t_all['record'] == rec_name]
            if not len(t_rows) or pd.isna(t_rows['ref_lead'].iloc[0]):
                raise ValueError(f'no T-stage anchors for {rec_name} (stale t-stage-dir?)')
            rec = parse_aecg(os.path.join(args.data_dir, fname))
            if not rec['signals']:
                raise ValueError('no lead signals parsed')
            rec_dir = os.path.join(per_lead_base, rec['filename'])
            os.makedirs(rec_dir, exist_ok=True)
            br, rr = process_record(rec, t_rows, prep, stage, prom,
                                    os.path.join(plots_dir, f"{rec['filename']}.png"), rec_dir)
            beat_rows.extend(br)
            rec_rows.append(rr)
        except Exception as e:
            errors.append((fname, str(e)))
        if i % 50 == 0 or i == len(files):
            print(f'[{i}/{len(files)}] elapsed {time.time() - t0:.0f}s, errors: {len(errors)}', flush=True)

    df_beat = pd.DataFrame(beat_rows)
    df = pd.DataFrame(rec_rows)
    p_beat = os.path.join(args.out_dir, 'p_by_beat.csv')
    p_rec = os.path.join(args.out_dir, 'summary_by_record.csv')
    df_beat.to_csv(p_beat, index=False, encoding='utf-8-sig')
    df.to_csv(p_rec, index=False, encoding='utf-8-sig')

    print(f"\nDone: {len(rec_rows)} records, {len(errors)} errors / {len(files)} files "
          f"in {time.time() - t0:.0f}s")
    n_seg = int(df['n_segments'].sum())
    n_f = int(df['n_p_found'].sum())
    n_p = int(df['n_p_pass'].sum())
    if n_seg:
        print(f"P found {n_f}/{n_seg} ({100 * n_f / n_seg:.1f}%), "
              f"pass SNR gate {n_p}/{n_f}" if n_f else f"P found {n_f}/{n_seg}")
    okb = df_beat[df_beat['passes_snr'] == True]  # noqa: E712
    if len(okb):
        print('P polarity:', okb['p_polarity'].value_counts().to_dict())
        print('P source:', okb['p_source'].value_counts().to_dict())
    ok = df[df['device_P_dur_ms'] > 0].dropna(subset=['p_dur_median_ms'])
    if len(ok):
        d = ok['p_dur_delta_ms'].abs()
        print(f"P duration vs device P_dur: median |d| {d.median():.1f} ms, P90 {d.quantile(0.9):.1f}, "
              f"within 30 ms {100 * (d <= 30).mean():.1f}% (n={len(ok)})")
    ok = df[df['device_PR_ms'] > 0].dropna(subset=['pr_median_ms'])
    if len(ok):
        d = ok['pr_delta_ms'].abs()
        s = ok['pr_delta_ms']
        print(f"PR vs device PR_ms: median |d| {d.median():.1f} ms, P90 {d.quantile(0.9):.1f}, "
              f"within 30 ms {100 * (d <= 30).mean():.1f}% (n={len(ok)}); "
              f"signed median {s.median():+.1f} ms")
    lowp = df[df['p_rate'] < 0.5]
    if len(lowp):
        af = lowp['interpretation'].str.contains('房颤|心房颤动|AF|atrial', case=False, na=False).sum()
        print(f"Records with P rate < 0.5: {len(lowp)} "
              f"({af} of them mention AF/房颤 in interpretation): "
              f"{lowp['record'].head(10).tolist()}{'...' if len(lowp) > 10 else ''}")
    print('Reference lead distribution:', df['ref_lead'].value_counts().to_dict())
    print(f"CSVs: {p_beat}\n      {p_rec}")
    for fname, err in errors[:10]:
        print(f'  ERROR {fname}: {err}')


if __name__ == '__main__':
    main()
