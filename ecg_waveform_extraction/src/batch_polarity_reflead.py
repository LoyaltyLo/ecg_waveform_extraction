"""Per-lead per-beat polarity for P / QRS / T (positive / negative / biphasic / flat).

Fifth stage of the reference-lead pipeline: propagate the per-beat wave
windows (already delineated on the reference lead and aligned beat-for-beat
across the three stage CSVs) onto ALL 12 leads and classify each wave's
polarity on every lead. Pure signal processing on library-given windows —
no detection algorithm is added.

Wave windows come from the upstream per-beat CSVs, keyed by (record, beat_i):
- P  : output/pwave_reflead/p_by_beat.csv      in_window AND passes_snr
- T  : output/twave_wavedet_reflead/t_by_beat.csv  in_window
- QRS: output/qrs_wavedet_reflead/qrs_by_beat.csv  valid qrs_on/qrs_off

Classification per (record, beat, lead, wave):
- Isoelectric baseline: 20 ms PR segment (the ECG convention), centered 30 ms
  before P onset; beats without P fall back to the midpoint of the T-end ->
  next-QRS interval; beats without either use the global minimum-|mean| 20 ms
  window of the lead.
- Window stats: pos_amp = max(v), neg_amp = -min(v), abs_amp = max(pos, neg),
  noise = 1.4826 * MAD(baseline window), snr = abs_amp / noise.
- biphasic: the second-largest SIGNIFICANT deflection (baseline-crossing
  runs >= MIN_DEFLECTION_MS long — noise slivers dropped) is OPPOSITE-signed
  to the largest and clears the floor / PHASE_RATIO / 2x-noise gates; the
  time order is kept as pattern '+-' / '-+' (in V1 the P pattern is the
  RA-LA reversal signature).  flat: primary excursion < FLAT_FLOOR_MV or
  snr < AMP_NOISE_RATIO;  otherwise positive / negative by the dominant
  excursion.

Validation (all 666 records, 2026-10, biphasic v2: deflection-level rules)
--------------------------------------
0 errors in 2711 s. 159,828 (beat x lead) rows = 13,319 beats x 12 leads;
8,658 figures. Polarity counts (pos / neg / biphasic / flat):
P 68,294 / 30,918 / 24,268 (15.2%, patterns +- 18,815 / -+ 5,453) / 1,440;
QRS 38,642 / 29,014 / 77,364 (48.4%, +- 66,331 / -+ 11,033) / 396;
T 88,902 / 32,604 / 20,444 (12.8%, +- 9,825 / -+ 10,619) / 838.

Physiology checks (the stage's own ground truth):
- aVR (P/QRS/T must be negative-or-biphasic in normal conduction): 84% /
  94% / 80% — the 94% QRS figure is the aVR-typical rs complex now
  correctly labeled biphasic by the 1/3-ratio second-deflection rule.
- Sinus-rhythm records (n=595) on lead II: P+ 77%, QRS+ 61%, T+ 77%.
- V1 biphasic P pattern: '+-' 3,240 vs '-+' 167 beats (19:1) — normal
  sinus P in V1 is positive-then-terminal-negative; the reversal-group
  (18xx/19xx) '-+' share is 7.5% vs 4.1% for 20xx+ (weak alone, one
  feature among many).
- Reversal probe: 7 records show aVR-dominant-positive QRS with device
  P_axis in the normal 0-90 range (report-only candidates).

Biphasic-rule calibration history (kept so the thresholds are traceable):
- v0 window-extremum both-excursion rules: 57-76% of QRS bogus-biphasic
  (the r/s notch of any normal qRs clears relative thresholds).
- v1 deflection-level without sign constraint / without sliver filter:
  same-sign r' counted as the second phase; noise wobble created phases.
- v2 (current): deflections = baseline-crossing runs >= MIN_DEFLECTION_MS
  (slivers dropped); biphasic needs the second-largest deflection
  OPPOSITE-signed and >= 1/3 of the primary for QRS (accompanying q/s of
  Rs/qRs is morphology, not a biphasic complex; at 1/10 the chest-lead
  rS population reads 69% biphasic); P/T keep 1/10 and 1/15 (their minor
  phase is genuinely small). Patterns '+-'/'-+' and the per-wave
  significant-deflection count (qRs -> 3) are exported.
- P polarity is PR-baseline-referenced and intentionally does NOT match
  the upstream single-sample peak polarity (66% agreement; V1 biphasic P
  is real morphology); T cross-check agrees 89.7%.

Usage:
    python src/batch_polarity_reflead.py [--limit N] [--records A,B,C]
                                         [--data-dir DIR] [--out-dir DIR]

Outputs
-------
    <out-dir>/polarity_by_beat_lead.csv   one row per (record, beat, lead) x wave
    <out-dir>/summary_by_record.csv       per-record polarity matrices
    <out-dir>/plots/<record>.png          4x3 overview, polarity chars on peaks
    <out-dir>/plots_per_lead/<record>/<lead>.png
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

# ---- Config ----
DEFAULT_DATA_DIR = r'C:\LoyaltyLo\datasets\RA-LA_Reversal\aECG'
DEFAULT_OUT_DIR = str(Path(__file__).resolve().parent.parent / 'output' / 'polarity_reflead')
BASE_DIR = Path(__file__).resolve().parent.parent / 'output'
DEFAULT_P_DIR = str(BASE_DIR / 'pwave_reflead')
DEFAULT_T_DIR = str(BASE_DIR / 'twave_wavedet_reflead')
DEFAULT_Q_DIR = str(BASE_DIR / 'qrs_wavedet_reflead')
QRS_SPAN = dict(color='#d32f2f', alpha=0.15, zorder=1)
T_SPAN = dict(color='#1976d2', alpha=0.15, zorder=1)
P_SPAN = dict(color='#2e7d32', alpha=0.18, zorder=1)
WAVE_COLOR = {'P': '#2e7d32', 'QRS': '#d32f2f', 'T': '#1976d2'}
# polarity thresholds (wave-aware: P is a small wave, QRS a large one — a
# single flat floor would call every low-amplitude P "significant" and every
# small-potential QRS "flat"). BIPHASIC is decided at the DEFLECTION level
# for every wave: deflections are baseline-crossing-separated runs, runs
# shorter than MIN_DEFLECTION_MS are noise slivers and dropped, and the
# second-largest surviving deflection must be OPPOSITE-signed to the largest
# (a same-sign r' never makes an rsR' complex "biphasic"; window-extremum
# rules measured 57-76% bogus biphasic on normal qRs). The time order of
# the two deflections is kept as the pattern '+-' / '-+' — in V1 the P
# pattern is the RA-LA reversal signature (normal sinus P is +- there).
PHASE_RATIO = {'P': 0.10, 'QRS': 0.33, 'T': 0.15}   # biphasic: secondary >= this x primary
# QRS note: the secondary deflection must reach ONE THIRD of the primary —
# the accompanying s/q of a normal Rs/qRs is typically 10-30% of R and is
# morphology, not a biphasic complex; at ratio 0.10 a correct
# second-deflection rule reads 69% of chest-lead QRS as biphasic (measured).
# True biphasic QRS = two comparable deflections (aVR-style qr/rs).
BIPHASIC_FLOOR_MV = {'P': 0.03, 'QRS': 0.10, 'T': 0.05}   # ... and >= this floor
FLAT_FLOOR_MV = {'P': 0.015, 'QRS': 0.1, 'T': 0.02}       # primary below -> 'flat'
AMP_NOISE_RATIO = 1.5       # primary below this SNR -> 'flat'
MIN_DEFLECTION_MS = {'P': 10.0, 'QRS': 8.0, 'T': 20.0}    # shorter runs = slivers
BASELINE_PRE_P_MS = 30.0    # baseline window centered this many ms before P onset
BASELINE_LEN_MS = 20.0


def _ms2idx(ms: float, fs: float) -> int:
    """CSV millisecond value -> sample index."""
    return int(round(ms / 1000.0 * fs))


def _baseline(x: np.ndarray, fs: float, p_on: int, t_off: int, qrs_on: int) -> tuple[float, float]:
    """(baseline_mV, noise_mV) for one beat on one lead.

    Priority: 20 ms PR window centered 30 ms before P onset; else the 20 ms
    window at the midpoint of [prev t_off, this qrs_on]; else the global
    minimum-|mean| 20 ms window of the lead. Noise is the MAD sigma of the
    baseline window.
    """
    n = len(x)
    w = int(BASELINE_LEN_MS / 1000.0 * fs)
    half = w // 2

    def _win(center: int):
        c = int(np.clip(center, half, n - 1 - half))
        seg = x[c - half:c + half]
        med = float(np.median(seg))
        noise = 1.4826 * float(np.median(np.abs(seg - med)))
        return med, noise

    if p_on > 0:
        c = p_on - int(BASELINE_PRE_P_MS / 1000.0 * fs)
        if 0 <= c - half:
            return _win(c)
    if t_off > 0 and qrs_on > t_off:
        c = (t_off + qrs_on) // 2
        return _win(c)
    # global fallback: minimum-|mean| 20 ms window (scan every w samples)
    best = (0.0, np.inf, np.inf)
    for c in range(half, n - half, w):
        seg = x[c - half:c + half]
        med = float(np.median(seg))
        noise = 1.4826 * float(np.median(np.abs(seg - med)))
        if abs(med) < best[2]:
            best = (med, noise, abs(med))
    return best[0], best[1]


def _deflections(v: np.ndarray, fs: float, min_ms: float) -> list[tuple[int, float]]:
    """(sign, amplitude) of the waveform's deflections, in time order.

    A deflection is a maximal run of samples on one side of the baseline
    (zero — ``v`` is baseline-centered); its amplitude is the extremum of
    the run. Runs shorter than ``min_ms`` are noise slivers and dropped:
    near-baseline wobble must not create phases.
    """
    min_n = max(1, int(min_ms / 1000.0 * fs))
    out = []
    start = 0
    for i in range(1, len(v) + 1):
        if i == len(v) or (v[i] >= 0) != (v[start] >= 0):
            seg = v[start:i]
            amp = float(seg.max() if seg[0] >= 0 else seg.min())
            if len(seg) >= min_n and amp != 0.0:
                out.append((1 if seg[0] >= 0 else -1, abs(amp)))
            start = i
    return out


def classify(v_pos: float, v_neg: float, noise: float, wave: str,
             defls: list[tuple[int, float]]) -> tuple[str, str, int]:
    """(polarity, biphasic_pattern, n_significant_deflections).

    flat: primary excursion below the wave's floor or SNR. biphasic: the
    SECOND-largest significant (sliver-filtered) deflection is OPPOSITE-signed
    to the largest, clears the floor / ratio / 2x-noise gates; the pattern
    '+-' / '-+' records which deflection comes first in time. Same-sign
    second deflections (r + r' in rsR') never create a biphasic label.
    n_significant counts deflections clearing the absolute floor (qRs -> 3,
    qr/rs -> 2) for morphology analysis.
    """
    primary = max(v_pos, v_neg)
    n_sig = sum(1 for _, a in defls if a >= BIPHASIC_FLOOR_MV[wave])
    if primary < FLAT_FLOOR_MV[wave] or (noise > 1e-9 and primary / noise < AMP_NOISE_RATIO):
        return 'flat', '', n_sig
    if len(defls) >= 2:
        order = sorted(range(len(defls)), key=lambda k: defls[k][1], reverse=True)
        s1, a1 = defls[order[0]]
        s2, a2 = defls[order[1]]
        if s1 != s2 and a2 >= max(BIPHASIC_FLOOR_MV[wave], PHASE_RATIO[wave] * a1) \
                and (noise <= 1e-9 or a2 / noise >= 2.0):
            first = s1 if order[0] < order[1] else s2
            return 'biphasic', '+-' if first > 0 else '-+', n_sig
    return ('positive' if v_pos >= v_neg else 'negative'), '', n_sig


def _wave_row(x: np.ndarray, fs: float, on: int, off: int,
              base: float, noise: float, wave: str) -> dict:
    """Window stats + polarity + biphasic pattern for one wave on one lead."""
    if on < 0 or off <= on or off >= len(x):
        return {f'{wave}_on_ms': np.nan, f'{wave}_off_ms': np.nan,
                f'{wave}_pos_amp': np.nan, f'{wave}_neg_amp': np.nan,
                f'{wave}_abs_amp': np.nan, f'{wave}_peak_rel_ms': np.nan,
                f'{wave}_polarity': '', f'{wave}_pattern': '',
                f'{wave}_n_defl': np.nan}
    v = x[on:off + 1] - base
    pos = float(v.max())
    neg = float(-v.min())
    pk = int(np.argmax(np.abs(v)))
    defls = _deflections(v, fs, MIN_DEFLECTION_MS[wave])
    pol, pat, n_defl = classify(pos, neg, noise, wave, defls)
    return {f'{wave}_on_ms': round(on / fs * 1000.0, 1),
            f'{wave}_off_ms': round(off / fs * 1000.0, 1),
            f'{wave}_pos_amp': round(pos, 4), f'{wave}_neg_amp': round(neg, 4),
            f'{wave}_abs_amp': round(max(pos, neg), 4),
            f'{wave}_peak_rel_ms': round(pk / fs * 1000.0, 1),
            f'{wave}_polarity': pol, f'{wave}_pattern': pat,
            f'{wave}_n_defl': n_defl}


def process_record(rec: dict, beats: pd.DataFrame, out_png: str,
                   per_lead_dir: str) -> tuple[list[dict], dict]:
    """Polarity for one record; returns (per-beat-lead rows, summary row)."""
    sigs = rec['signals']
    fs = rec['fs'] or 1000.0
    scale_uv = read_scale_uv(rec['filepath'])
    ref_lead = str(beats['ref_lead'].iloc[0])
    n_beats = len(beats)

    # per-beat window indices from the joined upstream rows
    bi = {}
    for col_on, col_off, key in (('qrs_on_ms', 'qrs_off_ms', 'qrs'),
                                 ('t_on_ms', 't_off_ms', 't'),
                                 ('p_on_ms', 'p_off_ms', 'p')):
        on, off = [], []
        for a, b in zip(beats[col_on], beats[col_off]):
            ok = a == a and b == b
            on.append(_ms2idx(a, fs) if ok else -1)
            off.append(_ms2idx(b, fs) if ok else -1)
        bi[key] = (on, off)
    in_win_p = [bool(x) for x in beats['p_in_window']]
    pass_p = [bool(x) for x in beats['passes_snr']]

    mvs = {lead: sigs[lead] * scale_uv / 1000.0 for lead in ALL_LEADS if lead in sigs}

    # ---- per (beat, lead) rows ----
    rows = []
    for j in range(n_beats):
        pon, poff = bi['p'][0][j], bi['p'][1][j]
        if not (in_win_p[j] and pass_p[j]):
            pon, poff = -1, -1
        prev_toff = bi['t'][1][j - 1] if j > 0 else -1
        qon, qoff = bi['qrs'][0][j], bi['qrs'][1][j]
        ton, toff = bi['t'][0][j], bi['t'][1][j]
        for lead, x in mvs.items():
            base, noise = _baseline(x, fs, pon, prev_toff, qon)
            row = {'record': rec['filename'], 'beat_i': j, 'lead': lead,
                   'baseline_mv': round(base, 4), 'noise_mv': round(noise, 4)}
            row.update(_wave_row(x, fs, qon, qoff, base, noise, 'QRS'))
            row.update(_wave_row(x, fs, ton, toff, base, noise, 'T'))
            row.update(_wave_row(x, fs, pon, poff, base, noise, 'P'))
            rows.append(row)

    # ---- figure: cumulative spans + polarity characters on peaks ----
    beats_df = beats
    r_idx = np.array([_ms2idx(m, fs) for m in beats_df['r_ms']], dtype=int)
    lead_major = {}
    for lead in mvs:
        lr = [r_ for r_ in rows if r_['lead'] == lead]
        for w in ('P', 'QRS', 'T'):
            ps = [r_[f'{w}_polarity'] for r_ in lr if r_[f'{w}_polarity']]
            lead_major[(lead, w)] = max(set(ps), key=ps.count) if ps else ''

    def _chars(lead, j):
        """(text, x_ms, color) for the polarity glyph of each wave, beat j."""
        out = []
        for w, (ons, offs) in bi.items():
            on = ons[j]
            if on < 0:
                continue
            lr = rows[j * len(mvs) + list(mvs).index(lead)]
            pol = lr[f'{w.upper()}_polarity']
            if not pol:
                continue
            ch = {'positive': '+', 'negative': '-', 'biphasic': '±', 'flat': 'o'}[pol]
            peak_ms = lr[f'{w.upper()}_on_ms'] + (lr[f'{w.upper()}_peak_rel_ms']
                                                  if lr[f'{w.upper()}_peak_rel_ms'] ==
                                                  lr[f'{w.upper()}_peak_rel_ms'] else 0)
            out.append((ch, peak_ms, WAVE_COLOR[w.upper()]))
        return out

    fig, axes = plt.subplots(4, 3, figsize=(16, 9))
    pol_txt = '  |  '.join(f"{w}: " + ''.join(f"{l}{lead_major.get((l, w), '')[0] if lead_major.get((l, w)) else ''} "
                                                for l in ('I', 'II', 'aVR', 'V1'))
                            for w in ('P', 'QRS', 'T'))
    fig.suptitle(f"{rec['filename']}  |  reference lead {ref_lead}  |  polarity "
                 f"(per lead, per beat)\n{pol_txt}", fontsize=12, fontweight='bold', y=0.99)
    for r in range(4):
        for c in range(3):
            ax = axes[r, c]
            lead = DISPLAY_LEADS[r][c]
            if lead not in mvs:
                ax.text(0.5, 0.5, 'lead missing', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='gray')
                ax.set_title(lead, fontsize=10, fontweight='bold', loc='left')
                continue
            x = mvs[lead]
            t = np.arange(len(x)) / fs
            ax.plot(t, x, color='k', linewidth=0.6)
            for j in range(n_beats):
                for key, span_kw in (('p', P_SPAN), ('t', T_SPAN), ('qrs', QRS_SPAN)):
                    on, off = bi[key][0][j], bi[key][1][j]
                    if on >= 0 and off > on:
                        ax.axvspan(on / fs, off / fs, **span_kw)
                for ch, xms, color in _chars(lead, j):
                    y = float(x[_ms2idx(xms, fs)]) if 0 <= _ms2idx(xms, fs) < len(x) else 0.0
                    ax.text(xms / 1000.0, y, ch, color=color, fontsize=9,
                            fontweight='bold', ha='center', va='center', zorder=6)
            if len(r_idx):
                ax.plot(t[r_idx], x[r_idx], 'o', color='#d32f2f', markersize=3, zorder=5)
            star = ' \u2605' if lead == ref_lead else ''
            maj = ''.join(lead_major.get((lead, w), '?')[0] if lead_major.get((lead, w)) else '?'
                          for w in ('P', 'QRS', 'T'))
            ax.set_title(f"{lead}{star}  {maj}", fontsize=10, fontweight='bold', loc='left')
            ax.grid(True, alpha=0.25)
            if r == 3:
                ax.set_xlabel('Time (s)')
            if c == 0:
                ax.set_ylabel('mV')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    if per_lead_dir:
        for lead, x in mvs.items():
            fig, ax = plt.subplots(figsize=(12, 3.5))
            t = np.arange(len(x)) / fs
            ax.plot(t, x, color='k', linewidth=0.7)
            for j in range(n_beats):
                for key, span_kw in (('p', P_SPAN), ('t', T_SPAN), ('qrs', QRS_SPAN)):
                    on, off = bi[key][0][j], bi[key][1][j]
                    if on >= 0 and off > on:
                        ax.axvspan(on / fs, off / fs, **span_kw)
                for ch, xms, color in _chars(lead, j):
                    y = float(x[_ms2idx(xms, fs)]) if 0 <= _ms2idx(xms, fs) < len(x) else 0.0
                    ax.text(xms / 1000.0, y, ch, color=color, fontsize=10,
                            fontweight='bold', ha='center', va='center', zorder=6)
            if len(r_idx):
                ax.plot(t[r_idx], x[r_idx], 'o', color='#d32f2f', markersize=4, zorder=5)
            tag = '  [reference]' if lead == ref_lead else ''
            maj = ''.join(lead_major.get((lead, w), '?')[0] if lead_major.get((lead, w)) else '?'
                          for w in ('P', 'QRS', 'T'))
            ax.set_title(f"{rec['filename']} — lead {lead}{tag}  |  major polarity P/QRS/T: {maj}",
                         fontsize=10, fontweight='bold')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('mV')
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(os.path.join(per_lead_dir, f'{lead}.png'), dpi=100)
            plt.close(fig)

    # ---- summary row ----
    row = {'record': rec['filename'], 'ref_lead': ref_lead, 'n_beats': n_beats,
           'device_P_axis': rec['measurements'].get('P_axis'),
           'device_QRS_axis': rec['measurements'].get('QRS_axis'),
           'device_T_axis': rec['measurements'].get('T_axis')}
    df_rows = pd.DataFrame(rows)
    for w in ('P', 'QRS', 'T'):
        col = f'{w}_polarity'
        vc = df_rows[col].value_counts()
        for pol in ('positive', 'negative', 'biphasic', 'flat'):
            row[f'{w}_{pol}'] = int(vc.get(pol, 0))
        row[f'{w}_n_classified'] = int(vc.sum())
        row[f'{w}_pm'] = int((df_rows[f'{w}_pattern'] == '+-').sum())
        row[f'{w}_mp'] = int((df_rows[f'{w}_pattern'] == '-+').sum())
    # majority polarity per lead x wave (12 x 3)
    for lead in mvs:
        for w in ('P', 'QRS', 'T'):
            row[f'{lead}_{w}'] = lead_major.get((lead, w), '')
    row['interpretation'] = rec['interpretation'][:120]
    return rows, row


def main():
    ap = argparse.ArgumentParser(
        description='Per-lead per-beat polarity for P/QRS/T on propagated windows')
    ap.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    ap.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    ap.add_argument('--p-dir', default=DEFAULT_P_DIR)
    ap.add_argument('--t-dir', default=DEFAULT_T_DIR)
    ap.add_argument('--q-dir', default=DEFAULT_Q_DIR)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--records', default=None)
    args = ap.parse_args()

    p_all = pd.read_csv(os.path.join(args.p_dir, 'p_by_beat.csv'))
    t_all = pd.read_csv(os.path.join(args.t_dir, 't_by_beat.csv'))
    q_all = pd.read_csv(os.path.join(args.q_dir, 'qrs_by_beat.csv'))
    ref = pd.read_csv(os.path.join(args.p_dir, 'summary_by_record.csv'))[['record', 'ref_lead']]

    beats = q_all[['record', 'beat_i', 'r_ms', 'qrs_on_ms', 'qrs_off_ms']].merge(
        t_all[['record', 'beat_i', 't_on_ms', 't_off_ms', 'in_window']]
        .rename(columns={'in_window': 't_in_window'}),
        on=['record', 'beat_i'], how='left').merge(
        p_all[['record', 'beat_i', 'p_on_ms', 'p_off_ms', 'in_window', 'passes_snr']]
        .rename(columns={'in_window': 'p_in_window'}),
        on=['record', 'beat_i'], how='left').merge(ref, on='record', how='left')

    plots_dir = os.path.join(args.out_dir, 'plots')
    per_lead_base = os.path.join(args.out_dir, 'plots_per_lead')
    os.makedirs(plots_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(args.data_dir) if f.endswith('.aECG'))
    if args.records:
        wanted = {x.strip().removesuffix('.aECG') for x in args.records.split(',')}
        files = [f for f in files if f[:-len('.aECG')] in wanted]
    if args.limit:
        files = files[:args.limit]

    rows, rec_rows, errors = [], [], []
    t0 = time.time()
    for i, fname in enumerate(files, 1):
        rec_name = fname[:-len('.aECG')]
        try:
            b = beats[beats['record'] == rec_name].sort_values('beat_i')
            if not len(b) or pd.isna(b['ref_lead'].iloc[0]):
                raise ValueError(f'no upstream beats for {rec_name} (stale upstream?)')
            rec = parse_aecg(os.path.join(args.data_dir, fname))
            if not rec['signals']:
                raise ValueError('no lead signals parsed')
            rec_dir = os.path.join(per_lead_base, rec['filename'])
            os.makedirs(rec_dir, exist_ok=True)
            br, rr = process_record(rec, b, os.path.join(plots_dir, f"{rec['filename']}.png"),
                                    rec_dir)
            rows.extend(br)
            rec_rows.append(rr)
        except Exception as e:
            errors.append((fname, str(e)))
        if i % 50 == 0 or i == len(files):
            print(f'[{i}/{len(files)}] elapsed {time.time() - t0:.0f}s, errors: {len(errors)}', flush=True)

    df = pd.DataFrame(rows)
    sm = pd.DataFrame(rec_rows)
    p_beat = os.path.join(args.out_dir, 'polarity_by_beat_lead.csv')
    p_rec = os.path.join(args.out_dir, 'summary_by_record.csv')

    def _save(frame, path):
        """to_csv that never loses a finished run to a locked target file
        (e.g. the previous CSV open in Excel): fall back to <name>.new.csv."""
        try:
            frame.to_csv(path, index=False, encoding='utf-8-sig')
        except PermissionError:
            alt = path.replace('.csv', '.new.csv')
            frame.to_csv(alt, index=False, encoding='utf-8-sig')
            print(f'WARNING: {path} is locked (open in Excel?); saved to {alt}')

    _save(df, p_beat)
    _save(sm, p_rec)

    print(f"\nDone: {len(rec_rows)} records, {len(errors)} errors / {len(files)} files "
          f"in {time.time() - t0:.0f}s")
    for w in ('P', 'QRS', 'T'):
        n = sm[f'{w}_n_classified'].sum()
        if n:
            dist = {pol: int(sm[f'{w}_{pol}'].sum()) for pol in
                    ('positive', 'negative', 'biphasic', 'flat')}
            print(f"{w}: {dist}  (biphasic {100 * dist['biphasic'] / n:.1f}%, "
                  f"flat {100 * dist['flat'] / n:.1f}%; patterns +- {int(sm[f'{w}_pm'].sum())}"
                  f" / -+ {int(sm[f'{w}_mp'].sum())})")
    # V1 P pattern is the RA-LA reversal signature (normal sinus: '+-')
    v1 = df[df['lead'] == 'V1'] if len(df) else pd.DataFrame()
    if len(v1):
        print('V1 P biphasic patterns:', v1['P_pattern'].value_counts().to_dict())
    # aVR reversal probe: device P_axis in normal range but aVR QRS/T dominant positive
    if {'device_P_axis', 'AVR_QRS', 'AVR_T'}.issubset(sm.columns):
        probe = sm[(sm['device_P_axis'].between(0, 90)) &
                   (sm['AVR_QRS'] == 'positive') & (sm['AVR_T'] == 'positive')]
        print(f"aVR-reversal probe (device P_axis 0-90, aVR QRS+/T+): {len(probe)} records: "
              f"{probe['record'].head(10).tolist()}{'...' if len(probe) > 10 else ''}")
    print(f"CSVs: {p_beat}\n      {p_rec}")
    for fname, err in errors[:10]:
        print(f'  ERROR {fname}: {err}')


if __name__ == '__main__':
    main()
