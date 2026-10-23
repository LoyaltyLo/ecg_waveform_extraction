#!/usr/bin/env python
"""Stage 7: RA-LA (left/right arm) limb-lead reversal detection.

Detects RA-LA reversal per record from the features the earlier stages
already measured (per-beat per-lead polarity CSV + record overview) —
pure decision logic on measured features, no new signal detection.

Physics: swapping RA/LA mirrors the frontal plane, I -> -I, II <-> III,
aVR <-> aVL, aVF and the chest leads unchanged.  Observable signatures
(and their measured quality against the 59 device-flagged records
"左右手臂导联可能接反" as a silver standard):

  S1  P frontal axis in [95, 215] deg   88.1% sens / 91.6% spec
      — THE discriminator: pathologic right axis (RVH, lateral MI) flips
      QRS/T but leaves the sinus P axis in 0-90; reversal flips P too
      (theta -> 180 - theta).  Axis needs a reliable P amplitude.
  S2  lead I inverted: P neg + T neg + QRS not positive
      86.4% / 95.7%
  S3  aVR P/QRS/T all majority-positive — rare but nearly conclusive
      (~29% / ~98%): after the swap aVR shows the old aVL, whose P is
      usually tiny, so S3 firing with a clear P is strong evidence.

Hard gates (score cannot fire without them):
  - P evidence gate: p_pass_rate >= 50% and median |P| in I/aVF above
    the axis floor; failing it caps the verdict at 'suspicious' (AF,
    low-amplitude P) and disables S1.
  - Differential: reversed R progression V1->V6 flags "dextrocardia?"
    (chest leads are untouched by arm reversal; 0/59 flagged records had
      R_V1 > R_V6).  Normal P axis + inverted lead-I QRS/T flags
      "pathologic RAD" (the aVR-probe mimics), not reversal.

Verdicts: definite / probable / suspicious / none / poor_P.  definite
includes the double signature S1 & S2 (user-adjudicated as true even
when the device missed it).  Evidence score (weights measured on the
silver standard): I_T neg +2, aVR QRS+T pos +2, I_P neg +1.5,
aVR_P pos +1.5, per-beat consistency >= 70% +1, device P axis flipped
or unmeasured +1, P amplitude adequate +0.5.

Inputs
------
output/polarity_reflead/polarity_by_beat_lead.csv
output/record_overview/record_overview.csv
raw .aECG XMLs (only for figures of triggered/flagged records)

Outputs (output/reversal_reflead/)
----------------------------------
reversal_by_record.csv : verdict, score, fired rules with values,
    agreement vs the device flag, differential notes (666 rows).
reversal_simple.csv : quick-read table — record, lead-I P/QRS/T
    polarity and the verdict, Chinese values, 5 columns total.
plots/<record>.png     : 12-lead figure for every triggered or
    device-flagged record — spans + polarity glyphs as in the polarity
    stage, lead I / aVR highlighted, verdict in the suptitle.
cohort_summary.txt     : verdict counts and device-flag agreement.

Validation (full 666-record run, 2026-10)
-----------------------------------------
666-row CSV + 111 figures in ~3 min.  Verdicts: definite 94 /
probable 9 / suspicious 6 / poor_P 50 / none 507.  Hard signatures:
S1 (P axis flip) 98, S2 (lead I inverted) 85, S3 (aVR all-positive)
25, double S1&S2 76.  vs the device flag (59) as silver standard:
sensitivity 86.4% (51/59); ours-only 52 of which 26 carry the double
signature (user-adjudicated true reversals the device missed — its
reports say "right axis / lateral MI", the classic RA-LA mimic);
device-only 8 = 4 poor-P (P coverage <50%, honestly capped) + 1
"none" (I_P positive, our P axis -51 deg — device false positive
suspected) + 3 suspicious.  8 definite/probable carry a
poor-R-progression (dextrocardia?) note for human review; the
flagged-59 all have normal R progression.  Figure check: I/aVR red
frames, spans+glyphs on waves, verdict/P-axis in the suptitle.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ecg_waveform_extraction.src.utils.aecg_parser import parse_aecg
from ecg_waveform_extraction.src.batch_rpeaks_pan_tompkins import (
    DISPLAY_LEADS, ALL_LEADS, read_scale_uv,
)
from ecg_waveform_extraction.src.batch_polarity_reflead import (
    QRS_SPAN, T_SPAN, P_SPAN, WAVE_COLOR,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = r'C:\LoyaltyLo\datasets\RA-LA_Reversal\aECG'
DEFAULT_POL = str(ROOT / 'output' / 'polarity_reflead')
DEFAULT_OVERVIEW = str(ROOT / 'output' / 'record_overview' / 'record_overview.csv')
DEFAULT_OUT = str(ROOT / 'output' / 'reversal_reflead')

DEVICE_FLAG_PHRASE = '左右手臂导联可能接反'
# ---- thresholds (calibrated against the 59 device-flagged records) ----
P_RATE_MIN_PCT = 50.0     # below: P evidence insufficient (AF, low P)
P_AMP_FLOOR_MV = 0.03     # below: our P axis is unreliable -> S1 disabled
P_AMP_GOOD_MV = 0.05      # adequate P amplitude (+0.5 evidence)
AXIS_FLIP_DEG = (95.0, 215.0)   # P frontal axis window after RA-LA swap
CONSISTENCY_MIN = 0.70    # share of beats with negative P in lead I
SCORE_DEFINITE = 4.0      # hard signature + this score -> definite
SCORE_SUSPICIOUS = 5.0    # no hard signature but this much evidence
R_PROG_MARGIN_MV = 0.2    # R_V1 - R_V6 above this -> poor progression
CHEST_LEADS = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# evidence weights (I_T neg measured 98.3%/91.1% on the silver standard,
# aVR QRS+T pos 33.9%/95.6% — specific; I_P neg 93.2%/87.3%)
W = {'I_T_neg': 2.0, 'avr_qt_pos': 2.0, 'I_P_neg': 1.5, 'avr_p_pos': 1.5,
     'consistency': 1.0, 'device_p_flip': 1.0, 'p_amp_good': 0.5}


def _signed_amp(g: pd.DataFrame, wave: str) -> np.ndarray:
    """Signed amplitude per beat: +amp if positive, -amp if negative, 0 else."""
    pos, neg = g[f'{wave}_pos_amp'], g[f'{wave}_neg_amp']
    pol = g[f'{wave}_polarity']
    return np.where(pol == 'positive', pos,
                    np.where(pol == 'negative', -neg, 0.0))


def record_features(pol: pd.DataFrame) -> dict:
    """Per-record reversal features from the per-beat per-lead polarity rows."""
    out = {}
    pw = pol[pol['P_on_ms'].notna()]
    gi, gavf = pw[pw['lead'] == 'I'], pw[pw['lead'] == 'AVF']
    a_i = float(np.median(_signed_amp(gi, 'P'))) if len(gi) else np.nan
    a_avf = float(np.median(_signed_amp(gavf, 'P'))) if len(gavf) else np.nan
    ax = np.degrees(np.arctan2(a_avf, a_i)) % 360.0 if a_i == a_i and a_avf == a_avf \
        and (a_i or a_avf) else np.nan
    neg = (gi['P_polarity'] == 'negative').mean() if len(gi) else np.nan

    qw = pol[pol['QRS_on_ms'].notna()]
    r_amp = {l: (float(np.median(_signed_amp(qw[qw['lead'] == l], 'QRS')))
                 if len(qw[qw['lead'] == l]) else np.nan) for l in CHEST_LEADS}
    out.update(p_ax_deg=ax, p_amp_i_mv=a_i, p_amp_avf_mv=a_avf,
               i_p_neg_share=float(neg) if neg == neg else np.nan,
               r_v1_mv=r_amp['V1'], r_v6_mv=r_amp['V6'])
    amps = [abs(a) for a in (a_i, a_avf) if a == a]
    out['p_amp_mv'] = max(amps) if amps else np.nan
    return out


def classify(f: dict) -> dict:
    """Verdict + score from one record's feature dict."""
    p_gate = f['p_pass_rate_pct'] >= P_RATE_MIN_PCT
    ax_ok = p_gate and f['p_ax_deg'] == f['p_ax_deg'] and f['p_amp_mv'] >= P_AMP_FLOOR_MV
    s1 = bool(ax_ok and AXIS_FLIP_DEG[0] <= f['p_ax_deg'] <= AXIS_FLIP_DEG[1])
    s2 = (f['I_P'] == 'negative' and f['I_T'] == 'negative'
          and f['I_QRS'] in ('negative', 'biphasic'))
    s3 = (f['AVR_P'] == 'positive' and f['AVR_QRS'] == 'positive'
          and f['AVR_T'] == 'positive')

    fired, score = [], 0.0
    def hit(name, cond, val):
        nonlocal score
        if cond:
            score += W[name]
            fired.append(f'{name}={val}')

    if p_gate:
        hit('I_T_neg', f['I_T'] == 'negative', f['I_T'])
        hit('I_P_neg', f['I_P'] == 'negative', f['I_P'])
        hit('avr_p_pos', f['AVR_P'] == 'positive', f['AVR_P'])
        hit('avr_qt_pos', f['AVR_QRS'] == 'positive' and f['AVR_T'] == 'positive',
            f"{f['AVR_QRS']}/{f['AVR_T']}")
        hit('consistency', f['i_p_neg_share'] == f['i_p_neg_share']
            and f['i_p_neg_share'] >= CONSISTENCY_MIN,
            f"{f['i_p_neg_share']:.0%}")
        dev = f['device_P_axis']
        hit('device_p_flip', dev == dev and (dev > 90.0 or dev <= 0.0), f'{dev:.0f}deg')
        hit('p_amp_good', f['p_amp_mv'] == f['p_amp_mv'] and f['p_amp_mv'] >= P_AMP_GOOD_MV,
            f"{f['p_amp_mv']:.2f}mV")
    else:  # QRS/T-only branch: weaker set, verdict stays capped
        hit('I_T_neg', f['I_T'] == 'negative', f['I_T'])
        hit('avr_qt_pos', f['AVR_QRS'] == 'positive' and f['AVR_T'] == 'positive',
            f"{f['AVR_QRS']}/{f['AVR_T']}")

    hard = (s1 or s2 or s3) and p_gate
    double = s1 and s2
    if double:
        verdict = 'definite'
    elif hard and score >= SCORE_DEFINITE:
        verdict = 'definite'
    elif hard:
        verdict = 'probable'
    elif p_gate and score >= SCORE_SUSPICIOUS:
        verdict = 'suspicious'
    elif p_gate:
        verdict = 'none'
    else:
        verdict = 'poor_P'

    notes = []
    if f['r_v1_mv'] == f['r_v1_mv'] and f['r_v6_mv'] == f['r_v6_mv'] \
            and f['r_v1_mv'] - f['r_v6_mv'] > R_PROG_MARGIN_MV:
        notes.append('poor_R_progression(dextrocardia?)')
    if not p_gate:
        notes.append('P_coverage_low')
    if ax_ok and not s1 and f['I_QRS'] == 'negative' and f['I_T'] == 'negative':
        notes.append('pathologic_RAD(P axis normal)')
    return {'verdict': verdict, 'score': round(score, 1),
            'S1_P_axis_flip': s1, 'S2_lead_I_inverted': bool(s2),
            'S3_aVR_all_positive': bool(s3), 'fired': '; '.join(fired),
            'notes': '; '.join(notes)}


def plot_record(fname: str, data_dir: str, pol: pd.DataFrame, f: dict,
                out_png: str) -> None:
    """12-lead reversal figure: spans + polarity glyphs, I / aVR framed."""
    rec = parse_aecg(os.path.join(data_dir, fname))
    sigs = rec['signals']
    fs = rec['fs'] or 1000.0
    scale_uv = read_scale_uv(rec['filepath'])
    mvs = {lead: sigs[lead] * scale_uv / 1000.0 for lead in ALL_LEADS if lead in sigs}

    spans = {w: pol[pol[f'{w}_on_ms'].notna()][[f'{w}_on_ms', f'{w}_off_ms']]
             .drop_duplicates()
             for w in ('P', 'QRS', 'T')}
    fig, axes = plt.subplots(4, 3, figsize=(16, 9))
    fig.suptitle(
        f"{rec['filename']}  |  RA-LA reversal: {f['verdict'].upper()}  "
        f"(score {f['score']}, S1={f['S1_P_axis_flip']} S2={f['S2_lead_I_inverted']} "
        f"S3={f['S3_aVR_all_positive']})\n"
        f"P axis ours {f['p_ax_deg']:.0f} deg (flip window {AXIS_FLIP_DEG[0]:.0f}-"
        f"{AXIS_FLIP_DEG[1]:.0f}) | device P/QRS/T axis "
        f"{f['device_P_axis']:.0f}/{f['device_QRS_axis']:.0f}/{f['device_T_axis']:.0f} | "
        f"P pass {f['p_pass_rate_pct']:.0f}%"
        + (f" | {f['notes']}" if f['notes'] else ''),
        fontsize=11, fontweight='bold', y=0.99)
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
            for w, span_kw in (('P', P_SPAN), ('QRS', QRS_SPAN), ('T', T_SPAN)):
                for _, srow in spans[w].iterrows():
                    ax.axvspan(srow[f'{w}_on_ms'] / 1000.0,
                               srow[f'{w}_off_ms'] / 1000.0, **span_kw)
            lr = pol[pol['lead'] == lead]
            for _, prow in lr.iterrows():
                for w in ('P', 'QRS', 'T'):
                    pol_w = prow[f'{w}_polarity']
                    if pd.isna(prow[f'{w}_on_ms']) or not pol_w:
                        continue
                    ch = {'positive': '+', 'negative': '-', 'biphasic': '±',
                          'flat': 'o'}[pol_w]
                    pk = prow[f'{w}_on_ms'] + (prow[f'{w}_peak_rel_ms']
                                               if pd.notna(prow[f'{w}_peak_rel_ms']) else 0)
                    idx = int(round(pk / 1000.0 * fs))
                    y = float(x[idx]) if 0 <= idx < len(x) else 0.0
                    ax.text(pk / 1000.0, y, ch, color=WAVE_COLOR[w], fontsize=9,
                            fontweight='bold', ha='center', va='center', zorder=6)
            maj = ''.join(f.get(f'{lead}_{w}', '?')[0] if f.get(f'{lead}_{w}') else '?'
                          for w in ('P', 'QRS', 'T'))
            ax.set_title(f"{lead}  {maj}", fontsize=10, fontweight='bold', loc='left')
            if lead in ('I', 'AVR'):  # the reversal leads, visually framed
                for sp in ax.spines.values():
                    sp.set_edgecolor('#d32f2f')
                    sp.set_linewidth(2.0)
            ax.grid(True, alpha=0.25)
            if r == 3:
                ax.set_xlabel('Time (s)')
            if c == 0:
                ax.set_ylabel('mV')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    ap.add_argument('--pol-dir', default=DEFAULT_POL)
    ap.add_argument('--overview', default=DEFAULT_OVERVIEW)
    ap.add_argument('--out-dir', default=DEFAULT_OUT)
    ap.add_argument('--records', default=None,
                    help='comma-separated record filter (debug)')
    ap.add_argument('--no-plots', action='store_true')
    args = ap.parse_args()

    t0 = time.time()
    pol_all = pd.read_csv(os.path.join(args.pol_dir, 'polarity_by_beat_lead.csv'),
                          encoding='utf-8-sig',
                          usecols=['record', 'beat_i', 'lead',
                                   'P_on_ms', 'P_off_ms', 'P_pos_amp', 'P_neg_amp',
                                   'P_polarity', 'P_peak_rel_ms',
                                   'QRS_on_ms', 'QRS_off_ms', 'QRS_pos_amp',
                                   'QRS_neg_amp', 'QRS_polarity', 'QRS_peak_rel_ms',
                                   'T_on_ms', 'T_off_ms', 'T_polarity',
                                   'T_peak_rel_ms'])
    ov = pd.read_csv(args.overview, encoding='utf-8-sig')
    lead_cols = [f'{l}_{w}' for l in ALL_LEADS for w in ('P', 'QRS', 'T')]
    base = ov[['record', 'ref_lead', 'p_pass_rate_pct',
               'device_P_axis', 'device_QRS_axis', 'device_T_axis']
              + lead_cols + ['interpretation']]
    if args.records:
        keep = {x.strip() for x in args.records.split(',')}
        base = base[base['record'].isin(keep)]

    feats = {rec: record_features(g) for rec, g in pol_all.groupby('record')}
    rows = []
    for _, r in base.iterrows():
        f = r.to_dict()
        f.update(feats.get(r['record'], {}))
        f['device_flag'] = DEVICE_FLAG_PHRASE in str(r['interpretation'])
        f.update(classify(f))
        rows.append({k: f[k] for k in (
            'record', 'verdict', 'score', 'S1_P_axis_flip', 'S2_lead_I_inverted',
            'S3_aVR_all_positive', 'fired', 'notes', 'p_ax_deg', 'p_amp_mv',
            'i_p_neg_share', 'r_v1_mv', 'r_v6_mv', 'p_pass_rate_pct',
            'device_P_axis', 'device_flag', 'interpretation')})
    df = pd.DataFrame(rows)

    detected = df['verdict'].isin(('definite', 'probable'))
    df['agreement'] = np.where(
        df['device_flag'] & detected, 'both',
        np.where(detected, 'ours_only',
                 np.where(df['device_flag'], 'device_only', 'neither')))
    df['double_signature'] = df['S1_P_axis_flip'] & df['S2_lead_I_inverted']

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(out_dir / 'reversal_by_record.csv', index=False,
                  encoding='utf-8-sig')
    except PermissionError:
        df.to_csv(out_dir / 'reversal_by_record.new.csv', index=False,
                  encoding='utf-8-sig')
        print('WARNING: reversal_by_record.csv locked; wrote .new.csv')

    # quick-read CSV: record + lead-I wave polarity + verdict, Chinese
    pol_cn = {'positive': '正向', 'negative': '负向', 'biphasic': '双相',
              'flat': '平坦'}
    verd_cn = {'definite': '确证反接', 'probable': '可能反接',
               'suspicious': '可疑反接', 'none': '正常连接',
               'poor_P': 'P证据不足'}
    simple = df[['record', 'verdict']].merge(
        base[['record', 'I_P', 'I_QRS', 'I_T']], on='record')
    for c in ('I_P', 'I_QRS', 'I_T'):
        simple[c] = simple[c].map(pol_cn).fillna('—')
    simple['verdict'] = simple['verdict'].map(verd_cn)
    simple.columns = ['记录号', 'P波(I导)', 'QRS波(I导)', 'T波(I导)', '结论']
    try:
        simple.to_csv(out_dir / 'reversal_simple.csv', index=False,
                      encoding='utf-8-sig')
    except PermissionError:
        simple.to_csv(out_dir / 'reversal_simple.new.csv', index=False,
                      encoding='utf-8-sig')
        print('WARNING: reversal_simple.csv locked; wrote .new.csv')

    lines = []
    add = lines.append
    add(f"records: {len(df)}  elapsed {time.time() - t0:.0f}s")
    add("verdicts: " + df['verdict'].value_counts().to_string()
        .replace('\n', '  '))
    add(f"hard signatures: S1 {int(df['S1_P_axis_flip'].sum())}, "
        f"S2 {int(df['S2_lead_I_inverted'].sum())}, "
        f"S3 {int(df['S3_aVR_all_positive'].sum())}, "
        f"double(S1&S2) {int(df['double_signature'].sum())}")
    add("agreement vs device flag: " + df['agreement'].value_counts().to_string()
        .replace('\n', '  '))
    flag = df['device_flag']
    add(f"vs device flag as silver standard: sensitivity "
        f"{100 * detected[flag].mean():.1f}% "
        f"({int(detected[flag].sum())}/{int(flag.sum())}), detected among "
        f"unflagged {int((detected & ~flag).sum())} "
        f"(of which double-signature, adjudicated true: "
        f"{int((df['double_signature'] & ~flag).sum())})")
    add("device-only records: "
        + ', '.join(df.loc[df['agreement'] == 'device_only', 'record'].head(15)))
    add("notes distribution: "
        + df['notes'].value_counts().head(6).to_string().replace('\n', '  '))
    report = '\n'.join(lines)
    print('\n' + report)
    (out_dir / 'cohort_summary.txt').write_text(report + '\n', encoding='utf-8')

    if not args.no_plots:
        plots_dir = out_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        wanted = df[detected | df['device_flag']]
        for i, (_, r) in enumerate(wanted.iterrows(), 1):
            try:
                plot_record(f"{r['record']}.aECG", args.data_dir,
                            pol_all[pol_all['record'] == r['record']],
                            {**r.to_dict(), **base[base['record'] == r['record']]
                             .iloc[0].to_dict()},
                            str(plots_dir / f"{r['record']}.png"))
            except Exception as e:
                print(f"  plot ERROR {r['record']}: {e}")
            if i % 25 == 0:
                print(f'  plots {i}/{len(wanted)}', flush=True)
        print(f"figures: {len(list(plots_dir.glob('*.png')))} -> {plots_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
