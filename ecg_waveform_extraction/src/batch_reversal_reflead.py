#!/usr/bin/env python
"""Stage 7: limb-lead reversal detection (RA-LA / RA-LL / LA-LL).

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
      86.4% / 95.7% — guarded: if the measured P axis is up-left
      (215-360 deg, retrograde/ectopic/noise) the P leg is void: a
      mirrored sinus P cannot land there (RAD/AF mimic guard).
  S3  aVR P/QRS/T all majority-positive — rare but nearly conclusive
      (~29% / ~98%): after the swap aVR shows the old aVL, whose P is
      usually tiny, so S3 firing with a clear P is strong evidence.

RA-LL (right arm <-> left leg): II -> -II, I -> -III, aVR <-> aVF,
aVL unchanged.  T1 lead II inverted (mirror triple, S2's shape on II);
junctional/retrograde P also gives II P neg but keeps II QRS positive
and aVR QRS negative, so the QRS legs separate the two.  T2 aVR
all-positive AND aVF QRS negative (aVR shows the old aVF, aVF the old
aVR — aVF all-positive is the normal sinus pattern, so aVR displaying
it is the clean RA-LL tell).  Ladder mirrors RA-LA (T1&T2 -> definite);
under RA-LL lead III shows -I, so U1 firing alongside T1 is EXPECTED
and corroborates ra_ll rather than contradicting it.

LA-LL (left arm <-> left leg): III -> -III, I <-> II, aVL <-> aVF,
aVR unchanged.  U1 lead III inverted; U2 aVL P/QRS positive AND aVF
QRS negative (the aVL/aVF swap).  definite needs U1 & U2 — III
inversion alone is common in left-axis normals, and LAFB mimics U2's
aVL+/aVF- without inverting III's P; partial evidence stays
suspicious.

RA-LL / LA-LL have no silver standard in this dataset (the device
never reports them), so their verdicts are UNCALIBRATED and held to a
stricter bar: (1) a verdict requires a hard signature — the soft score
never fires them alone (lead-III P-/T- with a positive QRS is the
textbook left-axis variant and would score 5.0 on ~19 otherwise-normal
records), and (2) every P leg must be per-beat stable (>= 70% of
P-detected beats) — a bare 57-64% majority reflects atrial bigeminy /
baseline wander (polarity alternating per beat), not wiring, and two
such near-fires (1807185Z4A, 2110148P2G) were caught exactly this way.

The three types are physically exclusive; the strongest verdict wins
(ties: score, then ra_la — the only silver-standard-calibrated type).
The P-evidence gate, dextrocardia differential and poor_P cap apply
to all three; the nonsinus-P guard is RA-LA only (the I/aVF axis
formula is not valid under the other swaps).

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
aVR_P pos +1.5, per-beat consistency >= 70% +1, P amplitude adequate
+0.5.  No device field feeds the verdict (user directive 2026-10):
device axes / interpretation are exported only as comparison columns,
and the device flag is only a reporting yardstick.

Inputs
------
output/polarity_reflead/polarity_by_beat_lead.csv
output/record_overview/record_overview.csv
raw .aECG XMLs (only for figures of triggered/flagged records)

Outputs (output/reversal_reflead/)
----------------------------------
reversal_by_record.csv : verdict, reversal_type (ra_la/ra_ll/la_ll),
    score, signature flags S1-S3 (RA-LA), T1/T2 (RA-LL), U1/U2 (LA-LL),
    fired rules with values, agreement vs the device flag, differential
    notes (666 rows).
reversal_simple.csv : quick-read table — record, lead-I and aVR
    P/QRS/T polarity and the verdict, Chinese values, 8 columns;
    detected verdicts carry the type (确证RA-LA反接 / 确证RA-LL反接 /
    确证LA-LL反接).  II/III/aVL/aVF polarity for the other two types is
    in reversal_by_record.csv / record_overview.csv.
plots/<record>.png     : 12-lead figure for every triggered or
    device-flagged record — spans + polarity glyphs as in the polarity
    stage, the active type's leads framed (RA-LA: I/aVR; RA-LL:
    II/aVR/aVF; LA-LL: III/aVL/aVF), type+verdict in the suptitle.
cohort_summary.txt     : verdict counts and device-flag agreement.

Validation (full 666-record run, 2026-10)
-----------------------------------------
666-row CSV + 109 figures in ~3 min.  Verdicts: definite 87 /
probable 14 / suspicious 3 / poor_P 50 / none 512.  Hard signatures:
S1 (P axis flip) 98, S2 (lead I inverted) 83, S3 (aVR all-positive)
25, double S1&S2 76.  vs the device flag (59) as reporting yardstick
only: sensitivity 86.4% (51/59); ours-only 50 of which 26 carry the
double signature (user-adjudicated true reversals the device missed —
its reports say "right axis / lateral MI", the classic RA-LA mimic);
device-only 8 = 4 poor-P (P coverage <50%, honestly capped) + 1
"none" (I_P positive, our P axis -51 deg — device false positive
suspected) + 3 suspicious.  8 definite/probable carry a
poor-R-progression (dextrocardia?) note for human review; the
flagged-59 all have normal R progression.  Figure check: I/aVR red
frames, spans+glyphs on waves, verdict/P-axis in the suptitle.
2026-10-27 the device_p_flip evidence item (+1) was removed (user
directive: no dataset conclusions in the verdict) — 5 definite demoted
to probable, 5 suspicious to none; the detected set, the double
signatures (76) and the headline numbers were unchanged.
2026-10-28 the RAD/AF mimic guard: a measurable P axis in 215-360 deg
(up-left = retrograde/ectopic/noise; a mirrored sinus P cannot land
there) voids S2's P leg.  Two records demoted definite -> suspicious
with a P_axis_nonsinus note (220316L6F9 is atrial fibrillation — its
"P" is fibrillatory noise; 2108190POK sinus tach with borderline RAD,
our P axis 222 vs device 59 deg).  Of the 331 records whose device
text says "右偏电轴", we call none/poor_P on 297 — the P-axis
gate is what separates pathologic RAD (QRS/T flip, P stays 0-90)
from true reversal.
2026-11-04 the three-type extension (RA-LL / LA-LL, no silver
standard exists for them in this dataset).  The RA-LA result is
bit-identical: definite 87 / probable 14 / suspicious-3 preserved,
detected set (101) unchanged.  RA-LL/LA-LL detections: 0 — the
strict bar (hard signature + >=70% per-beat P stability) killed both
near-fires that a first pass had called definite: 1807185Z4A (II/III
mirror triples but lead-II P only 4/7 beats negative, 0.07 mV) and
2110148P2G (per-beat P polarity ALTERNATES across all limb leads with
0.15-0.4 mV amplitudes on both signs — atrial bigeminy / baseline
wander, device gave no interpretation at all; it stays definite ra_la
via the silver-calibrated S3 ladder).  Survivors: 4x T1 and 1x T2,
all on records already definite ra_la (multi_type note — under RA-LA
lead II shows the old III, so T1 fires on right-axis RA-LA subjects);
4x U1 + 1x U2 with 3 la_ll suspicious partials (III inverted with
stable P) and 210316Y21L suspicious/LA-LL_P_unstable — full U1+U2
legs, per-beat aVL/III amplitude mirroring (0.49/0.51, 1.19/1.16 mV)
and normal R progression say true LA-LL on a +90-axis subject, but
lead-III P is 4/6 beats negative = 67% < 70%, so definite was not
defensible.  2012238UMB (U1&U2 both) honestly capped poor_P (P
coverage 45%).  The first pass also showed why score-only firing is
disabled for uncalibrated types: 19 normal left-axis variants score
exactly 5.0 on III P-/T- + consistency + amplitude.
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
AXIS_NONSINUS_DEG = (215.0, 360.0)  # up-left P: retrograde/ectopic/noise —
                                    # a mirrored sinus P cannot land here
CONSISTENCY_MIN = 0.70    # share of beats with negative P in lead I
SCORE_DEFINITE = 4.0      # hard signature + this score -> definite
SCORE_SUSPICIOUS = 5.0    # no hard signature but this much evidence
R_PROG_MARGIN_MV = 0.2    # R_V1 - R_V6 above this -> poor progression
CHEST_LEADS = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# evidence weights (I_T neg measured 98.3%/91.1% on the silver standard,
# aVR QRS+T pos 33.9%/95.6% — specific; I_P neg 93.2%/87.3%).
# Device-derived items are excluded: our measurements only.
W = {'I_T_neg': 2.0, 'avr_qt_pos': 2.0, 'I_P_neg': 1.5, 'avr_p_pos': 1.5,
     'consistency': 1.0, 'p_amp_good': 0.5}
# RA-LL / LA-LL: no silver standard on this dataset — weights mirror
# the RA-LA calibration with the leads substituted per type
W_RA_LL = {'II_T_neg': 2.0, 'II_P_neg': 1.5, 'avr_all_pos': 2.0,
           'avf_q_neg': 1.5, 'ii_consistency': 1.0, 'p_amp_good': 0.5}
W_LA_LL = {'III_T_neg': 2.0, 'III_P_neg': 1.5, 'avl_pq_pos': 1.5,
           'avf_q_neg': 1.5, 'iii_consistency': 1.0, 'p_amp_good': 0.5}
# leads framed red in figures, per detected type
FRAME_LEADS = {'ra_la': ('I', 'AVR'), 'ra_ll': ('II', 'AVR', 'AVF'),
               'la_ll': ('III', 'AVL', 'AVF')}


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

    def p_share(lead, value):
        gl = pw[pw['lead'] == lead]
        s = (gl['P_polarity'] == value).mean() if len(gl) else np.nan
        return float(s) if s == s else np.nan

    qw = pol[pol['QRS_on_ms'].notna()]
    r_amp = {l: (float(np.median(_signed_amp(qw[qw['lead'] == l], 'QRS')))
                 if len(qw[qw['lead'] == l]) else np.nan) for l in CHEST_LEADS}
    out.update(p_ax_deg=ax, p_amp_i_mv=a_i, p_amp_avf_mv=a_avf,
               i_p_neg_share=float(neg) if neg == neg else np.nan,
               ii_p_neg_share=p_share('II', 'negative'),
               iii_p_neg_share=p_share('III', 'negative'),
               avr_p_pos_share=p_share('AVR', 'positive'),
               avl_p_pos_share=p_share('AVL', 'positive'),
               r_v1_mv=r_amp['V1'], r_v6_mv=r_amp['V6'])
    amps = [abs(a) for a in (a_i, a_avf) if a == a]
    out['p_amp_mv'] = max(amps) if amps else np.nan
    return out


def classify(f: dict) -> dict:
    """Verdict + reversal type + score from one record's feature dict.

    Three physically exclusive arm swaps, each with its own hard
    signatures; the strongest verdict wins (ties: score, then ra_la —
    the only type calibrated against a silver standard).
    """
    p_gate = f['p_pass_rate_pct'] >= P_RATE_MIN_PCT
    ax_ok = p_gate and f['p_ax_deg'] == f['p_ax_deg'] and f['p_amp_mv'] >= P_AMP_FLOOR_MV

    def inverted(lead):  # mirror triple: P neg + T neg + QRS not positive
        return (f[f'{lead}_P'] == 'negative' and f[f'{lead}_T'] == 'negative'
                and f[f'{lead}_QRS'] in ('negative', 'biphasic'))

    # ---- RA-LA: I -> -I, II <-> III, aVR <-> aVL, aVF unchanged ----
    s1 = bool(ax_ok and AXIS_FLIP_DEG[0] <= f['p_ax_deg'] <= AXIS_FLIP_DEG[1])
    # RAD-mimic guard: the swap mirrors sinus 0-90 deg into 90-180 deg only;
    # a measurable P axis up-left (215-360 deg) is retrograde/ectopic/noise,
    # so lead-I P negativity must not count as a reversal signature there.
    p_nonsinus = ax_ok and (AXIS_NONSINUS_DEG[0] < f['p_ax_deg']
                            < AXIS_NONSINUS_DEG[1])
    s2 = (not p_nonsinus) and inverted('I')
    avr_all_pos = (f['AVR_P'] == 'positive' and f['AVR_QRS'] == 'positive'
                   and f['AVR_T'] == 'positive')

    # ---- RA-LL: II -> -II, aVR <-> aVF, aVL unchanged ----
    # junctional/retrograde P also gives II P neg + aVR P pos, but keeps
    # II QRS positive and aVR QRS negative — the QRS legs separate them.
    # P legs additionally need >=70% per-beat stability: without a silver
    # standard a 57-64% bare majority must not carry a hard signature
    # (atrial bigeminy / baseline wander alternates P polarity per beat).
    # Full-leg raw signatures whose P legs fail stability downgrade to
    # suspicious (the QRS legs are amplitude-robust; a human reviews).
    def _stable(share):
        return share == share and share >= CONSISTENCY_MIN

    t1_raw = inverted('II')
    t2_raw = avr_all_pos and f['AVF_QRS'] == 'negative'
    t1 = t1_raw and _stable(f['ii_p_neg_share'])
    t2 = (t2_raw and _stable(f['avr_p_pos_share']))

    # ---- LA-LL: III -> -III, aVL <-> aVF, aVR unchanged, I <-> II ----
    # LAFB mimics U2's aVL+/aVF- but never inverts III's P.
    u1_raw = inverted('III')
    u2_raw = (f['AVL_P'] == 'positive' and f['AVL_QRS'] == 'positive'
              and f['AVF_QRS'] == 'negative')
    u1 = u1_raw and _stable(f['iii_p_neg_share'])
    u2 = u2_raw and _stable(f['avl_p_pos_share'])

    def _score(weights, items):
        fired_l, sc = [], 0.0
        for name, cond, val in items:
            if cond:
                sc += weights[name]
                fired_l.append(f'{name}={val}')
        return sc, fired_l

    amp_ok = f['p_amp_mv'] == f['p_amp_mv'] and f['p_amp_mv'] >= P_AMP_GOOD_MV
    ii_share, iii_share = f['ii_p_neg_share'], f['iii_p_neg_share']
    ra_items = ([
        ('I_T_neg', f['I_T'] == 'negative', f['I_T']),
        ('I_P_neg', f['I_P'] == 'negative', f['I_P']),
        ('avr_p_pos', f['AVR_P'] == 'positive', f['AVR_P']),
        ('avr_qt_pos', f['AVR_QRS'] == 'positive' and f['AVR_T'] == 'positive',
         f"{f['AVR_QRS']}/{f['AVR_T']}"),
        ('consistency', f['i_p_neg_share'] == f['i_p_neg_share']
         and f['i_p_neg_share'] >= CONSISTENCY_MIN,
         f"{f['i_p_neg_share']:.0%}"),
        ('p_amp_good', amp_ok, f"{f['p_amp_mv']:.2f}mV"),
    ] if p_gate else [  # QRS/T-only branch: weaker set, verdict stays capped
        ('I_T_neg', f['I_T'] == 'negative', f['I_T']),
        ('avr_qt_pos', f['AVR_QRS'] == 'positive' and f['AVR_T'] == 'positive',
         f"{f['AVR_QRS']}/{f['AVR_T']}"),
    ])
    rall_items = [
        ('II_T_neg', f['II_T'] == 'negative', f['II_T']),
        ('II_P_neg', f['II_P'] == 'negative', f['II_P']),
        ('avr_all_pos', avr_all_pos, 'aVR'),
        ('avf_q_neg', f['AVF_QRS'] == 'negative', f['AVF_QRS']),
        ('ii_consistency', ii_share == ii_share and ii_share >= CONSISTENCY_MIN,
         f"{ii_share:.0%}"),
        ('p_amp_good', amp_ok, f"{f['p_amp_mv']:.2f}mV"),
    ]
    lall_items = [
        ('III_T_neg', f['III_T'] == 'negative', f['III_T']),
        ('III_P_neg', f['III_P'] == 'negative', f['III_P']),
        ('avl_pq_pos', f['AVL_P'] == 'positive' and f['AVL_QRS'] == 'positive',
         f"{f['AVL_P']}/{f['AVL_QRS']}"),
        ('avf_q_neg', f['AVF_QRS'] == 'negative', f['AVF_QRS']),
        ('iii_consistency', iii_share == iii_share
         and iii_share >= CONSISTENCY_MIN, f"{iii_share:.0%}"),
        ('p_amp_good', amp_ok, f"{f['p_amp_mv']:.2f}mV"),
    ]
    sc_ra, fired_ra = _score(W, ra_items)
    sc_rall, fired_rall = _score(W_RA_LL, rall_items)
    sc_lall, fired_lall = _score(W_LA_LL, lall_items)

    def _ladder(hard, double, sc):
        if double:
            return 'definite'
        if hard and sc >= SCORE_DEFINITE:
            return 'definite'
        if hard:
            return 'probable'
        return 'suspicious' if sc >= SCORE_SUSPICIOUS else 'none'

    v_ra = _ladder(s1 or s2 or avr_all_pos, s1 and s2, sc_ra)
    # RA-LL / LA-LL are uncalibrated (no device-flagged cases in the
    # dataset): a verdict requires a HARD signature — score alone never
    # fires.  Lead-III P-/T- with a POSITIVE QRS is the textbook normal
    # left-axis variant (~19 records score 5.0 on it); calling those
    # "suspicious LA-LL" would be noise, not evidence.
    if t1 or t2:
        v_rall = _ladder(True, t1 and t2, sc_rall)
    elif t1_raw and t2_raw:  # full legs, P unstable -> human review
        v_rall = 'suspicious'
    else:
        v_rall = 'none'
    # LA-LL: definite only on the full pair — III inversion alone is a
    # common left-axis/LAFB variant, so partial evidence stays suspicious
    if u1 and u2:
        v_lall = 'definite'
    elif u1 or (u1_raw and u2_raw):  # full legs, P unstable -> review
        v_lall = 'suspicious'
    else:
        v_lall = 'none'

    if not p_gate:
        verdict, rev_type = 'poor_P', ''
        fired_str, best_sc = '; '.join(fired_ra), sc_ra
    else:
        rank = {'none': 0, 'suspicious': 1, 'probable': 2, 'definite': 3}
        cands = [('ra_la', v_ra, sc_ra, fired_ra),
                 ('ra_ll', v_rall, sc_rall, fired_rall),
                 ('la_ll', v_lall, sc_lall, fired_lall)]
        rev_type, verdict, best_sc, fired_best = max(
            cands, key=lambda c: (rank[c[1]], c[2]))
        fired_str = '; '.join(fired_best)
        if rev_type != 'ra_la':
            fired_str = f'[{rev_type}] ' + fired_str
        if verdict == 'none':
            rev_type = ''

    notes = []
    if f['r_v1_mv'] == f['r_v1_mv'] and f['r_v6_mv'] == f['r_v6_mv'] \
            and f['r_v1_mv'] - f['r_v6_mv'] > R_PROG_MARGIN_MV:
        notes.append('poor_R_progression(dextrocardia?)')
    if not p_gate:
        notes.append('P_coverage_low')
    if ax_ok and not s1 and f['I_QRS'] == 'negative' and f['I_T'] == 'negative':
        notes.append('pathologic_RAD(P axis normal)')
    if p_nonsinus:
        notes.append('P_axis_nonsinus(retrograde/ectopic?)')
    hard_types = [t for t, h in (('ra_la', s1 or s2 or avr_all_pos),
                                 ('ra_ll', t1 or t2), ('la_ll', u1 or u2)) if h]
    if len(hard_types) > 1:
        notes.append('multi_type(' + '+'.join(hard_types) + ')')
    if t1 != t2:
        notes.append('RA-LL_partial')
    if u1 != u2:
        notes.append('LA-LL_partial')
    if (t1_raw and t2_raw) and not (t1 and t2):
        notes.append('RA-LL_P_unstable')
    if (u1_raw and u2_raw) and not (u1 and u2):
        notes.append('LA-LL_P_unstable')
    return {'verdict': verdict, 'reversal_type': rev_type,
            'score': round(best_sc, 1),
            'S1_P_axis_flip': s1, 'S2_lead_I_inverted': bool(s2),
            'S3_aVR_all_positive': bool(avr_all_pos),
            'T1_II_inverted': bool(t1), 'T2_aVR_aVF_swap': bool(t2),
            'U1_III_inverted': bool(u1), 'U2_aVL_aVF_swap': bool(u2),
            'fired': fired_str, 'notes': '; '.join(notes)}


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
    rtype = (f.get('reversal_type') or 'ra_la').replace('_', '-').upper()
    fig.suptitle(
        f"{rec['filename']}  |  {rtype} reversal: {f['verdict'].upper()}  "
        f"(score {f['score']}, S1={f['S1_P_axis_flip']} S2={f['S2_lead_I_inverted']} "
        f"S3={f['S3_aVR_all_positive']} T1={f.get('T1_II_inverted')} "
        f"T2={f.get('T2_aVR_aVF_swap')} U1={f.get('U1_III_inverted')} "
        f"U2={f.get('U2_aVL_aVF_swap')})\n"
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
            if lead in FRAME_LEADS.get(f.get('reversal_type') or 'ra_la', FRAME_LEADS['ra_la']):
                # this type's reversal leads, visually framed
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
            'record', 'verdict', 'reversal_type', 'score',
            'S1_P_axis_flip', 'S2_lead_I_inverted', 'S3_aVR_all_positive',
            'T1_II_inverted', 'T2_aVR_aVF_swap', 'U1_III_inverted',
            'U2_aVL_aVF_swap', 'fired', 'notes', 'p_ax_deg', 'p_amp_mv',
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
    wave_cols = [f'{l}_{w}' for l in ('I', 'AVR') for w in ('P', 'QRS', 'T')]
    simple = df[['record', 'verdict', 'reversal_type']].merge(
        base[['record'] + wave_cols], on='record')
    for c in wave_cols:
        simple[c] = simple[c].map(pol_cn).fillna('—')
    type_cn = {'ra_la': 'RA-LA', 'ra_ll': 'RA-LL', 'la_ll': 'LA-LL'}
    pre_cn = {'definite': '确证', 'probable': '可能', 'suspicious': '可疑'}
    simple['verdict'] = [
        pre_cn[v] + type_cn.get(str(t), '') + '反接' if v in pre_cn
        else verd_cn[v]
        for v, t in zip(simple['verdict'], simple['reversal_type'].fillna(''))]
    simple = simple[['record'] + wave_cols + ['verdict']]
    simple.columns = ['记录号', 'P波(I导)', 'QRS波(I导)', 'T波(I导)',
                      'P波(aVR)', 'QRS波(aVR)', 'T波(aVR)', '结论']
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
        f"double(S1&S2) {int(df['double_signature'].sum())}, "
        f"T1 {int(df['T1_II_inverted'].sum())}, "
        f"T2 {int(df['T2_aVR_aVF_swap'].sum())}, "
        f"U1 {int(df['U1_III_inverted'].sum())}, "
        f"U2 {int(df['U2_aVL_aVF_swap'].sum())}")
    add("reversal types among detected: "
        + df.loc[detected, 'reversal_type'].value_counts().to_string()
        .replace('\n', '  '))
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
