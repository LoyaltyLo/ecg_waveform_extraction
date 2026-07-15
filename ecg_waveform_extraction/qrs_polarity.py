"""QRS polarity detection using HSMM segmentation.

RA-LA Reversal key insight:
  Lead I QRS = most reliable indicator (inverted in RA-LA reversal).
  Lead II QRS = often stays positive even in reversal (NOT reliable).
  P-axis > 100° = gold standard for reversal confirmation.

Method:
  1. HSMM-segment both Lead I and Lead II
  2. Extract QRS metrics from each lead independently
  3. Compare Lead I QRS polarity vs Lead II QRS polarity
  4. RA-LA reversal pattern: Lead I inverted + Lead II upright/less positive
  5. Cross-validate with aECG P-axis annotation
"""

import sys
sys.path.insert(0, 'c:/LoyaltyLo/PythonProjects/ECG_engineering')

import os, json, re, time, gc
from collections import Counter
from dataclasses import dataclass, field
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ecg_waveform_extraction.preprocessing import ECGPreprocessor
from ecg_waveform_extraction.features import FeatureExtractor
from ecg_waveform_extraction.hsmm import HSMMModel, smart_initialize_gmms
from ecg_waveform_extraction.segmentation import ECGSegmenter

AECG_DIR = 'C:/LoyaltyLo/datasets/RA-LA_Reversal/aECG'
OUT_DIR = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/output_rala_full/_qrs_polarity'
os.makedirs(OUT_DIR, exist_ok=True)
N_FILES = 50
MAX_SAMPLES = 4000


@dataclass
class QRSResult:
    beat_id: int
    q_onset: int; r_peak: int; s_offset: int
    duration_ms: float
    r_amplitude: float; s_amplitude: float
    qrs_net_area: float
    rs_ratio: float
    polarity: str
    polarity_confidence: float
    lead_name: str


@dataclass
class BeatComparison:
    """Paired QRS from both leads for one beat."""
    beat_id: int
    lead_I_polarity: str; lead_II_polarity: str
    lead_I_rs: float; lead_II_rs: float
    lead_I_net: float; lead_II_net: float
    reversal_detected: bool  # Lead I inverted = RA-LA reversal
    reversal_confidence: float


@dataclass
class QRSRecordSummary:
    record: str
    n_beats: int
    p_axis: float | None
    # Lead II stats
    lead_II_pos: int; lead_II_neg: int; lead_II_bip: int
    # Lead I stats (the critical lead for reversal)
    lead_I_pos: int; lead_I_neg: int; lead_I_bip: int
    # Reversal detection
    reversal_ratio: float       # fraction of beats with Lead I inverted
    reversal_consensus: str      # 'normal' | 'reversed' | 'mixed'
    QRS_axis: float | None
    interpretation: str
    beat_comparisons: list = field(default_factory=list)


# =====================================================================
def parse_aecg(filepath):
    with open(filepath, 'rb') as f: raw = f.read()
    content = raw.decode('utf-8', errors='replace')
    r = {'fs': 1000.0, 'signals': {}}
    m = re.search('<increment[^>]*value="([^"]+)"[^>]*unit="s"'.encode(), raw)
    if m: r['fs'] = 1.0 / float(m.group(1))
    ss = content.find('<sequenceSet'); se = content.find('</sequenceSet>', ss)
    digits = re.findall(r'<digits[^>]*>([^<]+)</digits>', content[ss:se])
    for i, name in enumerate(['I','II','III','AVR','AVL','AVF']):
        if i < len(digits):
            r['signals'][name] = np.array([float(x) for x in digits[i].split()], dtype=np.float64)
    for key, pat in {
        'HR': 'HEART_RATE.*?value="([^"]+)"',
        'QRS_dur': 'TIME_PD_QRS\b(?!c).*?value="([^"]+)"',
        'QRS_axis': 'ANGLE_QRS_FRONT.*?value="([^"]+)"',
        'P_axis':   'ANGLE_P_FRONT.*?value="([^"]+)"',
    }.items():
        m = re.search(pat.encode(), raw, re.DOTALL)
        r[key] = float(m.group(1)) if m else None
    interp = re.search('INTERPRETATION_STATEMENT.*?xsi:type="ST"[^>]*>([^<]+)</value>'.encode(), raw, re.DOTALL)
    r['interpretation'] = interp.group(1).decode('utf-8',errors='replace').strip().replace('\n','; ') if interp else ''
    return r


def run_hsmm(signal, fs):
    """Run full HSMM pipeline on a single lead."""
    prep = ECGPreprocessor(fs=fs)
    clean = prep.preprocess(signal)
    fe = FeatureExtractor(fs=fs)
    features = fe.extract(clean)
    model = HSMMModel(fs=fs)
    model.initialize_with_priors()
    model.set_left_right_topology()
    smart_initialize_gmms(model, features)
    seg = ECGSegmenter(preprocessor=prep, feature_extractor=fe, model=model, fs=fs)
    return seg.segment(signal), clean


def refine_qrs_boundaries(ecg_clean, q_on_hsmm, r_peak_hsmm, s_off_hsmm, fs):
    """Refine HSMM QRS boundaries using signal derivative.

    Problems with raw HSMM boundaries:
      - Q onset may bleed into PR segment
      - S offset may bleed into ST segment
      - R peak is already good (midpoint of R state)

    Refinement strategy:
      1. R peak: find max |amplitude| in ±20ms window around HSMM estimate
      2. Q onset: walk RIGHT from HSMM Q-onset until |d1| exceeds noise floor
      3. S offset: walk LEFT from S-offset until |d1| returns to baseline
    """
    T = len(ecg_clean)
    window = int(0.005 * fs)  # 5ms smoothing

    # Detect noise floor from signal start (first 200ms, assumed quiet)
    quiet_end = min(200, T // 5)
    noise_d1_std = float(np.std(np.diff(ecg_clean[:quiet_end])) + 1e-6)
    threshold = noise_d1_std * 3.0  # 3σ of baseline noise

    # Step 1: Refine R peak
    r_search_start = max(0, r_peak_hsmm - int(0.02 * fs))
    r_search_end = min(T - 1, r_peak_hsmm + int(0.02 * fs))
    r_search = ecg_clean[r_search_start:r_search_end + 1]
    bl_local = float(np.median(r_search))
    r_peak = r_search_start + int(np.argmax(np.abs(r_search - bl_local)))
    r_val = float(ecg_clean[r_peak] - bl_local)

    # Step 2: Refine Q onset (walk right from HSMM onset to find true Q start)
    d1 = np.gradient(ecg_clean)
    q_on = q_on_hsmm
    # Walk right: find where slope first exceeds threshold
    for i in range(q_on_hsmm, min(r_peak, q_on_hsmm + int(0.08 * fs))):
        if abs(d1[i]) > threshold:
            # Found Q wave start — backtrack slightly to the quiescent point
            for j in range(i, max(q_on_hsmm, i - int(0.01 * fs)), -1):
                if abs(d1[j]) <= threshold * 0.5:
                    q_on = j
                    break
            else:
                q_on = i
            break
    else:
        q_on = q_on_hsmm  # no clear Q → keep HSMM onset

    # Step 3: Refine S offset (walk left from HSMM offset to J-point)
    s_off = s_off_hsmm
    for i in range(s_off_hsmm, max(r_peak + int(0.02 * fs), s_off_hsmm - int(0.10 * fs)), -1):
        if abs(d1[i]) > threshold:
            # Walk right from where slope returns to baseline
            for j in range(i, min(s_off_hsmm, i + int(0.02 * fs))):
                if abs(d1[j]) <= threshold * 0.5:
                    s_off = j
                    break
            else:
                s_off = i
            break
    else:
        s_off = s_off_hsmm

    # Sanity check: don't let refinement collapse the QRS
    min_qrs_samples = int(0.02 * fs)  # minimum 20ms
    if s_off - q_on < min_qrs_samples:
        q_on = q_on_hsmm
        s_off = s_off_hsmm

    # Ensure valid
    q_on = max(0, min(q_on, T - 1))
    s_off = max(q_on + min_qrs_samples, min(s_off, T - 1))
    r_peak = max(q_on, min(r_peak, s_off))

    return q_on, r_peak, s_off


def extract_qrs(seg_result, ecg_clean, fs, lead_name):
    """Extract QRS metrics from HSMM segmentation with refined boundaries."""
    results = []; T = len(ecg_clean)
    for b in seg_result.beats:
        if b.q_onset <= 0 or b.s_offset <= 0 or b.r_peak <= 0: continue
        # ---- Refine boundaries using derivative ----
        q_on, r_pk, s_off = refine_qrs_boundaries(
            ecg_clean, b.q_onset, b.r_peak, b.s_offset, fs)
        seg = ecg_clean[q_on:s_off+1]
        bl = float(np.mean(ecg_clean[max(0,q_on-30):q_on])) if q_on >= 30 else float(np.median(seg[:5]))
        detrend = seg - bl
        dur = len(seg) / fs * 1000.0
        r_amp = float(ecg_clean[r_pk] - bl) if 0 <= r_pk < T else 0.0
        # S after R
        r_idx = r_pk - q_on
        s_search = detrend[r_idx:] if r_idx < len(detrend) else detrend
        s_nadir = float(np.min(s_search)) if len(s_search) > 0 else 0.0
        # Q before R
        q_search = detrend[:r_idx+1] if r_idx < len(detrend) else detrend
        q_nadir = float(np.min(q_search)) if len(q_search) > 1 else 0.0

        qrs_net = float(np.sum(detrend))

        # R/S ratio (capped to avoid blowup from near-zero S)
        neg_mag = max(abs(s_nadir), abs(q_nadir), 0.001) if (s_nadir < 0 or q_nadir < 0) else 0.001
        if r_amp > 0:
            rs_ratio = float(min(r_amp / neg_mag, 1000.0))  # cap at 1000
        else:
            rs_ratio = float(max(r_amp / max(abs(r_amp) + neg_mag, 0.001), -1000.0))

        # Polarity classification
        if rs_ratio >= 1.5 and qrs_net > 0: polarity = 'positive'; conf = min(rs_ratio/3.0, 1.0)
        elif rs_ratio <= 0.5 and qrs_net < 0: polarity = 'negative'; conf = min(abs(qrs_net)/(abs(r_amp)+neg_mag+0.001), 1.0)
        elif abs(qrs_net) < abs(r_amp)*0.1: polarity = 'biphasic'; conf = 0.6
        elif qrs_net > 0: polarity = 'positive'; conf = 0.7
        else: polarity = 'negative'; conf = 0.7

        results.append(QRSResult(beat_id=b.beat_id, q_onset=q_on, r_peak=r_pk, s_offset=s_off,
            duration_ms=round(dur,2), r_amplitude=round(r_amp,4), s_amplitude=round(s_nadir,4),
            qrs_net_area=round(qrs_net,4), rs_ratio=round(rs_ratio,4),
            polarity=polarity, polarity_confidence=round(min(conf,1.0),3), lead_name=lead_name))
    return results


def plot_qrs(rec_name, rec_dir, ecg_clean, seg_result, qrs_results, fs, lead_label):
    beats_dir = os.path.join(rec_dir, 'beats'); os.makedirs(beats_dir, exist_ok=True)
    T = len(ecg_clean)

    # Overview
    fig, ax = plt.subplots(figsize=(18, 4))
    ps = min(3.8, T/fs); n_plot = int(ps*fs); t_plot = np.arange(n_plot)/fs
    e_plot = ecg_clean[:n_plot]; lbl = seg_result.state_labels[:n_plot]
    for q in qrs_results:
        if q.q_onset < n_plot and q.s_offset < n_plot:
            c = {'positive':'green','negative':'red','biphasic':'orange'}.get(q.polarity,'gray')
            ax.fill_between(t_plot[q.q_onset:q.s_offset+1], e_plot[q.q_onset:q.s_offset+1],
                            alpha=0.30, color=c, linewidth=0)
            mid = (q.q_onset+q.s_offset)//2
            if mid < n_plot:
                ax.annotate(f'{q.polarity[0].upper()}', (t_plot[mid], e_plot[mid]),
                            textcoords='offset points', xytext=(0,8 if q.qrs_net_area>0 else -14),
                            fontsize=7, ha='center', color=c, fontweight='bold')
    ax.plot(t_plot, e_plot, 'k-', linewidth=0.5)
    ax.set_xlim(t_plot[0],t_plot[-1]); ax.set_xlabel('Time(s)')
    pos=sum(1 for q in qrs_results if q.polarity=='positive')
    neg=sum(1 for q in qrs_results if q.polarity=='negative')
    bip=sum(1 for q in qrs_results if q.polarity=='biphasic')
    ax.set_title(f'{rec_name} — Lead {lead_label} QRS  |  +:{pos} -:{neg} ±:{bip}')
    ax.grid(True,alpha=0.15); fig.tight_layout()
    fig.savefig(os.path.join(rec_dir,f'qrs_overview_{lead_label}.png'),dpi=120,bbox_inches='tight'); plt.close(fig)

    # Per-beat (up to 6)
    for i in range(min(len(qrs_results), 6)):
        q = qrs_results[i]; margin = int(0.08*fs)
        ws=max(0,q.q_onset-margin); we=min(T-1,q.s_offset+margin)
        if we-ws<10: continue
        fig,ax=plt.subplots(figsize=(8,3)); t_win=np.arange(ws,we+1)/fs; e_win=ecg_clean[ws:we+1]
        ax.plot(t_win,e_win,'k-',linewidth=1.2)
        qrs_t=np.arange(q.q_onset,q.s_offset+1)/fs; qrs_v=ecg_clean[q.q_onset:q.s_offset+1]
        c={'positive':'#4caf50','negative':'#f44336','biphasic':'#ff9800'}[q.polarity]
        ax.fill_between(qrs_t,qrs_v,alpha=0.35,color=c,label=f'QRS({q.polarity})')
        if q.r_peak>0: ax.plot(q.r_peak/fs,ecg_clean[q.r_peak],'rv',markersize=8)
        bl=np.mean(ecg_clean[max(0,q.q_onset-30):q.q_onset]) if q.q_onset>=30 else np.median(e_win)
        ax.axhline(bl,color='gray',linestyle=':',linewidth=0.5,alpha=0.5)
        info=(f"Lead {lead_label}\nPolarity: {q.polarity.upper()}\nR/S: {q.rs_ratio:.2f}\n"
              f"QRS net: {q.qrs_net_area:.1f}\nDur: {q.duration_ms:.0f}ms\nConf: {q.polarity_confidence:.2f}")
        ax.text(0.98,0.95,info,transform=ax.transAxes,fontsize=9,va='top',ha='right',
                bbox=dict(boxstyle='round,pad=0.5',facecolor='white',alpha=0.9),fontfamily='monospace')
        ax.set_xlim(t_win[0],t_win[-1]); ax.set_xlabel('Time(s)'); ax.set_title(f'Beat {q.beat_id} Lead {lead_label}')
        ax.legend(fontsize=8,loc='upper left'); ax.grid(True,alpha=0.15); fig.tight_layout()
        fig.savefig(os.path.join(beats_dir,f'beat_{q.beat_id:03d}_qrs_{lead_label}.png'),dpi=120,bbox_inches='tight')
        plt.close(fig)


def align_beats(qrs_I, qrs_II):
    """Pair beats from Lead I and II by beat_id."""
    map_I = {q.beat_id: q for q in qrs_I}
    map_II = {q.beat_id: q for q in qrs_II}
    common = sorted(set(map_I.keys()) & set(map_II.keys()))
    comparisons = []
    for bid in common:
        qi = map_I[bid]; qii = map_II[bid]
        # Reversal pattern: Lead I inverted (negative or biphasic) while Lead II stays positive
        lead_I_neg = qi.polarity in ('negative', 'biphasic')
        lead_II_pos = qii.polarity == 'positive'
        reversal = lead_I_neg  # Lead I inverted = strong RA-LA reversal signal

        if lead_I_neg and (qii.polarity == 'positive' or qii.qrs_net_area > 0):
            conf = 0.90
        elif lead_I_neg:
            conf = 0.75
        elif not lead_I_neg and not lead_II_pos:
            conf = 0.85  # normal
        else:
            conf = 0.70

        comparisons.append(BeatComparison(
            beat_id=bid,
            lead_I_polarity=qi.polarity, lead_II_polarity=qii.polarity,
            lead_I_rs=qi.rs_ratio, lead_II_rs=qii.rs_ratio,
            lead_I_net=qi.qrs_net_area, lead_II_net=qii.qrs_net_area,
            reversal_detected=reversal,
            reversal_confidence=round(conf, 2),
        ))
    return comparisons


def process_record(fname):
    fpath = os.path.join(AECG_DIR, fname); rec_name = fname.replace('.aECG','')
    rec_dir = os.path.join(OUT_DIR, rec_name); os.makedirs(rec_dir, exist_ok=True)

    aecg = parse_aecg(fpath); fs = aecg['fs']
    sig_I = aecg['signals'].get('I'); sig_II = aecg['signals'].get('II')
    if sig_I is None or sig_II is None: return None

    n = min(len(sig_I), len(sig_II), MAX_SAMPLES)

    # ---- Lead I (primary for reversal detection) ----
    seg_I, clean_I = run_hsmm(sig_I[:n].astype(np.float64), fs)
    qrs_I = extract_qrs(seg_I, clean_I, fs, 'I')

    # ---- Lead II (for comparison) ----
    seg_II, clean_II = run_hsmm(sig_II[:n].astype(np.float64), fs)
    qrs_II = extract_qrs(seg_II, clean_II, fs, 'II')

    # ---- Cross-lead comparison ----
    comparisons = align_beats(qrs_I, qrs_II)
    n_beats = len(comparisons)
    n_rev = sum(1 for c in comparisons if c.reversal_detected)

    # ---- Save data ----
    np.save(os.path.join(rec_dir, 'raw_I.npy'), sig_I[:n])
    np.save(os.path.join(rec_dir, 'raw_II.npy'), sig_II[:n])
    np.save(os.path.join(rec_dir, 'filtered_I.npy'), clean_I)
    np.save(os.path.join(rec_dir, 'filtered_II.npy'), clean_II)
    np.save(os.path.join(rec_dir, 'state_labels_I.npy'), seg_I.state_labels)
    np.save(os.path.join(rec_dir, 'state_labels_II.npy'), seg_II.state_labels)

    # QRS per-beat JSON (both leads)
    qrs_json = [{
        'beat_id': c.beat_id,
        'lead_I': {'polarity': c.lead_I_polarity, 'rs_ratio': c.lead_I_rs, 'net': c.lead_I_net},
        'lead_II': {'polarity': c.lead_II_polarity, 'rs_ratio': c.lead_II_rs, 'net': c.lead_II_net},
        'reversal_detected': c.reversal_detected, 'reversal_confidence': c.reversal_confidence,
    } for c in comparisons]
    with open(os.path.join(rec_dir, 'qrs_comparison.json'), 'w') as f:
        json.dump(qrs_json, f, indent=2)

    # QRS samples
    samples_I = {str(q.beat_id): clean_I[q.q_onset:q.s_offset+1] for q in qrs_I}
    samples_II = {str(q.beat_id): clean_II[q.q_onset:q.s_offset+1] for q in qrs_II}
    np.savez(os.path.join(rec_dir, 'qrs_samples_I.npz'), **samples_I)
    np.savez(os.path.join(rec_dir, 'qrs_samples_II.npz'), **samples_II)

    # ---- Plots (both leads) ----
    plot_qrs(rec_name, rec_dir, clean_I, seg_I, qrs_I, fs, 'I')
    plot_qrs(rec_name, rec_dir, clean_II, seg_II, qrs_II, fs, 'II')

    # ---- Summary ----
    p_axis = aecg.get('P_axis')
    lead_I_neg = sum(1 for c in comparisons if c.lead_I_polarity in ('negative',))
    lead_I_pos = sum(1 for c in comparisons if c.lead_I_polarity == 'positive')
    lead_I_bip = sum(1 for c in comparisons if c.lead_I_polarity == 'biphasic')
    lead_II_pos = sum(1 for c in comparisons if c.lead_II_polarity == 'positive')
    lead_II_neg = sum(1 for c in comparisons if c.lead_II_polarity == 'negative')
    lead_II_bip = sum(1 for c in comparisons if c.lead_II_polarity == 'biphasic')

    # Consensus: if >50% of beats have Lead I inverted, it's reversed
    rev_frac = n_rev / max(n_beats, 1)
    if rev_frac >= 0.6:
        consensus = 'reversed'
    elif rev_frac <= 0.3:
        consensus = 'normal'
    else:
        consensus = 'mixed'

    summary = QRSRecordSummary(
        record=rec_name, n_beats=n_beats,
        p_axis=p_axis,
        lead_II_pos=lead_II_pos, lead_II_neg=lead_II_neg, lead_II_bip=lead_II_bip,
        lead_I_pos=lead_I_pos, lead_I_neg=lead_I_neg, lead_I_bip=lead_I_bip,
        reversal_ratio=round(rev_frac, 3),
        reversal_consensus=consensus,
        QRS_axis=aecg.get('QRS_axis'),
        interpretation=aecg.get('interpretation', ''),
        beat_comparisons=[{
            'beat_id': c.beat_id, 'lead_I_pol': c.lead_I_polarity,
            'lead_I_rs': c.lead_I_rs, 'lead_II_pol': c.lead_II_polarity,
            'reversal': c.reversal_detected, 'conf': c.reversal_confidence,
        } for c in comparisons],
    )

    sum_dict = {k: v for k, v in summary.__dict__.items()}
    with open(os.path.join(rec_dir, 'qrs_comparison_summary.json'), 'w') as f:
        json.dump(sum_dict, f, indent=2, default=str)

    return summary


def generate_dashboard(summaries):
    n = len(summaries); total_beats = sum(s.n_beats for s in summaries)
    lead_I_neg = sum(s.lead_I_neg for s in summaries)
    lead_I_pos = sum(s.lead_I_pos for s in summaries)
    lead_II_pos = sum(s.lead_II_pos for s in summaries)
    lead_II_neg = sum(s.lead_II_neg for s in summaries)
    rev_recs = sum(1 for s in summaries if s.reversal_consensus == 'reversed')
    norm_recs = sum(1 for s in summaries if s.reversal_consensus == 'normal')
    mixed_recs = sum(1 for s in summaries if s.reversal_consensus == 'mixed')

    # Cross-check with P-axis gold standard
    tp = sum(1 for s in summaries if s.reversal_consensus == 'reversed'
             and s.p_axis is not None and (s.p_axis > 100 or s.p_axis < -30))
    fn = sum(1 for s in summaries if s.reversal_consensus != 'reversed'
             and s.p_axis is not None and (s.p_axis > 100 or s.p_axis < -30))
    fp = sum(1 for s in summaries if s.reversal_consensus == 'reversed'
             and s.p_axis is not None and not (s.p_axis > 100 or s.p_axis < -30))
    tn = sum(1 for s in summaries if s.reversal_consensus != 'reversed'
             and s.p_axis is not None and not (s.p_axis > 100 or s.p_axis < -30))

    n_with_axis = tp + fn + fp + tn
    acc = round((tp + tn) / max(n_with_axis, 1) * 100, 1) if n_with_axis > 0 else 0

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('QRS Reversal Detection — Lead I vs Lead II Comparison', fontsize=14, fontweight='bold')

    # (0,0) Key metrics
    ax = axes[0, 0]; ax.axis('off')
    ax.text(0.1, 0.95,
        f"Records: {n}\nBeats: {total_beats}\n\n"
        f"Lead I (reversal lead):\n  Positive: {lead_I_pos} ({lead_I_pos/max(total_beats,1)*100:.0f}%)\n"
        f"  Negative: {lead_I_neg} ({lead_I_neg/max(total_beats,1)*100:.0f}%)\n\n"
        f"Lead II (comparison):\n  Positive: {lead_II_pos} ({lead_II_pos/max(total_beats,1)*100:.0f}%)\n"
        f"  Negative: {lead_II_neg} ({lead_II_neg/max(total_beats,1)*100:.0f}%)\n\n"
        f"vs P-axis gold standard:\n  Accuracy: {acc}% (n={n_with_axis})",
        transform=ax.transAxes, fontsize=10, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))

    # (0,1) Reversal consensus pie
    ax = axes[0, 1]
    ax.pie([norm_recs, rev_recs, mixed_recs],
           labels=[f'Normal\n({norm_recs})', f'Reversed\n({rev_recs})', f'Mixed\n({mixed_recs})'],
           colors=['#4caf50','#f44336','#ff9800'], autopct='%1.0f%%', startangle=90)
    ax.set_title('Record Consensus (Lead I QRS)', fontsize=11, fontweight='bold')

    # (0,2) Lead I vs Lead II polarity comparison
    ax = axes[0, 2]
    categories = ['Lead I + / Lead II +', 'Lead I - / Lead II +', 'Lead I - / Lead II -', 'Mixed']
    li_pos_ii_pos = sum(1 for s in summaries if s.lead_I_pos > s.lead_I_neg and s.lead_II_pos > s.lead_II_neg)
    li_neg_ii_pos = sum(1 for s in summaries if s.lead_I_neg > s.lead_I_pos and s.lead_II_pos > s.lead_II_neg)
    li_neg_ii_neg = sum(1 for s in summaries if s.lead_I_neg > s.lead_I_pos and s.lead_II_neg > s.lead_II_pos)
    other = n - li_pos_ii_pos - li_neg_ii_pos - li_neg_ii_neg
    ax.barh([0], [li_pos_ii_pos], color='#4caf50', label='Normal pattern')
    ax.barh([1], [li_neg_ii_pos], color='#f44336', label='REVERSAL pattern')
    ax.barh([2], [li_neg_ii_neg], color='#9c27b0', label='Both inverted')
    ax.barh([3], [other], color='#9e9e9e', label='Mixed')
    ax.set_yticks([0,1,2,3])
    ax.set_yticklabels([f'Li+ Lii+ ({li_pos_ii_pos})', f'Li- Lii+ ({li_neg_ii_pos})',
                        f'Li- Lii- ({li_neg_ii_neg})', f'Mixed ({other})'])
    ax.set_title('Lead I vs Lead II Pattern', fontsize=11, fontweight='bold'); ax.legend(fontsize=8)

    # (1,0) Lead I R/S histogram
    ax = axes[1, 0]
    li_rs = [s.lead_I_neg / max(s.n_beats, 1) for s in summaries if s.n_beats > 0]
    ax.hist(li_rs, bins=20, color='#2196f3', edgecolor='white', alpha=0.8)
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1.5, label='50% threshold')
    ax.set_xlabel('Fraction Lead I QRS negative'); ax.set_ylabel('Records')
    ax.set_title('Lead I Inversion Rate per Record'); ax.legend(); ax.grid(True, alpha=0.2, axis='y')

    # (1,1) vs P-axis scatter
    ax = axes[1, 1]
    for s in summaries:
        if s.p_axis is not None:
            c = 'red' if s.reversal_consensus == 'reversed' else ('green' if s.reversal_consensus == 'normal' else 'orange')
            ax.scatter(s.p_axis, s.reversal_ratio * 100, c=c, s=40, alpha=0.7, edgecolors='none')
    ax.axhline(50, color='red', linestyle='--', linewidth=0.8, alpha=0.5, label='Reversal threshold')
    ax.axvline(100, color='gray', linestyle=':', linewidth=0.8, alpha=0.5, label='P-axis >100°')
    ax.set_xlabel('P-axis (°)'); ax.set_ylabel('Lead I QRS negative %')
    ax.set_title('Lead I QRS vs P-axis Gold Standard'); ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

    # (1,2) Top reversed records
    ax = axes[1, 2]; ax.axis('off')
    top_rev = sorted(summaries, key=lambda s: -s.reversal_ratio)[:10]
    lines = ['Most REVERSED (Lead I QRS):'] + [
        f'  {s.record[:12]} {s.reversal_ratio*100:.0f}% inv P-axis={s.p_axis}°'
        for s in top_rev]
    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes, fontsize=8,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '_qrs_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    files = sorted([f for f in os.listdir(AECG_DIR) if f.endswith('.aECG')])[:N_FILES]
    print(f"{'='*65}")
    print(f"  QRS REVERSAL DETECTION — Lead I vs Lead II")
    print(f"  {N_FILES} records from RA-LA Reversal aECG")
    print(f"{'='*65}\n")

    summaries = []; t_start = time.time()
    for idx, fname in enumerate(files):
        print(f"[{idx+1:2d}/{N_FILES}] {fname[:14]}...", end=" ", flush=True)
        t0 = time.time(); s = process_record(fname); dt = time.time()-t0
        if s:
            summaries.append(s)
            print(f"OK beats={s.n_beats} LI+={s.lead_I_pos} LI-={s.lead_I_neg} "
                  f"rev={s.reversal_ratio*100:.0f}% cons={s.reversal_consensus} P-axis={s.p_axis}° ({dt:.0f}s)")
        else: print(f"SKIP")
        gc.collect()

    total_time = time.time()-t_start; generate_dashboard(summaries)

    # Save global
    class NpEnc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, (np.bool_, bool)): return bool(o)
            return super().default(o)

    # Cross-check with P-axis
    tp = sum(1 for s in summaries if s.reversal_consensus=='reversed' and s.p_axis is not None and (s.p_axis>100 or s.p_axis<-30))
    fn_rev = sum(1 for s in summaries if s.reversal_consensus!='reversed' and s.p_axis is not None and (s.p_axis>100 or s.p_axis<-30))
    n_with_paxis = sum(1 for s in summaries if s.p_axis is not None)

    global_summary = {
        'test_config': {'dataset':'RA-LA Reversal aECG','n_records':N_FILES,'lead_primary':'I','lead_compare':'II'},
        'aggregate': {
            'total_records': len(summaries), 'total_beats': sum(s.n_beats for s in summaries),
            'records_reversed': sum(1 for s in summaries if s.reversal_consensus=='reversed'),
            'records_normal': sum(1 for s in summaries if s.reversal_consensus=='normal'),
            'records_mixed': sum(1 for s in summaries if s.reversal_consensus=='mixed'),
            'lead_I_negative_beats': sum(s.lead_I_neg for s in summaries),
            'lead_I_positive_beats': sum(s.lead_I_pos for s in summaries),
            'lead_II_negative_beats': sum(s.lead_II_neg for s in summaries),
            'lead_II_positive_beats': sum(s.lead_II_pos for s in summaries),
            'p_axis_validated': {'tp':tp,'fn':fn_rev,'n_with_paxis':n_with_paxis,
                'sensitivity_vs_paxis': round(tp/max(tp+fn_rev,1)*100,1) if (tp+fn_rev)>0 else None},
            'total_time_sec': round(total_time,1),
        },
        'per_record': [{k:v for k,v in s.__dict__.items()} for s in summaries],
    }
    with open(os.path.join(OUT_DIR,'qrs_reversal_global.json'),'w',encoding='utf-8') as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False, cls=NpEnc)

    print(f"\n{'='*65}")
    print(f"  QRS REVERSAL DETECTION — COMPLETE")
    print(f"{'='*65}")
    print(f"  Records: {len(summaries)}")
    print(f"  Reversal consensus: {sum(1 for s in summaries if s.reversal_consensus=='reversed')} reversed, "
          f"{sum(1 for s in summaries if s.reversal_consensus=='normal')} normal, "
          f"{sum(1 for s in summaries if s.reversal_consensus=='mixed')} mixed")
    print(f"  vs P-axis gold standard: Se={global_summary['aggregate']['p_axis_validated']['sensitivity_vs_paxis']}%")
    print(f"  Lead I negative beats: {sum(s.lead_I_neg for s in summaries)}/{sum(s.n_beats for s in summaries)}")
    print(f"  Lead II negative beats: {sum(s.lead_II_neg for s in summaries)}/{sum(s.n_beats for s in summaries)}")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"  Output: {OUT_DIR}/")
    print(f"{'='*65}")
    print(f"\n  Per-record files:")
    print(f"    raw_I.npy, raw_II.npy, filtered_I.npy, filtered_II.npy")
    print(f"    state_labels_I.npy, state_labels_II.npy")
    print(f"    qrs_comparison.json — paired Lead I/II per beat")
    print(f"    qrs_samples_I.npz, qrs_samples_II.npz")
    print(f"    qrs_overview_I.png, qrs_overview_II.png")
    print(f"    beats/beat_###_qrs_I.png, beats/beat_###_qrs_II.png")


if __name__ == '__main__':
    main()
