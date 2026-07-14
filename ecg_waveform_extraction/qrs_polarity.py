"""QRS polarity detection using HSMM segmentation.

For each QRS complex:
  1. Extract the QRS segment from HSMM boundaries (Q_onset → S_offset)
  2. Compute direction metrics:
     - R/S ratio (R peak amplitude / S nadir amplitude)
     - QRS net area (integral over QRS window)
     - Dominant direction (positive net = upright, negative = inverted)
  3. Classify: positive (normal) or negative (inverted)

Uses Lead II by default (most common for rhythm analysis).
Optionally checks Lead I (most specific for RA-LA reversal).
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

# ---- Config ----
AECG_DIR = 'C:/LoyaltyLo/datasets/RA-LA_Reversal/aECG'
OUT_DIR = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/output_rala_full/_qrs_polarity'
os.makedirs(OUT_DIR, exist_ok=True)
N_FILES = 50
MAX_SAMPLES = 4000


# =====================================================================
@dataclass
class QRSResult:
    """QRS extraction and polarity for a single beat."""
    beat_id: int
    q_onset: int
    r_peak: int
    s_offset: int
    samples: np.ndarray
    duration_ms: float
    r_amplitude: float          # R peak value (signed, >0 = upright)
    s_amplitude: float          # S nadir value (signed, <0 = negative)
    rs_ratio: float             # R/S ratio (>1 = net positive, <1 = net negative)
    qrs_net_area: float         # Integrated area over QRS window
    polarity: str               # 'positive' | 'negative' | 'biphasic'
    polarity_confidence: float  # 0-1
    lead_name: str              # which lead was used


@dataclass
class QRSRecordSummary:
    record: str
    n_beats: int
    n_positive: int
    n_negative: int
    n_biphasic: int
    dominant_polarity: str
    polarity_agreement: float     # fraction of beats agreeing with dominant
    mean_rs_ratio: float
    mean_qrs_net: float
    mean_duration_ms: float
    interpretation: str
    beats: list = field(default_factory=list)


# =====================================================================
def parse_aecg(filepath):
    """Extract Lead I + Lead II signals + measurements from aECG XML."""
    with open(filepath, 'rb') as f:
        raw = f.read()
    content = raw.decode('utf-8', errors='replace')
    r = {'fs': 1000.0, 'signals': {}}
    m = re.search(rb'<increment[^>]*value="([^"]+)"[^>]*unit="s"', raw)
    if m: r['fs'] = 1.0 / float(m.group(1))
    ss = content.find('<sequenceSet')
    se = content.find('</sequenceSet>', ss)
    digits = re.findall(r'<digits[^>]*>([^<]+)</digits>', content[ss:se])
    lead_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF']
    for i, name in enumerate(lead_names):
        if i < len(digits):
            r['signals'][name] = np.array([float(x) for x in digits[i].split()], dtype=np.float64)
    for key, pat in {
        'HR': r'HEART_RATE.*?value="([^"]+)"[^>]*unit="bpm"',
        'QRS_dur': r'TIME_PD_QRS\b(?!c).*?value="([^"]+)"[^>]*unit="ms"',
        'QRS_axis': r'ANGLE_QRS_FRONT.*?value="([^"]+)"',
    }.items():
        m = re.search(pat.encode(), raw)
        r[key] = float(m.group(1)) if m else None
    interp = re.search(rb'INTERPRETATION_STATEMENT.*?xsi:type="ST"[^>]*>([^<]+)</value>', raw, re.DOTALL)
    r['interpretation'] = interp.group(1).decode('utf-8', errors='replace').strip().replace('\n','; ') if interp else ''
    return r


# =====================================================================
def extract_qrs_polarity(seg_result, ecg_clean, fs, lead_name='II') -> list[QRSResult]:
    """Extract QRS polarity from HSMM-segmented beats.

    Parameters
    ----------
    seg_result : SegmentResult
    ecg_clean : np.ndarray
    fs : float
    lead_name : str
        Which lead's signal is in ecg_clean.

    Returns
    -------
    list[QRSResult]
    """
    results = []
    T = len(ecg_clean)

    for b in seg_result.beats:
        if b.q_onset <= 0 or b.s_offset <= 0 or b.r_peak <= 0:
            continue
        q_on = b.q_onset
        s_off = b.s_offset
        r_pk = b.r_peak

        # Clamp
        q_on = max(0, min(q_on, T - 1))
        s_off = max(q_on + 2, min(s_off, T - 1))
        r_pk = max(q_on, min(r_pk, s_off))

        # ---- QRS waveform segment ----
        qrs_segment = ecg_clean[q_on:s_off + 1]
        baseline = np.mean(ecg_clean[max(0, q_on - 30):q_on]) if q_on >= 30 else float(np.median(qrs_segment[:5]))
        qrs_detrended = qrs_segment - baseline
        duration_ms = len(qrs_segment) / fs * 1000.0

        # ---- R amplitude (positive peak) ----
        r_idx = r_pk - q_on  # within QRS window
        r_amp = float(ecg_clean[r_pk] - baseline)

        # ---- S amplitude (negative nadir after R) ----
        s_search = qrs_detrended[r_idx:]
        s_nadir = float(np.min(s_search)) if len(s_search) > 0 else 0.0

        # ---- Q amplitude (negative deflection before R, if present) ----
        q_search = qrs_detrended[:r_idx + 1]
        q_nadir = float(np.min(q_search)) if len(q_search) > 1 else 0.0

        # ---- Net area (integral) ----
        qrs_net = float(np.sum(qrs_detrended))

        # ---- R/S ratio ----
        # R amplitude / max(|S|, |Q|, epsilon)
        s_mag = abs(s_nadir) if s_nadir < 0 else 0.0
        q_mag = abs(q_nadir) if q_nadir < 0 else 0.0
        neg_mag = max(s_mag, q_mag, 0.001)
        rs_ratio = float(r_amp / max(neg_mag, 0.001)) if r_amp > 0 else float(r_amp / max(r_amp + neg_mag, 0.001))

        # ---- Polarity classification ----
        if rs_ratio > 1.5 and qrs_net > 0:
            polarity = 'positive'
            conf = min(rs_ratio / 3.0, 1.0)
        elif rs_ratio < 0.5 and qrs_net < 0:
            polarity = 'negative'
            conf = min(abs(qrs_net) / (abs(r_amp) + abs(neg_mag) + 0.001), 1.0)
        elif rs_ratio > 0.5 and qrs_net < 0:
            polarity = 'biphasic'
            conf = 0.5
        elif rs_ratio < 0.5 and qrs_net > 0:
            polarity = 'biphasic'
            conf = 0.5
        elif abs(qrs_net) < abs(r_amp) * 0.1:
            polarity = 'biphasic'
            conf = 0.6
        elif qrs_net > 0:
            polarity = 'positive'
            conf = 0.7
        else:
            polarity = 'negative'
            conf = 0.7

        results.append(QRSResult(
            beat_id=b.beat_id,
            q_onset=q_on, r_peak=r_pk, s_offset=s_off,
            samples=qrs_segment.copy(),
            duration_ms=round(duration_ms, 2),
            r_amplitude=round(r_amp, 4),
            s_amplitude=round(s_nadir, 4),
            rs_ratio=round(rs_ratio, 4),
            qrs_net_area=round(qrs_net, 4),
            polarity=polarity,
            polarity_confidence=round(min(conf, 1.0), 3),
            lead_name=lead_name,
        ))

    return results


# =====================================================================
def plot_qrs_results(rec_name, rec_dir, ecg_clean, seg_result, qrs_results, fs):
    """Generate per-record and per-beat QRS plots."""
    T = len(ecg_clean)
    beats_dir = os.path.join(rec_dir, 'beats')
    os.makedirs(beats_dir, exist_ok=True)

    # ---- Segmentation overview ----
    fig, ax = plt.subplots(figsize=(18, 4))
    plot_sec = min(3.8, T / fs)
    n_plot = int(plot_sec * fs)
    t_plot = np.arange(n_plot) / fs
    e_plot = ecg_clean[:n_plot]
    lbl_plot = seg_result.state_labels[:n_plot]

    # QRS regions highlighted
    for q in qrs_results:
        if q.q_onset < n_plot and q.s_offset < n_plot:
            c = {'positive': 'green', 'negative': 'red', 'biphasic': 'orange'}.get(q.polarity, 'gray')
            ax.fill_between(t_plot[q.q_onset:q.s_offset + 1],
                            e_plot[q.q_onset:q.s_offset + 1],
                            alpha=0.30, color=c, linewidth=0)
            mid_q = (q.q_onset + q.s_offset) // 2
            if mid_q < n_plot:
                ax.annotate(f'{q.polarity[0].upper()}', (t_plot[mid_q], e_plot[mid_q]),
                            textcoords='offset points', xytext=(0, 8 if q.qrs_net_area > 0 else -14),
                            fontsize=7, ha='center', color=c, fontweight='bold')

    ax.plot(t_plot, e_plot, 'k-', linewidth=0.5)
    ax.set_xlim(t_plot[0], t_plot[-1])
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude')
    pos_n = sum(1 for q in qrs_results if q.polarity == 'positive')
    neg_n = sum(1 for q in qrs_results if q.polarity == 'negative')
    bip_n = sum(1 for q in qrs_results if q.polarity == 'biphasic')
    ax.set_title(f'QRS Polarity — {rec_name}  |  +:{pos_n}  -:{neg_n}  ±:{bip_n}')
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(rec_dir, 'qrs_overview.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ---- Per-beat QRS plots (up to 8) ----
    n_plot_beats = min(len(qrs_results), 8)
    for i in range(n_plot_beats):
        q = qrs_results[i]
        margin = int(0.08 * fs)
        ws = max(0, q.q_onset - margin)
        we = min(T - 1, q.s_offset + margin)
        if we - ws < 10: continue

        fig, ax = plt.subplots(figsize=(8, 3))
        t_win = np.arange(ws, we + 1) / fs
        e_win = ecg_clean[ws:we + 1]

        ax.plot(t_win, e_win, 'k-', linewidth=1.2)

        # Color QRS region
        qrs_t = np.arange(q.q_onset, q.s_offset + 1) / fs
        qrs_v = ecg_clean[q.q_onset:q.s_offset + 1]
        c_qrs = {'positive': '#4caf50', 'negative': '#f44336', 'biphasic': '#ff9800'}[q.polarity]
        ax.fill_between(qrs_t, qrs_v, alpha=0.35, color=c_qrs, label=f'QRS ({q.polarity})')

        # Mark R & S
        if q.r_peak > 0:
            ax.plot(q.r_peak / fs, ecg_clean[q.r_peak], 'rv', markersize=8, label=f'R={q.r_amplitude:.3f}')

        # Baseline
        bl = np.mean(ecg_clean[max(0, q.q_onset - 30):q.q_onset]) if q.q_onset >= 30 else np.median(e_win)
        ax.axhline(bl, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

        # Text box
        info = (
            f"Polarity: {q.polarity.upper()}\n"
            f"R/S ratio: {q.rs_ratio:.2f}\n"
            f"QRS net: {q.qrs_net_area:.1f}\n"
            f"Dur: {q.duration_ms:.0f}ms\n"
            f"Conf: {q.polarity_confidence:.2f}\n"
            f"Lead: {q.lead_name}"
        )
        ax.text(0.98, 0.95, info, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
                fontfamily='monospace')

        ax.set_xlim(t_win[0], t_win[-1])
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude')
        ax.set_title(f'Beat {q.beat_id} — QRS Polarity: {q.polarity}')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.15)
        fig.tight_layout()
        fig.savefig(os.path.join(beats_dir, f'beat_{q.beat_id:03d}_qrs.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)


# =====================================================================
def process_record(fname):
    """Full pipeline: parse → HSMM → QRS polarity → plots."""
    fpath = os.path.join(AECG_DIR, fname)
    rec_name = fname.replace('.aECG', '')
    rec_dir = os.path.join(OUT_DIR, rec_name)
    os.makedirs(rec_dir, exist_ok=True)

    aecg = parse_aecg(fpath)
    fs = aecg['fs']
    sig_II = aecg['signals'].get('II')
    if sig_II is None: return None

    n = min(len(sig_II), MAX_SAMPLES)
    sig = sig_II[:n].astype(np.float64)

    prep = ECGPreprocessor(fs=fs)
    clean = prep.preprocess(sig)
    fe = FeatureExtractor(fs=fs)
    features = fe.extract(clean)

    model = HSMMModel(fs=fs)
    model.initialize_with_priors()
    model.set_left_right_topology()
    smart_initialize_gmms(model, features)

    seg = ECGSegmenter(preprocessor=prep, feature_extractor=fe, model=model, fs=fs)
    seg_result = seg.segment(sig)

    # ---- QRS extraction + polarity on Lead II ----
    qrs_results = extract_qrs_polarity(seg_result, clean, fs, lead_name='II')

    # ---- Also do Lead I (if available) ----
    sig_I = aecg['signals'].get('I')
    qrs_I_results = []
    if sig_I is not None:
        sig_I = sig_I[:n].astype(np.float64)
        clean_I = prep.preprocess(sig_I)
        fe_I = FeatureExtractor(fs=fs)
        features_I = fe_I.extract(clean_I)
        model_I = HSMMModel(fs=fs)
        model_I.initialize_with_priors()
        model_I.set_left_right_topology()
        smart_initialize_gmms(model_I, features_I)
        seg_I = ECGSegmenter(preprocessor=prep, feature_extractor=fe_I, model=model_I, fs=fs)
        seg_I_result = seg_I.segment(sig_I)
        qrs_I_results = extract_qrs_polarity(seg_I_result, clean_I, fs, lead_name='I')

    # ---- Save data ----
    np.save(os.path.join(rec_dir, 'raw_ecg.npy'), sig)
    np.save(os.path.join(rec_dir, 'filtered_ecg.npy'), clean)
    np.save(os.path.join(rec_dir, 'state_labels.npy'), seg_result.state_labels)

    # QRS JSON per beat
    qrs_json = [{
        'beat_id': q.beat_id,
        'q_onset': q.q_onset, 'r_peak': q.r_peak, 's_offset': q.s_offset,
        'duration_ms': q.duration_ms,
        'r_amplitude': q.r_amplitude,
        's_amplitude': q.s_amplitude,
        'rs_ratio': q.rs_ratio,
        'qrs_net_area': q.qrs_net_area,
        'polarity': q.polarity,
        'polarity_confidence': q.polarity_confidence,
        'lead': q.lead_name,
    } for q in qrs_results]
    with open(os.path.join(rec_dir, 'qrs_polarity.json'), 'w') as f:
        json.dump(qrs_json, f, indent=2)

    # QRS samples
    samples_dict = {str(q.beat_id): q.samples for q in qrs_results}
    np.savez(os.path.join(rec_dir, 'qrs_samples.npz'), **samples_dict)

    # ---- Plots ----
    plot_qrs_results(rec_name, rec_dir, clean, seg_result, qrs_results, fs)

    # ---- Summary ----
    pos = sum(1 for q in qrs_results if q.polarity == 'positive')
    neg = sum(1 for q in qrs_results if q.polarity == 'negative')
    bip = sum(1 for q in qrs_results if q.polarity == 'biphasic')
    total = len(qrs_results)
    dominant = 'positive' if pos >= neg and pos >= bip else ('negative' if neg >= pos else 'biphasic')
    agreement = max(pos, neg, bip) / max(total, 1)

    summary = QRSRecordSummary(
        record=rec_name,
        n_beats=total,
        n_positive=pos, n_negative=neg, n_biphasic=bip,
        dominant_polarity=dominant,
        polarity_agreement=round(agreement, 3),
        mean_rs_ratio=round(float(np.mean([q.rs_ratio for q in qrs_results])), 4) if qrs_results else 0,
        mean_qrs_net=round(float(np.mean([q.qrs_net_area for q in qrs_results])), 2) if qrs_results else 0,
        mean_duration_ms=round(float(np.mean([q.duration_ms for q in qrs_results])), 1) if qrs_results else 0,
        interpretation=aecg.get('interpretation', ''),
        beats=[{
            'beat_id': q.beat_id, 'polarity': q.polarity,
            'rs_ratio': q.rs_ratio, 'qrs_net': q.qrs_net_area,
            'conf': q.polarity_confidence,
        } for q in qrs_results],
    )

    # Save summary JSON
    sum_dict = {k: v for k, v in summary.__dict__.items()}
    with open(os.path.join(rec_dir, 'qrs_summary.json'), 'w') as f:
        json.dump(sum_dict, f, indent=2, default=str)

    return summary


# =====================================================================
def generate_dashboard(summaries):
    """Aggregate dashboard for all records."""
    n = len(summaries)
    if n == 0: return

    pos_total = sum(s.n_positive for s in summaries)
    neg_total = sum(s.n_negative for s in summaries)
    bip_total = sum(s.n_biphasic for s in summaries)
    beat_total = pos_total + neg_total + bip_total
    pos_recs = sum(1 for s in summaries if s.dominant_polarity == 'positive')
    neg_recs = sum(1 for s in summaries if s.dominant_polarity == 'negative')

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('QRS Polarity Detection — RA-LA aECG (Lead II)', fontsize=14, fontweight='bold')

    # (0,0) Key metrics
    ax = axes[0, 0]
    ax.axis('off')
    ax.text(0.1, 0.95, f"Records: {n}\nBeats analyzed: {beat_total}\n\n"
            f"Lead II Dominant Polarity:\n  Positive: {pos_recs} records ({pos_recs/n*100:.0f}%)\n"
            f"  Negative: {neg_recs} records ({neg_recs/n*100:.0f}%)\n\n"
            f"Mean R/S ratio: {np.mean([s.mean_rs_ratio for s in summaries]):.2f}\n"
            f"Mean QRS net: {np.mean([s.mean_qrs_net for s in summaries]):.1f}",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))

    # (0,1) Per-beat polarity pie
    ax = axes[0, 1]
    labels = [f'Positive\n({pos_total})', f'Negative\n({neg_total})', f'Biphasic\n({bip_total})']
    ax.pie([pos_total, neg_total, bip_total], labels=labels, colors=['#4caf50','#f44336','#ff9800'],
           autopct='%1.0f%%', startangle=90)
    ax.set_title('Per-Beat QRS Polarity', fontsize=11, fontweight='bold')

    # (0,2) Per-record dominant polarity pie
    ax = axes[0, 2]
    other = n - pos_recs - neg_recs
    ax.pie([pos_recs, neg_recs, other], labels=[f'Pos dom\n({pos_recs})', f'Neg dom\n({neg_recs})', f'Mixed\n({other})'],
           colors=['#4caf50','#f44336','#9e9e9e'], autopct='%1.0f%%', startangle=90)
    ax.set_title('Per-Record Dominant Polarity', fontsize=11, fontweight='bold')

    # (1,0) R/S ratio histogram
    ax = axes[1, 0]
    rs_vals = [s.mean_rs_ratio for s in summaries]
    ax.hist(rs_vals, bins=25, color='#2196f3', edgecolor='white', alpha=0.8)
    ax.axvline(1.0, color='red', linestyle='--', linewidth=1.5, label='R/S=1 (threshold)')
    ax.set_xlabel('Mean R/S Ratio'); ax.set_ylabel('Records')
    ax.set_title('R/S Ratio Distribution'); ax.legend(); ax.grid(True, alpha=0.2, axis='y')

    # (1,1) QRS net area histogram
    ax = axes[1, 1]
    net_vals = [s.mean_qrs_net for s in summaries]
    ax.hist(net_vals, bins=25, color='#4caf50', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Net=0')
    ax.set_xlabel('Mean QRS Net Area'); ax.set_ylabel('Records')
    ax.set_title('QRS Net Area Distribution'); ax.legend(); ax.grid(True, alpha=0.2, axis='y')

    # (1,2) Example beats
    ax = axes[1, 2]
    ax.axis('off')
    top_neg = sorted(summaries, key=lambda s: s.mean_rs_ratio)[:5]
    top_pos = sorted(summaries, key=lambda s: -s.mean_rs_ratio)[:5]
    lines = ['Most NEGATIVE (inverted):'] + [f'  {s.record[:12]} R/S={s.mean_rs_ratio:.2f}' for s in top_neg]
    lines += ['\nMost POSITIVE (upright):'] + [f'  {s.record[:12]} R/S={s.mean_rs_ratio:.2f}' for s in top_pos]
    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '_qrs_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# =====================================================================
def main():
    files = sorted([f for f in os.listdir(AECG_DIR) if f.endswith('.aECG')])[:N_FILES]
    print(f"{'='*60}")
    print(f"  QRS POLARITY DETECTION — Lead II")
    print(f"  {N_FILES} records from RA-LA Reversal aECG")
    print(f"{'='*60}")
    print()

    summaries = []
    t_start = time.time()

    for idx, fname in enumerate(files):
        print(f"[{idx+1:2d}/{N_FILES}] {fname[:14]}...", end=" ", flush=True)
        t0 = time.time()
        s = process_record(fname)
        dt = time.time() - t0

        if s:
            summaries.append(s)
            print(f"OK beats={s.n_beats} +={s.n_positive} -={s.n_negative} ±={s.n_biphasic} "
                  f"dom={s.dominant_polarity} ({dt:.0f}s)")
        else:
            print(f"SKIP")
        gc.collect()

    total_time = time.time() - t_start

    # Dashboard
    if summaries:
        generate_dashboard(summaries)

    # Global summary JSON
    global_summary = {
        'test_config': {'dataset': 'RA-LA Reversal aECG', 'n_records': N_FILES,
                        'lead': 'II (primary)', 'max_samples': MAX_SAMPLES},
        'aggregate': {
            'total_records': len(summaries),
            'total_beats': sum(s.n_beats for s in summaries),
            'total_positive': sum(s.n_positive for s in summaries),
            'total_negative': sum(s.n_negative for s in summaries),
            'total_biphasic': sum(s.n_biphasic for s in summaries),
            'records_positive_dominant': sum(1 for s in summaries if s.dominant_polarity == 'positive'),
            'records_negative_dominant': sum(1 for s in summaries if s.dominant_polarity == 'negative'),
            'mean_polarity_agreement': round(float(np.mean([s.polarity_agreement for s in summaries])), 3),
            'mean_rs_ratio': round(float(np.mean([s.mean_rs_ratio for s in summaries])), 4),
            'mean_qrs_net': round(float(np.mean([s.mean_qrs_net for s in summaries])), 2),
            'total_time_sec': round(total_time, 1),
        },
        'per_record': [{k: v for k, v in s.__dict__.items()} for s in summaries],
    }
    class NpEnc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, (np.bool_, bool)): return bool(o)
            return super().default(o)
    with open(os.path.join(OUT_DIR, 'qrs_polarity_global.json'), 'w', encoding='utf-8') as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False, cls=NpEnc)

    # Print final report
    print(f"\n{'='*60}")
    print(f"  QRS POLARITY DETECTION — COMPLETE")
    print(f"{'='*60}")
    print(f"  Records: {len(summaries)}")
    beat_t = sum(s.n_beats for s in summaries)
    print(f"  Total beats: {beat_t}")
    print(f"  Positive: {sum(s.n_positive for s in summaries)} ({sum(s.n_positive for s in summaries)/max(beat_t,1)*100:.1f}%)")
    print(f"  Negative: {sum(s.n_negative for s in summaries)} ({sum(s.n_negative for s in summaries)/max(beat_t,1)*100:.1f}%)")
    print(f"  Biphasic: {sum(s.n_biphasic for s in summaries)} ({sum(s.n_biphasic for s in summaries)/max(beat_t,1)*100:.1f}%)")
    print(f"  Records pos-dominant: {sum(1 for s in summaries if s.dominant_polarity=='positive')}")
    print(f"  Records neg-dominant: {sum(1 for s in summaries if s.dominant_polarity=='negative')}")
    print(f"  Mean R/S ratio: {np.mean([s.mean_rs_ratio for s in summaries]):.3f}")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"  Output: {OUT_DIR}/")
    print(f"  Summary: {OUT_DIR}/qrs_polarity_global.json")
    print(f"  Dashboard: {OUT_DIR}/_qrs_dashboard.png")
    print(f"{'='*60}")

    print(f"\n{'─'*60}")
    print(f"  PER-RECORD PER-BEAT FILES SAVED:")
    print(f"  {OUT_DIR}/<record>/")
    print(f"    raw_ecg.npy, filtered_ecg.npy, state_labels.npy")
    print(f"    qrs_polarity.json      — per-beat QRS polarity + metrics")
    print(f"    qrs_samples.npz        — QRS waveform segments")
    print(f"    qrs_summary.json       — record-level summary")
    print(f"    qrs_overview.png       — full waveform with QRS highlighting")
    print(f"    beats/beat_###_qrs.png — per-beat QRS with R/S/ratio/polarity box")
    print(f"{'─'*60}")


if __name__ == '__main__':
    main()
