"""QRS extraction and polarity detection using refined HSMM boundaries.

For each beat in Lead I and Lead II:
  1. HSMM Viterbi -> Q/R/S state boundaries
  2. refine_qrs_boundaries() -> derivative-based Q-onset/S-offset correction
  3. Compute: R/S ratio, QRS net area, polarity (positive/negative/biphasic)
  4. Save per-beat JSON, waveform segments, plots
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
class QRSBeat:
    beat_id: int
    q_onset: int; r_peak: int; s_offset: int
    duration_ms: float
    r_amplitude: float; s_amplitude: float; q_amplitude: float
    qrs_net_area: float
    rs_ratio: float
    polarity: str          # positive | negative | biphasic
    confidence: float
    lead_name: str


# =====================================================================
def parse_aecg(filepath):
    with open(filepath, 'rb') as f: raw = f.read()
    content = raw.decode('utf-8', errors='replace')
    r = {'fs': 1000.0, 'signals': {}}
    m = re.search(rb'<increment[^>]*value="([^"]+)"[^>]*unit="s"', raw)
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
        'P_axis': 'ANGLE_P_FRONT.*?value="([^"]+)"',
    }.items():
        m = re.search(pat.encode(), raw, re.DOTALL)
        r[key] = float(m.group(1)) if m else None
    interp = re.search(rb'INTERPRETATION_STATEMENT.*?xsi:type="ST"[^>]*>([^<]+)</value>', raw, re.DOTALL)
    r['interpretation'] = (interp.group(1).decode('utf-8',errors='replace').strip().replace('\n','; ')
                           if interp else '')
    return r


def run_hsmm(signal, fs):
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

    1. R peak: max |amplitude| in ±20ms window
    2. Q onset: walk RIGHT until |d1| exceeds 3σ noise floor
    3. S offset (J-point): walk LEFT until |d1| returns to baseline
    """
    T = len(ecg_clean)
    d1 = np.gradient(ecg_clean)

    # Noise floor from quiet early segment
    quiet_end = min(int(0.2 * fs), T // 5)
    noise_sigma = float(np.std(d1[:quiet_end]) + 1e-6)
    threshold = noise_sigma * 3.0

    # Step 1: R peak refinement
    r_search_start = max(0, r_peak_hsmm - int(0.02 * fs))
    r_search_end = min(T - 1, r_peak_hsmm + int(0.02 * fs))
    r_search = ecg_clean[r_search_start:r_search_end + 1]
    bl_local = float(np.median(r_search))
    r_peak = r_search_start + int(np.argmax(np.abs(r_search - bl_local)))

    # Step 2: Q onset refinement (walk right from HSMM estimate)
    q_on = q_on_hsmm
    for i in range(q_on_hsmm, min(r_peak, q_on_hsmm + int(0.08 * fs))):
        if abs(d1[i]) > threshold:
            for j in range(i, max(q_on_hsmm, i - int(0.01 * fs)), -1):
                if abs(d1[j]) <= threshold * 0.5:
                    q_on = j; break
            else: q_on = i
            break

    # Step 3: S offset (J-point) refinement (walk left from HSMM estimate)
    s_off = s_off_hsmm
    for i in range(s_off_hsmm, max(r_peak + int(0.02 * fs), s_off_hsmm - int(0.10 * fs)), -1):
        if abs(d1[i]) > threshold:
            for j in range(i, min(s_off_hsmm, i + int(0.02 * fs))):
                if abs(d1[j]) <= threshold * 0.5:
                    s_off = j; break
            else: s_off = i
            break

    # Sanity minimum 20ms QRS
    min_qrs = int(0.02 * fs)
    if s_off - q_on < min_qrs: q_on, s_off = q_on_hsmm, s_off_hsmm
    q_on = max(0, min(q_on, T - 1))
    s_off = max(q_on + min_qrs, min(s_off, T - 1))
    r_peak = max(q_on, min(r_peak, s_off))
    return q_on, r_peak, s_off


def extract_qrs(seg_result, ecg_clean, fs, lead_name):
    """Extract refined QRS metrics from HSMM segmentation."""
    beats = []; T = len(ecg_clean)
    for b in seg_result.beats:
        if b.q_onset <= 0 or b.s_offset <= 0 or b.r_peak <= 0: continue

        # Refine boundaries
        q_on, r_pk, s_off = refine_qrs_boundaries(ecg_clean, b.q_onset, b.r_peak, b.s_offset, fs)

        seg = ecg_clean[q_on:s_off + 1]
        bl = float(np.mean(ecg_clean[max(0, q_on - 30):q_on])) if q_on >= 30 else float(np.median(seg[:5]))
        detrend = seg - bl
        dur = len(seg) / fs * 1000.0
        r_amp = float(ecg_clean[r_pk] - bl) if 0 <= r_pk < T else 0.0

        # S nadir after R
        r_idx = r_pk - q_on
        s_nadir = float(np.min(detrend[r_idx:])) if r_idx < len(detrend) else 0.0

        # Q nadir before R
        q_nadir = float(np.min(detrend[:r_idx + 1])) if r_idx + 1 <= len(detrend) and r_idx >= 0 else 0.0

        qrs_net = float(np.sum(detrend))

        # R/S ratio
        neg_mag = max(abs(s_nadir), abs(q_nadir), 0.001) if (s_nadir < 0 or q_nadir < 0) else 0.001
        if r_amp > 0:
            rs_ratio = float(min(r_amp / neg_mag, 100.0))
        else:
            rs_ratio = float(max(r_amp / max(abs(r_amp) + neg_mag, 0.001), -100.0))

        # Polarity
        if rs_ratio >= 1.5 and qrs_net > 0:
            pol, conf = 'positive', min(rs_ratio / 3.0, 1.0)
        elif rs_ratio <= 0.5 and qrs_net < 0:
            pol, conf = 'negative', min(abs(qrs_net) / (abs(r_amp) + neg_mag + 0.001), 1.0)
        elif abs(qrs_net) < abs(r_amp) * 0.1:
            pol, conf = 'biphasic', 0.6
        elif qrs_net > 0:
            pol, conf = 'positive', 0.7
        else:
            pol, conf = 'negative', 0.7

        beats.append(QRSBeat(
            beat_id=b.beat_id, q_onset=q_on, r_peak=r_pk, s_offset=s_off,
            duration_ms=round(dur, 2),
            r_amplitude=round(r_amp, 4), s_amplitude=round(s_nadir, 4),
            q_amplitude=round(q_nadir, 4),
            qrs_net_area=round(qrs_net, 4), rs_ratio=round(rs_ratio, 4),
            polarity=pol, confidence=round(min(conf, 1.0), 3), lead_name=lead_name))
    return beats


def plot_qrs(rec_name, rec_dir, ecg_clean, seg_result, qrs_beats, fs, lead_label):
    beats_dir = os.path.join(rec_dir, 'beats'); os.makedirs(beats_dir, exist_ok=True)
    T = len(ecg_clean); n_beats = len(qrs_beats)

    # ---- Overview ----
    fig, ax = plt.subplots(figsize=(18, 4))
    ps = min(3.8, T / fs); n_plot = int(ps * fs)
    t_plot, e_plot = np.arange(n_plot) / fs, ecg_clean[:n_plot]
    for q in qrs_beats:
        if q.q_onset < n_plot and q.s_offset < n_plot:
            c = {'positive': '#4caf50', 'negative': '#f44336', 'biphasic': '#ff9800'}[q.polarity]
            ax.fill_between(t_plot[q.q_onset:q.s_offset + 1], e_plot[q.q_onset:q.s_offset + 1],
                            alpha=0.32, color=c, linewidth=0)
            mid = (q.q_onset + q.s_offset) // 2
            if mid < n_plot:
                y_offset = 8 if q.qrs_net_area > 0 else -14
                ax.annotate(f'{q.polarity[0].upper()}', (t_plot[mid], e_plot[mid]),
                            textcoords='offset points', xytext=(0, y_offset),
                            fontsize=7, ha='center', color=c, fontweight='bold')
    ax.plot(t_plot, e_plot, 'k-', linewidth=0.5)
    ax.set_xlim(t_plot[0], t_plot[-1]); ax.set_xlabel('Time (s)')
    pos = sum(1 for q in qrs_beats if q.polarity == 'positive')
    neg = sum(1 for q in qrs_beats if q.polarity == 'negative')
    bip = sum(1 for q in qrs_beats if q.polarity == 'biphasic')
    ax.set_title(f'{rec_name} — Lead {lead_label} QRS (refined)  |  +:{pos}  -:{neg}  bip:{bip}')
    ax.grid(True, alpha=0.15); fig.tight_layout()
    fig.savefig(os.path.join(rec_dir, f'qrs_overview_{lead_label}.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ---- Per-beat detail (up to 8) ----
    for i in range(min(n_beats, 8)):
        q = qrs_beats[i]; margin = int(0.08 * fs)
        ws, we = max(0, q.q_onset - margin), min(T - 1, q.s_offset + margin)
        if we - ws < 10: continue
        fig, ax = plt.subplots(figsize=(8, 3))
        t_win, e_win = np.arange(ws, we + 1) / fs, ecg_clean[ws:we + 1]
        ax.plot(t_win, e_win, 'k-', linewidth=1.2)
        qrs_t = np.arange(q.q_onset, q.s_offset + 1) / fs
        qrs_v = ecg_clean[q.q_onset:q.s_offset + 1]
        c = {'positive': '#4caf50', 'negative': '#f44336', 'biphasic': '#ff9800'}[q.polarity]
        ax.fill_between(qrs_t, qrs_v, alpha=0.35, color=c, label=f'QRS ({q.polarity})')
        if q.r_peak > 0: ax.plot(q.r_peak / fs, ecg_clean[q.r_peak], 'rv', markersize=8, label='R')
        ax.plot(q.q_onset / fs, ecg_clean[q.q_onset], 'g<', markersize=6)
        ax.plot(q.s_offset / fs, ecg_clean[q.s_offset], 'b>', markersize=6)
        bl = float(np.mean(ecg_clean[max(0, q.q_onset - 30):q.q_onset])) if q.q_onset >= 30 else float(np.median(e_win))
        ax.axhline(bl, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        info = (f"Lead {lead_label} QRS\n"
                f"Polarity: {q.polarity.upper()}\n"
                f"R/S ratio: {q.rs_ratio:.2f}\n"
                f"QRS net: {q.qrs_net_area:.1f}\n"
                f"R amp: {q.r_amplitude:.3f}  S amp: {q.s_amplitude:.3f}\n"
                f"Dur: {q.duration_ms:.0f}ms  Conf: {q.confidence:.2f}")
        ax.text(0.98, 0.95, info, transform=ax.transAxes, fontsize=9, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9), fontfamily='monospace')
        ax.set_xlim(t_win[0], t_win[-1]); ax.set_xlabel('Time (s)')
        ax.set_title(f'Beat {q.beat_id} — Lead {lead_label} QRS')
        ax.legend(fontsize=8, loc='upper left'); ax.grid(True, alpha=0.15)
        fig.tight_layout()
        fig.savefig(os.path.join(beats_dir, f'beat_{q.beat_id:03d}_qrs_{lead_label}.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)


def process_record(fname):
    fpath = os.path.join(AECG_DIR, fname); rec_name = fname.replace('.aECG', '')
    rec_dir = os.path.join(OUT_DIR, rec_name); os.makedirs(rec_dir, exist_ok=True)

    aecg = parse_aecg(fpath); fs = aecg['fs']
    sig_I = aecg['signals'].get('I'); sig_II = aecg['signals'].get('II')
    if sig_I is None or sig_II is None: return None

    n = min(len(sig_I), len(sig_II), MAX_SAMPLES)

    # ---- Lead I ----
    seg_I, clean_I = run_hsmm(sig_I[:n].astype(np.float64), fs)
    qrs_I = extract_qrs(seg_I, clean_I, fs, 'I')

    # ---- Lead II ----
    seg_II, clean_II = run_hsmm(sig_II[:n].astype(np.float64), fs)
    qrs_II = extract_qrs(seg_II, clean_II, fs, 'II')

    # ---- Save numpy ----
    np.save(os.path.join(rec_dir, 'raw_I.npy'), sig_I[:n])
    np.save(os.path.join(rec_dir, 'raw_II.npy'), sig_II[:n])
    np.save(os.path.join(rec_dir, 'filtered_I.npy'), clean_I)
    np.save(os.path.join(rec_dir, 'filtered_II.npy'), clean_II)
    np.save(os.path.join(rec_dir, 'state_labels_I.npy'), seg_I.state_labels)
    np.save(os.path.join(rec_dir, 'state_labels_II.npy'), seg_II.state_labels)

    # QRS samples
    samp_I = {str(q.beat_id): clean_I[q.q_onset:q.s_offset + 1] for q in qrs_I}
    samp_II = {str(q.beat_id): clean_II[q.q_onset:q.s_offset + 1] for q in qrs_II}
    np.savez(os.path.join(rec_dir, 'qrs_samples_I.npz'), **samp_I)
    np.savez(os.path.join(rec_dir, 'qrs_samples_II.npz'), **samp_II)

    # ---- Per-beat JSON ----
    beats_json = []
    for qi, qii in zip(qrs_I, qrs_II):
        if qi.beat_id == qii.beat_id:
            beats_json.append({
                'beat_id': qi.beat_id,
                'lead_I': {
                    'q_onset': qi.q_onset, 'r_peak': qi.r_peak, 's_offset': qi.s_offset,
                    'duration_ms': qi.duration_ms,
                    'r_amplitude': qi.r_amplitude, 's_amplitude': qi.s_amplitude,
                    'q_amplitude': qi.q_amplitude,
                    'qrs_net_area': qi.qrs_net_area, 'rs_ratio': qi.rs_ratio,
                    'polarity': qi.polarity, 'confidence': qi.confidence,
                },
                'lead_II': {
                    'q_onset': qii.q_onset, 'r_peak': qii.r_peak, 's_offset': qii.s_offset,
                    'duration_ms': qii.duration_ms,
                    'r_amplitude': qii.r_amplitude, 's_amplitude': qii.s_amplitude,
                    'q_amplitude': qii.q_amplitude,
                    'qrs_net_area': qii.qrs_net_area, 'rs_ratio': qii.rs_ratio,
                    'polarity': qii.polarity, 'confidence': qii.confidence,
                },
            })
    with open(os.path.join(rec_dir, 'qrs_beats.json'), 'w') as f:
        json.dump(beats_json, f, indent=2)

    # ---- Plots ----
    plot_qrs(rec_name, rec_dir, clean_I, seg_I, qrs_I, fs, 'I')
    plot_qrs(rec_name, rec_dir, clean_II, seg_II, qrs_II, fs, 'II')

    # ---- Summary ----
    I_pos = sum(1 for q in qrs_I if q.polarity == 'positive')
    I_neg = sum(1 for q in qrs_I if q.polarity == 'negative')
    I_bip = sum(1 for q in qrs_I if q.polarity == 'biphasic')
    II_pos = sum(1 for q in qrs_II if q.polarity == 'positive')
    II_neg = sum(1 for q in qrs_II if q.polarity == 'negative')
    II_bip = sum(1 for q in qrs_II if q.polarity == 'biphasic')
    n_beats = len(qrs_I)

    summary = {
        'record': rec_name,
        'n_beats': n_beats,
        'P_axis': aecg.get('P_axis'), 'QRS_axis': aecg.get('QRS_axis'),
        'HR': aecg.get('HR'), 'ann_QRS_dur_ms': aecg.get('QRS_dur'),
        'lead_I': {
            'n_positive': I_pos, 'n_negative': I_neg, 'n_biphasic': I_bip,
            'mean_rs_ratio': round(float(np.mean([q.rs_ratio for q in qrs_I])), 4) if qrs_I else 0,
            'mean_qrs_net': round(float(np.mean([q.qrs_net_area for q in qrs_I])), 1) if qrs_I else 0,
            'mean_duration_ms': round(float(np.mean([q.duration_ms for q in qrs_I])), 1) if qrs_I else 0,
        },
        'lead_II': {
            'n_positive': II_pos, 'n_negative': II_neg, 'n_biphasic': II_bip,
            'mean_rs_ratio': round(float(np.mean([q.rs_ratio for q in qrs_II])), 4) if qrs_II else 0,
            'mean_qrs_net': round(float(np.mean([q.qrs_net_area for q in qrs_II])), 1) if qrs_II else 0,
            'mean_duration_ms': round(float(np.mean([q.duration_ms for q in qrs_II])), 1) if qrs_II else 0,
        },
        'interpretation': aecg.get('interpretation', ''),
    }
    with open(os.path.join(rec_dir, 'qrs_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def generate_dashboard(summaries):
    n = len(summaries)
    total_beats = sum(s['n_beats'] for s in summaries)
    I_pos = sum(s['lead_I']['n_positive'] for s in summaries)
    I_neg = sum(s['lead_I']['n_negative'] for s in summaries)
    II_pos = sum(s['lead_II']['n_positive'] for s in summaries)
    II_neg = sum(s['lead_II']['n_negative'] for s in summaries)
    I_dur = [s['lead_I']['mean_duration_ms'] for s in summaries if s['lead_I']['mean_duration_ms'] > 0]
    II_dur = [s['lead_II']['mean_duration_ms'] for s in summaries if s['lead_II']['mean_duration_ms'] > 0]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('QRS Extraction — Refined Boundaries (Lead I + Lead II)', fontsize=14, fontweight='bold')

    # (0,0) Key metrics
    ax = axes[0, 0]; ax.axis('off')
    ax.text(0.1, 0.95,
        f"Records: {n}  |  Total beats: {total_beats}\n\n"
        f"Lead I QRS:\n  Positive: {I_pos} ({I_pos/max(total_beats,1)*100:.0f}%)\n"
        f"  Negative: {I_neg} ({I_neg/max(total_beats,1)*100:.0f}%)\n"
        f"  Mean dur: {np.mean(I_dur):.0f}ms\n\n"
        f"Lead II QRS:\n  Positive: {II_pos} ({II_pos/max(total_beats,1)*100:.0f}%)\n"
        f"  Negative: {II_neg} ({II_neg/max(total_beats,1)*100:.0f}%)\n"
        f"  Mean dur: {np.mean(II_dur):.0f}ms",
        transform=ax.transAxes, fontsize=10, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))

    # (0,1) Lead I polarity pie
    ax = axes[0, 1]
    ax.pie([I_pos, I_neg], labels=[f'Positive\n({I_pos})', f'Negative\n({I_neg})'],
           colors=['#4caf50', '#f44336'], autopct='%1.0f%%', startangle=90)
    ax.set_title('Lead I QRS Polarity', fontsize=11, fontweight='bold')

    # (0,2) Lead II polarity pie
    ax = axes[0, 2]
    ax.pie([II_pos, II_neg], labels=[f'Positive\n({II_pos})', f'Negative\n({II_neg})'],
           colors=['#4caf50', '#f44336'], autopct='%1.0f%%', startangle=90)
    ax.set_title('Lead II QRS Polarity', fontsize=11, fontweight='bold')

    # (1,0) Lead I R/S histogram
    ax = axes[1, 0]
    I_rs = [s['lead_I']['mean_rs_ratio'] for s in summaries]
    ax.hist(I_rs, bins=25, color='#2196f3', edgecolor='white', alpha=0.8)
    ax.axvline(1.0, color='red', linestyle='--', linewidth=1.5, label='R/S=1')
    ax.set_xlabel('Mean R/S Ratio (Lead I)'); ax.set_ylabel('Records')
    ax.set_title('Lead I R/S Ratio Distribution'); ax.legend(); ax.grid(True, alpha=0.2, axis='y')

    # (1,1) QRS duration comparison
    ax = axes[1, 1]
    ax.hist(I_dur, bins=20, color='#2196f3', edgecolor='white', alpha=0.6, label='Lead I')
    ax.hist(II_dur, bins=20, color='#ff9800', edgecolor='white', alpha=0.6, label='Lead II')
    ax.set_xlabel('Mean QRS Duration (ms)'); ax.set_ylabel('Records')
    ax.set_title('QRS Duration Distribution'); ax.legend(); ax.grid(True, alpha=0.2, axis='y')

    # (1,2) Per-record breakdown
    ax = axes[1, 2]; ax.axis('off')
    lines = ['Most LEAD I NEGATIVE:'] + [
        f'  {s["record"][:12]} I:{s["lead_I"]["n_negative"]}/{s["n_beats"]}neg'
        for s in sorted(summaries, key=lambda s: -s['lead_I']['n_negative'])[:8]]
    lines += ['\nMost LEAD I POSITIVE:'] + [
        f'  {s["record"][:12]} I:{s["lead_I"]["n_positive"]}/{s["n_beats"]}pos'
        for s in sorted(summaries, key=lambda s: -s['lead_I']['n_positive'])[:6]]
    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes, fontsize=8,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '_qrs_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    files = sorted([f for f in os.listdir(AECG_DIR) if f.endswith('.aECG')])[:N_FILES]
    print(f"{'='*60}")
    print(f"  QRS EXTRACTION — Refined Boundaries (Lead I + Lead II)")
    print(f"  {N_FILES} records from RA-LA Reversal aECG")
    print(f"  Method: HSMM Viterbi + derivative-based Q/S refinement")
    print(f"{'='*60}\n")

    summaries = []; t_start = time.time()
    for idx, fname in enumerate(files):
        print(f"[{idx+1:2d}/{N_FILES}] {fname[:14]}...", end=" ", flush=True)
        t0 = time.time(); s = process_record(fname); dt = time.time() - t0
        if s:
            summaries.append(s)
            I_sum = s['lead_I']; II_sum = s['lead_II']
            print(f"OK beats={s['n_beats']} "
                  f"I: +{I_sum['n_positive']} -{I_sum['n_negative']} "
                  f"dur={I_sum['mean_duration_ms']:.0f}ms | "
                  f"II: +{II_sum['n_positive']} -{II_sum['n_negative']} "
                  f"dur={II_sum['mean_duration_ms']:.0f}ms "
                  f"({dt:.0f}s)")
        else: print(f"SKIP")
        gc.collect()

    total_time = time.time() - t_start
    generate_dashboard(summaries)

    # Global summary JSON
    class NpEnc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, (np.bool_, bool)): return bool(o)
            return super().default(o)

    total_beats = sum(s['n_beats'] for s in summaries)
    global_summary = {
        'method': 'HSMM Viterbi + refine_qrs_boundaries (derivative-based)',
        'dataset': 'RA-LA Reversal aECG', 'n_records': N_FILES,
        'n_leads': 2, 'leads': ['I', 'II'],
        'aggregate': {
            'total_records': len(summaries), 'total_beats': total_beats,
            'lead_I': {
                'positive': sum(s['lead_I']['n_positive'] for s in summaries),
                'negative': sum(s['lead_I']['n_negative'] for s in summaries),
                'mean_duration_ms': round(float(np.mean([s['lead_I']['mean_duration_ms'] for s in summaries if s['lead_I']['mean_duration_ms'] > 0])), 1),
            },
            'lead_II': {
                'positive': sum(s['lead_II']['n_positive'] for s in summaries),
                'negative': sum(s['lead_II']['n_negative'] for s in summaries),
                'mean_duration_ms': round(float(np.mean([s['lead_II']['mean_duration_ms'] for s in summaries if s['lead_II']['mean_duration_ms'] > 0])), 1),
            },
            'total_time_sec': round(total_time, 1),
        },
        'per_record': summaries,
    }
    with open(os.path.join(OUT_DIR, 'qrs_global_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False, cls=NpEnc)

    I_pos_total = sum(s['lead_I']['n_positive'] for s in summaries)
    I_neg_total = sum(s['lead_I']['n_negative'] for s in summaries)
    II_pos_total = sum(s['lead_II']['n_positive'] for s in summaries)
    II_neg_total = sum(s['lead_II']['n_negative'] for s in summaries)

    print(f"\n{'='*60}")
    print(f"  QRS EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Records: {len(summaries)}  |  Total beats: {total_beats}")
    print(f"  Lead I  —  +{I_pos_total}  -{I_neg_total}")
    print(f"  Lead II —  +{II_pos_total}  -{II_neg_total}")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"  Output: {OUT_DIR}/")
    print(f"  Dashboard: {OUT_DIR}/_qrs_dashboard.png")
    print(f"  Summary: {OUT_DIR}/qrs_global_summary.json")
    print(f"{'='*60}")
    print(f"\n  Per-record files:")
    print(f"    raw_I/II.npy, filtered_I/II.npy, state_labels_I/II.npy")
    print(f"    qrs_beats.json — per-beat Lead I + II QRS metrics")
    print(f"    qrs_samples_I/II.npz — QRS waveform segments")
    print(f"    qrs_summary.json — record-level summary")
    print(f"    qrs_overview_I/II.png, beats/beat_###_qrs_I/II.png")


if __name__ == '__main__':
    main()
