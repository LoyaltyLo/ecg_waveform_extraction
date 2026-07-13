"""Process RA-LA Reversal aECG dataset with HSMM segmentation.

Each aECG file contains:
- 12-lead ECG: 10,000 samples @ 1000Hz (10 seconds)
- 1 representative beat with P/QRS/T boundary annotations (in ms, relative)
- Global measurements: HR, PR, QRS, QT intervals, axis angles

Pipeline:
1. Parse aECG XML -> extract lead II signal + annotations
2. Run HSMM segmentation (smart init per record)
3. Compare HSMM P/QRS/T boundaries against annotations
4. Save per-file results, metrics, and plots
"""

import sys
sys.path.insert(0, 'c:/LoyaltyLo/PythonProjects/ECG_engineering')

import os, json, re, time, gc
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ecg_waveform_extraction.preprocessing import ECGPreprocessor
from ecg_waveform_extraction.features import FeatureExtractor
from ecg_waveform_extraction.hsmm import HSMMModel, HSMMDecoder, smart_initialize_gmms
from ecg_waveform_extraction.segmentation import ECGSegmenter
from ecg_waveform_extraction.extraction import PWaveExtractor, PWaveAnalyzer
from ecg_waveform_extraction.utils.vis import plot_segmentation, plot_p_wave_detail, STATE_COLORS
from ecg_waveform_extraction.hsmm.hsmm_model import STATE_LABELS


# =====================================================================
# Config
# =====================================================================
AECG_DIR = 'C:/LoyaltyLo/datasets/RA-LA_Reversal/aECG'
OUT_DIR = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/output_rala'
os.makedirs(OUT_DIR, exist_ok=True)

SKIP_PLOTS = True  # Skip per-file plots for speed in batch mode
MAX_SAMPLES = 4000  # Process first 4s for faster processing


# =====================================================================
# aECG XML Parser
# =====================================================================
def parse_aecg(filepath):
    """Parse HL7 aECG XML. Returns dict with signal, annotations, measurements."""
    # Try UTF-8 first, fall back to GBK for Chinese-encoded XML
    for enc in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                content = f.read()
            if '<?xml' in content[:100]:
                break
        except (UnicodeDecodeError, UnicodeError):
            continue

    result = {
        'filename': os.path.basename(filepath),
        'fs': None,
        'n_leads': 12,
        'lead_names': ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
        'signals': {},
        'annotations': {},  # {P_on_ms, P_off_ms, QRS_on_ms, QRS_off_ms, T_on_ms, T_off_ms}
        'measurements': {},  # {HR_bpm, PR_ms, QRS_ms, QT_ms, QTc_ms, P_axis, QRS_axis, T_axis}
        'interpretation': '',
    }

    # ---- Sampling rate ----
    inc_match = re.search(r'<increment[^>]*value="([^"]+)"[^>]*unit="s"', content)
    if inc_match:
        increment = float(inc_match.group(1))
        result['fs'] = 1.0 / increment

    # ---- Extract lead signals (first sequenceSet with MDC_ECG_LEAD codes) ----
    # Find the rhythm waveform section: all 12 leads in one sequenceSet
    # Each lead's samples are in a <sequence> containing <digits>
    all_digits_matches = list(re.finditer(r'<digits[^>]*>([^<]*)</digits>', content))

    if len(all_digits_matches) >= 12:
        # First 12 digit blocks are the 12 leads of rhythm waveform
        for i, m in enumerate(all_digits_matches[:12]):
            text = m.group(1).strip()
            samples = np.array([float(x) for x in text.split()], dtype=np.float64)
            lead_name = result['lead_names'][i]
            result['signals'][lead_name] = samples
        result['n_samples'] = len(samples)

    # ---- P-wave boundary ----
    pw_match = re.search(
        r'MDC_ECG_WAVC_PWAVE.*?<low value="([^"]+)" unit="ms".*?<high value="([^"]+)" unit="ms"',
        content, re.DOTALL)
    if pw_match:
        result['annotations']['P_on_ms'] = float(pw_match.group(1))
        result['annotations']['P_off_ms'] = float(pw_match.group(2))

        # Convert to sample indices (relative to representative beat start)
        if result['fs']:
            result['annotations']['P_on_sample'] = int(result['annotations']['P_on_ms'] / 1000.0 * result['fs'])
            result['annotations']['P_off_sample'] = int(result['annotations']['P_off_ms'] / 1000.0 * result['fs'])

    # ---- QRS boundary ----
    qrs_match = re.search(
        r'MDC_ECG_WAVC_QRSWAVE.*?<low value="([^"]+)" unit="ms".*?<high value="([^"]+)" unit="ms"',
        content, re.DOTALL)
    if qrs_match:
        result['annotations']['QRS_on_ms'] = float(qrs_match.group(1))
        result['annotations']['QRS_off_ms'] = float(qrs_match.group(2))
        if result['fs']:
            result['annotations']['QRS_on_sample'] = int(float(qrs_match.group(1)) / 1000.0 * result['fs'])
            result['annotations']['QRS_off_sample'] = int(float(qrs_match.group(2)) / 1000.0 * result['fs'])

    # ---- T-wave boundary ----
    t_match = re.search(
        r'MDC_ECG_WAVC_TWAVE.*?<low value="([^"]+)" unit="ms".*?<high value="([^"]+)" unit="ms"',
        content, re.DOTALL)
    if t_match:
        result['annotations']['T_on_ms'] = float(t_match.group(1))
        result['annotations']['T_off_ms'] = float(t_match.group(2))
        if result['fs']:
            result['annotations']['T_on_sample'] = int(float(t_match.group(1)) / 1000.0 * result['fs'])
            result['annotations']['T_off_sample'] = int(float(t_match.group(2)) / 1000.0 * result['fs'])

    # ---- Global measurements ----
    measurements = {
        'HR_bpm': r'MDC_ECG_HEART_RATE.*?<value[^>]*value="([^"]+)"[^>]*unit="bpm"',
        'PR_ms': r'MDC_ECG_TIME_PD_PR.*?<value[^>]*value="([^"]+)"[^>]*unit="ms"',
        'QRS_ms': r'MDC_ECG_TIME_PD_QRS.*?<value[^>]*value="([^"]+)"[^>]*unit="ms"',
        'QT_ms': r'MDC_ECG_TIME_PD_QT\b(?!c).*?<value[^>]*value="([^"]+)"[^>]*unit="ms"',
        'QTc_ms': r'MDC_ECG_TIME_PD_QTc.*?<value[^>]*value="([^"]+)"[^>]*unit="ms"',
        'P_dur_ms': r'MDC_ECG_TIME_PD_P\b(?!R).*?<value[^>]*value="([^"]+)"[^>]*unit="ms"',
    }
    for key, pattern in measurements.items():
        m = re.search(pattern, content, re.DOTALL)
        if m:
            result['measurements'][key] = float(m.group(1))

    # Axis angles
    axis_map = {
        'P_axis': r'MDC_ECG_ANGLE_P_FRONT.*?<value[^>]*value="([^"]+)"[^>]*unit="deg"',
        'QRS_axis': r'MDC_ECG_ANGLE_QRS_FRONT.*?<value[^>]*value="([^"]+)"[^>]*unit="deg"',
        'T_axis': r'MDC_ECG_ANGLE_T_FRONT.*?<value[^>]*value="([^"]+)"[^>]*unit="deg"',
    }
    for key, pattern in axis_map.items():
        m = re.search(pattern, content, re.DOTALL)
        if m:
            result['measurements'][key] = float(m.group(1))

    # Interpretation text - find the actual statement value
    interp_match = re.search(
        r'MDC_ECG_INTERPRETATION_STATEMENT.*?<value[^>]*xsi:type="ST"[^>]*>([^<]+)</value>',
        content, re.DOTALL)
    if interp_match:
        result['interpretation'] = interp_match.group(1).strip()

    return result


# =====================================================================
# Process one aECG file
# =====================================================================
def process_file(filepath):
    """Parse aECG, run HSMM, compare with annotations."""
    try:
        aecg = parse_aecg(filepath)
    except Exception as e:
        return {'filename': os.path.basename(filepath), 'error': f'parse: {e}'}

    rec_name = aecg['filename'].replace('.aECG', '')
    rec_dir = os.path.join(OUT_DIR, rec_name)
    os.makedirs(rec_dir, exist_ok=True)

    fs = aecg['fs']
    if fs is None:
        fs = 1000.0  # default

    # Get lead II (primary lead for HSMM)
    lead_ii = aecg['signals'].get('II')
    if lead_ii is None:
        return {'filename': rec_name, 'error': 'No lead II signal'}

    # Truncate to MAX_SAMPLES (first 8s)
    n = min(len(lead_ii), MAX_SAMPLES)
    signal = lead_ii[:n].astype(np.float64)

    # ---- Preprocess ----
    prep = ECGPreprocessor(fs=fs)
    clean = prep.preprocess(signal)
    fe = FeatureExtractor(fs=fs)
    features = fe.extract(clean)

    # Save
    np.save(os.path.join(rec_dir, 'raw_ecg.npy'), signal)
    np.save(os.path.join(rec_dir, 'filtered_ecg.npy'), clean)

    # ---- Build & decode HSMM ----
    model = HSMMModel(fs=fs)
    model.initialize_with_priors()
    model.set_left_right_topology()
    smart_initialize_gmms(model, features)

    segmenter = ECGSegmenter(preprocessor=prep, feature_extractor=fe, model=model, fs=fs)
    seg_result = segmenter.segment(signal)
    np.save(os.path.join(rec_dir, 'state_labels.npy'), seg_result.state_labels)

    # ---- P-wave extraction ----
    p_ext = PWaveExtractor(fs=fs)
    p_waves = p_ext.extract(seg_result)
    analyzer = PWaveAnalyzer(fs=fs)
    p_feats = analyzer.analyze(p_waves, clean, seg_result.beats)
    p_summary = analyzer.summarize(p_feats)

    # ---- Compare HSMM boundaries with annotations ----
    # Annotations are RELATIVE to the representative beat template (not absolute time).
    # The template P/QRS/T boundaries are in ms from template start.
    # We compare: P duration, QRS duration, PR interval (relative measures).
    ann = aecg['annotations']

    ann_qrs_on = ann.get('QRS_on_ms')
    ann_qrs_off = ann.get('QRS_off_ms')
    ann_p_on = ann.get('P_on_ms')
    ann_p_off = ann.get('P_off_ms')
    ann_t_off = ann.get('T_off_ms')

    # Annotation-derived template measurements
    ann_p_dur = ((ann_p_off or 0) - (ann_p_on or 0)) if ann_p_on and ann_p_off else None
    ann_qrs_dur = ((ann_qrs_off or 0) - (ann_qrs_on or 0)) if ann_qrs_on and ann_qrs_off else None
    ann_pr = ((ann_qrs_on or 0) - (ann_p_on or 0)) if ann_p_on and ann_qrs_on else None

    # For each HSMM beat, compute P-dur, QRS-dur, PR.
    # Find the beat whose QRS duration best matches the annotation template.
    hsmm_beat_metrics = []
    for b in seg_result.beats:
        if b.q_onset <= 0 or b.s_offset <= 0:
            continue
        p_dur = ((b.p_offset - b.p_onset) / fs * 1000
                 if b.p_onset > 0 and b.p_offset > 0 else None)
        qrs_dur = (b.s_offset - b.q_onset) / fs * 1000
        pr = ((b.q_onset - b.p_onset) / fs * 1000
              if b.p_onset > 0 and b.q_onset > 0 else None)
        hsmm_beat_metrics.append({
            'beat_id': b.beat_id,
            'p_dur_ms': round(p_dur, 1) if p_dur else None,
            'qrs_dur_ms': round(qrs_dur, 1),
            'pr_ms': round(pr, 1) if pr else None,
        })

    # Find best-matching beat (by QRS duration similarity)
    best_match = None
    best_match_idx = -1
    if ann_qrs_dur and hsmm_beat_metrics:
        best_err = float('inf')
        for i, bm in enumerate(hsmm_beat_metrics):
            err = abs(bm['qrs_dur_ms'] - ann_qrs_dur)
            if err < best_err:
                best_err = err
                best_match = bm
                best_match_idx = i

    # Compute HSMM mean metrics across all beats (for comparison)
    hsmm_mean_p_dur = np.mean([bm['p_dur_ms'] for bm in hsmm_beat_metrics if bm['p_dur_ms'] is not None]) if hsmm_beat_metrics else None
    hsmm_mean_qrs_dur = np.mean([bm['qrs_dur_ms'] for bm in hsmm_beat_metrics]) if hsmm_beat_metrics else None
    hsmm_mean_pr = np.mean([bm['pr_ms'] for bm in hsmm_beat_metrics if bm['pr_ms'] is not None]) if hsmm_beat_metrics else None

    hsmm_metrics = {
        'n_beats_with_boundaries': len(hsmm_beat_metrics),
        'hsmm_mean_p_dur_ms': round(hsmm_mean_p_dur, 1) if hsmm_mean_p_dur else None,
        'hsmm_mean_qrs_dur_ms': round(hsmm_mean_qrs_dur, 1) if hsmm_mean_qrs_dur else None,
        'hsmm_mean_pr_ms': round(hsmm_mean_pr, 1) if hsmm_mean_pr else None,
        'best_match_beat': best_match,
        'best_match_qrs_dur_err_ms': round(abs(best_match['qrs_dur_ms'] - ann_qrs_dur), 1) if best_match and ann_qrs_dur else None,
    }

    # ---- Save result ----
    result = {
        'filename': aecg['filename'],
        'record': rec_name,
        'fs': fs, 'n_samples': n,
        'annotations': {k: v for k, v in ann.items() if not k.endswith('_sample')},
        'ann_P_dur_ms': ann_p_dur,
        'ann_QRS_dur_ms': ann_qrs_dur,
        'ann_PR_ms': ann_pr,
        'global_measurements': aecg['measurements'],
        'interpretation': aecg['interpretation'],
        'n_beats_detected': len(seg_result.beats),
        'n_p_waves': len(p_waves),
        'p_duration_mean_ms': p_summary.duration_mean_ms,
        'p_dispersion_ms': p_summary.dispersion_ms,
        'pr_mean_ms': p_summary.pr_mean_ms,
        'hsmm_metrics': hsmm_metrics,
        'first_beat': {
            'p_onset': seg_result.beats[0].p_onset if seg_result.beats else -1,
            'p_offset': seg_result.beats[0].p_offset if seg_result.beats else -1,
            'q_onset': seg_result.beats[0].q_onset if seg_result.beats else -1,
            'r_peak': seg_result.beats[0].r_peak if seg_result.beats else -1,
            's_offset': seg_result.beats[0].s_offset if seg_result.beats else -1,
            't_offset': seg_result.beats[0].t_offset if seg_result.beats else -1,
        },
    }

    # ---- Plot (skip in batch mode for speed) ----
    if not SKIP_PLOTS:
        try:
            fig, ax = plt.subplots(figsize=(18, 5))
            plot_time = min(8.0, n / fs)
            plot_segmentation(clean, seg_result.state_labels, seg_result.state_names,
                              fs=fs, title=f'{rec_name} — HSMM (Lead II)',
                              time_range=(0, plot_time), ax=ax)
            bm_beat = hsmm_metrics.get('best_match_beat')
            if bm_beat and hsmm_metrics.get('best_match_qrs_dur_err_ms') is not None:
                for b in seg_result.beats:
                    if b.beat_id == bm_beat['beat_id'] and b.q_onset > 0:
                        t_qrs = b.q_onset / fs
                        ax.axvline(t_qrs, color='orange', linewidth=1.5, linestyle='-', alpha=0.8)
                        err_val = hsmm_metrics['best_match_qrs_dur_err_ms']
                        ax.text(t_qrs, clean.max() * 0.8, f'Match(err={err_val}ms)',
                                fontsize=8, color="orange", fontweight="bold")
                        break
            fig.savefig(os.path.join(rec_dir, 'segmentation.png'), dpi=120, bbox_inches='tight')
            plt.close(fig)

            if p_waves and p_waves[0].onset_sample > 0:
                pw = p_waves[0]
                fig, ax = plt.subplots(figsize=(7, 3))
                plot_p_wave_detail(clean, pw.onset_sample, pw.offset_sample,
                                   fs=fs, title=f'{rec_name} — P-Wave (Beat {pw.beat_id})', ax=ax)
                fig.savefig(os.path.join(rec_dir, 'p_wave_detail.png'), dpi=120, bbox_inches='tight')
                plt.close(fig)
        except Exception as e:
            print(f'    plot warning: {e}')

    # Save JSON
    class NpEnc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, np.bool_): return bool(o)
            return super().default(o)

    with open(os.path.join(rec_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2, cls=NpEnc)

    return result


# =====================================================================
# Main
# =====================================================================
def main():
    files = sorted([f for f in os.listdir(AECG_DIR) if f.endswith('.aECG')])
    print(f"{'='*65}")
    print(f"  RA-LA REVERSAL aECG DATASET — HSMM SEGMENTATION")
    print(f"{'='*65}")
    print(f"  Files found: {len(files)}")
    print(f"  Output: {OUT_DIR}/")
    print()

    all_results = []
    total_beats = 0
    total_matched = 0
    start_time = time.time()

    for idx, fname in enumerate(files):
        fpath = os.path.join(AECG_DIR, fname)
        rec_name = fname.replace('.aECG', '')
        print(f"[{idx+1}/{len(files)}] {rec_name}...", end=" ", flush=True)

        t0 = time.time()
        res = process_file(fpath)
        dt = time.time() - t0

        if 'error' in res:
            print(f"FAIL: {res['error']}")
            all_results.append(res)
            continue

        all_results.append(res)
        n_beats = res['n_beats_detected']
        total_beats += n_beats

        hm_info = res.get('hsmm_metrics', {})
        if hm_info.get('best_match_beat'):
            total_matched += 1

        # Summary line
        ann_info = res.get('annotations', {})
        interp = res.get('interpretation', '')[:50]
        hm = res.get('hsmm_metrics', {})
        bm = hm.get('best_match_beat', {}) or {}
        print(f"OK beats={n_beats}  "
              f"ANN[P={ann_info.get('P_on_ms','?')}-{ann_info.get('P_off_ms','?')}ms]  "
              f"HSMM[Pdur={hm.get('hsmm_mean_p_dur_ms','?')}ms "
              f"QRSdur={hm.get('hsmm_mean_qrs_dur_ms','?')}ms "
              f"QRSerr={hm.get('best_match_qrs_dur_err_ms','?')}ms]  "
              f"({dt:.0f}s)")

        gc.collect()

    total_time = time.time() - start_time

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*65}")
    print(f"  DATASET SUMMARY")
    print(f"{'='*65}")
    print(f"  Files processed: {len(all_results)}")
    print(f"  Total HSMM beats detected: {total_beats}")
    print(f"  Beat match rate: {total_matched}/{len(files)} ({total_matched/max(len(files),1)*100:.1f}%)")

    # Comparison of HSMM vs annotation measurements
    hsmm_p_durs = []
    ann_p_durs = []
    hsmm_qrs_durs = []
    ann_qrs_durs = []
    hsmm_prs = []
    ann_prs = []
    qrs_dur_errors = []

    for res in all_results:
        if 'error' in res: continue
        hm = res.get('hsmm_metrics', {})
        # P-wave dur
        hp = hm.get('hsmm_mean_p_dur_ms')
        ap = res.get('ann_P_dur_ms')
        if hp and ap:
            hsmm_p_durs.append(hp)
            ann_p_durs.append(ap)
        # QRS dur
        hq = hm.get('hsmm_mean_qrs_dur_ms')
        aq = res.get('ann_QRS_dur_ms')
        if hq and aq:
            hsmm_qrs_durs.append(hq)
            ann_qrs_durs.append(aq)
            e = hm.get('best_match_qrs_dur_err_ms')
            if e is not None:
                qrs_dur_errors.append(e)
        # PR
        hpr_val = hm.get('hsmm_mean_pr_ms')
        apr_val = res.get('ann_PR_ms')
        if hpr_val and apr_val:
            hsmm_prs.append(hpr_val)
            ann_prs.append(apr_val)

    if hsmm_p_durs:
        print(f"\n  P-wave duration:")
        print(f"    HSMM: {np.mean(hsmm_p_durs):.1f} ± {np.std(hsmm_p_durs):.1f} ms")
        if ann_p_durs:
            print(f"    ANN:  {np.mean(ann_p_durs):.1f} ± {np.std(ann_p_durs):.1f} ms")
            errors = [abs(h - a) for h, a in zip(hsmm_p_durs, ann_p_durs)]
            print(f"    MAE: {np.mean(errors):.1f} ms")

    if hsmm_qrs_durs:
        print(f"\n  QRS duration:")
        print(f"    HSMM: {np.mean(hsmm_qrs_durs):.1f} ± {np.std(hsmm_qrs_durs):.1f} ms")
        if ann_qrs_durs:
            print(f"    ANN:  {np.mean(ann_qrs_durs):.1f} ± {np.std(ann_qrs_durs):.1f} ms")
            errors = [abs(h - a) for h, a in zip(hsmm_qrs_durs, ann_qrs_durs)]
            print(f"    MAE: {np.mean(errors):.1f} ms")

    if hsmm_prs:
        print(f"\n  PR interval:")
        print(f"    HSMM: {np.mean(hsmm_prs):.1f} ± {np.std(hsmm_prs):.1f} ms")
        if ann_prs:
            print(f"    ANN:  {np.mean(ann_prs):.1f} ± {np.std(ann_prs):.1f} ms")
            errors = [abs(h - a) for h, a in zip(hsmm_prs, ann_prs)]
            print(f"    MAE: {np.mean(errors):.1f} ms")

    # Interpretation distribution
    interp_counter = defaultdict(int)
    for res in all_results:
        if 'error' in res: continue
        interp = res.get('interpretation', '')
        if interp:
            interp_counter[interp] += 1
    if interp_counter:
        print(f"\n  ECG Interpretation distribution:")
        for interp, count in interp_counter.most_common():
            print(f"    {interp}: {count}")

    # Save global summary
    global_summary = {
        'dataset': 'RA-LA Reversal aECG',
        'n_files': len(files),
        'n_processed': len([r for r in all_results if 'error' not in r]),
        'n_matched': total_matched,
        'total_beats_detected': total_beats,
        'hsmm_p_dur_mean': round(float(np.mean(hsmm_p_durs)), 1) if hsmm_p_durs else None,
        'ann_p_dur_mean': round(float(np.mean(ann_p_durs)), 1) if ann_p_durs else None,
        'hsmm_qrs_dur_mean': round(float(np.mean(hsmm_qrs_durs)), 1) if hsmm_qrs_durs else None,
        'ann_qrs_dur_mean': round(float(np.mean(ann_qrs_durs)), 1) if ann_qrs_durs else None,
        'hsmm_pr_mean': round(float(np.mean(hsmm_prs)), 1) if hsmm_prs else None,
        'ann_pr_mean': round(float(np.mean(ann_prs)), 1) if ann_prs else None,
        'p_dur_mae_ms': round(float(np.mean([abs(h - a) for h, a in zip(hsmm_p_durs, ann_p_durs)])), 1) if hsmm_p_durs and ann_p_durs else None,
        'qrs_dur_mae_ms': round(float(np.mean([abs(h - a) for h, a in zip(hsmm_qrs_durs, ann_qrs_durs)])), 1) if hsmm_qrs_durs and ann_qrs_durs else None,
        'pr_mae_ms': round(float(np.mean([abs(h - a) for h, a in zip(hsmm_prs, ann_prs)])), 1) if hsmm_prs and ann_prs else None,
        'interpretation_distribution': dict(interp_counter),
        'per_file': all_results,
        'total_time_sec': round(total_time, 1),
    }

    class NpEnc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, np.bool_): return bool(o)
            return super().default(o)

    with open(os.path.join(OUT_DIR, 'global_summary.json'), 'w') as f:
        json.dump(global_summary, f, indent=2, cls=NpEnc)

    print(f"\n  Results saved: {os.path.join(OUT_DIR, 'global_summary.json')}")
    print(f"  Per-file output: {OUT_DIR}/<record>/*")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
