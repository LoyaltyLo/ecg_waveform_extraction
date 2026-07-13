"""Reprocess RA-LA aECG data with full output matching output_test_only format.

For each file saves:
  raw_ecg.npy, filtered_ecg.npy, state_labels.npy, features.npy
  segmentation.json, p_waves.json, p_wave_metrics.json, p_wave_samples.npz
  summary.json, segmentation.png
  beats/beat_###_waveform.png, beats/beat_###_p_wave.png

Format matches output_test_only/<record>/ exactly.
"""

import sys
sys.path.insert(0, 'c:/LoyaltyLo/PythonProjects/ECG_engineering')

import os, json, time, gc, traceback
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ecg_waveform_extraction.preprocessing import ECGPreprocessor
from ecg_waveform_extraction.features import FeatureExtractor
from ecg_waveform_extraction.hsmm import HSMMModel, HSMMDecoder, smart_initialize_gmms
from ecg_waveform_extraction.segmentation import ECGSegmenter
from ecg_waveform_extraction.extraction import PWaveExtractor, PWaveAnalyzer
from ecg_waveform_extraction.utils.vis import plot_segmentation, plot_p_wave_detail

# ---- Config ----
AECG_DIR = 'C:/LoyaltyLo/datasets/RA-LA_Reversal/aECG'
OUT_DIR = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/output_rala_full'
os.makedirs(OUT_DIR, exist_ok=True)

MAX_SAMPLES = 4000        # 4s for speed
PLOT_PER_BEAT = True      # generate per-beat waveform + P-wave plots
BATCH_PRINT_EVERY = 10

# JSON encoder
class NpEnc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, np.bool_): return bool(o)
        return super().default(o)


# =====================================================================
# aECG Parser (lightweight, regex-based)
# =====================================================================
def parse_aecg(filepath):
    import re
    for enc in ['utf-8', 'gbk', 'latin-1']:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                content = f.read()
            if '<?xml' in content[:100]: break
        except: continue

    result = {'fs': None, 'signals': {}, 'annotations': {}, 'measurements': {}, 'interpretation': ''}

    inc = re.search(r'<increment[^>]*value="([^"]+)"[^>]*unit="s"', content)
    if inc: result['fs'] = 1.0 / float(inc.group(1))

    lead_names = ['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
    digits = list(re.finditer(r'<digits[^>]*>([^<]*)</digits>', content))
    for i, m in enumerate(digits[:12]):
        samples = np.array([float(x) for x in m.group(1).split()], dtype=np.float64)
        result['signals'][lead_names[i]] = samples

    # Annotations (P/QRS/T boundaries in ms)
    for key, tag in [('P_on_ms','PWAVE'),('P_off_ms','PWAVE'),('QRS_on_ms','QRSWAVE'),('QRS_off_ms','QRSWAVE'),
                      ('T_on_ms','TWAVE'),('T_off_ms','TWAVE')]:
        if 'on' in key:
            m = re.search(rf'MDC_ECG_WAVC_{tag}.*?<low value="([^"]+)" unit="ms"', content, re.DOTALL)
        else:
            m = re.search(rf'MDC_ECG_WAVC_{tag}.*?<high value="([^"]+)" unit="ms"', content, re.DOTALL)
        if m: result['annotations'][key] = float(m.group(1))

    # Global measurements
    for key, pat in {
        'HR_bpm': r'MDC_ECG_HEART_RATE.*?<value[^>]*value="([^"]+)"[^>]*unit="bpm"',
        'PR_ms': r'MDC_ECG_TIME_PD_PR.*?<value[^>]*value="([^"]+)"[^>]*unit="ms"',
        'QRS_ms': r'MDC_ECG_TIME_PD_QRS\b(?!c).*?<value[^>]*value="([^"]+)"[^>]*unit="ms"',
        'QT_ms': r'MDC_ECG_TIME_PD_QT\b(?!c).*?<value[^>]*value="([^"]+)"[^>]*unit="ms"',
        'QTc_ms': r'MDC_ECG_TIME_PD_QTc.*?<value[^>]*value="([^"]+)"[^>]*unit="ms"',
    }.items():
        m = re.search(pat, content, re.DOTALL)
        if m: result['measurements'][key] = float(m.group(1))

    interp = re.search(r'MDC_ECG_INTERPRETATION_STATEMENT.*?xsi:type="ST"[^>]*>([^<]+)</value>', content, re.DOTALL)
    if interp: result['interpretation'] = interp.group(1).strip()

    return result


# =====================================================================
# Process one file
# =====================================================================
def process_file(filepath):
    aecg = parse_aecg(filepath)
    rec_name = os.path.basename(filepath).replace('.aECG', '')
    rec_dir = os.path.join(OUT_DIR, rec_name)
    os.makedirs(rec_dir, exist_ok=True)
    beats_dir = os.path.join(rec_dir, 'beats')
    os.makedirs(beats_dir, exist_ok=True)

    fs = aecg['fs'] or 1000.0
    lead_ii = aecg['signals'].get('II')
    if lead_ii is None:
        lead_ii = next(iter(aecg['signals'].values()))
    n = min(len(lead_ii), MAX_SAMPLES)
    signal = lead_ii[:n].astype(np.float64)

    result = {
        'record': rec_name, 'status': 'ok', 'error': None,
        'fs': fs, 'n_samples': n, 'duration_sec': round(n/fs, 1),
        'n_beats': 0, 'n_p_waves': 0,
        'p_duration_mean_ms': None, 'p_duration_std_ms': None,
        'p_dispersion_ms': None, 'pr_interval_mean_ms': None,
        'pr_interval_std_ms': None, 'p_amplitude_mean': None,
        'processing_time_sec': 0,
        'ann_interpretation': aecg.get('interpretation', ''),
        'ann_measurements': aecg.get('measurements', {}),
        'ann_qrs_ms': aecg['annotations'].get('QRS_on_ms'),
        'ann_p_ms': aecg['annotations'].get('P_on_ms'),
        'ann_t_ms': aecg['annotations'].get('T_off_ms'),
    }

    t0 = time.time()

    try:
        # Preprocess
        prep = ECGPreprocessor(fs=fs)
        clean = prep.preprocess(signal)
        fe = FeatureExtractor(fs=fs)
        features = fe.extract(clean)

        np.save(os.path.join(rec_dir, 'raw_ecg.npy'), signal)
        np.save(os.path.join(rec_dir, 'filtered_ecg.npy'), clean)
        np.save(os.path.join(rec_dir, 'features.npy'), features)

        # HSMM
        model = HSMMModel(fs=fs)
        model.initialize_with_priors()
        model.set_left_right_topology()
        smart_initialize_gmms(model, features)

        segmenter = ECGSegmenter(preprocessor=prep, feature_extractor=fe, model=model, fs=fs)
        seg_result = segmenter.segment(signal)
        np.save(os.path.join(rec_dir, 'state_labels.npy'), seg_result.state_labels)

        # ---- segmentation.json ----
        seg_data = [{
            'beat_id': int(b.beat_id),
            'iso_start': int(b.iso_start) if b.iso_start > 0 else -1,
            'p_onset': int(b.p_onset) if b.p_onset > 0 else -1,
            'p_offset': int(b.p_offset) if b.p_offset > 0 else -1,
            'pr_start': int(b.pr_start) if b.pr_start > 0 else -1,
            'q_onset': int(b.q_onset) if b.q_onset > 0 else -1,
            'r_peak': int(b.r_peak) if b.r_peak > 0 else -1,
            's_offset': int(b.s_offset) if b.s_offset > 0 else -1,
            'st_start': int(b.st_start) if b.st_start > 0 else -1,
            't_onset': int(b.t_onset) if b.t_onset > 0 else -1,
            't_offset': int(b.t_offset) if b.t_offset > 0 else -1,
            'tp_start': int(b.tp_start) if b.tp_start > 0 else -1,
        } for b in seg_result.beats]
        with open(os.path.join(rec_dir, 'segmentation.json'), 'w') as f:
            json.dump(seg_data, f, indent=2, cls=NpEnc)

        n_beats = len(seg_result.beats)
        result['n_beats'] = n_beats

        # ---- P-wave extraction ----
        p_ext = PWaveExtractor(fs=fs)
        p_waves = p_ext.extract(seg_result)
        result['n_p_waves'] = len(p_waves)

        # p_waves.json
        pw_data = [{
            'beat_id': pw.beat_id,
            'onset_sample': pw.onset_sample,
            'offset_sample': pw.offset_sample,
            'peak_sample': pw.peak_sample,
            'duration_ms': round(pw.duration_ms, 2),
            'confidence': round(pw.confidence, 4),
        } for pw in p_waves]
        with open(os.path.join(rec_dir, 'p_waves.json'), 'w') as f:
            json.dump(pw_data, f, indent=2, cls=NpEnc)

        # p_wave_samples.npz
        if p_waves:
            pw_samp = {}
            for pw in p_waves:
                if pw.onset_sample >= 0 and pw.offset_sample >= pw.onset_sample:
                    pw_samp[str(pw.beat_id)] = clean[pw.onset_sample:pw.offset_sample+1]
            np.savez(os.path.join(rec_dir, 'p_wave_samples.npz'), **pw_samp)

        # ---- P-wave analysis ----
        analyzer = PWaveAnalyzer(fs=fs)
        p_feats = analyzer.analyze(p_waves, clean, seg_result.beats)

        # p_wave_metrics.json
        pm = [{
            'beat_id': pf.beat_id,
            'onset_sample': pf.onset_sample,
            'offset_sample': pf.offset_sample,
            'peak_sample': pf.peak_sample,
            'duration_ms': pf.duration_ms,
            'peak_amplitude': pf.peak_amplitude,
            'area': pf.area,
            'morphology_score': pf.morphology_score,
            'pr_interval_ms': pf.pr_interval_ms,
        } for pf in p_feats]
        with open(os.path.join(rec_dir, 'p_wave_metrics.json'), 'w') as f:
            json.dump(pm, f, indent=2, cls=NpEnc)

        p_summary = analyzer.summarize(p_feats)
        result.update({
            'p_duration_mean_ms': p_summary.duration_mean_ms,
            'p_duration_std_ms': p_summary.duration_std_ms,
            'p_dispersion_ms': p_summary.dispersion_ms,
            'pr_interval_mean_ms': p_summary.pr_mean_ms,
            'pr_interval_std_ms': p_summary.pr_std_ms,
            'p_amplitude_mean': p_summary.amplitude_mean,
        })

        # ---- summary.json ----
        with open(os.path.join(rec_dir, 'summary.json'), 'w') as f:
            json.dump(result, f, indent=2, cls=NpEnc)

        # ---- Per-beat plots ----
        if PLOT_PER_BEAT:
            from ecg_waveform_extraction.utils.vis import STATE_COLORS
            from ecg_waveform_extraction.hsmm.hsmm_model import STATE_LABELS
            from matplotlib.patches import Rectangle

            for b in seg_result.beats:
                bid = b.beat_id
                if b.p_onset <= 0 or b.t_offset <= 0 or b.q_onset <= 0:
                    continue

                margin = int(0.15 * fs)
                ws = max(0, b.p_onset - margin)
                we = min(n - 1, b.t_offset + margin)
                if we - ws < 30:
                    continue

                # -- waveform plot --
                fig, ax = plt.subplots(figsize=(12, 4))
                t_win = np.arange(ws, we + 1) / fs
                e_win = clean[ws:we + 1]
                l_win = seg_result.state_labels[ws:we + 1]

                if len(l_win) > 0:
                    prev = l_win[0]; seg_start = 0
                    for i in range(1, len(l_win)):
                        if l_win[i] != prev:
                            c = STATE_COLORS.get(STATE_LABELS[prev] if 0<=prev<9 else 'UNKNOWN','#9e9e9e')
                            ax.axvspan(t_win[seg_start], t_win[i], alpha=0.25, color=c)
                            seg_start = i; prev = l_win[i]
                    c = STATE_COLORS.get(STATE_LABELS[prev] if 0<=prev<9 else 'UNKNOWN','#9e9e9e')
                    ax.axvspan(t_win[seg_start], t_win[-1], alpha=0.25, color=c)

                ax.plot(t_win, e_win, 'k-', linewidth=0.8)
                ylo, yhi = e_win.min(), e_win.max()
                yr = max(yhi - ylo, 0.01)

                for lbl, idx, color in [('P on', b.p_onset, 'green'), ('P off', b.p_offset, 'green'),
                                          ('QRS on', b.q_onset, 'red'), ('QRS off', b.s_offset, 'red'),
                                          ('T off', b.t_offset, 'blue')]:
                    if idx > 0 and ws <= idx <= we:
                        tx = idx / fs
                        ax.axvline(tx, color=color, linestyle='--', linewidth=0.8, alpha=0.7)
                        ax.text(tx, yhi + 0.05*yr, lbl, fontsize=7, color=color, ha='center')

                ax.set_xlim(t_win[0], t_win[-1])
                ax.set_title(f'Record {rec_name} — Beat {bid} — P-QRS-T Waveform')
                ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude')
                handles = [Rectangle((0,0),1,1,facecolor=STATE_COLORS[s],alpha=0.25,label=s) for s in STATE_LABELS]
                ax.legend(handles=handles, loc='upper right', ncol=9, fontsize=6)
                ax.grid(True, alpha=0.2)
                fig.tight_layout()
                fig.savefig(os.path.join(beats_dir, f'beat_{bid:03d}_waveform.png'), dpi=120, bbox_inches='tight')
                plt.close(fig)

                # -- P-wave detail plot --
                pw_match = next((pw for pw in p_waves if pw.beat_id == bid), None)
                if pw_match and pw_match.onset_sample > 0 and pw_match.offset_sample > pw_match.onset_sample:
                    pw_on, pw_off = pw_match.onset_sample, pw_match.offset_sample
                    pw_peak = pw_match.peak_sample
                    pmg = int(0.1 * fs)
                    pws, pwe = max(0, pw_on - pmg), min(n - 1, pw_off + pmg)
                    if pws < pwe:
                        fig, ax = plt.subplots(figsize=(7, 3))
                        t_pw = np.arange(pws, pwe + 1) / fs
                        ax.plot(t_pw, clean[pws:pwe + 1], 'k-', linewidth=1.0)
                        p_idx = np.arange(pw_on, pw_off + 1)
                        if len(p_idx) <= len(clean):
                            ax.fill_between(p_idx / fs, clean[p_idx], alpha=0.3, color='#4caf50', label='P wave')
                        ax.axvline(pw_on / fs, color='green', linestyle='--', linewidth=1.0)
                        ax.axvline(pw_off / fs, color='red', linestyle='--', linewidth=1.0)
                        if 0 <= pw_peak < n:
                            ax.axvline(pw_peak / fs, color='blue', linestyle=':', linewidth=0.8)
                        mid_t = (pw_on + pw_off) // 2
                        if mid_t < n:
                            ax.annotate(f'{pw_match.duration_ms:.0f}ms', (mid_t/fs, clean[mid_t]),
                                        textcoords='offset points', xytext=(0, 10),
                                        fontsize=9, ha='center', fontweight='bold')
                        ax.set_title(f'Record {rec_name} — Beat {bid} — P-Wave Detail')
                        ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude')
                        ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
                        fig.tight_layout()
                        fig.savefig(os.path.join(beats_dir, f'beat_{bid:03d}_p_wave.png'), dpi=120, bbox_inches='tight')
                        plt.close(fig)

        # ---- segmentation overview ----
        fig, ax = plt.subplots(figsize=(18, 5))
        plot_time = min(n / fs, 4.0)
        plot_segmentation(clean, seg_result.state_labels, seg_result.state_names,
                          fs=fs, title=f'{rec_name} — HSMM Segmentation',
                          time_range=(0, plot_time), ax=ax)
        fig.savefig(os.path.join(rec_dir, 'segmentation.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

        result['status'] = 'ok'

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()

    result['processing_time_sec'] = round(time.time() - t0, 1)
    return result


# =====================================================================
# Main
# =====================================================================
def main():
    files = sorted([f for f in os.listdir(AECG_DIR) if f.endswith('.aECG')])
    print(f"{'='*65}")
    print(f"  RA-LA REVERSAL — FULL OUTPUT (matching output_test_only format)")
    print(f"{'='*65}")
    print(f"  Files: {len(files)}")
    print(f"  Output: {OUT_DIR}")
    print()

    all_results = []
    ok_count = 0
    t_start = time.time()

    for idx, fname in enumerate(files):
        fpath = os.path.join(AECG_DIR, fname)
        rec_name = fname.replace('.aECG', '')
        res = process_file(fpath)
        all_results.append(res)

        if res['status'] == 'ok':
            ok_count += 1

        if (idx + 1) % BATCH_PRINT_EVERY == 0:
            elapsed = time.time() - t_start
            avg = elapsed / (idx + 1)
            remaining = avg * (len(files) - idx - 1)
            n_beats = res.get('n_beats', '?')
            print(f"[{idx+1}/{len(files)}] {rec_name} "
                  f"beats={n_beats} dur={res.get('p_duration_mean_ms','?')}ms "
                  f"({res['processing_time_sec']}s) "
                  f"[{elapsed/60:.0f}m elapsed, ~{remaining/60:.0f}m remaining]", flush=True)

    total_time = time.time() - t_start

    # Global summary
    with open(os.path.join(OUT_DIR, 'global_summary.json'), 'w') as f:
        json.dump({
            'dataset': 'RA-LA Reversal aECG',
            'total_files': len(files),
            'processed_ok': ok_count,
            'total_time_sec': round(total_time, 1),
            'output_format': 'matches output_test_only',
        }, f, indent=2, cls=NpEnc)

    print(f"\n{'='*65}")
    print(f"  DONE: {ok_count}/{len(files)} OK")
    print(f"  Time: {total_time/60:.1f} min")
    print(f"  Output: {OUT_DIR}/")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
