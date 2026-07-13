"""Per-beat-type evaluation on MIT-BIH Arrhythmia records.

For each MIT-BIH record with beat-type annotations:
1. Smart-initialize GMMs from the signal (per-record, avoids cross-freq issues)
2. Viterbi decode → R-peak positions
3. Match HSMM R-peaks to annotated beats → per-type sensitivity/PPV
4. For matched beats, check if P-wave was detected → P-wave rate per type

Reports broken down by: N, L, R, V, A, /, F, f, J, E, a, S, !
"""

import sys
sys.path.insert(0, 'c:/LoyaltyLo/PythonProjects/ECG_engineering')

import os, json, time, gc
from collections import defaultdict, Counter
import numpy as np
import wfdb

from ecg_waveform_extraction.preprocessing import ECGPreprocessor
from ecg_waveform_extraction.features import FeatureExtractor
from ecg_waveform_extraction.hsmm import HSMMModel, HSMMDecoder, smart_initialize_gmms
from ecg_waveform_extraction.segmentation import ECGSegmenter
from ecg_waveform_extraction.extraction import PWaveExtractor, PWaveAnalyzer
from ecg_waveform_extraction.utils.vis import plot_segmentation, plot_p_wave_detail
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ecg_waveform_extraction.hsmm.hsmm_model import STATE_LABELS, N_STATES

# =====================================================================
# Config
# =====================================================================
DATA_DIR = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/data'
OUT_DIR = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/output_arrhythmia'
os.makedirs(OUT_DIR, exist_ok=True)

MAX_SEC = 25.0            # 25s per record for more beats
MAX_SAMPLES = 12000
R_MATCH_TOL_MS = 150      # 150ms tolerance for R-peak matching

BEAT_NAMES = {
    'N': 'Normal', 'L': 'LBBB', 'R': 'RBBB', 'V': 'PVC',
    'A': 'Atrial Premature', '/': 'Paced', 'F': 'Fusion V-N',
    'f': 'Fusion P-N', 'J': 'Nodal Premature', 'E': 'Vent Escape',
    'a': 'Aberrated Atrial', 'S': 'SV Premature', '!': 'Vent Flutter',
    'Q': 'Unclassifiable', 'j': 'Nodal Escape',
}

# =====================================================================
# Main processing function
# =====================================================================
def process_mitbih_record(rec_name):
    """Process one MIT-BIH record: segment, match beats, report per-type."""
    rec_path = os.path.join(DATA_DIR, rec_name)
    if not os.path.exists(rec_path + '.dat'):
        return None

    try:
        # ---- Load ----
        record = wfdb.rdrecord(rec_path)
        sig = record.p_signal
        if sig.ndim > 1: sig = sig[:, 0]
        sig = sig.astype(np.float64)
        fs = float(record.fs)
        n = min(int(MAX_SEC * fs), MAX_SAMPLES, len(sig))
        sig = sig[:n]
        T = len(sig)

        # ---- Annotations with beat types ----
        ann_beats = []  # list of (sample, type_symbol)
        try:
            ann = wfdb.rdann(rec_path, 'atr')
            for i, sym in enumerate(ann.symbol):
                sample = int(ann.sample[i])
                if sym != '+' and sample < n and sym.strip():
                    ann_beats.append((sample, sym))
        except Exception:
            pass

        if not ann_beats:
            return None

        # ---- Preprocess + features ----
        prep = ECGPreprocessor(fs=fs)
        clean = prep.preprocess(sig)
        fe = FeatureExtractor(fs=fs)
        features = fe.extract(clean)

        # ---- Build model (per-record smart init) ----
        model = HSMMModel(fs=fs)
        model.initialize_with_priors()
        model.set_left_right_topology()
        smart_initialize_gmms(model, features)

        # ---- Decode ----
        decoder = HSMMDecoder()
        result = decoder.decode(model, features)

        # Extract R-peaks (state 4 = R)
        hsmm_r = np.array(
            [(s + e) // 2 for state, s, e in result['state_sequence'] if state == 4],
            dtype=int,
        )

        # ---- Segment for P-wave extraction ----
        segmenter = ECGSegmenter(preprocessor=prep, feature_extractor=fe,
                                 model=model, fs=fs)
        seg_result = segmenter.segment(sig)

        # P-wave extraction
        p_ext = PWaveExtractor(fs=fs)
        p_waves = p_ext.extract(seg_result)

        # Map beat_id -> p_wave
        p_wave_by_beat = {pw.beat_id: pw for pw in p_waves}

        # ---- Match HSMM R-peaks to annotated beats ----
        tol = int(R_MATCH_TOL_MS / 1000.0 * fs)

        # For each annotated beat, find the closest HSMM peak
        ann_to_hsmm = {}  # ann_index -> hsmm_peak_index (or -1 if unmatched)
        hsmm_used = set()

        for ai, (ar, atype) in enumerate(ann_beats):
            best_dist = tol + 1
            best_hi = -1
            for hi, hr in enumerate(hsmm_r):
                if hi in hsmm_used:
                    continue
                dist = abs(hr - ar)
                if dist <= tol and dist < best_dist:
                    best_dist = dist
                    best_hi = hi
            if best_hi >= 0:
                ann_to_hsmm[ai] = best_hi
                hsmm_used.add(best_hi)
            else:
                ann_to_hsmm[ai] = -1

        # ---- Per-type statistics ----
        type_stats = defaultdict(lambda: {'total': 0, 'detected': 0, 'p_wave_found': 0})

        for ai, (ar, atype) in enumerate(ann_beats):
            type_stats[atype]['total'] += 1
            if ann_to_hsmm[ai] >= 0:
                type_stats[atype]['detected'] += 1

                # Check P-wave: find the beat in seg_result that contains this R-peak
                hr_peak = hsmm_r[ann_to_hsmm[ai]]
                for b in seg_result.beats:
                    if (b.r_peak > 0 and abs(b.r_peak - hr_peak) <= tol):
                        if b.beat_id in p_wave_by_beat:
                            pw = p_wave_by_beat[b.beat_id]
                            if pw.onset_sample > 0 and pw.offset_sample > pw.onset_sample:
                                type_stats[atype]['p_wave_found'] += 1
                        break

        # Overall
        total_ann = len(ann_beats)
        total_det = sum(1 for v in ann_to_hsmm.values() if v >= 0)
        n_hsmm = len(hsmm_r)
        sensitivity = round(total_det / max(total_ann, 1) * 100, 1)
        ppv = round(total_det / max(n_hsmm, 1) * 100, 1)

        # Type distribution
        type_counter = Counter(s for _, s in ann_beats)

        # P-wave analysis
        analyzer = PWaveAnalyzer(fs=fs)
        p_feats = analyzer.analyze(p_waves, clean, seg_result.beats)
        p_summary = analyzer.summarize(p_feats)

        # Save per-record beat-type details
        beat_details = []
        for ai, (ar, atype) in enumerate(ann_beats):
            matched = ann_to_hsmm[ai] >= 0
            detail = {
                'ann_sample': int(ar), 'type': atype,
                'type_name': BEAT_NAMES.get(atype, atype),
                'detected': matched,
            }
            if matched:
                hr = hsmm_r[ann_to_hsmm[ai]]
                detail['hsmm_peak'] = int(hr)
                detail['error_ms'] = round(abs(hr - ar) / fs * 1000, 1)

                # Find associated beat
                for b in seg_result.beats:
                    if b.r_peak > 0 and abs(b.r_peak - hr) <= tol:
                        detail['beat_id'] = b.beat_id
                        detail['p_onset'] = int(b.p_onset) if b.p_onset > 0 else -1
                        detail['p_offset'] = int(b.p_offset) if b.p_offset > 0 else -1
                        detail['p_detected'] = (b.beat_id in p_wave_by_beat and
                            p_wave_by_beat[b.beat_id].onset_sample > 0)
                        break
            beat_details.append(detail)

        # Per-beat waveform plots
        beats_dir = os.path.join(OUT_DIR, rec_name, 'beats')
        os.makedirs(beats_dir, exist_ok=True)

        # Plot up to 5 beats per type for this record
        plotted = Counter()
        for ai, (ar, atype) in enumerate(ann_beats):
            if plotted[atype] >= 3:
                continue
            if ann_to_hsmm[ai] < 0:
                continue
            hr = hsmm_r[ann_to_hsmm[ai]]
            # Find the beat
            for b in seg_result.beats:
                if b.r_peak > 0 and abs(b.r_peak - hr) <= tol:
                    bid = b.beat_id
                    break
            else:
                continue

            if b.p_onset <= 0 or b.t_offset <= 0:
                continue

            margin = int(0.15 * fs)
            ws = max(0, b.p_onset - margin)
            we = min(T - 1, b.t_offset + margin)
            if we - ws < 30:
                continue

            fig, ax = plt.subplots(figsize=(12, 4))
            t_win = np.arange(ws, we + 1) / fs
            e_win = clean[ws:we + 1]
            l_win = result['state_labels'][ws:we + 1]

            from ecg_waveform_extraction.utils.vis import STATE_COLORS
            if len(l_win) > 0:
                prev = l_win[0]; seg_start = 0
                for ii in range(1, len(l_win)):
                    if l_win[ii] != prev:
                        c = STATE_COLORS.get(STATE_LABELS[prev] if 0<=prev<9 else 'UNKNOWN','#9e9e9e')
                        ax.axvspan(t_win[seg_start], t_win[ii], alpha=0.25, color=c)
                        seg_start = ii; prev = l_win[ii]
                c = STATE_COLORS.get(STATE_LABELS[prev] if 0<=prev<9 else 'UNKNOWN','#9e9e9e')
                ax.axvspan(t_win[seg_start], t_win[-1], alpha=0.25, color=c)

            ax.plot(t_win, e_win, 'k-', linewidth=0.8)
            ylo, yhi = e_win.min(), e_win.max()
            yr = max(yhi - ylo, 0.01)

            # Mark key boundaries
            for lbl, idx, color in [
                ('P on', b.p_onset, 'green'), ('P off', b.p_offset, 'green'),
                ('QRS on', b.q_onset, 'red'), ('QRS off', b.s_offset, 'red'),
                ('T off', b.t_offset, 'blue')]:
                if idx > 0:
                    tx = idx / fs
                    ax.axvline(tx, color=color, linestyle='--', linewidth=0.8, alpha=0.7)

            ax.set_title(f"{rec_name} Beat {bid}  |  Type: {atype} ({BEAT_NAMES.get(atype,atype)})")
            ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude')
            ax.set_xlim(t_win[0], t_win[-1])
            ax.grid(True, alpha=0.2)
            from matplotlib.patches import Rectangle
            handles = [Rectangle((0,0),1,1,facecolor=STATE_COLORS[s],alpha=0.25,label=s) for s in STATE_LABELS]
            ax.legend(handles=handles, loc='upper right', ncol=9, fontsize=6)
            fig.tight_layout()
            import re
            fname_type = re.sub(r'[<>:"/\\|?*\'"~]', '_', atype)
            fig.savefig(os.path.join(beats_dir, f'type_{fname_type}_beat{bid:03d}.png'), dpi=120, bbox_inches='tight')
            plt.close(fig)
            plotted[atype] += 1

        # Segmentation overview
        fig, ax = plt.subplots(figsize=(18, 5))
        plot_time = min(15.0, T / fs)
        plot_segmentation(clean, result['state_labels'],
                          [STATE_LABELS[l] if l>=0 else 'UNK' for l in result['state_labels']],
                          fs=fs, title=f"{rec_name} — HSMM Segmentation",
                          time_range=(0, plot_time), ax=ax)
        fig.savefig(os.path.join(OUT_DIR, rec_name, 'segmentation.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

        # Save per-record results
        rec_result = {
            'record': rec_name,
            'fs': fs, 'n_samples': T, 'duration_sec': round(T/fs, 1),
            'n_annotated': total_ann, 'n_hsmm': n_hsmm,
            'n_matched': total_det,
            'sensitivity': sensitivity, 'ppv': ppv,
            'n_p_waves': len(p_waves),
            'p_duration_mean_ms': p_summary.duration_mean_ms,
            'p_dispersion_ms': p_summary.dispersion_ms,
            'pr_interval_mean_ms': p_summary.pr_mean_ms,
            'beat_type_distribution': dict(type_counter.most_common()),
            'per_type': {
                t: {'total': s['total'], 'detected': s['detected'],
                    'sensitivity': round(s['detected']/max(s['total'],1)*100, 1),
                    'p_wave_rate': round(s['p_wave_found']/max(s['detected'],1)*100, 1) if s['detected']>0 else 0}
                for t, s in sorted(type_stats.items())
            },
            'beat_details': beat_details,
        }

        # Save
        class NpEnc(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, (np.integer,)): return int(o)
                if isinstance(o, (np.floating,)): return float(o)
                if isinstance(o, np.ndarray): return o.tolist()
                if isinstance(o, np.bool_): return bool(o)
                if isinstance(o, (set,)): return sorted(o)
                return super().default(o)

        with open(os.path.join(OUT_DIR, rec_name, 'result.json'), 'w') as f:
            json.dump(rec_result, f, indent=2, cls=NpEnc)

        # Save numpy data
        np.save(os.path.join(OUT_DIR, rec_name, 'filtered_ecg.npy'), clean)
        np.save(os.path.join(OUT_DIR, rec_name, 'state_labels.npy'), result['state_labels'])

        return rec_result

    except Exception as e:
        import traceback
        print(f"    FAIL: {e}")
        traceback.print_exc()
        return {'record': rec_name, 'error': str(e)}


# =====================================================================
# Main
# =====================================================================
def main():
    # Find all MIT-BIH records with .atr files (beat type annotations available)
    mitbih_recs = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith('.atr'):
            continue
        rec = fname[:-4]
        if not rec.isdigit():
            continue
        rec_num = int(rec)
        if not ((100 <= rec_num < 110) or (111 <= rec_num < 125) or (200 <= rec_num < 235)):
            continue
        if os.path.exists(os.path.join(DATA_DIR, rec + '.dat')):
            mitbih_recs.append(rec)

    print(f"{'='*65}")
    print(f"  PER-BEAT-TYPE EVALUATION ON MIT-BIH ARRHYTHMIA")
    print(f"{'='*65}")
    print(f"  Records with beat-type annotations: {len(mitbih_recs)}")
    print(f"  Tolerance: {R_MATCH_TOL_MS}ms for R-peak matching")
    print(f"  Max: {MAX_SEC}s per record")
    print()

    all_results = []
    global_type_stats = defaultdict(lambda: {'total': 0, 'detected': 0, 'p_wave_found': 0})
    total_beats = 0
    total_matched = 0
    total_hsmm_peaks = 0
    total_p_waves = 0
    ok_count = 0

    t_start = time.time()

    for idx, rec in enumerate(mitbih_recs):
        print(f"[{idx+1}/{len(mitbih_recs)}] {rec}...", end=" ", flush=True)
        t0 = time.time()
        res = process_mitbih_record(rec)

        if res is None or 'error' in res:
            print(f"SKIP ({time.time()-t0:.0f}s)")
            continue

        ok_count += 1
        total_beats += res['n_annotated']
        total_matched += res['n_matched']
        total_hsmm_peaks += res['n_hsmm']
        total_p_waves += res['n_p_waves']
        all_results.append(res)

        # Aggregate global per-type
        for t, s in res['per_type'].items():
            global_type_stats[t]['total'] += s['total']
            global_type_stats[t]['detected'] += s['detected']
            # P-wave found tracked per-record but not accumulated globally without match info
            # We'll compute from beat_details

        # Print summary line
        type_se_str = ' '.join(
            f"{t}={res['per_type'][t]['sensitivity']:.0f}%"
            for t in sorted(res['per_type'].keys(),
                          key=lambda x: -res['per_type'][x]['total'])[:3]
        )
        print(f"OK  Se={res['sensitivity']:.1f}% PPV={res['ppv']:.1f}% "
              f"beats={res['n_annotated']} P={res['n_p_waves']} "
              f"[{type_se_str}] ({time.time()-t0:.0f}s)")

        gc.collect()

    total_time = time.time() - t_start

    # ==================================================================
    # Global per-type aggregation (from all beat_details across records)
    # ==================================================================
    global_beat_type = defaultdict(lambda: {'total': 0, 'detected': 0, 'p_detected': 0})
    for res in all_results:
        for bd in res.get('beat_details', []):
            t = bd['type']
            global_beat_type[t]['total'] += 1
            if bd['detected']:
                global_beat_type[t]['detected'] += 1
                if bd.get('p_detected'):
                    global_beat_type[t]['p_detected'] += 1

    # ==================================================================
    # Final report
    # ==================================================================
    print(f"\n{'='*65}")
    print(f"  GLOBAL PER-BEAT-TYPE RESULTS")
    print(f"{'='*65}")
    overall_se = round(total_matched / max(total_beats, 1) * 100, 1)
    overall_ppv = round(total_matched / max(total_hsmm_peaks, 1) * 100, 1)
    print(f"  Records processed: {ok_count}/{len(mitbih_recs)}")
    print(f"  Total annotated beats: {total_beats}")
    print(f"  Total HSMM peaks: {total_hsmm_peaks}")
    print(f"  Total matched: {total_matched}")
    print(f"  OVERALL R-peak Sensitivity: {overall_se}%")
    print(f"  OVERALL R-peak PPV: {overall_ppv}%")
    print(f"  Total P-waves extracted: {total_p_waves}")
    print(f"  Total time: {total_time:.0f}s")

    print(f"\n  {'Type':<6} {'Name':<28} {'Total':>6} {'Detected':>8} {'Se%':>7} {'P-wave%':>8}")
    print(f"  {'-'*63}")

    for t in sorted(global_beat_type.keys(),
                     key=lambda x: -global_beat_type[x]['total']):
        s = global_beat_type[t]
        se = round(s['detected'] / max(s['total'], 1) * 100, 1)
        pw = round(s['p_detected'] / max(s['detected'], 1) * 100, 1) if s['detected'] > 0 else 0
        print(f"  {t:<6} {BEAT_NAMES.get(t, t):<28} {s['total']:>6} {s['detected']:>8} {se:>6.1f}% {pw:>7.1f}%")

    print(f"  {'-'*63}")
    print(f"  {'ALL':<6} {'Overall':<28} {total_beats:>6} {total_matched:>8} {overall_se:>6.1f}%")

    # P-wave analysis per type explanation
    print(f"\n  NOTES:")
    print(f"  - P-wave% = fraction of DETECTED beats where P-wave was found")
    print(f"  - Beats without preceding P-wave (PVC, Paced) expected to have low P-wave%")
    print(f"  - Normal (N) beats should have high P-wave%")

    # Save global
    class NpEnc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, np.bool_): return bool(o)
            if isinstance(o, (set,)): return sorted(o)
            if isinstance(o, defaultdict): return dict(o)
            if isinstance(o, Counter): return dict(o)
            return super().default(o)

    global_result = {
        'dataset': 'MIT-BIH Arrhythmia',
        'n_records': ok_count,
        'n_annotated': total_beats,
        'n_hsmm_peaks': total_hsmm_peaks,
        'n_matched': total_matched,
        'overall_sensitivity': overall_se,
        'overall_ppv': overall_ppv,
        'n_p_waves_total': total_p_waves,
        'r_match_tolerance_ms': R_MATCH_TOL_MS,
        'per_type_summary': {
            t: {
                'name': BEAT_NAMES.get(t, t),
                'total': s['total'],
                'detected': s['detected'],
                'sensitivity': round(s['detected']/max(s['total'],1)*100, 1),
                'p_wave_rate': round(s['p_detected']/max(s['detected'],1)*100, 1) if s['detected']>0 else 0,
            }
            for t, s in sorted(global_beat_type.items(), key=lambda x: -x[1]['total'])
        },
        'per_record_results': all_results,
        'total_time_sec': round(total_time, 1),
    }

    with open(os.path.join(OUT_DIR, 'arrhythmia_per_type_summary.json'), 'w') as f:
        json.dump(global_result, f, indent=2, cls=NpEnc)

    print(f"\n  Saved: {os.path.join(OUT_DIR, 'arrhythmia_per_type_summary.json')}")
    print(f"  Per-record output: {OUT_DIR}/<record>/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
