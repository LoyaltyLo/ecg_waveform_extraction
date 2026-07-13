"""Evaluate trained HSMM strictly on unseen test records."""

import sys
sys.path.insert(0, 'c:/LoyaltyLo/PythonProjects/ECG_engineering')

import os
import json
import time
import numpy as np
import wfdb

from ecg_waveform_extraction.preprocessing import ECGPreprocessor
from ecg_waveform_extraction.features import FeatureExtractor
from ecg_waveform_extraction.hsmm import HSMMModel
from ecg_waveform_extraction.segmentation import ECGSegmenter
from ecg_waveform_extraction.extraction import PWaveExtractor, PWaveAnalyzer
from ecg_waveform_extraction.utils.vis import plot_segmentation, plot_p_wave_detail
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# JSON encoder that handles numpy types
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, set): return sorted(obj)
        return super().default(obj)

def _jwrite(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, cls=NpEncoder)

# =====================================================================
# Config
# =====================================================================
DATA_DIR = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/data'
MODEL_PATH = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/models/hsmm_trained.npz'
OUT_DIR = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/output_test_only'
os.makedirs(OUT_DIR, exist_ok=True)

MAX_SEC = 15.0
MAX_SAMPLES = 10000

# =====================================================================
# Reproduce exact train/test split
# =====================================================================
# Collect all .hea+.dat record names (same as training did)
all_records = sorted([
    f[:-4] for f in os.listdir(DATA_DIR)
    if f.endswith('.hea') and os.path.exists(os.path.join(DATA_DIR, f[:-4] + '.dat'))
])

# Reproduce exact split
np.random.seed(42)
indices = np.random.permutation(len(all_records))
n_train = min(30, len(all_records))
train_records = {all_records[i] for i in indices[:n_train]}
test_records = [all_records[i] for i in indices[n_train:]]

print(f"All records: {len(all_records)}")
print(f"Train records: {len(train_records)}")
print(f"Test records: {len(test_records)} (UNSEEN)")
print()

# =====================================================================
# Load model once
# =====================================================================
print(f"Loading model: {MODEL_PATH}")
model = HSMMModel.load(MODEL_PATH)
print(f"Model: {model}\n")

# =====================================================================
# Evaluate one record
# =====================================================================
def eval_record(rec, model):
    """Run full pipeline on one record, save all outputs."""
    import traceback
    rec_dir = os.path.join(OUT_DIR, rec)
    os.makedirs(rec_dir, exist_ok=True)
    beats_dir = os.path.join(rec_dir, 'beats')
    os.makedirs(beats_dir, exist_ok=True)

    result = {
        "record": rec, "status": "error", "in_training": rec in train_records,
        "n_beats": 0, "n_p_waves": 0,
        "r_peak_sensitivity": None, "r_peak_ppv": None,
        "p_duration_mean_ms": None, "p_duration_std_ms": None,
        "p_dispersion_ms": None, "pr_interval_mean_ms": None,
        "processing_time_sec": 0,
    }

    try:
        t0 = time.time()

        # -- Load --
        record = wfdb.rdrecord(os.path.join(DATA_DIR, rec))
        sig = record.p_signal
        if sig.ndim > 1: sig = sig[:, 0]
        sig = sig.astype(np.float64)
        fs = float(record.fs)
        n = min(int(MAX_SEC * fs), MAX_SAMPLES, len(sig))
        sig = sig[:n]

        # -- Annotations --
        ann_r = None
        try:
            ann = wfdb.rdann(os.path.join(DATA_DIR, rec), 'atr')
            a_s = np.asarray(ann.sample)[np.asarray(ann.symbol) != '+']
            ann_r = a_s[a_s < n]
        except: pass

        # -- Preprocess + features --
        prep = ECGPreprocessor(fs=fs)
        clean = prep.preprocess(sig)
        fe = FeatureExtractor(fs=fs)
        features = fe.extract(clean)

        # Save
        np.save(os.path.join(rec_dir, 'raw_ecg.npy'), sig)
        np.save(os.path.join(rec_dir, 'filtered_ecg.npy'), clean)
        np.save(os.path.join(rec_dir, 'features.npy'), features)

        # -- HSMM Segment --
        segmenter = ECGSegmenter(preprocessor=prep, feature_extractor=fe, model=model, fs=fs)
        seg = segmenter.segment(sig)
        np.save(os.path.join(rec_dir, 'state_labels.npy'), seg.state_labels)

        n_beats = len(seg.beats)
        result['n_beats'] = n_beats

        # R-peak comparison
        if ann_r is not None and len(ann_r) > 0 and n_beats > 0:
            hsmm_r = np.array([b.r_peak for b in seg.beats if b.r_peak > 0], dtype=int)
            tol = int(0.15 * fs)
            if len(hsmm_r) > 0:
                matched_se = sum(1 for ar in ann_r if np.any(np.abs(hsmm_r - int(ar)) <= tol))
                result['r_peak_sensitivity'] = round(matched_se / len(ann_r) * 100, 1)
                matched_ppv = sum(1 for hr in hsmm_r if np.any(np.abs(ann_r - hr) <= tol))
                result['r_peak_ppv'] = round(matched_ppv / len(hsmm_r) * 100, 1)

        # Save segmentation JSON (cast numpy ints to native int)
        seg_data = [{k: int(getattr(b, k)) for k in [
            "beat_id","p_onset","p_offset","q_onset","r_peak","s_offset",
            "t_onset","t_offset","iso_start","pr_start","st_start","tp_start"
        ]} for b in seg.beats]
        _jwrite(seg_data, os.path.join(rec_dir, 'segmentation.json'))

        # -- P-wave extraction --
        p_ext = PWaveExtractor(fs=fs)
        p_waves = p_ext.extract(seg)
        result['n_p_waves'] = len(p_waves)

        # Save P-wave JSON
        pw_data = [{
            "beat_id": pw.beat_id,
            "onset_sample": pw.onset_sample,
            "offset_sample": pw.offset_sample,
            "peak_sample": pw.peak_sample,
            "duration_ms": round(pw.duration_ms, 2),
            "confidence": round(pw.confidence, 4),
        } for pw in p_waves]
        _jwrite(pw_data, os.path.join(rec_dir, 'p_waves.json'))

        # P-wave samples
        if p_waves:
            pw_samp = {}
            for pw in p_waves:
                if pw.onset_sample >= 0 and pw.offset_sample >= pw.onset_sample:
                    pw_samp[str(pw.beat_id)] = clean[pw.onset_sample:pw.offset_sample+1]
            np.savez(os.path.join(rec_dir, 'p_wave_samples.npz'), **pw_samp)

        # -- P-wave analysis --
        analyzer = PWaveAnalyzer(fs=fs)
        p_feats = analyzer.analyze(p_waves, clean, seg.beats)
        pm = [{
            "beat_id": pf.beat_id, "onset_sample": pf.onset_sample,
            "offset_sample": pf.offset_sample, "peak_sample": pf.peak_sample,
            "duration_ms": pf.duration_ms, "peak_amplitude": pf.peak_amplitude,
            "area": pf.area, "morphology_score": pf.morphology_score,
            "pr_interval_ms": pf.pr_interval_ms,
        } for pf in p_feats]
        _jwrite(pm, os.path.join(rec_dir, 'p_wave_metrics.json'))

        s = analyzer.summarize(p_feats)
        result.update({
            "p_duration_mean_ms": s.duration_mean_ms,
            "p_duration_std_ms": s.duration_std_ms,
            "p_dispersion_ms": s.dispersion_ms,
            "pr_interval_mean_ms": s.pr_mean_ms,
            "pr_interval_std_ms": s.pr_std_ms,
            "p_amplitude_mean": s.amplitude_mean,
        })

        _jwrite(result, os.path.join(rec_dir, 'summary.json'))

        # -- Per-beat plots --
        for b in seg.beats:
            bid = b.beat_id
            if b.p_onset <= 0 or b.q_onset <= 0 or b.t_offset <= 0:
                continue

            margin = int(0.15 * fs)
            ws = max(0, b.p_onset - margin)
            we = min(n - 1, b.t_offset + margin)
            if we - ws < 30: continue

            # Waveform plot
            fig, ax = plt.subplots(figsize=(12, 4))
            t_win = np.arange(ws, we + 1) / fs
            e_win = clean[ws:we + 1]
            l_win = seg.state_labels[ws:we + 1]

            # State color bands
            from ecg_waveform_extraction.utils.vis import STATE_COLORS
            from ecg_waveform_extraction.hsmm.hsmm_model import STATE_LABELS
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
            yr = yhi - ylo or 1

            for lbl, idx, color in [('P on', b.p_onset, 'green'), ('P off', b.p_offset, 'green'),
                                      ('QRS on', b.q_onset, 'red'), ('QRS off', b.s_offset, 'red'),
                                      ('T off', b.t_offset, 'blue')]:
                if idx > 0:
                    tx = idx / fs
                    ax.axvline(tx, color=color, linestyle='--', linewidth=0.8, alpha=0.7)
                    ax.text(tx, yhi + 0.05*yr, lbl, fontsize=7, color=color, ha='center')

            ax.set_xlim(t_win[0], t_win[-1])
            ax.set_title(f"Record {rec} Beat {bid}  |  P-QRS-T Waveform (UNSEEN TEST)")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
            from matplotlib.patches import Rectangle
            handles = [Rectangle((0,0),1,1,facecolor=STATE_COLORS[s],alpha=0.25,label=s) for s in STATE_LABELS]
            ax.legend(handles=handles, loc='upper right', ncol=9, fontsize=6)
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            fig.savefig(os.path.join(beats_dir, f'beat_{bid:03d}_waveform.png'), dpi=120, bbox_inches='tight')
            plt.close(fig)

            # P-wave detail plot
            pw_match = next((pw for pw in p_waves if pw.beat_id == bid), None)
            if pw_match and pw_match.onset_sample > 0 and pw_match.offset_sample > pw_match.onset_sample:
                pw_on, pw_off = pw_match.onset_sample, pw_match.offset_sample
                pw_peak = pw_match.peak_sample
                pmg = int(0.1 * fs)
                pws, pwe = max(0, pw_on - pmg), min(n - 1, pw_off + pmg)
                fig, ax = plt.subplots(figsize=(7, 3))
                t_pw = np.arange(pws, pwe + 1) / fs
                ax.plot(t_pw, clean[pws:pwe + 1], 'k-', linewidth=1.0)
                p_idx = np.arange(pw_on, pw_off + 1)
                ax.fill_between(p_idx / fs, clean[p_idx], alpha=0.3, color='#4caf50', label='P wave')
                ax.axvline(pw_on / fs, color='green', linestyle='--', linewidth=1.0)
                ax.axvline(pw_off / fs, color='red', linestyle='--', linewidth=1.0)
                if 0 <= pw_peak < n:
                    ax.axvline(pw_peak / fs, color='blue', linestyle=':', linewidth=0.8)
                dur_str = f'{pw_match.duration_ms:.0f}ms'
                mid_t = (pw_on + pw_off) // 2
                ax.annotate('onset', (pw_on/fs, clean[pw_on]), textcoords='offset points', xytext=(-5, -15), fontsize=8, color='green', ha='right')
                ax.annotate('offset', (pw_off/fs, clean[pw_off]), textcoords='offset points', xytext=(5, -15), fontsize=8, color='red', ha='left')
                ax.annotate(dur_str, (mid_t/fs, clean[mid_t]), textcoords='offset points', xytext=(0, 10), fontsize=9, ha='center', fontweight='bold')
                ax.set_title(f"Record {rec} Beat {bid}  |  P-Wave Detail (UNSEEN TEST)")
                ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
                ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
                fig.tight_layout()
                fig.savefig(os.path.join(beats_dir, f'beat_{bid:03d}_p_wave.png'), dpi=120, bbox_inches='tight')
                plt.close(fig)

        # -- Overview plots --
        fig, ax = plt.subplots(figsize=(18, 5))
        plot_time = min(10.0, n / fs)
        plot_segmentation(clean, seg.state_labels, seg.state_names, fs=fs,
                          title=f"Record {rec} — UNSEEN TEST — Trained HSMM",
                          time_range=(0, plot_time), ax=ax)
        fig.savefig(os.path.join(rec_dir, 'segmentation.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

        result["status"] = "ok"
        result["processing_time_sec"] = round(time.time() - t0, 1)

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    return result


# =====================================================================
# Main
# =====================================================================
print(f"{'='*65}")
print(f"  EVALUATION ON UNSEEN TEST RECORDS ONLY")
print(f"{'='*65}")
print(f"  Train (excluded): {len(train_records)} records")
print(f"  Test (evaluated): {len(test_records)} records")
print()

test_results = []
total_beats = 0
total_p_waves = 0
t_start = time.time()

for idx, rec in enumerate(test_records):
    print(f"[{idx+1}/{len(test_records)}] {rec}...", end=" ", flush=True)
    res = eval_record(rec, model)
    test_results.append(res)

    if res["status"] == "ok":
        total_beats += res["n_beats"]
        total_p_waves += res["n_p_waves"]
        se = res.get("r_peak_sensitivity", "N/A")
        ppv = res.get("r_peak_ppv", "N/A")
        pd = res.get("p_duration_mean_ms", "N/A")
        pr = res.get("pr_interval_mean_ms", "N/A")
        print(f"OK  beats={res['n_beats']} P={res['n_p_waves']} "
              f"dur={pd}ms PR={pr}ms Se={se} PPV={ppv} ({res['processing_time_sec']}s)")
    else:
        print(f"FAIL: {res['error']}")

total_time = time.time() - t_start

# =====================================================================
# Aggregate test-only results
# =====================================================================
ok_results = [r for r in test_results if r["status"] == "ok"]

se_vals = [r["r_peak_sensitivity"] for r in ok_results if r.get("r_peak_sensitivity") is not None]
ppv_vals = [r["r_peak_ppv"] for r in ok_results if r.get("r_peak_ppv") is not None]
p_durs = [r["p_duration_mean_ms"] for r in ok_results if r.get("p_duration_mean_ms") is not None]
p_disps = [r["p_dispersion_ms"] for r in ok_results if r.get("p_dispersion_ms") is not None]
prs = [r["pr_interval_mean_ms"] for r in ok_results if r.get("pr_interval_mean_ms") is not None]

summary = {
    "split_info": {
        "total_records": len(all_records),
        "train_records": len(train_records),
        "test_records": len(test_records),
        "train_set": sorted(train_records),
        "test_set": sorted(test_records),
    },
    "test_results": {
        "total_evaluated": len(test_records),
        "processed_ok": len(ok_results),
        "failed": len(test_records) - len(ok_results),
        "total_beats": total_beats,
        "total_p_waves": total_p_waves,
        "total_time_sec": round(total_time, 1),
        "avg_time_sec": round(total_time / max(len(ok_results), 1), 1),
        "sensitivity_mean": round(float(np.mean(se_vals)), 1) if se_vals else None,
        "sensitivity_std": round(float(np.std(se_vals)), 1) if se_vals else None,
        "sensitivity_min": round(float(np.min(se_vals)), 1) if se_vals else None,
        "sensitivity_max": round(float(np.max(se_vals)), 1) if se_vals else None,
        "ppv_mean": round(float(np.mean(ppv_vals)), 1) if ppv_vals else None,
        "ppv_std": round(float(np.std(ppv_vals)), 1) if ppv_vals else None,
        "ppv_min": round(float(np.min(ppv_vals)), 1) if ppv_vals else None,
        "ppv_max": round(float(np.max(ppv_vals)), 1) if ppv_vals else None,
        "p_duration_mean_ms": round(float(np.mean(p_durs)), 2) if p_durs else None,
        "p_duration_std_ms": round(float(np.std(p_durs)), 2) if p_durs else None,
        "p_dispersion_mean_ms": round(float(np.mean(p_disps)), 2) if p_disps else None,
        "pr_interval_mean_ms": round(float(np.mean(prs)), 2) if prs else None,
        "pr_interval_std_ms": round(float(np.std(prs)), 2) if prs else None,
        "per_record": test_results,
    },
}

# Save
with open(os.path.join(OUT_DIR, 'test_only_summary.json'), 'w') as f:
    _jwrite(summary, os.path.join(OUT_DIR, 'test_only_summary.json'))

# =====================================================================
# Final report
# =====================================================================
n_with_ann = len(se_vals)
print(f"\n{'='*65}")
print(f"  TEST-ONLY RESULTS (records NEVER seen during training)")
print(f"{'='*65}")
print(f"  Test records: {len(test_records)}")
print(f"  OK: {len(ok_results)}, Failed: {len(test_records) - len(ok_results)}")
print(f"  Total beats: {total_beats}")
print(f"  Total P-waves: {total_p_waves}")
if n_with_ann > 0:
    print(f"  Records with annotations: {n_with_ann}")
    print(f"  R-peak Sensitivity: {summary['test_results']['sensitivity_mean']}% ± {summary['test_results']['sensitivity_std']}%")
    print(f"  R-peak PPV:         {summary['test_results']['ppv_mean']}% ± {summary['test_results']['ppv_std']}%")
    print(f"  Sensitivity range:  [{summary['test_results']['sensitivity_min']}% – {summary['test_results']['sensitivity_max']}%]")
    print(f"  P-wave duration:    {summary['test_results']['p_duration_mean_ms']} ± {summary['test_results']['p_duration_std_ms']} ms")
    print(f"  P-wave dispersion:  {summary['test_results']['p_dispersion_mean_ms']} ms (mean)")
    print(f"  PR interval:        {summary['test_results']['pr_interval_mean_ms']} ± {summary['test_results']['pr_interval_std_ms']} ms")
print(f"  Total time: {total_time:.0f}s")
print(f"  Output: {OUT_DIR}/")
print(f"  Summary: {os.path.join(OUT_DIR, 'test_only_summary.json')}")
print(f"{'='*65}")
