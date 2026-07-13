"""Batch process all records through the trained HSMM pipeline.

Uses the pre-trained HSMM model from models/hsmm_trained.npz.
Saves per-record: segmentation, P-waves, metrics, plots, waveform data.
"""

import sys
sys.path.insert(0, 'c:/LoyaltyLo/PythonProjects/ECG_engineering')

import os
import json
import time
import traceback
import numpy as np

from ecg_waveform_extraction.preprocessing import ECGPreprocessor
from ecg_waveform_extraction.features import FeatureExtractor
from ecg_waveform_extraction.hsmm import HSMMModel
from ecg_waveform_extraction.segmentation import ECGSegmenter
from ecg_waveform_extraction.extraction import PWaveExtractor, PWaveAnalyzer
from ecg_waveform_extraction.utils.vis import plot_segmentation, plot_p_wave_detail
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =====================================================================
# Config
# =====================================================================
DATA_DIR = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/data'
OUTPUT_DIR = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/output_trained'
MODEL_PATH = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/models/hsmm_trained.npz'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_DURATION_SEC = 15.0
MAX_SAMPLES = 10000


# =====================================================================
# Helpers
# =====================================================================
def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# =====================================================================
# Process one record
# =====================================================================
def process_record(rec_name, model, decoder, p_extractor, analyzer):
    import wfdb

    rec_dir = os.path.join(OUTPUT_DIR, rec_name)
    os.makedirs(rec_dir, exist_ok=True)

    record_path = os.path.join(DATA_DIR, rec_name)
    result = {
        "record": rec_name, "status": "error", "error": None,
        "n_samples": 0, "fs": None, "n_beats": 0, "n_p_waves": 0,
        "r_peak_sensitivity": None, "r_peak_ppv": None,
        "p_duration_mean_ms": None, "p_duration_std_ms": None,
        "p_dispersion_ms": None, "pr_interval_mean_ms": None,
        "processing_time_sec": 0.0,
    }

    t0 = time.time()

    try:
        # Load
        if not os.path.exists(record_path + ".hea"):
            result["error"] = "No .hea file"
            return result

        record = wfdb.rdrecord(record_path)
        sig = record.p_signal
        if sig.ndim > 1:
            sig = sig[:, 0]
        sig = sig.astype(np.float64)
        fs = float(record.fs)
        max_n = min(int(MAX_DURATION_SEC * fs), MAX_SAMPLES)
        if len(sig) > max_n:
            sig = sig[:max_n]
        n = len(sig)
        result["n_samples"] = n
        result["fs"] = fs

        # Save raw
        np.save(os.path.join(rec_dir, "raw_ecg.npy"), sig)

        # Annotations
        ann_r = None
        try:
            ann = wfdb.rdann(record_path, 'atr')
            a_s = np.asarray(ann.sample)[np.asarray(ann.symbol) != '+']
            ann_r = a_s[a_s < n]
        except Exception:
            pass

        # Preprocess
        prep = ECGPreprocessor(fs=fs)
        clean = prep.preprocess(sig)
        np.save(os.path.join(rec_dir, "filtered_ecg.npy"), clean)

        # Features
        fe = FeatureExtractor(fs=fs)
        features = fe.extract(clean)

        # Segment with trained model
        segmenter = ECGSegmenter(preprocessor=prep, feature_extractor=fe,
                                 model=model, fs=fs)
        seg_result = segmenter.segment(sig)

        np.save(os.path.join(rec_dir, "state_labels.npy"), seg_result.state_labels)
        n_beats = len(seg_result.beats)
        result["n_beats"] = n_beats

        # R-peak comparison
        if ann_r is not None and len(ann_r) > 0 and n_beats > 0:
            hsmm_r = np.array([b.r_peak for b in seg_result.beats if b.r_peak > 0], dtype=int)
            tol = int(0.15 * fs)
            if len(hsmm_r) > 0:
                matched_se = sum(1 for ar in ann_r if np.any(np.abs(hsmm_r - int(ar)) <= tol))
                result["r_peak_sensitivity"] = round(matched_se / len(ann_r) * 100, 1)
                matched_ppv = sum(1 for hr in hsmm_r if np.any(np.abs(ann_r - hr) <= tol))
                result["r_peak_ppv"] = round(matched_ppv / len(hsmm_r) * 100, 1)

        # Save segmentation
        seg_data = [{k: getattr(b, k) for k in [
            "beat_id","iso_start","p_onset","p_offset","pr_start",
            "q_onset","r_peak","s_offset","st_start","t_onset",
            "t_offset","tp_start"]} for b in seg_result.beats]
        with open(os.path.join(rec_dir, "segmentation.json"), 'w') as f:
            json.dump(_json_safe(seg_data), f, indent=2)

        # P-wave extraction
        p_waves = p_extractor.extract(seg_result)
        result["n_p_waves"] = len(p_waves)

        pw_data = [{
            "beat_id": pw.beat_id, "onset_sample": pw.onset_sample,
            "offset_sample": pw.offset_sample, "peak_sample": pw.peak_sample,
            "duration_ms": round(pw.duration_ms, 2),
            "confidence": round(pw.confidence, 4),
        } for pw in p_waves]
        with open(os.path.join(rec_dir, "p_waves.json"), 'w') as f:
            json.dump(_json_safe(pw_data), f, indent=2)

        # P-wave samples
        if p_waves:
            pw_samp = {}
            for pw in p_waves:
                if pw.onset_sample >= 0 and pw.offset_sample >= pw.onset_sample:
                    pw_samp[str(pw.beat_id)] = clean[pw.onset_sample:pw.offset_sample + 1]
            np.savez(os.path.join(rec_dir, "p_wave_samples.npz"), **pw_samp)

        # P-wave analysis
        p_features = analyzer.analyze(p_waves, clean, seg_result.beats)
        pm = [{
            "beat_id": pf.beat_id, "onset_sample": pf.onset_sample,
            "offset_sample": pf.offset_sample, "peak_sample": pf.peak_sample,
            "duration_ms": pf.duration_ms, "peak_amplitude": pf.peak_amplitude,
            "area": pf.area, "morphology_score": pf.morphology_score,
            "pr_interval_ms": pf.pr_interval_ms,
        } for pf in p_features]
        with open(os.path.join(rec_dir, "p_wave_metrics.json"), 'w') as f:
            json.dump(_json_safe(pm), f, indent=2)

        summary = analyzer.summarize(p_features)
        result.update({
            "p_duration_mean_ms": summary.duration_mean_ms,
            "p_duration_std_ms": summary.duration_std_ms,
            "p_dispersion_ms": summary.dispersion_ms,
            "pr_interval_mean_ms": summary.pr_mean_ms,
            "pr_interval_std_ms": summary.pr_std_ms,
            "p_amplitude_mean": summary.amplitude_mean,
        })

        with open(os.path.join(rec_dir, "summary.json"), 'w') as f:
            json.dump(_json_safe(result), f, indent=2)

        # Plots
        try:
            fig, ax = plt.subplots(figsize=(18, 5))
            plot_time = min(10.0, n / fs)
            plot_segmentation(clean, seg_result.state_labels,
                              seg_result.state_names, fs=fs,
                              title=f"Record {rec_name} - Trained HSMM",
                              time_range=(0, plot_time), ax=ax)
            fig.savefig(os.path.join(rec_dir, "segmentation.png"), dpi=120, bbox_inches='tight')
            plt.close(fig)

            if p_waves:
                pw = p_waves[0]
                fig, ax = plt.subplots(figsize=(8, 3))
                plot_p_wave_detail(clean, pw.onset_sample, pw.offset_sample,
                                   fs=fs, title=f"Record {rec_name} - P-Wave (Beat {pw.beat_id})",
                                   ax=ax)
                fig.savefig(os.path.join(rec_dir, "p_wave_detail.png"), dpi=120, bbox_inches='tight')
                plt.close(fig)
        except Exception:
            pass

        result["status"] = "ok"

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    result["processing_time_sec"] = round(time.time() - t0, 1)
    return result


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 60)
    print("  Batch Process with TRAINED HSMM Model")
    print("=" * 60)

    # Load trained model
    print(f"\nLoading trained model: {MODEL_PATH}")
    model = HSMMModel.load(MODEL_PATH)
    print(f"  Model: {model}")

    # Shared components
    decoder = None  # not needed separately; segmenter creates its own
    p_extractor = PWaveExtractor(fs=model.fs)
    analyzer_instance = PWaveAnalyzer(fs=model.fs)

    # Find records
    records = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.endswith('.hea'):
            rec = fname[:-4]
            if os.path.exists(os.path.join(DATA_DIR, rec + '.dat')):
                records.append(rec)

    print(f"\nRecords: {len(records)}")
    print(f"Output: {OUTPUT_DIR}/<record>/")
    print()

    all_results = []
    ok_count = 0
    t_start = time.time()

    for idx, rec in enumerate(records):
        print(f"[{idx+1}/{len(records)}] {rec}...", end=" ", flush=True)
        res = process_record(rec, model, decoder, p_extractor, analyzer_instance)
        all_results.append(res)

        if res["status"] == "ok":
            ok_count += 1
            se = res.get("r_peak_sensitivity", "N/A")
            ppv = res.get("r_peak_ppv", "N/A")
            pd = res.get("p_duration_mean_ms", "N/A")
            print(f"OK  beats={res['n_beats']} P={res['n_p_waves']} "
                  f"dur={pd}ms Se={se} PPV={ppv} ({res['processing_time_sec']}s)")
        else:
            print(f"FAIL: {res['error']}")

    total_time = time.time() - t_start

    # Global summary
    ok_results = [r for r in all_results if r["status"] == "ok"]

    global_summary = {
        "model": MODEL_PATH,
        "total_records": len(records),
        "processed_ok": ok_count,
        "total_time_sec": round(total_time, 1),
        "avg_time_sec": round(total_time / max(ok_count, 1), 1),
        "records": all_results,
    }

    if ok_results:
        se_vals = [r["r_peak_sensitivity"] for r in ok_results if r.get("r_peak_sensitivity") is not None]
        ppv_vals = [r["r_peak_ppv"] for r in ok_results if r.get("r_peak_ppv") is not None]
        p_durs = [r["p_duration_mean_ms"] for r in ok_results if r.get("p_duration_mean_ms") is not None]
        prs = [r["pr_interval_mean_ms"] for r in ok_results if r.get("pr_interval_mean_ms") is not None]
        beats = [r["n_beats"] for r in ok_results]
        pws = [r["n_p_waves"] for r in ok_results]

        global_summary["aggregates"] = {
            "total_beats": sum(beats),
            "total_p_waves": sum(pws),
            "sensitivity_mean": round(np.mean(se_vals), 1) if se_vals else None,
            "sensitivity_std": round(np.std(se_vals), 1) if se_vals else None,
            "ppv_mean": round(np.mean(ppv_vals), 1) if ppv_vals else None,
            "ppv_std": round(np.std(ppv_vals), 1) if ppv_vals else None,
            "p_duration_mean_ms": round(np.mean(p_durs), 2) if p_durs else None,
            "p_duration_std_ms": round(np.std(p_durs), 2) if p_durs else None,
            "pr_interval_mean_ms": round(np.mean(prs), 2) if prs else None,
            "pr_interval_std_ms": round(np.std(prs), 2) if prs else None,
        }

    with open(os.path.join(OUTPUT_DIR, "global_summary.json"), 'w') as f:
        json.dump(_json_safe(global_summary), f, indent=2)

    # Report
    print(f"\n{'=' * 60}")
    print(f"  BATCH COMPLETE (TRAINED MODEL)")
    print(f"{'=' * 60}")
    print(f"  Records: {ok_count}/{len(records)} OK")
    print(f"  Time: {total_time:.1f}s ({total_time/max(ok_count,1):.1f}s avg)")
    if ok_results and "aggregates" in global_summary:
        a = global_summary["aggregates"]
        print(f"  Total beats: {a['total_beats']}, P-waves: {a['total_p_waves']}")
        print(f"  Sensitivity: {a['sensitivity_mean']}% ± {a['sensitivity_std']}%")
        print(f"  PPV: {a['ppv_mean']}% ± {a['ppv_std']}%")
        print(f"  P-wave duration: {a['p_duration_mean_ms']} ± {a['p_duration_std_ms']} ms")
        print(f"  PR interval: {a['pr_interval_mean_ms']} ± {a['pr_interval_std_ms']} ms")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
