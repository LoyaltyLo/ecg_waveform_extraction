"""Test HSMM pipeline on MIT-BIH real ECG data."""

import sys
sys.path.insert(0, 'c:/LoyaltyLo/PythonProjects/ECG_engineering')

import os
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

# ---- Setup ----
DATA_DIR = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/data'
OUT_DIR = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/output'
os.makedirs(OUT_DIR, exist_ok=True)

RECORD_PATH = os.path.join(DATA_DIR, '100')

# ---- Step 0: Load MIT-BIH 100 ----
print("=== 0. Loading MIT-BIH record 100 ===\n")
try:
    record = wfdb.rdrecord(RECORD_PATH)
    signal_raw = record.p_signal
    if signal_raw.ndim > 1:
        signal_raw = signal_raw[:, 0]  # MLII lead
    signal = signal_raw.astype(np.float64)
    fs = record.fs
    print(f"  Record: {len(signal)} samples @ {fs}Hz ({len(signal)/fs:.1f}s)")
    print(f"  Leads: {record.sig_name}")
    using_real = True
except Exception as e:
    print(f"  FAIL: {e}")
    print("  Falling back to synthetic ECG...")
    from ecg_waveform_extraction.utils.data_loader import generate_synthetic_ecg
    data = generate_synthetic_ecg(fs=250.0, duration_sec=30.0, heart_rate=72.0,
                                   noise_std=0.02, random_state=123)
    signal = data['ecg']
    fs = data['fs']
    using_real = False

# Use a segment (first 2 minutes for speed)
max_duration = 15.0  # seconds (keep reasonable for Viterbi decode time)
max_samples = int(max_duration * fs)
if len(signal) > max_samples:
    signal = signal[:max_samples]
print(f"  Processing: {len(signal)} samples ({len(signal)/fs:.1f}s)")

# ---- Step 1: Annotations ----
print("\n=== 1. Loading annotations ===")
try:
    ann = wfdb.rdann(RECORD_PATH, 'atr')
    ann_samples = np.asarray(ann.sample, dtype=int)
    ann_symbols = np.asarray(ann.symbol)

    # Filter to beats only (exclude rhythm change markers '+')
    is_beat = ann_symbols != '+'
    beat_samples = ann_samples[is_beat]
    beat_symbols = ann_symbols[is_beat]

    # Also restrict to our time window
    beat_mask = beat_samples < len(signal)
    beat_samples = beat_samples[beat_mask]
    beat_symbols = beat_symbols[beat_mask]

    print(f"  {len(beat_samples)} beats in window")
    syms_unique, syms_counts = np.unique(beat_symbols, return_counts=True)
    for s, c in zip(syms_unique, syms_counts):
        print(f"    {s}: {c}")
    print(f"  First 10: {list(zip(beat_symbols[:10], beat_samples[:10]))}")
except Exception as e:
    print(f"  No annotations: {e}")
    beat_samples = None

# ---- Step 2: Preprocess ----
print("\n=== 2. Preprocessing ===")
prep = ECGPreprocessor(fs=fs)
clean = prep.preprocess(signal)
print(f"  Clean: mean={clean.mean():.6f}, std={clean.std():.6f}")

# ---- Step 3: Extract features ----
print("\n=== 3. Extracting features ===")
fe = FeatureExtractor(fs=fs)
features = fe.extract(clean)
print(f"  Features: {features.shape}")

# ---- Step 4: Build and initialize model ----
print("\n=== 4. Building HSMM model ===")
model = HSMMModel(fs=fs)
model.initialize_with_priors()

T = len(features)
for i in range(model.n_states):
    start = i * T // 9
    end = min((i + 1) * T // 9, T)
    seg = features[start:end]
    if len(seg) > model.n_gmm_components:
        try:
            model.obs_dists[i].fit(seg, max_iter=30)
        except:
            pass
print(f"  Model: {model}")

# ---- Step 5: HSMM Viterbi decoding ----
print("\n=== 5. HSMM Viterbi decoding ===")
segmenter = ECGSegmenter(preprocessor=prep, feature_extractor=fe, model=model, fs=fs)
result = segmenter.segment(signal)

n_state_types = len(set(lbl for lbl in result.state_labels if lbl >= 0))
print(f"  {len(result.beats)} beats, {n_state_types} state types")
print(f"  Log-likelihood: {result.log_likelihood:.1f}")

# R-peak comparison
if beat_samples is not None and len(beat_samples) > 0:
    hsmm_r = np.array([b.r_peak for b in result.beats if b.r_peak > 0], dtype=int)

    print(f"\n  === R-peak Comparison ===")
    print(f"  MIT-BIH annotations: {len(beat_samples)} beats")
    print(f"  HSMM detected: {len(hsmm_r)} beats")

    # Match within 150ms tolerance
    tol = int(0.15 * fs)

    # Sensitivity: fraction of annotated beats matched by HSMM
    matched_se = 0
    for ar in beat_samples:
        ar_int = int(ar)
        if np.any(np.abs(hsmm_r - ar_int) <= tol):
            matched_se += 1
    se = matched_se / len(beat_samples) * 100
    print(f"  Sensitivity: {matched_se}/{len(beat_samples)} ({se:.1f}%)")

    # PPV: fraction of HSMM beats matched by annotation
    matched_ppv = 0
    for hr in hsmm_r:
        if np.any(np.abs(beat_samples - hr) <= tol):
            matched_ppv += 1
    ppv = matched_ppv / len(hsmm_r) * 100
    print(f"  PPV: {matched_ppv}/{len(hsmm_r)} ({ppv:.1f}%)")

for b in result.beats[:5]:
    print(f"  Beat {b.beat_id}: P=[{b.p_onset}:{b.p_offset}], "
          f"Q=[{b.q_onset}], R={b.r_peak}, S=[{b.s_offset}], "
          f"T=[{b.t_onset}:{b.t_offset}]")

# ---- Step 6: P-wave extraction ----
print("\n=== 6. P-wave extraction ===")
p_extractor = PWaveExtractor(fs=fs)
p_waves = p_extractor.extract(result)
print(f"  {len(p_waves)} P-waves found")
for pw in p_waves[:5]:
    print(f"  Beat {pw.beat_id}: onset={pw.onset_sample}, offset={pw.offset_sample}, "
          f"dur={pw.duration_ms:.1f}ms, conf={pw.confidence:.2f}")

# ---- Step 7: P-wave analysis ----
print("\n=== 7. P-wave analysis ===")
analyzer = PWaveAnalyzer(fs=fs)
p_features = analyzer.analyze(p_waves, result.filtered_ecg, result.beats)
summary = analyzer.summarize(p_features)
print(f"  N beats: {summary.n_beats}")
print(f"  P duration: {summary.duration_mean_ms:.1f} ± {summary.duration_std_ms:.1f} ms")
print(f"  P dispersion: {summary.dispersion_ms:.1f} ms")
print(f"  Amplitude mean: {summary.amplitude_mean:.4f}")
if summary.pr_mean_ms:
    print(f"  PR interval: {summary.pr_mean_ms:.1f} ± {summary.pr_std_ms:.1f} ms")
if summary.flagged_beats:
    print(f"  Flagged: {summary.flagged_beats}")

# ---- Step 8: Visualize ----
print("\n=== 8. Generating plots ===")
tag = "mitdb100" if using_real else "synthetic"

# Segmentation (first 10 seconds)
fig, ax = plt.subplots(figsize=(18, 5))
secs = min(10.0, len(signal) / fs)
plot_segmentation(result.filtered_ecg, result.state_labels, result.state_names,
                  fs=fs, title=f"HSMM Segmentation - {tag.upper()} (first {secs:.0f}s)",
                  time_range=(0, secs), ax=ax)
fig.savefig(os.path.join(OUT_DIR, f'segmentation_{tag}.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: output/segmentation_{tag}.png")

# P-wave detail
if p_waves:
    pw = p_waves[0]
    fig, ax = plt.subplots(figsize=(8, 3))
    plot_p_wave_detail(result.filtered_ecg, pw.onset_sample, pw.offset_sample,
                       fs=fs, title=f"P-Wave Detail ({tag.upper()}, Beat {pw.beat_id})", ax=ax)
    fig.savefig(os.path.join(OUT_DIR, f'p_wave_{tag}.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: output/p_wave_{tag}.png")

plt.close('all')

print(f"\n{'='*60}")
print(f"Test Complete: {tag}")
print(f"  {'Real MIT-BIH 100' if using_real else 'Synthetic ECG'}")
print(f"  {len(signal)/fs:.1f}s @ {fs}Hz")
print(f"  Beats: {len(result.beats)}, P-waves: {len(p_waves)}")
