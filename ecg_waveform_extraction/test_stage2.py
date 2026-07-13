"""Stage 2 integration test: segmenter -> P-wave extraction -> analysis."""
import sys
sys.path.insert(0, 'c:/LoyaltyLo/PythonProjects/ECG_engineering')

import numpy as np
from ecg_waveform_extraction.preprocessing import ECGPreprocessor
from ecg_waveform_extraction.features import FeatureExtractor
from ecg_waveform_extraction.hsmm import HSMMModel, HSMMDecoder
from ecg_waveform_extraction.segmentation import ECGSegmenter
from ecg_waveform_extraction.extraction import PWaveExtractor, PWaveAnalyzer
from ecg_waveform_extraction.utils.data_loader import generate_synthetic_ecg

print('=== Stage 2 Integration Test ===\n')

# 1. Generate data
data = generate_synthetic_ecg(fs=250.0, duration_sec=15.0, heart_rate=60.0, noise_std=0.01, random_state=42)
ecg = data['ecg']
print(f'Generated {len(ecg)} samples, {len(data["true_boundaries"])} beats')

# 2. Setup model
prep = ECGPreprocessor(fs=250.0)
fe = FeatureExtractor(fs=250.0)
model = HSMMModel(fs=250.0)
model.initialize_with_priors()

# Train GMMs quickly on data subsets
clean = prep.preprocess(ecg)
features = fe.extract(clean)
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

# 3. Segment
segmenter = ECGSegmenter(preprocessor=prep, feature_extractor=fe, model=model, fs=250.0)
result = segmenter.segment(ecg)

print(f'Segmentation: {len(result.beats)} beats detected')
print(f'Log-likelihood: {result.log_likelihood:.1f}')
for b in result.beats[:3]:
    print(f'  Beat {b.beat_id}: P=[{b.p_onset}:{b.p_offset}], Q=[{b.q_onset}], R=[{b.r_peak}], S=[{b.s_offset}], T=[{b.t_onset}:{b.t_offset}]')

# 4. P-wave extraction
p_extractor = PWaveExtractor(fs=250.0)
p_waves = p_extractor.extract(result)
print(f'\nP-wave extraction: {len(p_waves)} P-waves found')
for pw in p_waves[:3]:
    print(f'  Beat {pw.beat_id}: onset={pw.onset_sample}, offset={pw.offset_sample}, '
          f'dur={pw.duration_ms:.1f}ms, peak={pw.peak_sample}, conf={pw.confidence:.2f}')

# 5. P-wave analysis
analyzer = PWaveAnalyzer(fs=250.0)
p_features = analyzer.analyze(p_waves, result.filtered_ecg, result.beats)
print(f'\nP-wave analysis: {len(p_features)} features computed')
for pf in p_features[:3]:
    print(f'  Beat {pf.beat_id}: dur={pf.duration_ms:.1f}ms, amp={pf.peak_amplitude:.4f}, '
          f'area={pf.area:.4f}, PR={pf.pr_interval_ms}ms')

summary = analyzer.summarize(p_features)
print(f'\nSummary:')
print(f'  N beats: {summary.n_beats}')
print(f'  Duration: {summary.duration_mean_ms:.1f} ± {summary.duration_std_ms:.1f} ms')
print(f'  Dispersion: {summary.dispersion_ms:.1f} ms')
print(f'  Amplitude mean: {summary.amplitude_mean:.4f}')
print(f'  Flagged: {summary.flagged_beats}')

print('\n=== Stage 2 test passed! ===')
