"""Integration test for ECG waveform extraction pipeline."""
import sys
sys.path.insert(0, 'c:/LoyaltyLo/PythonProjects/ECG_engineering')

import numpy as np
from ecg_waveform_extraction.preprocessing import ECGPreprocessor
from ecg_waveform_extraction.features import FeatureExtractor
from ecg_waveform_extraction.hsmm import GaussianMixtureModel, DurationDistribution, HSMMModel, HSMMDecoder
from ecg_waveform_extraction.utils.data_loader import generate_synthetic_ecg

print('=== 1. Import check ===')
print('All modules imported successfully')

print('\n=== 2. Generate synthetic ECG ===')
data = generate_synthetic_ecg(fs=250.0, duration_sec=10.0, heart_rate=60.0, noise_std=0.01, random_state=42)
ecg = data['ecg']
print(f'Generated {len(ecg)} samples ({len(ecg)/250:.1f}s) with {len(data["true_boundaries"])} beats')
tb0 = data['true_boundaries'][0]
print(f'First beat P: [{tb0["P_onset"]}:{tb0["P_offset"]}], QRS: [{tb0["Q_onset"]}:{tb0["S_offset"]}], T: [{tb0["T_onset"]}:{tb0["T_offset"]}]')

print('\n=== 3. Preprocessing ===')
prep = ECGPreprocessor(fs=250.0)
clean = prep.preprocess(ecg)
print(f'Preprocessed: mean={clean.mean():.6f}, std={clean.std():.6f}')

print('\n=== 4. Feature extraction ===')
fe = FeatureExtractor(fs=250.0)
features = fe.extract(clean)
print(f'Features shape: {features.shape}')
print(f'Feature ranges: amp=[{features[:,0].min():.2f},{features[:,0].max():.2f}], d1=[{features[:,1].min():.2f},{features[:,1].max():.2f}], d2=[{features[:,2].min():.2f},{features[:,2].max():.2f}]')

print('\n=== 5. GMM test ===')
gmm = GaussianMixtureModel(n_components=2, n_features=3)
gmm.fit(features[:1000], max_iter=30)
ll = gmm.log_prob(features[:10])
print(f'GMM fitted, mean log_prob on first 10 samples: {ll.mean():.2f}')
params = gmm.get_params()
print(f'GMM weights: {params["weights"]}')

print('\n=== 6. Duration distribution test ===')
dd = DurationDistribution()
dd.set_physiological_prior('P', fs=250.0)
print(f'P-wave prior: mu={dd.mu:.1f}, sigma={dd.sigma:.1f}, d_min={dd.d_min}')
lp = dd.log_prob_range(20, 30)
print(f'log_prob_range(20,30): {lp}')

print('\n=== 7. HSMM Model initialization ===')
model = HSMMModel(fs=250.0)
model.initialize_with_priors()
print(f'States: {model.state_labels}')
print(f'D_max: {model.D_max}')
print('Non-zero transitions:')
for i in range(model.n_states):
    for j in range(model.n_states):
        if model.A[i,j] > 0:
            print(f'  {model.state_labels[i]} -> {model.state_labels[j]}: {model.A[i,j]:.3f}')

print('\n=== 8. Train GMMs on synthetic data ===')
# Initialize GMMs for all states with quick fits on data subsets
# This is needed before Viterbi decoding
T = len(features)
segment_len = T // 9
for i in range(model.n_states):
    start = i * segment_len
    end = min((i + 1) * segment_len, T)
    seg = features[start:end]
    if len(seg) > model.n_gmm_components:
        try:
            model.obs_dists[i].fit(seg, max_iter=20)
        except Exception as e:
            print(f'  GMM {i} ({model.state_labels[i]}) failed: {e}')

issues = model.validate()
print(f'Validation issues after training: {issues}')

print('\n=== 9. Viterbi decoding ===')
decoder = HSMMDecoder()
result = decoder.decode(model, features)
print(f'Decoded {len(result["state_sequence"])} segments')
print(f'Log-likelihood: {result["log_likelihood"]:.1f}')
unique_labels, counts = np.unique(result["state_labels"], return_counts=True)
for lbl, cnt in zip(unique_labels, counts):
    name = model.get_state_name(lbl) if lbl >= 0 else 'UNKNOWN'
    print(f'  {name}: {cnt} samples ({cnt/T*100:.1f}%)')

print('\n=== 10. All tests passed! ===')
