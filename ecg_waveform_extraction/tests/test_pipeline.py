"""End-to-end pipeline tests on synthetic ECG with known ground truth.

Covers:
- GMM weighted-EM regression (sample_weight actually weights)
- Full segmentation on synthetic ECG (beat count, P onset accuracy)
- HSMM decode structural constraints (durations, transitions)
- R-peak localization against ground truth
"""
import numpy as np
import pytest

from ecg_waveform_extraction.preprocessing import ECGPreprocessor
from ecg_waveform_extraction.features import FeatureExtractor
from ecg_waveform_extraction.hsmm import HSMMModel, HSMMDecoder, smart_initialize_gmms
from ecg_waveform_extraction.hsmm.distributions import GaussianMixtureModel
from ecg_waveform_extraction.hsmm.hsmm_model import ALLOWED_TRANSITIONS
from ecg_waveform_extraction.segmentation import ECGSegmenter
from ecg_waveform_extraction.utils.data_loader import generate_synthetic_ecg

FS = 250.0


# ----------------------------------------------------------------------
# A1 regression: weighted GMM EM must honor sample_weight
# ----------------------------------------------------------------------
def test_gmm_weighted_em_respects_sample_weight():
    """Two clusters; heavy weight on the right one -> fitted mean near it."""
    rng = np.random.RandomState(0)
    left = rng.normal(0.0, 0.1, size=(100, 1))
    right = rng.normal(10.0, 0.1, size=(100, 1))
    X = np.vstack([left, right])
    # 1e6 : 1 weight ratio — mean must be pulled to ~10
    w = np.concatenate([np.full(100, 1e-6), np.ones(100)])

    gmm = GaussianMixtureModel(n_components=1, n_features=1, random_state=0)
    gmm.fit(X, max_iter=100, tol=1e-6, sample_weight=w)

    assert gmm.means[0, 0] > 9.0, (
        f"weighted mean should be near 10, got {gmm.means[0, 0]:.3f} — "
        "sample_weight is being ignored (row re-normalization bug?)"
    )


# ----------------------------------------------------------------------
# Shared segmentation fixture
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_result():
    """Segment a 10s synthetic ECG; returns (SegmentResult, truth dict)."""
    data = generate_synthetic_ecg(fs=FS, duration_sec=10.0, heart_rate=60.0,
                                  noise_std=0.02, random_state=42)
    model = HSMMModel(fs=FS)
    model.initialize_with_priors()
    # Deterministic GMM init for reproducible tests
    for g in model.obs_dists:
        g._rng = np.random.RandomState(42)

    prep = ECGPreprocessor(fs=FS)
    clean = prep.preprocess(data["ecg"])
    feats = FeatureExtractor(fs=FS).extract(clean)
    smart_initialize_gmms(model, feats)

    seg = ECGSegmenter(model=model, fs=FS)
    result = seg.segment(data["ecg"])
    return result, data


def test_segmentation_beat_count(synthetic_result):
    result, data = synthetic_result
    n_true = len(data["true_boundaries"])
    n_det = len(result.beats)
    assert abs(n_det - n_true) <= 1, f"detected {n_det} beats, true {n_true}"


def test_segmentation_p_onset_accuracy(synthetic_result):
    result, data = synthetic_result
    truth = data["true_boundaries"]
    errors = []
    for beat in result.beats:
        if beat.p_onset < 0 or beat.r_peak < 0:
            continue
        # match to nearest true beat by R peak
        i = int(np.argmin([abs(t["R_peak"] - beat.r_peak) for t in truth]))
        if abs(truth[i]["R_peak"] - beat.r_peak) > int(0.05 * FS):
            continue  # no plausible match
        errors.append(abs(beat.p_onset - truth[i]["P_onset"]) / FS * 1000.0)
    assert len(errors) >= 5, f"only {len(errors)} matched beats"
    med = float(np.median(errors))
    assert med < 20.0, f"median P-onset error {med:.1f} ms >= 20 ms"


def test_decode_structural_constraints(synthetic_result):
    """Every decoded segment: duration >= d_min, transitions in topology."""
    # rebuild the decode directly to inspect segments
    data = generate_synthetic_ecg(fs=FS, duration_sec=10.0, heart_rate=60.0,
                                  noise_std=0.02, random_state=42)
    m = HSMMModel(fs=FS)
    m.initialize_with_priors()
    for g in m.obs_dists:
        g._rng = np.random.RandomState(42)
    clean = ECGPreprocessor(fs=FS).preprocess(data["ecg"])
    feats = FeatureExtractor(fs=FS).extract(clean)
    smart_initialize_gmms(m, feats)
    dec = HSMMDecoder().decode(m, feats)
    segs = dec["state_sequence"]
    assert len(segs) >= 5
    for (state, start, end) in segs:
        dur = end - start + 1
        assert dur >= m.dur_dists[state].d_min, (
            f"state {state} duration {dur} < d_min {m.dur_dists[state].d_min}")
    for (s0, _, _), (s1, _, _) in zip(segs, segs[1:]):
        assert (s0, s1) in ALLOWED_TRANSITIONS, f"illegal transition {s0}->{s1}"


def test_r_peak_accuracy(synthetic_result):
    result, data = synthetic_result
    truth = data["true_boundaries"]
    errors = []
    for beat in result.beats:
        if beat.r_peak < 0:
            continue
        i = int(np.argmin([abs(t["R_peak"] - beat.r_peak) for t in truth]))
        err = abs(truth[i]["R_peak"] - beat.r_peak)
        if err <= int(0.05 * FS):
            errors.append(err)
    assert len(errors) >= 5, f"only {len(errors)} matched beats"
    med = float(np.median(errors))
    assert med <= 2.0, f"median R-peak error {med:.1f} samples > 2"
