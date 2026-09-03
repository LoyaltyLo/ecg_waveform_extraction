"""End-to-end pipeline tests on synthetic ECG with known ground truth.

Covers:
- GMM weighted-EM regression (sample_weight actually weights)
- Full segmentation on synthetic ECG (beat count, P onset accuracy)
- HSMM decode structural constraints (durations, transitions)
- R-peak localization against ground truth
- fs threading: D_max cap, full 1 kHz pipeline, prominence P/T refinement
"""
import numpy as np
import pytest

from ecg_waveform_extraction.src.preprocessing import ECGPreprocessor
from ecg_waveform_extraction.src.features import FeatureExtractor
from ecg_waveform_extraction.src.hsmm import HSMMModel, HSMMDecoder, smart_initialize_gmms
from ecg_waveform_extraction.src.hsmm.distributions import GaussianMixtureModel
from ecg_waveform_extraction.src.hsmm.hsmm_model import ALLOWED_TRANSITIONS
from ecg_waveform_extraction.src.segmentation import ECGSegmenter
from ecg_waveform_extraction.src.utils.data_loader import generate_synthetic_ecg
from ecg_waveform_extraction.src.delineation.prominence_stage import (
    ProminenceStage, refine_p_t_boundaries,
)

FS = 250.0
FS_HR = 1000.0  # real aECG sampling rate


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


# ----------------------------------------------------------------------
# fs threading: D_max cap, 1 kHz pipeline, prominence P/T refinement
# ----------------------------------------------------------------------
def test_d_max_cap_scales_with_fs():
    """GLOBAL_D_MAX_SECONDS=2.0: 500 samples at 250 Hz, 2000 at 1 kHz.

    The old GLOBAL_D_MAX=500 clamped TP/ISO to 0.5 s at 1 kHz, forcing
    spurious states on slow rhythms.
    """
    m250 = HSMMModel(fs=250.0)
    m250.initialize_with_priors()
    assert m250._d_max_cap == 500, f"250 Hz cap {m250._d_max_cap} != 500"

    m1000 = HSMMModel(fs=FS_HR)
    m1000.initialize_with_priors()
    assert m1000._d_max_cap == 2000, f"1 kHz cap {m1000._d_max_cap} != 2000"
    # TP prior (mu=200ms, sigma=200ms) must survive uncapped at 1 kHz:
    # mu+4*sigma = 1000 samples > 500 (the old flat cap)
    tp_idx = m1000.state_labels.index("TP")
    assert m1000.D_max[tp_idx] > 500, (
        f"TP D_max {m1000.D_max[tp_idx]} still clamped to the old 250 Hz-era cap")


def test_notch_defaults_to_50hz():
    """Mains frequency is not inferable from fs; default is 50 Hz (CN/EU)."""
    assert ECGPreprocessor(fs=1000.0).notch_freq == 50.0
    assert ECGPreprocessor(fs=360.0, notch_freq=60.0).notch_freq == 60.0


@pytest.fixture(scope="module")
def synthetic_result_1khz():
    """Segment a 10 s synthetic ECG at the real 1 kHz rate."""
    data = generate_synthetic_ecg(fs=FS_HR, duration_sec=10.0, heart_rate=60.0,
                                  noise_std=0.02, random_state=42)
    model = HSMMModel(fs=FS_HR)
    model.initialize_with_priors()
    for g in model.obs_dists:
        g._rng = np.random.RandomState(42)

    clean = ECGPreprocessor(fs=FS_HR).preprocess(data["ecg"])
    feats = FeatureExtractor(fs=FS_HR).extract(clean)
    smart_initialize_gmms(model, feats)

    seg = ECGSegmenter(model=model, fs=FS_HR)
    result = seg.segment(data["ecg"])
    return result, data


def test_segmentation_beat_count_1khz(synthetic_result_1khz):
    result, data = synthetic_result_1khz
    n_true = len(data["true_boundaries"])
    n_det = len(result.beats)
    assert abs(n_det - n_true) <= 1, f"1 kHz: detected {n_det} beats, true {n_true}"


def test_prominence_refines_p_boundaries_1khz(synthetic_result_1khz):
    """Prominence refinement produces valid P/T boundaries at 1 kHz."""
    result, data = synthetic_result_1khz
    r_peaks = [b.r_peak for b in result.beats if b.r_peak > 0]
    assert len(r_peaks) >= 5, f"only {len(r_peaks)} R-peaks at 1 kHz"

    stage = ProminenceStage(FS_HR)
    pbeats = stage.delineate(result.filtered_ecg, r_peaks)
    assert len(pbeats) == len(set(r_peaks)), "output must be beat-aligned"

    n_p = sum(1 for pb in pbeats if pb.p_onset >= 0)
    assert n_p >= len(pbeats) * 0.5, f"only {n_p}/{len(pbeats)} beats have P"
    for pb in pbeats:
        if pb.p_onset >= 0:
            assert pb.p_onset < pb.p_offset < pb.r_peak, (
                f"P window [{pb.p_onset},{pb.p_offset}] not before R={pb.r_peak}")
        if pb.t_onset >= 0:
            assert pb.r_peak < pb.t_onset < pb.t_offset, (
                f"T window [{pb.t_onset},{pb.t_offset}] not after R={pb.r_peak}")

    # ---- In-place write-back: assert on the MUTATED beats, not the wrapper ----
    import copy
    beats = copy.deepcopy(result.beats)

    def _truth_errors(bs):
        truth = data["true_boundaries"]
        errs = []
        for b in bs:
            if b.p_onset < 0 or b.r_peak < 0:
                continue
            i = int(np.argmin([abs(t["R_peak"] - b.r_peak) for t in truth]))
            if abs(truth[i]["R_peak"] - b.r_peak) > int(0.05 * FS_HR):
                continue
            errs.append(abs(b.p_onset - truth[i]["P_onset"]) / FS_HR * 1000.0)
        return errs

    pre_windows = {b.r_peak: (b.p_onset, b.p_offset) for b in beats}
    pre_median = float(np.median(_truth_errors(beats)))

    n_refined = refine_p_t_boundaries(beats, result.filtered_ecg, FS_HR)
    assert n_refined >= 1, "no beat was refined"

    # At least one P window actually moved (guards against silent no-ops)
    moved = sum(1 for b in beats
                if pre_windows.get(b.r_peak, (-1, -1)) != (b.p_onset, b.p_offset))
    assert moved >= 1, "refinement count > 0 but no window changed"

    # Structural invariants on the WRITTEN-BACK windows
    for i, b in enumerate(beats):
        if b.p_source == 'prominence':
            assert b.p_onset < b.p_offset < b.r_peak, (
                f"refined P [{b.p_onset},{b.p_offset}] not before R={b.r_peak}")
            if b.q_onset > 0:
                assert b.p_offset <= b.q_onset, (
                    f"refined P offset {b.p_offset} intrudes into QRS "
                    f"(q_onset={b.q_onset})")
        if b.t_source == 'prominence':
            assert b.r_peak < b.t_onset < b.t_offset, (
                f"refined T [{b.t_onset},{b.t_offset}] not after R={b.r_peak}")
            # T must not run into the next beat
            if i + 1 < len(beats):
                nb = beats[i + 1]
                anchor = next((v for v in (nb.p_onset, nb.q_onset, nb.r_peak)
                               if v > b.r_peak), -1)
                if anchor > b.r_peak:
                    assert b.t_offset <= anchor, (
                        f"refined T offset {b.t_offset} runs into next beat "
                        f"(anchor={anchor})")

    # Accuracy must IMPROVE over the HSMM baseline (the 20 ms gate alone
    # cannot distinguish a working writer from a no-op: HSMM med is ~18 ms)
    post_errors = _truth_errors(beats)
    assert len(post_errors) >= 5, f"only {len(post_errors)} matched beats at 1 kHz"
    post_median = float(np.median(post_errors))
    assert post_median < pre_median, (
        f"refinement did not improve P onsets: {pre_median:.1f} -> {post_median:.1f} ms")
    assert post_median < 12.0, (
        f"1 kHz refined P-onset median error {post_median:.1f} ms >= 12 ms")


def test_prominence_stage_guards():
    """<2 R-peaks must not crash (the package indexes rr[0]/rr[-1])."""
    stage = ProminenceStage(FS_HR)
    ecg = np.zeros(2000)
    assert stage.delineate(ecg, []) == []
    assert stage.delineate(ecg, [500]) == []
    assert refine_p_t_boundaries([], ecg, FS_HR) == 0
