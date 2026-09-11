"""Tests for the self-contained classical DSP delineator (DSPECGSegmenter).

Mirrors the test_pipeline.py contract on the same synthetic ECG:
- beat count, R-peak and P-onset accuracy vs ground truth
- structural invariants on every beat
- polarity freedom (inverted record still delineated)
- degenerate-input guards (never an exception, empty beats)
- SegmentResult contract (state labels, features, read-only input)
- LimbLeadProcessor integration flag (default off)
"""
import numpy as np
import pytest

from ecg_waveform_extraction.src.segmentation import (
    DSPECGSegmenter, DSPDelineationConfig,
)
from ecg_waveform_extraction.src.segmentation.segmenter import SegmentResult
from ecg_waveform_extraction.src.limb_lead_processor import LimbLeadProcessor
from ecg_waveform_extraction.src.utils.data_loader import generate_synthetic_ecg

FS = 250.0
FS_HR = 1000.0  # real aECG sampling rate


# ----------------------------------------------------------------------
# Shared fixtures: one segmentation per sampling rate
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def dsp_result_250():
    data = generate_synthetic_ecg(fs=FS, duration_sec=10.0, heart_rate=60.0,
                                  noise_std=0.02, random_state=42)
    result = DSPECGSegmenter(fs=FS).segment(data["ecg"])
    return result, data


@pytest.fixture(scope="module")
def dsp_result_1khz():
    data = generate_synthetic_ecg(fs=FS_HR, duration_sec=10.0, heart_rate=60.0,
                                  noise_std=0.02, random_state=42)
    result = DSPECGSegmenter(fs=FS_HR).segment(data["ecg"])
    return result, data


def _match_errors(beats, truth, fs, attr, truth_key):
    """Per-beat |beat.attr - truth[truth_key]| in ms, matched by nearest R."""
    errors = []
    for b in beats:
        if getattr(b, attr) < 0 or b.r_peak < 0:
            continue
        i = int(np.argmin([abs(t["R_peak"] - b.r_peak) for t in truth]))
        if abs(truth[i]["R_peak"] - b.r_peak) > int(0.05 * fs):
            continue  # no plausible match
        errors.append(abs(getattr(b, attr) - truth[i][truth_key]) / fs * 1000.0)
    return errors


# ----------------------------------------------------------------------
# 1. Beat count
# ----------------------------------------------------------------------
def test_dsp_beat_count_250(dsp_result_250):
    result, data = dsp_result_250
    n_true = len(data["true_boundaries"])
    assert abs(len(result.beats) - n_true) <= 1, (
        f"250 Hz: detected {len(result.beats)} beats, true {n_true}")


def test_dsp_beat_count_1khz(dsp_result_1khz):
    result, data = dsp_result_1khz
    n_true = len(data["true_boundaries"])
    assert abs(len(result.beats) - n_true) <= 1, (
        f"1 kHz: detected {len(result.beats)} beats, true {n_true}")


# ----------------------------------------------------------------------
# 2. R-peak accuracy
# ----------------------------------------------------------------------
@pytest.mark.parametrize("fixture_name,fs",
                         [("dsp_result_250", FS), ("dsp_result_1khz", FS_HR)])
def test_dsp_r_peak_accuracy(fixture_name, fs, request):
    result, data = request.getfixturevalue(fixture_name)
    errors = []
    for b in result.beats:
        if b.r_peak < 0:
            continue
        truth = data["true_boundaries"]
        i = int(np.argmin([abs(t["R_peak"] - b.r_peak) for t in truth]))
        err = abs(truth[i]["R_peak"] - b.r_peak)
        if err <= int(0.05 * fs):
            errors.append(err)
    assert len(errors) >= 5, f"only {len(errors)} matched beats"
    med = float(np.median(errors))
    assert med <= 2.0, f"median R-peak error {med:.1f} samples > 2"


# ----------------------------------------------------------------------
# 3. P-onset accuracy (the hard contract: <20 ms @250 Hz, <12 ms @1 kHz)
# ----------------------------------------------------------------------
def test_dsp_p_onset_accuracy_250(dsp_result_250):
    result, data = dsp_result_250
    errors = _match_errors(result.beats, data["true_boundaries"], FS,
                           "p_onset", "P_onset")
    assert len(errors) >= 5, f"only {len(errors)} matched beats with P"
    med = float(np.median(errors))
    assert med < 20.0, f"250 Hz P-onset median error {med:.1f} ms >= 20 ms"


def test_dsp_p_onset_accuracy_1khz(dsp_result_1khz):
    result, data = dsp_result_1khz
    errors = _match_errors(result.beats, data["true_boundaries"], FS_HR,
                           "p_onset", "P_onset")
    assert len(errors) >= 5, f"only {len(errors)} matched beats with P"
    med = float(np.median(errors))
    assert med < 12.0, f"1 kHz P-onset median error {med:.1f} ms >= 12 ms"


# ----------------------------------------------------------------------
# 4. Structural invariants on every beat
# ----------------------------------------------------------------------
@pytest.mark.parametrize("fixture_name", ["dsp_result_250", "dsp_result_1khz"])
def test_dsp_structural_invariants(fixture_name, request):
    result, _ = request.getfixturevalue(fixture_name)
    beats = result.beats
    assert len(beats) >= 5
    for b in beats:
        # QRS anchors are always set (a beat without them is dropped)
        assert 0 < b.q_onset < b.r_peak <= b.s_offset, (
            f"QRS anchors broken: q={b.q_onset} r={b.r_peak} s={b.s_offset}")
        if b.p_onset >= 0:
            assert b.p_onset < b.p_offset <= b.q_onset, (
                f"P window [{b.p_onset},{b.p_offset}] not before "
                f"q_onset={b.q_onset}")
            assert b.p_source == 'dsp'
        if b.t_onset >= 0:
            assert b.s_offset < b.t_onset < b.t_offset, (
                f"T window [{b.t_onset},{b.t_offset}] not after "
                f"s_offset={b.s_offset}")
            assert b.t_source == 'dsp'
    # T must not run into the next beat
    for i in range(len(beats) - 1):
        nb = beats[i + 1]
        anchor = next((v for v in (nb.p_onset, nb.q_onset, nb.r_peak)
                       if v > beats[i].r_peak), -1)
        if anchor > 0 and beats[i].t_offset > 0:
            assert beats[i].t_offset <= anchor, (
                f"beat {i} T offset {beats[i].t_offset} runs into next beat "
                f"(anchor={anchor})")


# ----------------------------------------------------------------------
# 5. Polarity freedom: inverted record still delineated
# ----------------------------------------------------------------------
def test_dsp_inverted_signal(dsp_result_1khz):
    _, data = dsp_result_1khz
    inv = DSPECGSegmenter(fs=FS_HR).segment(-data["ecg"])
    n_true = len(data["true_boundaries"])
    assert abs(len(inv.beats) - n_true) <= 1, (
        f"inverted: {len(inv.beats)} beats, true {n_true}")
    n_p = sum(1 for b in inv.beats if b.p_onset >= 0)
    n_t = sum(1 for b in inv.beats if b.t_onset >= 0)
    assert n_p >= len(inv.beats) * 0.5, (
        f"inverted: only {n_p}/{len(inv.beats)} beats have P")
    assert n_t >= len(inv.beats) * 0.5, (
        f"inverted: only {n_t}/{len(inv.beats)} beats have T")


# ----------------------------------------------------------------------
# 6. Degenerate inputs: never an exception, always a valid empty result
# ----------------------------------------------------------------------
@pytest.mark.parametrize("signal", [
    np.zeros(5000),                       # all zeros
    np.full(5000, 3.7),                   # constant
    np.zeros(1),                          # single sample
    np.zeros((10, 2)),                    # wrong ndim
])
def test_dsp_degenerate_guards(signal):
    result = DSPECGSegmenter(fs=FS).segment(signal)
    assert isinstance(result, SegmentResult)
    assert result.beats == []


def test_dsp_nan_spike_guard(dsp_result_250):
    """NaNs are sanitized in-place of crashing; beats still detected."""
    _, data = dsp_result_250
    ecg = data["ecg"].copy()
    ecg[100], ecg[700], ecg[1234] = np.nan, np.inf, -np.inf
    result = DSPECGSegmenter(fs=FS).segment(ecg)
    assert len(result.beats) >= 5
    assert np.all(np.isfinite(result.filtered_ecg))


def test_dsp_bad_fs_guard():
    with pytest.raises(ValueError):
        DSPECGSegmenter(fs=0.0)


# ----------------------------------------------------------------------
# 7. SegmentResult contract
# ----------------------------------------------------------------------
def test_dsp_result_contract(dsp_result_1khz):
    result, data = dsp_result_1khz
    T = len(data["ecg"])
    labels = result.state_labels
    assert labels.shape == (T,)
    assert np.issubdtype(labels.dtype, np.integer)
    assert set(np.unique(labels)) <= set(range(9)), (
        f"state labels outside 0..8: {sorted(set(np.unique(labels)))}")
    assert len(result.state_names) == T
    assert result.filtered_ecg.shape == (T,)
    assert result.features.shape == (T, 3)
    assert result.fs == FS_HR
    # beat ids are dense and ordered
    assert [b.beat_id for b in result.beats] == list(range(len(result.beats)))


# ----------------------------------------------------------------------
# 8. Input array is read-only
# ----------------------------------------------------------------------
def test_dsp_input_readonly(dsp_result_250):
    _, data = dsp_result_250
    ecg = data["ecg"].copy()
    snapshot = ecg.copy()
    DSPECGSegmenter(fs=FS).segment(ecg)
    assert np.array_equal(ecg, snapshot, equal_nan=True), (
        "segment() modified its input array")


# ----------------------------------------------------------------------
# 9. LimbLeadProcessor integration flag (default off)
# ----------------------------------------------------------------------
def test_dsp_integration_flag():
    data = generate_synthetic_ecg(fs=FS_HR, duration_sec=10.0, heart_rate=60.0,
                                  noise_std=0.02, random_state=42)
    aecg = {"signals": {"II": data["ecg"]}, "fs": FS_HR,
            "n_samples": len(data["ecg"]), "filename": "synthetic"}

    proc = LimbLeadProcessor(fs=FS_HR, use_dsp_delineator=True,
                             use_wavelet_crosscheck=False)
    _, seg_data = proc.process_record(aecg, record_name="synthetic")
    beats = seg_data["II"]["beats"]
    assert len(beats) >= 5, f"DSP path produced only {len(beats)} beats"
    p_srcs = {b.p_source for b in beats if b.p_onset >= 0}
    t_srcs = {b.t_source for b in beats if b.t_onset >= 0}
    assert p_srcs == {"dsp"}, f"DSP path P provenance: {p_srcs}"
    assert t_srcs == {"dsp"}, f"DSP path T provenance: {t_srcs}"

    # Default path unchanged: no 'dsp' provenance without the flag.
    proc0 = LimbLeadProcessor(fs=FS_HR, use_wavelet_crosscheck=False)
    _, seg0 = proc0.process_record(aecg, record_name="synthetic")
    beats0 = seg0["II"]["beats"]
    assert beats0, "HSMM path produced no beats"
    assert all(b.p_source != "dsp" for b in beats0)
    assert all(b.t_source != "dsp" for b in beats0)


def test_dsp_config_override():
    """Custom config is honored (spot-check one constant actually moves the
    output: a 5 ms P-duration cap rejects every real ~90 ms P wave)."""
    data = generate_synthetic_ecg(fs=FS, duration_sec=10.0, heart_rate=60.0,
                                  noise_std=0.02, random_state=42)
    cfg = DSPDelineationConfig(p_dur_max_ms=5.0)
    result = DSPECGSegmenter(fs=FS, config=cfg).segment(data["ecg"])
    assert len(result.beats) >= 5  # QRS detection unaffected
    assert all(b.p_onset < 0 for b in result.beats), (
        "p_dur_max_ms=5 should reject every P wave")
