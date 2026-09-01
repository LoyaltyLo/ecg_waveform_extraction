"""Time-frequency P/QRS/T segmentation.

Segments an ECG into P wave, QRS complex and T wave directly from the
CWT scalogram (time-frequency domain), with no trained model:

- QRS:   envelope of the 8-30 Hz scalogram rows (QRS energy band),
         adaptive threshold + duration/gap constraints.
- P / T: low-frequency (1.5-8 Hz) envelope evaluated inside physiological
         windows anchored on each R peak (P before, T after),
         threshold-crossing for onsets/offsets.

Designed for 1 kHz aECG records after ECGPreprocessor (0.5-40 Hz passband).
The output carries a per-sample 4-class label array (0=none, 1=P, 2=QRS, 3=T)
aligned 1:1 with the signal, so results can be compared sample-by-sample
against the main package's HSMM state labels
(ecg_waveform_extraction/output/rala_full/_limb_leads/...).

Known limitation (2026-08-31, RA-LA_Reversal dataset, 10 records x 6 limb
leads vs HSMM cache): QRS precision/recall 0.60-0.92 / 0.75-1.00, T
0.37-0.68 / 0.60-0.96, but P waves are almost never detected — in this
dataset the P amplitude is only ~10-25% of the QRS and produces NO local
peak in the CWT envelope at any tested band (1.5-8 / 3-10 / 5-12 / 6-14 Hz)
or transform (CWT morlet/ricker, short-window STFT); it is swamped by the
QRS low-frequency skirt. Undetected P waves are left unmarked rather than
fabricated at the expected timing.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from .spectrogram import compute_cwt_complex

# ---------------------------------------------------------------------------
# Parameters (fs = 1000 Hz; all time constants in ms are converted via fs)
# ---------------------------------------------------------------------------
QRS_BAND = (8.0, 30.0)     # Hz — QRS energy band in the scalogram
LOW_BAND = (1.5, 8.0)      # Hz — P/T energy band
CWT_FREQ_RANGE = (1.0, 40.0)  # Hz — CWT coverage (signal is 0.5-40 Hz bandpassed)
N_VOICES = 48

SMOOTH_MS = 40.0     # envelope moving-average window
MIN_QRS_MS = 60.0    # discard QRS regions shorter than this
MAX_GAP_MS = 80.0    # merge QRS regions separated by less than this
QRS_THR_MEDIAN = 3.0  # QRS envelope threshold: max(K_MED*median, K_P95*p95)
QRS_THR_P95 = 0.30

P_WIN_MS = (-280.0, -60.0)   # relative to R peak
T_WIN_MS = (120.0, 420.0)
# P/T detection is relative to the *global* low-freq envelope baseline
# (LOW_BASE_PCT percentile — near the TP floor). A wave must rise by at
# least WAVE_MIN_RISE x the typical bump height, and its extent is traced
# down to baseline + WAVE_PROMINENCE x (peak - baseline).
LOW_BASE_PCT = 10.0
WAVE_MIN_RISE = 0.20
WAVE_PROMINENCE = 0.35
MIN_WAVE_MS = 20.0    # discard P/T detections narrower than this
# P must be a real local peak of the low-freq envelope inside its window,
# chosen by prominence weighted with a Gaussian timing prior centered at
# P_PRIOR_MS before R. Small P waves that produce no local TF peak are NOT
# marked (an inherent limit of amplitude-based TF analysis).
P_PEAK_PROMINENCE = 0.15  # x low-freq envelope scale
P_PRIOR_MS = -160.0
P_PRIOR_SIGMA_MS = 60.0

# 4-class label codes (match the HSMM-mapped band codes used in compare_with_hsmm)
LABEL_NONE, LABEL_P, LABEL_QRS, LABEL_T = 0, 1, 2, 3

BAND_COLORS = {LABEL_P: '#4C72B0', LABEL_QRS: '#C44E52', LABEL_T: '#55A868'}
BAND_NAMES = {LABEL_P: 'P', LABEL_QRS: 'QRS', LABEL_T: 'T'}


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class TFBeat:
    """One detected beat with P/QRS/T boundaries (sample indices)."""

    beat_id: int
    r_peak: int
    qrs_onset: int
    qrs_offset: int
    p_onset: int | None = None
    p_offset: int | None = None
    t_onset: int | None = None
    t_offset: int | None = None


@dataclass
class TFSegmentation:
    """Result of time-frequency segmentation of one lead.

    Attributes
    ----------
    labels : (N,) int8 per-sample band labels (0=none, 1=P, 2=QRS, 3=T)
    beats : list[TFBeat] per-beat boundaries
    cwt_mag, cwt_freqs : scalogram magnitude and its frequency axis (Hz)
    qrs_env, low_env : (N,) smoothed band envelopes
    qrs_threshold : scalar threshold used for the QRS envelope
    fs : sampling rate
    """

    labels: np.ndarray
    beats: list[TFBeat]
    cwt_mag: np.ndarray
    cwt_freqs: np.ndarray
    qrs_env: np.ndarray
    low_env: np.ndarray
    qrs_threshold: float
    fs: float

    @property
    def n_beats(self) -> int:
        return len(self.beats)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """Centered moving average; win always odd and >=1."""
    if win <= 1:
        return x
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win) / win
    return np.convolve(x, kernel, mode='same')


def _find_regions(mask: np.ndarray, min_len: int, merge_gap: int) -> list[tuple[int, int]]:
    """Boolean mask -> [(start, end)] contiguous regions.

    Merges regions whose gap is < merge_gap, then drops regions
    shorter than min_len. end is exclusive.
    """
    if not mask.any():
        return []
    edges = np.flatnonzero(np.diff(mask.astype(np.int8)))
    starts = [0] if mask[0] else []
    ends = []
    for i in edges:
        (ends if mask[i] else starts).append(i + 1)
    if mask[-1]:
        ends.append(len(mask))
    regions = list(zip(starts, ends))

    merged: list[list[int]] = []
    for s, e in regions:
        if merged and s - merged[-1][1] < merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if e - s >= min_len]


def _band_envelope(coeffs: np.ndarray, freqs: np.ndarray,
                   band: tuple[float, float], smooth_win: int) -> np.ndarray:
    """Mean magnitude over the scalogram rows inside [band], smoothed."""
    fmask = (freqs >= band[0]) & (freqs <= band[1])
    env = np.abs(coeffs[fmask, :]).mean(axis=0) if fmask.any() \
        else np.zeros(coeffs.shape[1])
    return _moving_average(env, smooth_win)


def _wave_bounds_in_window(env_w: np.ndarray, baseline: float,
                           scale: float) -> tuple[int, int] | None:
    """Onset/offset (window-relative, exclusive end) around the envelope peak.

    The peak must rise at least WAVE_MIN_RISE x `scale` above `baseline`
    (global envelope statistics); the extent is traced from the peak down
    to baseline + WAVE_PROMINENCE x (peak - baseline) on both sides.
    """
    if len(env_w) < 5 or scale <= 0:
        return None
    peak = int(np.argmax(env_w))
    rise = env_w[peak] - baseline
    if rise < WAVE_MIN_RISE * scale:
        return None
    thr = baseline + WAVE_PROMINENCE * rise
    lo = peak
    while lo > 0 and env_w[lo - 1] > thr:
        lo -= 1
    hi = peak + 1
    while hi < len(env_w) and env_w[hi] > thr:
        hi += 1
    return lo, hi


# ---------------------------------------------------------------------------
# Core segmentation
# ---------------------------------------------------------------------------
def segment_tf(ecg_signal: np.ndarray, fs: float) -> TFSegmentation:
    """Segment one preprocessed lead into P/QRS/T from the CWT scalogram.

    Parameters
    ----------
    ecg_signal : (N,) preprocessed signal (bandpass 0.5-40 Hz, z-scored).
    fs : sampling rate in Hz.

    Returns
    -------
    TFSegmentation with per-sample labels and per-beat boundaries.
    """
    ecg_signal = np.asarray(ecg_signal, dtype=np.float64)
    N = len(ecg_signal)
    empty = TFSegmentation(
        labels=np.zeros(N, dtype=np.int8), beats=[],
        cwt_mag=np.zeros((0, N)), cwt_freqs=np.array([]),
        qrs_env=np.zeros(N), low_env=np.zeros(N),
        qrs_threshold=0.0, fs=fs,
    )
    if N < int(1.0 * fs):  # need at least ~1 s of signal
        return empty

    # ---- CWT (complex, magnitude used here) ----
    cwt_spec = compute_cwt_complex(
        ecg_signal, fs=fs, wavelet='morlet',
        freq_range=CWT_FREQ_RANGE, n_voices=N_VOICES,
    )
    coeffs = np.asarray(cwt_spec.data)
    cwt_mag = np.abs(coeffs)

    smooth_win = max(1, int(SMOOTH_MS * fs / 1000.0))
    qrs_env = _band_envelope(coeffs, cwt_spec.freqs, QRS_BAND, smooth_win)
    low_env = _band_envelope(coeffs, cwt_spec.freqs, LOW_BAND, smooth_win)

    # ---- QRS regions ----
    qrs_thr = float(max(QRS_THR_MEDIAN * np.median(qrs_env),
                        QRS_THR_P95 * np.percentile(qrs_env, 95)))
    qrs_regions = _find_regions(
        qrs_env > qrs_thr,
        min_len=int(MIN_QRS_MS * fs / 1000.0),
        merge_gap=int(MAX_GAP_MS * fs / 1000.0),
    )
    if not qrs_regions:
        return TFSegmentation(
            labels=np.zeros(N, dtype=np.int8), beats=[],
            cwt_mag=cwt_mag, cwt_freqs=cwt_spec.freqs,
            qrs_env=qrs_env, low_env=low_env, qrs_threshold=qrs_thr, fs=fs,
        )

    # ---- R peaks ----
    r_peaks = np.array([s + int(np.argmax(np.abs(ecg_signal[s:e])))
                        for s, e in qrs_regions])

    # ---- Labels + P/T boundaries per beat ----
    labels = np.zeros(N, dtype=np.int8)
    min_wave = int(MIN_WAVE_MS * fs / 1000.0)
    # Global statistics of the low-frequency envelope: P/T are detected as
    # bumps relative to this near-TP-floor baseline, not relative to their
    # own window (the window may sit on the QRS low-frequency skirt).
    low_base = float(np.percentile(low_env, LOW_BASE_PCT))
    low_scale = float(np.percentile(low_env, 95)) - low_base
    beats: list[TFBeat] = []

    for i, (s, e) in enumerate(qrs_regions):
        labels[s:e] = LABEL_QRS
        r = int(r_peaks[i])
        prev_r = int(r_peaks[i - 1]) if i > 0 else None
        next_r = int(r_peaks[i + 1]) if i < len(r_peaks) - 1 else None

        beat = TFBeat(beat_id=i, r_peak=r, qrs_onset=s, qrs_offset=e)

        # ---- P wave: window before QRS ----
        p_lo = max(0, r + int(P_WIN_MS[0] * fs / 1000.0))
        p_hi = max(p_lo, r + int(P_WIN_MS[1] * fs / 1000.0))
        if prev_r is not None:                      # don't bleed into the previous T
            p_lo = max(p_lo, prev_r + int(120.0 * fs / 1000.0))
        if p_hi - p_lo >= min_wave:
            pw = low_env[p_lo:p_hi]
            cand, props = find_peaks(pw, prominence=P_PEAK_PROMINENCE * low_scale)
            if len(cand):
                # Prominence weighted by a Gaussian timing prior around the
                # expected P location — rejects picks on the QRS skirt.
                prior_center = r + int(P_PRIOR_MS * fs / 1000.0) - p_lo
                prior_sigma = max(1.0, P_PRIOR_SIGMA_MS * fs / 1000.0)
                score = props['prominences'] * np.exp(
                    -0.5 * ((cand - prior_center) / prior_sigma) ** 2)
                pk = int(cand[np.argmax(score)])
                thr = low_base + WAVE_PROMINENCE * (pw[pk] - low_base)
                on = pk
                while on > 0 and pw[on - 1] > thr:
                    on -= 1
                off = pk + 1
                while off < len(pw) and pw[off] > thr:
                    off += 1
                if off - on >= min_wave:
                    beat.p_onset, beat.p_offset = p_lo + on, p_lo + off
                    labels[beat.p_onset:beat.p_offset] = LABEL_P

        # ---- T wave: window after QRS ----
        t_lo = min(N - 1, e + int(T_WIN_MS[0] * fs / 1000.0))
        t_hi = min(N, e + int(T_WIN_MS[1] * fs / 1000.0))
        if next_r is not None:                      # don't bleed into the next P
            t_hi = min(t_hi, next_r - int(80.0 * fs / 1000.0))
        if t_hi - t_lo >= min_wave:
            bounds = _wave_bounds_in_window(low_env[t_lo:t_hi], low_base, low_scale)
            if bounds is not None:
                on, off = bounds
                if off - on >= min_wave:
                    beat.t_onset, beat.t_offset = t_lo + on, t_lo + off
                    labels[beat.t_onset:beat.t_offset] = LABEL_T

        beats.append(beat)

    return TFSegmentation(
        labels=labels, beats=beats,
        cwt_mag=cwt_mag, cwt_freqs=cwt_spec.freqs,
        qrs_env=qrs_env, low_env=low_env, qrs_threshold=qrs_thr, fs=fs,
    )


# ---------------------------------------------------------------------------
# Comparison against the main package's HSMM labels
# ---------------------------------------------------------------------------
# HSMM 9-state -> 4-class band mapping (hsmm_model.STATE_LABELS):
#   P=1, QRS=3/4/5, T=7; everything else (ISO/PR/ST/TP) = none
_HSMM_TO_BAND = {1: LABEL_P, 3: LABEL_QRS, 4: LABEL_QRS, 5: LABEL_QRS, 7: LABEL_T}


def hsmm_to_bands(hsmm_labels: np.ndarray) -> np.ndarray:
    """Map 9-state HSMM labels to 4-class band labels (0/1/2/3)."""
    out = np.zeros(len(hsmm_labels), dtype=np.int8)
    for state, band in _HSMM_TO_BAND.items():
        out[hsmm_labels == state] = band
    return out


def compare_with_hsmm(labels_tf: np.ndarray, hsmm_labels: np.ndarray) -> dict:
    """Sample-level comparison of TF segmentation vs cached HSMM labels.

    Compares on the overlapping prefix of the two label arrays (the HSMM
    cache covers only the first 4000 samples of each record).
    Returns overall agreement and per-band precision/recall
    (TF = prediction, HSMM = reference).
    """
    n = min(len(labels_tf), len(hsmm_labels))
    tf = np.asarray(labels_tf[:n])
    ref = hsmm_to_bands(np.asarray(hsmm_labels[:n]))

    agree = float(np.mean(tf == ref))
    report: dict = {'n_samples': int(n), 'agreement': round(agree, 4), 'per_band': {}}
    for code in (LABEL_P, LABEL_QRS, LABEL_T):
        tp = int(np.sum((tf == code) & (ref == code)))
        fp = int(np.sum((tf == code) & (ref != code)))
        fn = int(np.sum((tf != code) & (ref == code)))
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        iou = tp / (tp + fp + fn) if tp + fp + fn else None
        report['per_band'][BAND_NAMES[code]] = {
            'precision': None if precision is None else round(precision, 4),
            'recall': None if recall is None else round(recall, 4),
            'iou': None if iou is None else round(iou, 4),
            'support_ref_samples': int(np.sum(ref == code)),
        }
    return report


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------
def segmentation_to_dict(seg: TFSegmentation) -> dict:
    """JSON-safe summary of a TFSegmentation (boundaries in ms)."""
    fs = seg.fs
    beats = []
    for b in seg.beats:

        def ms(v):
            return None if v is None else round(v / fs * 1000.0, 1)

        beats.append({
            'beat_id': b.beat_id,
            'r_peak_ms': ms(b.r_peak),
            'qrs_onset_ms': ms(b.qrs_onset), 'qrs_offset_ms': ms(b.qrs_offset),
            'p_onset_ms': ms(b.p_onset), 'p_offset_ms': ms(b.p_offset),
            't_onset_ms': ms(b.t_onset), 't_offset_ms': ms(b.t_offset),
        })
    return {
        'fs': fs,
        'n_samples': int(len(seg.labels)),
        'n_beats': seg.n_beats,
        'qrs_threshold': round(seg.qrs_threshold, 6),
        'beats': beats,
    }
