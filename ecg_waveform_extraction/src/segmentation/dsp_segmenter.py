"""DSPECGSegmenter: self-contained classical ECG waveform delineator.

A training-free, deterministic alternative to the HSMM pipeline
(:mod:`ecg_waveform_extraction.src.segmentation.segmenter`).  It produces the
same ``SegmentResult`` / ``BeatBoundary`` contract, so the existing runner
(:mod:`ecg_waveform_extraction.src.limb_lead_processor`), caches, audits and
lead-first export consume it unchanged.

Pipeline (all stages on the preprocessed signal ``clean``):

1. QRS energy envelope: narrow bandpass (default 6-18 Hz) -> square ->
   moving average.  Squaring makes the envelope non-negative, so peak
   ``prominence`` is insensitive to baseline offsets.
2. R detection: ``find_peaks`` with a MAD-scaled prominence floor and a
   refractory ``distance``, a median-based amplitude screen, RR-plausibility
   / shape joint rejection, then refine each candidate to ``argmax |clean|``
   in +-50 ms.  Everything uses |amplitude| / non-negative energy, so the
   stage is lead-polarity agnostic.
3. QRS onset/offset: walk the smoothed derivative outward from the R peak
   until it drops below a MAD noise floor for a sustained quiet run.  The
   walk deliberately stops *just outside* the wave: the downstream
   ``refine_qrs_boundaries`` can only move the given onset rightward and the
   given offset leftward, so an inward-biased seed would be unrecoverable.
4. P wave: R-anchored search window before the QRS; mirrored (upright +
   inverted) prominence peaks on a P-band signal; SNR / edge / duration
   gates; onset/offset by the tangent method on the band signal (the
   clinical T-end construction, immune to baseline depression).
5. T wave: same recipe after the QRS, capped by the next beat's anchor;
   flat T fails the SNR gate and is simply absent (``-1``).
6. Assembly: fills the derived ``iso/pr/st/tp`` markers, stamps
   ``p_source='dsp'`` / ``t_source='dsp'``, and paints a per-sample state
   label array using the same 9-label order as
   ``ecg_waveform_extraction.src.hsmm.hsmm_model.STATE_LABELS`` so the
   spectral audit and plotting keep working.

Only numpy + scipy are used.  No HSMM model, no prominence-delineator, no
wavedet.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, find_peaks

from ..preprocessing.filters import ECGPreprocessor
from .segmenter import BeatBoundary, SegmentResult

# Same 9-state order as src/hsmm/hsmm_model.STATE_LABELS.  Duplicated here on
# purpose: this module must not import the HSMM machinery, and the painted
# label array has to stay index-compatible with the spectral audit and the
# plotting helpers.
DSP_STATE_LABELS = ["ISO", "P", "PR", "Q", "R", "S", "ST", "T", "TP"]

_LABEL_ISO, _LABEL_P, _LABEL_PR, _LABEL_Q, _LABEL_R = 0, 1, 2, 3, 4
_LABEL_S, _LABEL_ST, _LABEL_T, _LABEL_TP = 5, 6, 7, 8


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
@dataclass
class DSPDelineationConfig:
    """Every threshold of the DSP delineator.

    Time constants are stored in seconds / milliseconds and converted to
    samples at use, so one config serves any sampling rate.  Amplitude
    thresholds are MAD- or fraction-based (scale free after z-scoring).
    """

    # -- QRS envelope / R detection --------------------------------------
    qrs_env_band_hz: tuple[float, float] = (6.0, 18.0)
    env_filter_order: int = 3
    env_smooth_s: float = 0.030       # moving-average window of the energy
    env_prom_mad_k: float = 6.0       # find_peaks prominence = k * MAD(env)
    env_peak_med_frac: float = 0.25   # keep peaks >= frac * median(peak env)
    qrs_refractory_s: float = 0.25    # min RR (max ~240 bpm)
    r_refine_half_s: float = 0.05     # argmax|clean| window around candidate
    dedup_merge_s: float = 0.10       # merge refined R peaks closer than this
    # -- plausibility / shape gates --------------------------------------
    rr_plaus_lo: float = 0.40         # RR < lo * median(RR) is implausible
    rr_plaus_hi: float = 2.50         # RR > hi * median(RR) is implausible
    amp_weak_frac: float = 0.35       # weak peak: env < frac * median(peak)
    shape_min_ratio: float = 2.5      # env peak / mean(env, +-0.1 s)
    # -- QRS boundaries ---------------------------------------------------
    qrs_deriv_smooth_s: float = 0.010
    qrs_noise_k: float = 3.0          # |d1| threshold = k * MAD(d1)
    qrs_quiet_run_s: float = 0.005    # sustained quiet run that ends the walk
    qrs_on_max_back_s: float = 0.12   # hard clamp: Q onset <= R - 120 ms
    qrs_off_max_fwd_s: float = 0.16   # hard clamp: S offset >= R + 160 ms
    qrs_min_dur_s: float = 0.02       # structural minimum QRS duration
    qrs_fallback_on_s: float = 0.04   # fixed geometry if the walk never quits
    qrs_fallback_off_s: float = 0.06
    # -- P wave ------------------------------------------------------------
    p_band_hz: tuple[float, float] = (0.5, 10.0)
    p_search_before_s: float = 0.32   # window: [R - 320 ms, q_onset - guard]
    p_end_guard_s: float = 0.01
    p_snr_factor: float = 2.5         # peak >= k * MAD(p_sig, TP reference)
    p_prom_mad_k: float = 3.0         # find_peaks prominence floor
    p_edge_guard_ms: float = 10.0     # reject peaks pinned at the window edge
    p_dur_min_ms: float = 40.0
    p_dur_max_ms: float = 160.0
    # -- T wave --------------------------------------------------------------
    t_band_hz: tuple[float, float] = (0.5, 8.0)
    t_on_guard_s: float = 0.02        # window starts at s_offset + 20 ms
    t_search_abs_max_s: float = 0.55  # absolute cap on the search end
    t_search_rr_frac: float = 0.50    # ... and R + frac * RR
    t_search_min_lead_s: float = 0.28  # but never earlier than R + 280 ms
    t_snr_factor: float = 2.5
    t_prom_mad_k: float = 3.0
    t_edge_guard_ms: float = 15.0     # peak at the cap is the cap, not a wave
    t_tail_k: float = 2.0             # tangent offset <= peak + k * up-flank
    t_dur_min_ms: float = 80.0
    t_dur_max_ms: float = 350.0
    # -- assembly ---------------------------------------------------------
    iso_lead_pad_s: float = 0.20      # iso_start of the first beat
    # Record-level TP fallback window (fractions of the preceding RR), used
    # per signal as the noise/baseline reference when a beat's own TP gap
    # is too short to be quiet (fast rhythms).
    tp_gap_after_r: float = 0.45      # window starts 45% into the previous RR
    tp_gap_before_r: float = 0.30     # ... and ends 30% before this R


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------
def robust_sigma(x: np.ndarray) -> float:
    """MAD-based standard deviation estimate: 1.4826 * MAD(x).

    Returns 0.0 for empty input; callers treat a 0.0 sigma as "gate passed"
    (a constant window contains no wave to reject anyway).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def bandpass(x: np.ndarray, fs: float, band_hz: tuple[float, float],
             order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth bandpass.  Zeros out on degenerate bands."""
    x = np.asarray(x, dtype=np.float64)
    nyq = fs / 2.0
    lo, hi = band_hz
    if lo <= 0 or hi >= nyq or lo >= hi or x.size < 4 * (order + 1):
        return np.zeros_like(x)
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x)


def _odd_window(n_samples: float, minimum: int = 3) -> int:
    """Round a sample count to an odd int >= minimum (for filter sizes)."""
    n = max(int(round(n_samples)), minimum)
    return n if n % 2 == 1 else n + 1


# ---------------------------------------------------------------------------
# Stage 2: QRS energy envelope and R detection
# ---------------------------------------------------------------------------
def build_qrs_envelope(clean: np.ndarray, fs: float,
                       cfg: DSPDelineationConfig) -> np.ndarray:
    """Non-negative QRS energy envelope: bandpass -> square -> smooth.

    The 6-18 Hz default band keeps wide/bundle-branch QRS visible while
    excluding the T wave (<10 Hz of energy per the spectral audit bands).
    """
    bp = bandpass(clean, fs, cfg.qrs_env_band_hz, cfg.env_filter_order)
    env = bp * bp
    win = _odd_window(cfg.env_smooth_s * fs)
    return uniform_filter1d(env, size=win)


def detect_r_candidates(env: np.ndarray, fs: float,
                        cfg: DSPDelineationConfig) -> np.ndarray:
    """Refractory + prominence + median-amplitude peak picking on the envelope."""
    prom = max(cfg.env_prom_mad_k * robust_sigma(env), 0.0)
    distance = max(1, int(round(cfg.qrs_refractory_s * fs)))
    peaks, _ = find_peaks(env, distance=distance, prominence=prom)
    if peaks.size == 0:
        return peaks
    # Median-based amplitude screen: one outsized PVC must not erase normal
    # beats, and one missed weak beat must not lower the bar.
    med = float(np.median(env[peaks]))
    return peaks[env[peaks] >= cfg.env_peak_med_frac * med]


def refine_r_peak(clean: np.ndarray, cand: int, fs: float,
                  cfg: DSPDelineationConfig) -> int:
    """Snap an envelope candidate to argmax |detrended clean| within +-50 ms."""
    T = len(clean)
    half = max(1, int(round(cfg.r_refine_half_s * fs)))
    i0 = max(0, cand - half)
    i1 = min(T, cand + half + 1)
    seg = clean[i0:i1]
    if seg.size == 0:
        return int(cand)
    base = float(np.median(seg))
    return i0 + int(np.argmax(np.abs(seg - base)))


def dedup_r_peaks(r_peaks: np.ndarray, env: np.ndarray, fs: float,
                  cfg: DSPDelineationConfig) -> np.ndarray:
    """Merge refined R peaks closer than ``dedup_merge_s``, keeping the
    larger envelope (two candidates can refine onto one wide QRS)."""
    merge = max(1, int(round(cfg.dedup_merge_s * fs)))
    kept: list[int] = []
    for p in sorted(int(p) for p in r_peaks):
        if kept and p - kept[-1] < merge:
            if env[p] > env[kept[-1]]:
                kept[-1] = p
        else:
            kept.append(p)
    return np.asarray(kept, dtype=int)


def rr_plausibility_filter(r_peaks: np.ndarray, env: np.ndarray, fs: float,
                           cfg: DSPDelineationConfig) -> np.ndarray:
    """Drop candidates whose RR is implausible *and* whose spike looks wrong.

    Implausibility alone is not enough (real beats skip); the candidate must
    also be weak or shapeless.  Disabled below 4 candidates (unstable
    median) and never allowed to leave fewer than 2 beats.
    """
    if len(r_peaks) < 4:
        return r_peaks
    rr = np.diff(r_peaks)
    med_rr = float(np.median(rr))
    if med_rr <= 0:
        return r_peaks
    lo, hi = cfg.rr_plaus_lo * med_rr, cfg.rr_plaus_hi * med_rr
    med_amp = float(np.median(env[r_peaks]))
    shape_w = max(1, int(round(0.10 * fs)))

    kept: list[int] = []
    for i, p in enumerate(r_peaks):
        rr_prev = rr[i - 1] if i > 0 else rr[i]
        rr_next = rr[i] if i < len(rr) else rr[i - 1]
        implausible = not (lo <= rr_prev <= hi) or not (lo <= rr_next <= hi)
        if not implausible:
            kept.append(int(p))
            continue
        weak = env[p] < cfg.amp_weak_frac * med_amp
        i0, i1 = max(0, p - shape_w), min(len(env), p + shape_w + 1)
        local_mean = float(np.mean(env[i0:i1]))
        shapeless = env[p] / max(local_mean, 1e-12) < cfg.shape_min_ratio
        if weak or shapeless:
            continue  # drop the suspect candidate
        kept.append(int(p))
    if len(kept) < 2:
        return r_peaks
    return np.asarray(kept, dtype=int)


# ---------------------------------------------------------------------------
# Stage 3: QRS onset / offset
# ---------------------------------------------------------------------------
def _walk_qrs_edge(d1s: np.ndarray, r_peak: int, fs: float, sigma_d: float,
                   cfg: DSPDelineationConfig, backward: bool) -> int:
    """Walk the smoothed derivative outward from the R peak.

    Stops after a sustained quiet run (``|d1| < k * MAD``).  Returns the
    quiet-run sample closest to the wave, i.e. *just outside* the QRS — the
    bias the downstream ``refine_qrs_boundaries`` expects (it can only walk
    the onset rightward / the offset leftward).  Falls back to fixed
    geometry when the walk never leaves the active region.
    """
    T = len(d1s)
    thr = cfg.qrs_noise_k * max(sigma_d, 1e-12)
    quiet = max(1, int(round(cfg.qrs_quiet_run_s * fs)))
    step = -1 if backward else 1
    clamp = int(round(cfg.qrs_on_max_back_s * fs)) if backward \
        else int(round(cfg.qrs_off_max_fwd_s * fs))
    fallback = int(round(cfg.qrs_fallback_on_s * fs)) if backward \
        else int(round(cfg.qrs_fallback_off_s * fs))

    limit = r_peak - clamp if backward else r_peak + clamp
    run = 0
    i = r_peak + step
    while (i >= limit) if backward else (i <= limit):
        if i < 0 or i > T - 1:
            break
        if abs(d1s[i]) < thr:
            run += 1
            if run >= quiet:
                # quiet run spans i .. i - step*(quiet-1); keep the sample
                # closest to R (just outside the wave)
                return i - step * (quiet - 1)
        else:
            run = 0
        i += step
    return r_peak - fallback if backward else r_peak + fallback


def estimate_qrs_boundaries(clean: np.ndarray, r_peak: int, fs: float, cfg:
                            DSPDelineationConfig,
                            noise_sigma: float | None = None
                            ) -> tuple[int, int]:
    """Outward-biased (q_onset, s_offset) around one R peak."""
    d1s = uniform_filter1d(
        np.gradient(clean), size=_odd_window(cfg.qrs_deriv_smooth_s * fs))
    if noise_sigma is None:
        noise_sigma = robust_sigma(d1s)
    min_dur = max(1, int(round(cfg.qrs_min_dur_s * fs)))

    q_on = _walk_qrs_edge(d1s, r_peak, fs, noise_sigma, cfg, backward=True)
    s_off = _walk_qrs_edge(d1s, r_peak, fs, noise_sigma, cfg, backward=False)

    # Structural guarantees: 1 <= q_onset < r_peak <= s_offset, min duration.
    q_on = max(1, min(int(q_on), r_peak - min_dur))
    s_off = min(len(clean) - 1, max(int(s_off), r_peak, q_on + min_dur))
    return int(q_on), int(s_off)


# ---------------------------------------------------------------------------
# Stage 4/5: P and T waves
# ---------------------------------------------------------------------------
def local_baseline(x: np.ndarray, i0: int, i1: int) -> float:
    """Median of ``x[i0:i1]`` (clamped); NaN-safe fallback to 0.0."""
    i0, i1 = max(0, int(i0)), min(len(x), int(i1))
    if i1 - i0 < 1:
        return 0.0
    return float(np.median(x[i0:i1]))


def quietest_tp_ref(r_arr: np.ndarray, sig: np.ndarray, fs: float,
                    cfg: DSPDelineationConfig) -> tuple[int, int] | None:
    """The quietest record-level TP plateau on band signal ``sig``.

    Scans one candidate window per inter-beat gap (``tp_gap_after_r`` into
    the previous RR through ``tp_gap_before_r`` before the next R) and
    returns the window with the smallest MAD sigma.  This is the fallback
    isoelectric reference for beats whose own TP gap is too short to be
    quiet (fast rhythms): the whole-record fallback formerly used instead
    includes every QRS and T, inflating sigma until the SNR and prominence
    gates rejected nearly every wave on the beat.
    """
    min_len = int(round(0.05 * fs))
    best: tuple[int, int] | None = None
    best_sigma = np.inf
    for k in range(1, len(r_arr)):
        rr = float(r_arr[k] - r_arr[k - 1])
        w0 = int(r_arr[k - 1] + cfg.tp_gap_after_r * rr)
        w1 = int(r_arr[k] - cfg.tp_gap_before_r * rr)
        if w1 - w0 < min_len:
            continue
        s = robust_sigma(sig[w0:w1])
        if s < best_sigma:
            best_sigma = s
            best = (w0, w1)
    return best


def tangent_edge(sig: np.ndarray, d1: np.ndarray, c: int, base: float,
                 pol: int, lo: int, hi: int, backward: bool) -> int:
    """Wave edge by the tangent method (the clinical T-end approach).

    Draw the tangent at the steepest-slope point of the wave flank and take
    its crossing of the isoelectric ``base``.  For a Gaussian bump this
    lands exactly at peak ± 2·sigma — the visually-marked onset/offset —
    and, unlike amplitude-fraction crossings, it is immune to both the
    rounded wave foot and the baseline depression the preprocessor's
    bandpass ripple leaves in the TP segments.

    ``pol`` is +1 for a bump above base, -1 for a dip below it; ``lo/hi``
    bound the flank search (the wave's own search window, which keeps the
    next wave's steep edge out of the argmax).
    """
    lo, hi = max(1, int(lo)), min(len(sig) - 2, int(hi))
    if backward:
        flank = d1[lo:c]
        if flank.size == 0:
            return int(c)
        im = lo + int(np.argmax(flank * pol))
        slope = float(d1[im]) * pol
        val = pol * (float(sig[im]) - base)
        if abs(slope) < 1e-9:
            return int(im)
        return int(round(im - val / slope))
    flank = d1[c + 1:hi + 1]
    if flank.size == 0:
        return int(c)
    im = c + 1 + int(np.argmax(-flank * pol))
    slope = -float(d1[im]) * pol
    val = pol * (float(sig[im]) - base)
    if abs(slope) < 1e-9:
        return int(im)
    return int(round(im + val / slope))


def _band_candidates(band_sig: np.ndarray, i0: int, i1: int, prom: float,
                     band_base: float) -> list[int]:
    """Prominence peaks inside ``[i0, i1]`` on a band signal (absolute idx),
    best-first by |deviation from the band baseline|."""
    i0, i1 = max(0, i0), min(len(band_sig) - 1, i1)
    if i1 - i0 < 3:
        return []
    seg = band_sig[i0:i1 + 1]
    peaks, _ = find_peaks(seg, prominence=max(prom, 1e-12))
    if peaks.size == 0:
        return []
    order = np.argsort(-np.abs(seg[peaks] - band_base))
    return [i0 + int(peaks[k]) for k in order]


def find_p_wave(p_sig: np.ndarray, d1_p: np.ndarray,
                quiet_ref: tuple[int, int], fs: float, r_peak: int,
                q_onset: int, search_start: int,
                cfg: DSPDelineationConfig) -> tuple[int, int, int]:
    """Delineate one P wave.  Returns ``(p_onset, p_offset, p_peak)`` or
    ``(-1, -1, -1)`` when no wave passes the gates.

    Mirrored prominence passes (signal and its negation) make the detection
    polarity free — inverted and retrograde P waves are found the same way.
    ``quiet_ref`` is the isoelectric TP/ISO window where the noise sigma
    and the baseline are measured.  Edges come from the tangent method on
    the band signal (empirically 2-4 ms from ground truth at 1 kHz — the
    preprocessor's bandpass ripple depresses the TP baseline by ~12% of a P
    amplitude, which systematically biases amplitude-fraction crossings
    ~13 ms into the wave; the tangent is immune to a DC shift).
    """
    T = len(p_sig)
    search_end = q_onset - int(round(cfg.p_end_guard_s * fs))
    if search_end - search_start < int(round(0.05 * fs)):
        return -1, -1, -1

    sigma_p = robust_sigma(p_sig[quiet_ref[0]:quiet_ref[1]])
    p_base = local_baseline(p_sig, quiet_ref[0], quiet_ref[1])
    prom = cfg.p_prom_mad_k * max(sigma_p, 0.0)
    edge = int(round(cfg.p_edge_guard_ms / 1000.0 * fs))

    # Two polarity passes: upright peaks, then inverted (negated) peaks.
    cand_lists = [
        _band_candidates(p_sig, search_start, search_end, prom, p_base),
        _band_candidates(-p_sig, search_start, search_end, prom, -p_base),
    ]
    for candidates in cand_lists:
        for c in candidates:
            if c - search_start < edge or search_end - c < edge:
                continue  # window-pinned bumps are not waves
            if abs(float(p_sig[c]) - p_base) < cfg.p_snr_factor * sigma_p:
                continue  # SNR gate
            pol = 1 if float(p_sig[c]) > p_base else -1
            p_onset = tangent_edge(p_sig, d1_p, c, p_base, pol,
                                   search_start, search_end, backward=True)
            p_offset = tangent_edge(p_sig, d1_p, c, p_base, pol,
                                    search_start, search_end, backward=False)
            p_onset = max(search_start, min(int(p_onset), c - 1))
            p_offset = min(int(p_offset), q_onset, T - 1)
            dur_ms = (p_offset - p_onset) / fs * 1000.0
            if cfg.p_dur_min_ms <= dur_ms <= cfg.p_dur_max_ms \
                    and p_onset < p_offset < r_peak:
                return int(p_onset), int(p_offset), int(c)
    return -1, -1, -1


def find_t_wave(t_sig: np.ndarray, d1_t: np.ndarray,
                quiet_ref: tuple[int, int], fs: float, r_peak: int,
                s_offset: int, search_end: int,
                cfg: DSPDelineationConfig) -> tuple[int, int, int]:
    """Delineate one T wave after the QRS.  Returns
    ``(t_onset, t_offset, t_peak)`` or ``(-1, -1, -1)`` (flat T fails the
    SNR gate and is simply absent).

    Baseline and noise come from the same isoelectric ``quiet_ref`` as the
    P wave.  Using an ST-segment reference instead (the naive choice) is
    what breaks low rates: the S-wave tail rings in the T band for tens of
    ms, dragging the "baseline" down into the S nadir, which corrupts every
    threshold derived from it.  Edges use the tangent method on the band
    signal — the band removes the QRS ringing that contaminates ``clean``,
    and the tangent extrapolates the steep flank across the rounded foot.
    """
    T = len(t_sig)
    search_start = s_offset + int(round(cfg.t_on_guard_s * fs))
    if search_end - search_start < int(round(0.10 * fs)):
        return -1, -1, -1

    sigma_st = robust_sigma(t_sig[quiet_ref[0]:quiet_ref[1]])
    t_base = local_baseline(t_sig, quiet_ref[0], quiet_ref[1])
    prom = cfg.t_prom_mad_k * max(sigma_st, 0.0)
    edge = int(round(cfg.t_edge_guard_ms / 1000.0 * fs))

    cand_lists = [
        _band_candidates(t_sig, search_start, search_end, prom, t_base),
        _band_candidates(-t_sig, search_start, search_end, prom, -t_base),
    ]
    for candidates in cand_lists:
        for c in candidates:
            if search_end - c < edge:
                continue  # a peak at the cap is the cap, not a wave
            if abs(float(t_sig[c]) - t_base) < cfg.t_snr_factor * sigma_st:
                continue  # flat / absent T
            pol = 1 if float(t_sig[c]) > t_base else -1
            t_onset = tangent_edge(t_sig, d1_t, c, t_base, pol,
                                   search_start, search_end, backward=True)
            t_offset = tangent_edge(t_sig, d1_t, c, t_base, pol,
                                    search_start, search_end, backward=False)
            t_onset = max(search_start, min(int(t_onset), c - 1))
            # Tail guard: on a symmetric wave the tangent lands at
            # peak + (peak - onset); a landing much farther out means the
            # steep flank extrapolated across a shallow, band-smoothed tail.
            # Trim to k * up-flank (bounded by the search end) so the
            # duration gate judges the wave, not the extrapolation.
            t_offset = min(int(t_offset),
                           c + int(round(cfg.t_tail_k * (c - t_onset))),
                           search_end, T - 1)
            dur_ms = (t_offset - t_onset) / fs * 1000.0
            if cfg.t_dur_min_ms <= dur_ms <= cfg.t_dur_max_ms \
                    and t_onset < t_offset and s_offset < t_onset < T:
                return int(t_onset), int(t_offset), int(c)
    return -1, -1, -1


# ---------------------------------------------------------------------------
# Stage 6: assembly
# ---------------------------------------------------------------------------
def next_anchor(beats: list[BeatBoundary], i: int, n_samples: int) -> int:
    """First anchor of beat ``i+1``: q_onset -> r_peak -> end of signal.

    Used to cap the T-wave search window so a T can never reach the next
    beat (the audit's t_offset -> next-p collision is made structurally
    impossible: the next beat's P search starts after this beat's T).
    """
    if i + 1 < len(beats):
        nb = beats[i + 1]
        if nb.q_onset > 0:
            return nb.q_onset
        if nb.r_peak > 0:
            return nb.r_peak
    return n_samples


def paint_state_labels(beats: list[BeatBoundary], T: int) -> np.ndarray:
    """Paint the per-sample 9-label state array (``DSP_STATE_LABELS`` order).

    Regions are painted from the validated beat geometry only; a beat whose
    P/T were rejected contributes no P/T runs.  Inter-beat diastole after
    each beat's last wave is TP; everything before the first wave is ISO.
    """
    labels = np.zeros(T, dtype=np.int64)

    def _paint(a: int, b: int, label: int) -> None:
        a, b = max(0, int(a)), min(T - 1, int(b))
        if a <= b:
            labels[a:b + 1] = label

    # First pass: beat-local regions in label order.
    for b in beats:
        if b.p_onset >= 0 and b.p_offset > b.p_onset:
            _paint(b.p_onset, b.p_offset, _LABEL_P)
            _paint(b.p_offset + 1, b.q_onset - 1, _LABEL_PR)
        _paint(b.q_onset, b.r_peak - 1, _LABEL_Q)
        _paint(b.r_peak, b.r_peak, _LABEL_R)
        _paint(b.r_peak + 1, b.s_offset, _LABEL_S)
        if b.t_onset >= 0 and b.t_offset > b.t_onset:
            _paint(b.s_offset + 1, b.t_onset - 1, _LABEL_ST)
            _paint(b.t_onset, b.t_offset, _LABEL_T)
        else:
            _paint(b.s_offset + 1, b.s_offset + 1, _LABEL_ST)

    # Second pass: TP diastole between this beat's last wave and the next
    # beat's first painted sample.
    for i, b in enumerate(beats):
        end_of_wave = b.t_offset if (b.t_onset >= 0 and b.t_offset > b.t_onset) \
            else b.s_offset
        nxt = beats[i + 1] if i + 1 < len(beats) else None
        next_start = T
        if nxt is not None:
            next_start = nxt.p_onset if nxt.p_onset >= 0 else nxt.q_onset
        _paint(end_of_wave + 1, next_start - 1, _LABEL_TP)
    return labels


# ---------------------------------------------------------------------------
# The segmenter
# ---------------------------------------------------------------------------
class DSPECGSegmenter:
    """Training-free classical ECG delineator with the ``ECGSegmenter`` API.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    preprocessor : ECGPreprocessor or None
        Created with defaults if None (0.5-40 Hz bandpass, 50 Hz notch,
        baseline removal, z-score).
    config : DSPDelineationConfig or None
        Thresholds; defaults if None.
    """

    def __init__(self, fs: float,
                 preprocessor: ECGPreprocessor | None = None,
                 config: DSPDelineationConfig | None = None) -> None:
        if fs <= 0:
            raise ValueError(f"Sampling frequency must be positive, got {fs}")
        self.fs = float(fs)
        self.preprocessor = preprocessor or ECGPreprocessor(fs=self.fs)
        self.config = config or DSPDelineationConfig()

    # ------------------------------------------------------------------
    def _empty_result(self, T: int, filtered: np.ndarray) -> SegmentResult:
        """Valid, beat-free result for degenerate inputs (never an exception)."""
        return SegmentResult(
            state_labels=np.zeros(T, dtype=np.int64),
            state_names=[DSP_STATE_LABELS[0]] * T,
            beats=[],
            log_likelihood=0.0,
            filtered_ecg=filtered,
            features=np.zeros((T, 3), dtype=np.float64),
            fs=self.fs,
        )

    # ------------------------------------------------------------------
    def segment(self, raw_ecg: np.ndarray) -> SegmentResult:
        """Preprocess -> R detection -> QRS/P/T delineation -> boundaries.

        Parameters
        ----------
        raw_ecg : np.ndarray, shape (N,)
            Raw single-lead ECG.  The input array is never modified.

        Returns
        -------
        SegmentResult
            ``beats`` is a list of :class:`BeatBoundary` with
            ``q_onset/r_peak/s_offset`` always set (else the beat is
            dropped) and P/T fiducials set where waves passed the gates
            (``-1`` otherwise, provenance ``'dsp'``).
        """
        cfg = self.config
        raw = np.asarray(raw_ecg, dtype=np.float64)
        if raw.ndim != 1 or raw.size < max(int(0.5 * self.fs), 8) \
                or float(np.std(raw)) < 1e-12:
            return self._empty_result(max(raw.size, 1),
                                      np.zeros(max(raw.size, 1)))
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        clean = self.preprocessor.preprocess(raw)
        T = len(clean)

        # ---- Stage 2: R peaks ------------------------------------------
        env = build_qrs_envelope(clean, self.fs, cfg)
        cands = detect_r_candidates(env, self.fs, cfg)
        if cands.size == 0:
            return self._empty_result(T, clean)
        r_peaks = [refine_r_peak(clean, int(c), self.fs, cfg) for c in cands]
        r_arr = np.asarray(sorted(r_peaks), dtype=np.int64)
        r_arr = dedup_r_peaks(r_arr, env, self.fs, cfg)
        r_arr = rr_plausibility_filter(r_arr, env, self.fs, cfg)
        if len(r_arr) == 0:
            return self._empty_result(T, clean)

        # ---- Stage 3: provisional QRS boundaries for every beat ---------
        d1s = uniform_filter1d(
            np.gradient(clean), size=_odd_window(cfg.qrs_deriv_smooth_s * self.fs))
        sigma_d = robust_sigma(d1s)
        beats: list[BeatBoundary] = []
        for k, r in enumerate(r_arr):
            q_on, s_off = estimate_qrs_boundaries(
                clean, int(r), self.fs, cfg, noise_sigma=sigma_d)
            beats.append(BeatBoundary(beat_id=k, q_onset=q_on,
                                      r_peak=int(r), s_offset=s_off))

        # ---- Stage 4/5: P and T waves, forward pass ---------------------
        p_sig = bandpass(clean, self.fs, cfg.p_band_hz)
        t_sig = bandpass(clean, self.fs, cfg.t_band_hz)
        d1_p = np.gradient(p_sig)
        d1_t = np.gradient(t_sig)
        # Record-level fallback references (quietest TP plateau per signal),
        # used when a beat's own TP gap is too short to measure noise on.
        min_quiet = int(round(0.05 * self.fs))
        gref_p = quietest_tp_ref(r_arr, p_sig, self.fs, cfg)
        gref_t = quietest_tp_ref(r_arr, t_sig, self.fs, cfg)
        rrs = np.diff(r_arr)
        rr_med = float(np.median(rrs)) if rrs.size else 1.0 * self.fs

        prev_end = 0  # last sample occupied by the previous beat (T or QRS)
        for k, b in enumerate(beats):
            # Isoelectric reference shared by the P and T gates: the TP/ISO
            # quiet stretch before this beat's P wave.
            search_start = max(b.r_peak - int(round(cfg.p_search_before_s * self.fs)),
                               prev_end + 1, 1)
            quiet_ref = (0, search_start) if k == 0 \
                else (prev_end + 1, search_start)
            gap_ok = quiet_ref[1] - quiet_ref[0] >= min_quiet
            if not gap_ok:
                # this beat's TP gap collapsed (fast rhythm): use the
                # quietest record-level plateau instead of the whole record
                # (which contains every QRS/T and would blind the gates)
                quiet_ref = gref_p or (0, T - 1)
            quiet_ref_t = quiet_ref if gap_ok else (gref_t or (0, T - 1))

            # -- P wave ---------------------------------------------------
            p = find_p_wave(p_sig, d1_p, quiet_ref, self.fs,
                            b.r_peak, b.q_onset, search_start, cfg)
            b.p_onset, b.p_offset = p[0], p[1]

            # -- T wave ---------------------------------------------------
            rr_k = float(rrs[k - 1]) if k > 0 else rr_med
            t_cap = min(
                next_anchor(beats, k, T) - 1,
                b.r_peak + int(round(max(cfg.t_search_rr_frac * rr_k,
                                         cfg.t_search_min_lead_s) * self.fs)),
                b.r_peak + int(round(cfg.t_search_abs_max_s * self.fs)),
            )
            if quiet_ref_t[1] - quiet_ref_t[0] < min_quiet:
                quiet_ref_t = gref_t or (0, T - 1)
            t = find_t_wave(t_sig, d1_t, quiet_ref_t, self.fs,
                            b.r_peak, b.s_offset,
                            max(t_cap, b.s_offset + 1), cfg)
            b.t_onset, b.t_offset = t[0], t[1]

            # -- P wave ---------------------------------------------------
            p = find_p_wave(p_sig, d1_p, quiet_ref, self.fs,
                            b.r_peak, b.q_onset, search_start, cfg)
            b.p_onset, b.p_offset = p[0], p[1]

            # -- T wave ---------------------------------------------------
            rr_k = float(rrs[k - 1]) if k > 0 else rr_med
            t_cap = min(
                next_anchor(beats, k, T) - 1,
                b.r_peak + int(round(max(cfg.t_search_rr_frac * rr_k,
                                         cfg.t_search_min_lead_s) * self.fs)),
                b.r_peak + int(round(cfg.t_search_abs_max_s * self.fs)),
            )
            t = find_t_wave(t_sig, d1_t, quiet_ref, self.fs,
                            b.r_peak, b.s_offset,
                            max(t_cap, b.s_offset + 1), cfg)
            b.t_onset, b.t_offset = t[0], t[1]

            prev_end = max(prev_end, b.t_offset if b.t_offset >= 0 else b.s_offset)

        # ---- Stage 6: derived markers, hygiene, labels ------------------
        prev_tail = 0  # first sample after the previous beat's last wave
        for k, b in enumerate(beats):
            b.p_source = 'dsp'
            b.t_source = 'dsp'
            b.pr_start = b.p_offset + 1 if b.p_offset >= 0 else -1
            b.st_start = b.s_offset + 1
            b.tp_start = b.t_offset + 1 if b.t_offset >= 0 else -1
            if k == 0:
                b.iso_start = max(0, (b.p_onset if b.p_onset >= 0 else b.q_onset)
                                  - int(round(cfg.iso_lead_pad_s * self.fs)))
            else:
                b.iso_start = prev_tail
            prev_tail = (b.t_offset if b.t_offset >= 0 else b.s_offset) + 1

        # Cross-beat hygiene: a P must not touch the previous beat's tail.
        for k in range(1, len(beats)):
            b, prev = beats[k], beats[k - 1]
            prev_t = prev.t_offset if prev.t_offset >= 0 else -1
            if b.p_onset >= 0 and prev_t >= 0 and b.p_onset <= prev_t:
                b.p_onset = b.p_offset = b.pr_start = -1  # false P: drop it

        # Reindex and paint.
        for k, b in enumerate(beats):
            b.beat_id = k
        state_labels = paint_state_labels(beats, T)
        state_names = [DSP_STATE_LABELS[l] for l in state_labels]

        features = np.column_stack([
            clean,
            np.gradient(clean),
            np.gradient(np.gradient(clean)),
        ])
        return SegmentResult(
            state_labels=state_labels,
            state_names=state_names,
            beats=beats,
            log_likelihood=0.0,
            filtered_ecg=clean,
            features=features,
            fs=self.fs,
        )
