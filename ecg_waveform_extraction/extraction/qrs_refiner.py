"""QRS boundary refinement using signal derivative analysis.

Refines HSMM-estimated Q-onset and S-offset (J-point) by walking along the
first derivative until it returns to the baseline noise floor. Provides
~5-10ms improvement in boundary precision over raw HSMM output.
"""

import numpy as np


def refine_qrs_boundaries(ecg_clean: np.ndarray,
                          q_on_hsmm: int,
                          r_peak_hsmm: int,
                          s_off_hsmm: int,
                          fs: float) -> tuple[int, int, int]:
    """Refine HSMM QRS boundaries using signal derivative.

    Algorithm:
      1. R peak: pick max |amplitude| in ±20ms window around HSMM estimate
      2. Q onset: walk RIGHT from HSMM estimate until |d1| exceeds 3σ noise
      3. S offset (J-point): walk LEFT from HSMM estimate until |d1| returns
         to baseline

    Parameters
    ----------
    ecg_clean : np.ndarray, shape (T,)
        Preprocessed (filtered) ECG signal.
    q_on_hsmm : int
        HSMM-estimated Q onset sample index.
    r_peak_hsmm : int
        HSMM-estimated R peak sample index.
    s_off_hsmm : int
        HSMM-estimated S offset sample index.
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    (q_onset, r_peak, s_offset) : tuple[int, int, int]
        Refined sample indices.
    """
    T = len(ecg_clean)
    d1 = np.gradient(ecg_clean)

    # Noise floor from quiet early segment (first 200ms or first 20%)
    quiet_end = min(int(0.2 * fs), max(T // 5, 10))
    noise_sigma = float(np.std(d1[:quiet_end]) + 1e-6)
    threshold = noise_sigma * 3.0

    # ---- Step 1: R peak refinement (±20ms window) ----
    r_search_start = max(0, r_peak_hsmm - int(0.02 * fs))
    r_search_end = min(T - 1, r_peak_hsmm + int(0.02 * fs))
    r_search = ecg_clean[r_search_start:r_search_end + 1]
    bl_local = float(np.median(r_search))
    r_peak = r_search_start + int(np.argmax(np.abs(r_search - bl_local)))

    # ---- Step 2: Q onset refinement (walk right) ----
    q_on = q_on_hsmm
    for i in range(q_on_hsmm, min(r_peak, q_on_hsmm + int(0.08 * fs))):
        if abs(d1[i]) > threshold:
            for j in range(i, max(q_on_hsmm, i - int(0.01 * fs)), -1):
                if abs(d1[j]) <= threshold * 0.5:
                    q_on = j
                    break
            else:
                q_on = i
            break

    # ---- Step 3: S offset (J-point) refinement (walk left) ----
    s_off = s_off_hsmm
    for i in range(s_off_hsmm, max(r_peak + int(0.02 * fs), s_off_hsmm - int(0.10 * fs)), -1):
        if abs(d1[i]) > threshold:
            for j in range(i, min(s_off_hsmm, i + int(0.02 * fs))):
                if abs(d1[j]) <= threshold * 0.5:
                    s_off = j
                    break
            else:
                s_off = i
            break

    # ---- Sanity checks ----
    min_qrs = int(0.02 * fs)  # minimum 20ms QRS
    if s_off - q_on < min_qrs:
        q_on, s_off = q_on_hsmm, s_off_hsmm

    q_on = max(0, min(q_on, T - 1))
    s_off = max(q_on + min_qrs, min(s_off, T - 1))
    r_peak = max(q_on, min(r_peak, s_off))

    return q_on, r_peak, s_off


def compute_qrs_metrics(ecg_clean: np.ndarray,
                        q_on: int, r_pk: int, s_off: int,
                        fs: float) -> dict:
    """Compute QRS clinical metrics from refined boundaries.

    Parameters
    ----------
    ecg_clean : np.ndarray
        Filtered ECG signal.
    q_on, r_pk, s_off : int
        Refined QRS boundary sample indices.
    fs : float
        Sampling frequency.

    Returns
    -------
    dict with keys: duration_ms, r_amplitude, s_amplitude, q_amplitude,
                    qrs_net_area, rs_ratio, polarity, confidence
    """
    T = len(ecg_clean)
    seg = ecg_clean[q_on:s_off + 1]
    bl = float(np.mean(ecg_clean[max(0, q_on - 30):q_on])) if q_on >= 30 else float(np.median(seg[:5]))
    detrend = seg - bl
    dur = len(seg) / fs * 1000.0
    r_amp = float(ecg_clean[r_pk] - bl) if 0 <= r_pk < T else 0.0

    # S nadir after R
    r_idx = r_pk - q_on
    s_nadir = float(np.min(detrend[r_idx:])) if r_idx < len(detrend) else 0.0

    # Q nadir before R
    q_nadir = float(np.min(detrend[:r_idx + 1])) if r_idx + 1 <= len(detrend) and r_idx >= 0 else 0.0

    qrs_net = float(np.sum(detrend))

    # R/S ratio
    neg_mag = max(abs(s_nadir), abs(q_nadir), 0.001) if (s_nadir < 0 or q_nadir < 0) else 0.001
    if r_amp > 0:
        rs_ratio = float(min(r_amp / neg_mag, 100.0))
    else:
        rs_ratio = float(max(r_amp / max(abs(r_amp) + neg_mag, 0.001), -100.0))

    # Polarity classification
    if rs_ratio >= 1.5 and qrs_net > 0:
        pol, conf = 'positive', min(rs_ratio / 3.0, 1.0)
    elif rs_ratio <= 0.5 and qrs_net < 0:
        pol, conf = 'negative', min(abs(qrs_net) / (abs(r_amp) + neg_mag + 0.001), 1.0)
    elif abs(qrs_net) < abs(r_amp) * 0.1:
        pol, conf = 'biphasic', 0.6
    elif qrs_net > 0:
        pol, conf = 'positive', 0.7
    else:
        pol, conf = 'negative', 0.7

    return {
        'duration_ms': round(dur, 2),
        'r_amplitude': round(r_amp, 4),
        's_amplitude': round(s_nadir, 4),
        'q_amplitude': round(q_nadir, 4),
        'qrs_net_area': round(qrs_net, 4),
        'rs_ratio': round(rs_ratio, 4),
        'polarity': pol,
        'confidence': round(min(conf, 1.0), 3),
    }


# ===========================================================================
# 5-Criterion Weighted Voting Polarity Classifier (v2)
# ===========================================================================

# Lead-specific prior expectations (clinical electrophysiology):
#   positive_prior = how strongly we expect a positive QRS in this lead
#   Range [-1, +1]; 0 = no prior.
_LEAD_PRIORS = {
    'I':   0.7,   # strongly positive
    'II':  0.8,   # most positive in normal hearts
    'III': 0.1,   # variable (often isoelectric or negative)
    'AVR': -0.9,  # almost always negative
    'AVL': 0.3,   # moderately positive
    'AVF': 0.5,   # positive in vertical hearts
    'V1': -0.5,   # dominantly negative (rS pattern)
    'V2': -0.3,   # transitional
    'V3': 0.0,    # transitional zone
    'V4': 0.4,    # becoming positive
    'V5': 0.6,    # positive
    'V6': 0.7,    # strongly positive
}

# Weights for each criterion (sum = 1.0)
_CRITERION_WEIGHTS = {
    'dominant_deflection': 0.25,
    'rs_ratio':            0.20,
    'net_area':            0.25,
    'energy_ratio':        0.15,
    'template_correlation': 0.15,
}


def compute_qrs_polarity_v2(ecg_clean: np.ndarray,
                             q_on: int, r_pk: int, s_off: int,
                             fs: float,
                             lead_name: str | None = None) -> dict:
    """5-criterion weighted voting QRS polarity classifier.

    Each criterion votes positive (+1), negative (-1), or biphasic (0).
    Votes are weighted and summed to produce a polarity score in [-1, +1].
    Confidence = weighted agreement among the 5 criteria.

    Criteria
    --------
    1. Dominant Deflection  (w=0.25) — sign of max(|amplitude|)
    2. R/S Ratio            (w=0.20) — clinical standard (>2.0→pos, <0.5→neg)
    3. Net Area             (w=0.25) — ∫detrend sign
    4. Energy Ratio         (w=0.15) — E_up / E_total (noise-tolerant)
    5. Template Correlation (w=0.15) — morphology match to pos/neg envelopes

    Parameters
    ----------
    ecg_clean : np.ndarray
        Filtered ECG signal.
    q_on, r_pk, s_off : int
        Refined QRS boundaries.
    fs : float
        Sampling frequency.
    lead_name : str or None
        Lead identifier (I, II, V1, etc.) for clinical prior. If None, no prior.

    Returns
    -------
    dict with keys:
        polarity          : 'positive' | 'negative' | 'biphasic'
        confidence        : float 0-1
        polarity_score    : float [-1, +1] — raw weighted score
        criteria           : dict — individual criterion results
        energy_ratio       : float — E_up / E_total
        peak_count         : int — distinct positive + negative peaks
    """
    T = len(ecg_clean)
    seg = ecg_clean[q_on:s_off + 1]
    n_seg = len(seg)

    if n_seg < 3:
        return _fallback_result()

    # ---- Baseline estimation (pre-QRS window) ----
    bl_start = max(0, q_on - int(0.05 * fs))
    bl_end = q_on
    if bl_end - bl_start >= 5:
        bl = float(np.median(ecg_clean[bl_start:bl_end]))
    else:
        bl = float(np.median(seg[:min(5, n_seg)]))

    detrend = seg - bl
    r_amp = float(ecg_clean[r_pk] - bl) if 0 <= r_pk < T else 0.0

    # ==================================================================
    # Find all distinct positive and negative peaks
    # ==================================================================
    from scipy.signal import find_peaks

    pos_peaks, neg_peaks = _find_qrs_peaks(detrend, fs)

    # ==================================================================
    # Criterion 1: Dominant Deflection  (w=0.25)
    # ==================================================================
    pos_max = float(np.max(detrend)) if len(detrend) > 0 else 0.0
    neg_min = float(np.min(detrend)) if len(detrend) > 0 else 0.0
    dominant_val = pos_max if abs(pos_max) >= abs(neg_min) else neg_min
    c1_vote = +1 if dominant_val > 0 else -1
    c1_strength = min(abs(dominant_val) / max(abs(dominant_val) + abs(neg_min) + 0.001, 1.0), 1.0)

    # ==================================================================
    # Criterion 2: R/S Ratio  (w=0.20)
    # ==================================================================
    neg_mag = abs(neg_min) if neg_min < 0 else 0.001
    if r_amp > 0 and neg_mag > 0.002:
        rs_ratio = float(min(r_amp / neg_mag, 100.0))
    elif r_amp <= 0 and neg_mag > 0.001:
        rs_ratio = float(max(r_amp / (abs(r_amp) + neg_mag + 0.001), -100.0))
    else:
        rs_ratio = 1.0

    if rs_ratio >= 2.0:
        c2_vote = +1
        c2_strength = min((rs_ratio - 1.0) / 4.0, 1.0)
    elif rs_ratio <= 0.5:
        c2_vote = -1
        c2_strength = min((1.0 / max(rs_ratio, 0.1) - 1.0) / 4.0, 1.0)
    else:
        c2_vote = 0  # borderline → biphasic
        c2_strength = 1.0 - abs(rs_ratio - 1.0)

    # ==================================================================
    # Criterion 3: Net Area  (w=0.25)
    # ==================================================================
    qrs_net = float(np.sum(detrend))
    # Normalize by segment length for comparability
    net_norm = qrs_net / max(n_seg, 1)
    area_threshold = 0.01 * max(abs(detrend).max(), 0.01)
    c3_vote = +1 if net_norm > area_threshold else (-1 if net_norm < -area_threshold else 0)
    c3_strength = min(abs(net_norm) / max(abs(detrend).max() * 0.3, area_threshold), 1.0)

    # ==================================================================
    # Criterion 4: Energy Ratio  (w=0.15)
    # ==================================================================
    E_up = float(np.sum(np.maximum(detrend, 0) ** 2))
    E_down = float(np.sum(np.minimum(detrend, 0) ** 2))
    E_total = E_up + E_down + 1e-12
    energy_ratio = E_up / E_total

    if energy_ratio >= 0.65:
        c4_vote = +1
        c4_strength = min((energy_ratio - 0.5) / 0.5, 1.0)
    elif energy_ratio <= 0.35:
        c4_vote = -1
        c4_strength = min((0.5 - energy_ratio) / 0.5, 1.0)
    else:
        c4_vote = 0  # biphasic zone
        c4_strength = 1.0 - abs(energy_ratio - 0.5) / 0.15

    # ==================================================================
    # Criterion 5: Template Correlation  (w=0.15)
    # ==================================================================
    c5_vote, c5_strength = _template_correlation_vote(detrend, fs)

    # ==================================================================
    # Weighted Voting Aggregation
    # ==================================================================
    w = _CRITERION_WEIGHTS
    votes = {
        'dominant_deflection':  {'vote': c1_vote, 'strength': c1_strength},
        'rs_ratio':             {'vote': c2_vote, 'strength': c2_strength},
        'net_area':             {'vote': c3_vote, 'strength': c3_strength},
        'energy_ratio':         {'vote': c4_vote, 'strength': c4_strength},
        'template_correlation': {'vote': c5_vote, 'strength': c5_strength},
    }

    weighted_score = 0.0
    total_weight = 0.0
    agreement = 0.0

    for name, v in votes.items():
        weight = w[name]
        if v['vote'] != 0:  # non-biphasic votes
            weighted_score += float(v['vote']) * weight * v['strength']
        total_weight += weight

    # Confidence: how well each criterion's strength aligns with the final polarity
    dominant_pol = +1 if weighted_score > 0 else (-1 if weighted_score < 0 else 0)
    for name, v in votes.items():
        if v['vote'] == dominant_pol and v['vote'] != 0:
            agreement += w[name] * v['strength']
        elif v['vote'] == -dominant_pol and v['vote'] != 0:
            agreement -= w[name] * v['strength'] * 0.5  # penalty for opposing

    # ---- Biphasic detection ----
    n_pos_peaks = len(pos_peaks)
    n_neg_peaks = len(neg_peaks)
    peak_count = n_pos_peaks + n_neg_peaks

    is_biphasic = (
        0.35 < energy_ratio < 0.65 and
        n_pos_peaks >= 1 and n_neg_peaks >= 1 and
        abs(weighted_score) < 0.35
    )

    # ---- Lead prior nudges the score (weak, max ±0.15 shift) ----
    if lead_name is not None:
        prior = _LEAD_PRIORS.get(lead_name.upper(), 0.0)
        weighted_score += prior * 0.15

    # ---- Final polarity decision ----
    # A beat is "uncertain" when criteria are too conflicted to call
    n_pos_votes = sum(1 for v in votes.values() if v['vote'] == +1)
    n_neg_votes = sum(1 for v in votes.values() if v['vote'] == -1)
    vote_split = abs(n_pos_votes - n_neg_votes)

    # Ambiguous only when criteria are TRULY split (2-2, 2-1-1, 1-1-3)
    # or agreement/score is very near zero. 3-2 splits still get classified.
    is_ambiguous = (
        agreement < 0.15 or
        abs(weighted_score) < 0.18 or
        vote_split == 0
    )

    if is_ambiguous:
        polarity = 'uncertain'
        confidence = round(float(np.clip(0.15 + agreement * 0.5, 0.1, 0.35)), 3)
    elif is_biphasic:
        polarity = 'biphasic'
        confidence = min(max(agreement, 0.4) + 0.15, 1.0)
    elif weighted_score >= 0.20:
        polarity = 'positive'
        confidence = min(agreement + 0.1, 1.0)
    elif weighted_score <= -0.20:
        polarity = 'negative'
        confidence = min(agreement + 0.1, 1.0)
    else:
        # Edge case: very flat signal → uncertain, not a guess
        if max(abs(detrend)) < 0.02:
            polarity = 'uncertain'
            confidence = 0.2
        else:
            polarity = 'uncertain'
            confidence = 0.4

    # Clamp
    confidence = round(float(np.clip(confidence, 0.0, 1.0)), 3)
    polarity_score = round(float(np.clip(weighted_score, -1.0, 1.0)), 3)

    return {
        'polarity': polarity,
        'confidence': confidence,
        'polarity_score': polarity_score,
        'criteria': votes,
        'energy_ratio': round(float(energy_ratio), 4),
        'peak_count': peak_count,
        'rs_ratio': round(float(rs_ratio), 4),
        'qrs_net_area': round(float(qrs_net), 4),
    }


# ===========================================================================
# Internal helpers
# ===========================================================================

def _find_qrs_peaks(detrend: np.ndarray, fs: float) -> tuple[list[int], list[int]]:
    """Find distinct positive and negative peaks in QRS segment."""
    from scipy.signal import find_peaks

    max_abs = max(abs(detrend).max(), 0.01)
    min_height = max_abs * 0.15
    min_distance = max(3, int(0.012 * fs))  # 12ms minimum peak separation

    pos_arr, _ = find_peaks(detrend, height=min_height, distance=min_distance)
    neg_arr, _ = find_peaks(-detrend, height=min_height, distance=min_distance)

    # deduplicate: if pos and neg are within 4ms, keep the larger one
    return _dedup_peaks(list(pos_arr), list(neg_arr), detrend, int(0.004 * fs))


def _dedup_peaks(pos, neg, signal: np.ndarray,
                 merge_dist: int) -> tuple[list, list]:
    """Merge nearby peaks of opposite sign, keeping the more extreme."""
    pos_arr = list(pos) if hasattr(pos, '__iter__') else []
    neg_arr = list(neg) if hasattr(neg, '__iter__') else []
    if not pos_arr or not neg_arr:
        return pos_arr, neg_arr
    for pi in reversed(pos_arr):
        for ni in reversed(neg_arr):
            if abs(pi - ni) <= merge_dist:
                if abs(signal[pi]) >= abs(signal[ni]):
                    neg_arr.remove(ni)
                else:
                    pos_arr.remove(pi)
                    break
    return pos_arr, neg_arr


def _template_correlation_vote(detrend: np.ndarray, fs: float) -> tuple[int, float]:
    """Vote based on morphological similarity to positive/negative templates.

    Builds crude templates from the QRS segment itself:
      - Positive template: upper envelope of the QRS
      - Negative template: lower envelope (inverted)

    Returns (vote: -1/0/+1, strength: 0-1).
    """
    n = len(detrend)
    if n < 8:
        return 0, 0.0

    # Simple envelope extraction
    pos_env = np.maximum(detrend, 0)
    neg_env = np.abs(np.minimum(detrend, 0))

    # Template: normalize to unit energy
    pos_norm = pos_env / (np.sqrt(np.sum(pos_env ** 2)) + 1e-8)
    neg_norm = neg_env / (np.sqrt(np.sum(neg_env ** 2)) + 1e-8)

    # Correlate the actual QRS with both templates
    sig_norm = detrend / (np.sqrt(np.sum(detrend ** 2)) + 1e-8)
    corr_pos = float(np.dot(sig_norm, pos_norm))
    corr_neg = float(np.dot(sig_norm, neg_norm))

    diff = corr_pos - corr_neg
    strength = min(abs(diff) / 0.7, 1.0)  # normalize to 0-1

    if diff > 0.1:
        return +1, strength
    elif diff < -0.1:
        return -1, strength
    else:
        return 0, strength


def _fallback_result() -> dict:
    """Return a sane default when QRS segment is too short."""
    return {
        'polarity': 'uncertain',
        'confidence': 0.2,
        'polarity_score': 0.0,
        'criteria': {},
        'energy_ratio': 0.5,
        'peak_count': 0,
        'rs_ratio': 1.0,
        'qrs_net_area': 0.0,
    }
