"""Smart HSMM observation-model initialization from signal characteristics.

Uses the feature space (amplitude, d1, d2) to roughly partition the signal
into ECG-state regions without hard thresholds or manual search windows.
This seeds the GMMs so Viterbi decoding can produce meaningful segmentations.

Principle: high |d1| → QRS complex, low |d1| + low |amp| → isoelectric,
medium |amp| + moderate |d1| → P/T waves. All done via relative ranking,
not absolute thresholds.
"""

import numpy as np
from .hsmm_model import HSMMModel


def smart_initialize_gmms(model: HSMMModel, features: np.ndarray):
    """Initialize all state GMMs from signal characteristics without annotations.

    Uses the inherent structure of the feature space:
      - Features are [amplitude, d1 (velocity), d2 (acceleration)]
      - QRS: highest |d1|, highest |amplitude|
      - P/T: moderate |d1|, moderate |amplitude|
      - ISO/PR/ST/TP: lowest |d1|, lowest |amplitude|

    Parameters
    ----------
    model : HSMMModel
        Model whose obs_dists will be initialized. Must already have topology set.
    features : np.ndarray, shape (T, 3)
        Feature vectors [amplitude, d1, d2].
    """
    T = features.shape[0]
    N = model.n_states

    if T < 100:
        return  # Too short

    amp = features[:, 0]
    d1 = features[:, 1]
    d2 = features[:, 2]

    # ---- 1. Compute "activity" score for each sample ----
    # Combines amplitude deviation and derivative energy
    amp_abs = np.abs(amp - np.median(amp))
    d1_abs = np.abs(d1)
    d2_abs = np.abs(d2)

    # Activity score: normalized combination
    activity = amp_abs + d1_abs * 0.5 + d2_abs * 0.3
    # Normalize to [0, 1]
    act_max = np.percentile(activity, 99)
    if act_max > 0:
        activity = activity / act_max

    # ---- 2. Find R-peak candidates: local maxima of |d1| in high-activity regions ----
    from scipy.signal import find_peaks

    # Adaptive prominence: 50th percentile of |d1| in active regions
    active_mask = activity > 0.3
    if active_mask.sum() < 10:
        return  # Not enough activity

    prominence = np.percentile(d1_abs[active_mask], 70)
    if prominence <= 0:
        prominence = 1e-3

    r_peaks, _ = find_peaks(np.abs(d1), height=prominence * 0.5, distance=int(model.fs * 0.2))
    # Also find on raw d1 (positive peaks for R)
    r_peaks2, _ = find_peaks(d1, height=prominence * 0.5, distance=int(model.fs * 0.2))

    # Merge and deduplicate
    all_peaks = np.unique(np.concatenate([r_peaks, r_peaks2]))
    if len(all_peaks) < 2:
        return

    # ---- 3. Assign rough state labels to windows around R-peaks ----
    fs = model.fs
    # Window sizes in samples (approximate physiological ranges)
    qrs_half = int(0.06 * fs)   # 60ms each side of R
    p_window = int(0.12 * fs)   # 120ms for P wave, centered ~160ms before R
    t_window = int(0.20 * fs)   # 200ms for T wave, centered ~250ms after R
    iso_pad = int(0.05 * fs)    # 50ms of ISO between waves

    # Collect feature samples per state
    samples_per_state = {j: [] for j in range(N)}

    for r in all_peaks:
        # Q wave: before R peak
        q_start = max(0, r - qrs_half)
        q_end = max(0, r - int(0.02 * fs))
        if q_end > q_start:
            sample_q = features[q_start:q_end]
            if len(sample_q) > 2:
                samples_per_state[3].append(sample_q)  # Q state

        # R wave: narrow window around R peak
        r_start = max(0, r - int(0.03 * fs))
        r_end = min(T - 1, r + int(0.03 * fs))
        if r_end > r_start:
            sample_r = features[r_start:r_end]
            if len(sample_r) > 2:
                samples_per_state[4].append(sample_r)

        # S wave: after R
        s_start = min(T - 1, r + int(0.02 * fs))
        s_end = min(T - 1, r + qrs_half)
        if s_end > s_start:
            sample_s = features[s_start:s_end]
            if len(sample_s) > 2:
                samples_per_state[5].append(sample_s)

        # P wave: ~160ms before R
        p_center = max(0, r - int(0.16 * fs))
        p_start = max(0, p_center - p_window // 2)
        p_end = min(T - 1, p_center + p_window // 2)
        if p_end > p_start:
            sample_p = features[p_start:p_end]
            if len(sample_p) > 2:
                samples_per_state[1].append(sample_p)

        # PR segment: between P and Q
        pr_start = min(T - 1, p_end + iso_pad)
        pr_end = max(0, q_start - iso_pad)
        if pr_end > pr_start:
            sample_pr = features[pr_start:pr_end]
            if len(sample_pr) > 2:
                samples_per_state[2].append(sample_pr)

        # ST segment: between S and T
        st_start = min(T - 1, s_end + iso_pad)
        t_center = min(T - 1, r + int(0.25 * fs))
        st_end = min(T - 1, t_center - t_window // 2)
        if st_end > st_start:
            sample_st = features[st_start:st_end]
            if len(sample_st) > 2:
                samples_per_state[6].append(sample_st)

        # T wave: ~250ms after R
        t_start = max(0, t_center - t_window // 2)
        t_end = min(T - 1, t_center + t_window // 2)
        if t_end > t_start:
            sample_t = features[t_start:t_end]
            if len(sample_t) > 2:
                samples_per_state[7].append(sample_t)

        # TP segment: between T end and next P
        tp_start = min(T - 1, t_end + iso_pad)
        tp_end = min(T - 1, r + int(0.8 * fs))  # ~next P area
        if tp_end > tp_start:
            sample_tp = features[tp_start:tp_end]
            if len(sample_tp) > 2:
                samples_per_state[8].append(sample_tp)

    # ISO: collect from flat regions (low activity)
    iso_mask = activity < np.percentile(activity, 20)
    if iso_mask.sum() > 10:
        iso_features = features[iso_mask]
        samples_per_state[0].append(iso_features)

    # ---- 4. Fit GMMs from collected samples ----
    for j in range(N):
        if not samples_per_state[j]:
            continue

        # Concatenate all samples for this state
        state_features = np.concatenate(samples_per_state[j], axis=0)

        if len(state_features) > model.n_gmm_components * 2:
            try:
                model.obs_dists[j].fit(state_features, max_iter=40, tol=1e-3)
            except Exception:
                pass  # Keep default random init

    # ---- 5. For states with no samples, use nearest neighbor state's GMM ----
    # (This helps when the recording is very short)
    for j in range(N):
        if not model.obs_dists[j]._fitted:
            # Copy from an adjacent state
            for nb in [j - 1, j + 1, j - 2, j + 2]:
                if 0 <= nb < N and model.obs_dists[nb]._fitted:
                    params = model.obs_dists[nb].get_params()
                    model.obs_dists[j].set_params(params)
                    break
