"""Chest Lead (V1-V6) Misplacement Detection.

Detects common chest lead placement errors:
  1. V1/V2 swap — V1 R > V2 R
  2. R-wave progression break — non-monotonic
  3. High placement — P-wave negative in V1-V2, early transition
  4. Low placement — large R in V1-V2, late transition
  5. Transition zone shift — R=S point outside V3-V4
  6. Single lead outlier — one lead doesn't fit the smooth curve

Usage:
    from ecg_waveform_extraction.src.chest_lead_analyzer import ChestLeadAnalyzer
    analyzer = ChestLeadAnalyzer()
    result = analyzer.analyze(aecg_data)
"""

import numpy as np
from .preprocessing import ECGPreprocessor

CHEST_LEADS = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']


class ChestLeadResult:
    """Chest lead analysis result."""
    def __init__(self):
        self.record = ''
        self.r_wave = {}       # lead -> max amplitude
        self.s_wave = {}       # lead -> abs(min amplitude)
        self.rs_ratio = {}     # lead -> R/(R+S)
        self.r_progression_ok = True
        self.v1_v2_swapped = False
        self.transition_zone = ''  # e.g. 'V3-V4'
        self.high_placement = False
        self.low_placement = False
        self.single_outlier = None  # lead name or None
        self.flags = []        # list of warning strings


class ChestLeadAnalyzer:
    """Analyze V1-V6 chest leads for misplacement patterns.

    Parameters
    ----------
    fs : float — sampling frequency
    max_samples : int — truncate signals
    """

    def __init__(self, fs: float = 250.0, max_samples: int = 16000):
        self.fs = fs
        self.max_samples = max_samples

    def analyze(self, aecg_data: dict) -> ChestLeadResult:
        """Run full chest lead analysis on one record.

        Parameters
        ----------
        aecg_data : dict — output of parse_aecg()

        Returns
        -------
        ChestLeadResult
        """
        result = ChestLeadResult()
        result.record = aecg_data.get('filename', 'unknown')

        signals = aecg_data.get('signals', {})
        fs_actual = aecg_data.get('fs') or self.fs  # None-safe fallback
        prep = ECGPreprocessor(fs=fs_actual)

        # ---- Step 1: Extract R and S for each chest lead ----
        for ln in CHEST_LEADS:
            sig = signals.get(ln)
            if sig is None:
                result.r_wave[ln] = 0
                result.s_wave[ln] = 0
                result.rs_ratio[ln] = 0
                continue

            clean = prep.preprocess(sig[:self.max_samples].astype(np.float64))

            r = float(np.max(clean))
            s = float(np.abs(np.min(clean)))
            ratio = r / (r + s + 0.001)

            result.r_wave[ln] = round(r, 2)
            result.s_wave[ln] = round(s, 2)
            result.rs_ratio[ln] = round(ratio, 3)

            # P-wave check for high placement (V1-V2)
            if ln in ('V1', 'V2'):
                # P-wave should be in the first 200ms before QRS
                # Simple check: early signal polarity
                early_seg = clean[:int(0.12 * fs_actual)]  # first 120ms
                if len(early_seg) > 10:
                    p_min = float(np.min(early_seg))
                    p_max = float(np.max(early_seg))
                    # If negative peak dominates → P is negative → high placement
                    if p_min < -p_max * 1.5:
                        result.flags.append(f'{ln} P-wave negative (high placement)')

        # ---- Step 2: V1/V2 swap detection ----
        if result.r_wave['V1'] > result.r_wave['V2'] * 1.3:
            result.v1_v2_swapped = True
            result.flags.append(
                f'V1 R({result.r_wave["V1"]:.1f}) > V2 R({result.r_wave["V2"]:.1f}) — SWAP suspected')

        # ---- Step 3: R-wave progression monotonicity ----
        r_vals = [result.r_wave[ln] for ln in CHEST_LEADS]
        # Check monotonic increase (allow 10% tolerance for noise)
        dips = []
        for i in range(len(r_vals) - 1):
            if r_vals[i] > r_vals[i + 1] * 1.15:
                dips.append(f'{CHEST_LEADS[i]}→{CHEST_LEADS[i+1]}')
        if dips:
            result.r_progression_ok = False
            result.flags.append(f'R-wave dips: {", ".join(dips)}')

        # ---- Step 4: Transition zone (where R/S ≈ 1) ----
        transition_idx = None
        for i, ln in enumerate(CHEST_LEADS):
            if result.rs_ratio[ln] >= 0.45:  # R >= S approximate
                transition_idx = i
                break
        if transition_idx is None:
            result.transition_zone = 'V5-V6 (late)'
            result.flags.append('Transition zone late — possible low placement')
        elif transition_idx <= 1:
            result.transition_zone = f'{CHEST_LEADS[transition_idx]} (early)'
            result.flags.append('Transition zone early — possible high placement')
        else:
            # Find the exact pair
            for i in range(len(CHEST_LEADS) - 1):
                if result.rs_ratio[CHEST_LEADS[i]] < 0.45 <= result.rs_ratio[CHEST_LEADS[i + 1]]:
                    result.transition_zone = f'{CHEST_LEADS[i]}-{CHEST_LEADS[i+1]}'
                    break
            if not result.transition_zone:
                result.transition_zone = 'V3-V4'

        if result.transition_zone not in ('V2-V3', 'V3-V4', 'V3-V4'):
            # Only flag if clearly abnormal
            if 'early' in result.transition_zone:
                result.high_placement = True
            if 'late' in result.transition_zone:
                result.low_placement = True

        # ---- Step 5: Single outlier detection ----
        # Check if one lead deviates significantly from neighbors
        for i in range(1, len(CHEST_LEADS) - 1):
            prev_r = result.rs_ratio[CHEST_LEADS[i - 1]]
            curr_r = result.rs_ratio[CHEST_LEADS[i]]
            next_r = result.rs_ratio[CHEST_LEADS[i + 1]]
            expected = (prev_r + next_r) / 2
            if abs(curr_r - expected) > 0.25:
                result.single_outlier = CHEST_LEADS[i]
                result.flags.append(f'{CHEST_LEADS[i]} outlier (R/S={curr_r:.2f}, expected ~{expected:.2f})')

        return result


# ---------------------------------------------------------------------------
# Quick summary for xlsx export
# ---------------------------------------------------------------------------
def chest_result_to_dict(r: ChestLeadResult) -> dict:
    """Serialize ChestLeadResult to a flat dict for xlsx."""
    return {
        'V1_R': r.r_wave.get('V1', 0), 'V1_S': r.s_wave.get('V1', 0),
        'V2_R': r.r_wave.get('V2', 0), 'V2_S': r.s_wave.get('V2', 0),
        'V3_R': r.r_wave.get('V3', 0), 'V3_S': r.s_wave.get('V3', 0),
        'V4_R': r.r_wave.get('V4', 0), 'V4_S': r.s_wave.get('V4', 0),
        'V5_R': r.r_wave.get('V5', 0), 'V5_S': r.s_wave.get('V5', 0),
        'V6_R': r.r_wave.get('V6', 0), 'V6_S': r.s_wave.get('V6', 0),
        'V1_R/S': r.rs_ratio.get('V1', 0), 'V2_R/S': r.rs_ratio.get('V2', 0),
        'V3_R/S': r.rs_ratio.get('V3', 0), 'V4_R/S': r.rs_ratio.get('V4', 0),
        'V5_R/S': r.rs_ratio.get('V5', 0), 'V6_R/S': r.rs_ratio.get('V6', 0),
        'V1-V2_SWAP': r.v1_v2_swapped,
        'R_Progression': 'OK' if r.r_progression_ok else 'BROKEN',
        'Transition': r.transition_zone,
        'High_Place': r.high_placement,
        'Low_Place': r.low_placement,
        'Outlier': r.single_outlier or '',
        'Chest_Flags': '; '.join(r.flags) if r.flags else 'OK',
    }
