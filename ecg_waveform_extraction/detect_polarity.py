"""RA-LA Lead Reversal Detection via ECG Polarity Analysis.

Methods for detecting electrode reversal without manual review:
1. P-wave polarity in Lead II (should be positive in normal sinus rhythm)
2. QRS axis from multi-lead reconstruction
3. Lead I QRS net area sign
4. P-wave axis comparison

Uses existing HSMM segmentation output (p_waves.json + segmentation.json + filtered_ecg.npy).
"""

import sys
sys.path.insert(0, 'c:/LoyaltyLo/PythonProjects/ECG_engineering')

import os, json
import numpy as np

# ---- Config ----
DATA_DIR = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/output_rala_full'
AECG_DIR = 'C:/LoyaltyLo/datasets/RA-LA_Reversal/aECG'


def detect_by_p_wave(record_dir: str, fs: float = 1000.0) -> dict:
    """Method 1: P-wave polarity check in Lead II.

    Normal sinus: P-wave is positive (upward) in Lead II.
    RA-LA reversal: P-wave becomes negative (downward/inverted) in Lead II.

    Uses p_waves.json + filtered_ecg.npy from HSMM output.
    """
    pw_path = os.path.join(record_dir, 'p_waves.json')
    ecg_path = os.path.join(record_dir, 'filtered_ecg.npy')

    if not os.path.exists(pw_path) or not os.path.exists(ecg_path):
        return {'status': 'error', 'reason': 'missing files'}

    with open(pw_path) as f:
        p_waves = json.load(f)
    ecg = np.load(ecg_path)

    # For each P-wave, compute net area (integral)
    p_areas = []
    p_peak_signs = []
    for pw in p_waves:
        onset = pw['onset_sample']
        offset = pw['offset_sample']
        if onset < 0 or offset <= onset:
            continue
        segment = ecg[onset:offset + 1]
        baseline = np.mean(ecg[max(0, onset - 50):onset]) if onset >= 50 else np.mean(segment[:10])
        area = np.sum(segment - baseline)  # net area (positive = upward P)
        peak_val = np.max(np.abs(segment - baseline))  # check sign via peak
        peak_sign = 1 if np.max(segment - baseline) > abs(np.min(segment - baseline)) else -1
        p_areas.append(area)
        p_peak_signs.append(peak_sign)

    if not p_areas:
        return {'status': 'error', 'reason': 'no valid P-waves'}

    # Decision: if majority of P-waves have negative area → reversed
    n_negative = sum(1 for a in p_areas if a < 0)
    n_positive = sum(1 for a in p_areas if a > 0)
    mean_area = np.mean(p_areas)
    agreement = max(n_negative, n_positive) / len(p_areas)

    polarity = 'normal' if n_positive > n_negative else 'reversed'
    confidence = agreement

    return {
        'status': 'ok',
        'method': 'P_wave_polarity',
        'polarity': polarity,
        'confidence': round(confidence, 3),
        'n_p_waves': len(p_areas),
        'n_positive': n_positive,
        'n_negative': n_negative,
        'mean_area': round(float(mean_area), 4),
        'mean_peak_sign': round(float(np.mean(p_peak_signs)), 3),
    }


def detect_by_qrs_lead_i_vs_ii(aecg_filepath: str) -> dict:
    """Method 2: QRS morphology comparison between Lead I and Lead II.

    Normal: Lead I has positive QRS, Lead II has positive QRS.
    RA-LA reversal: Lead I QRS becomes inverted (negative dominant).
    Lead II and III swap roles.

    Uses raw aECG XML data (multi-lead).
    """
    import re
    for enc in ['utf-8', 'gbk', 'latin-1']:
        try:
            with open(aecg_filepath, 'r', encoding=enc) as f:
                content = f.read()
            if '<?xml' in content[:100]:
                break
        except:
            continue

    # Extract Lead I and Lead II signals
    lead_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF']
    digits_matches = list(re.finditer(r'<digits[^>]*>([^<]*)</digits>', content))
    if len(digits_matches) < 2:
        return {'status': 'error', 'reason': 'not enough leads'}

    signals = {}
    for i, m in enumerate(digits_matches[:6]):
        samples = np.array([float(x) for x in m.group(1).split()], dtype=np.float64)
        if i < len(lead_names):
            signals[lead_names[i]] = samples

    lead_I = signals.get('I')
    lead_II = signals.get('II')
    if lead_I is None or lead_II is None:
        return {'status': 'error', 'reason': 'no lead I/II'}

    # Take first 4 seconds
    n = min(4000, len(lead_I), len(lead_II))
    lead_I = lead_I[:n]
    lead_II = lead_II[:n]

    # Find QRS regions (simple: look for max |amplitude| in filtered data)
    # Use a crude approach: slice into windows and compute net area
    # A proper approach uses HSMM segmentation, but for Lead I we don't have it.
    # Instead, compute global signal asymmetry.

    # Lead I net area (positive in normal, negative in reversal)
    li_net = np.sum(lead_I - np.median(lead_I)) / n
    lii_net = np.sum(lead_II - np.median(lead_II)) / n

    # Lead I dominant direction of large deflections
    li_absmax_idx = np.argmax(np.abs(lead_I - np.median(lead_I)))
    li_dominant_sign = 1 if (lead_I[li_absmax_idx] - np.median(lead_I)) > 0 else -1

    # Lead II P-wave region is typically in the first 600ms
    # Check lead II early-segment polarity
    early_II = lead_II[200:600]  # 200-600ms (typical P-wave region)
    early_II_net = np.sum(early_II - np.median(early_II))

    # Decision rules
    lead_I_inverted = li_dominant_sign < 0
    lead_II_p_inverted = early_II_net < 0

    if lead_I_inverted and lead_II_p_inverted:
        polarity = 'reversed'
    elif not lead_I_inverted and not lead_II_p_inverted:
        polarity = 'normal'
    elif lead_I_inverted:
        polarity = 'likely_reversed'
    else:
        polarity = 'uncertain'

    return {
        'status': 'ok',
        'method': 'QRS_lead_comparison',
        'polarity': polarity,
        'lead_I_net': round(float(li_net), 4),
        'lead_II_net': round(float(lii_net), 4),
        'lead_I_dominant_sign': li_dominant_sign,
        'lead_II_early_net': round(float(early_II_net), 4),
        'lead_I_inverted': lead_I_inverted,
        'lead_II_P_inverted': lead_II_p_inverted,
    }


def detect_by_p_axis(aecg_filepath: str) -> dict:
    """Method 3: P-wave axis from aECG annotations.

    Normal P-axis: 0° to +75°
    RA-LA reversal P-axis: approximately +120° to -150° (extreme right axis)

    Uses P-axis value already in the aECG XML measurements.
    """
    import re
    for enc in ['utf-8', 'gbk', 'latin-1']:
        try:
            with open(aecg_filepath, 'r', encoding=enc) as f:
                content = f.read()
            if '<?xml' in content[:100]:
                break
        except:
            continue

    m = re.search(r'MDC_ECG_ANGLE_P_FRONT.*?<value[^>]*value="([^"]+)"[^>]*unit="deg"',
                  content, re.DOTALL)
    if not m:
        return {'status': 'error', 'reason': 'no P-axis in file'}

    p_axis = float(m.group(1))

    # Normal P-axis range
    if 0 <= p_axis <= 75:
        polarity = 'normal'
    elif p_axis > 100 or p_axis < -30:
        polarity = 'reversed'
    elif 75 < p_axis <= 100:
        polarity = 'borderline_right'
    else:
        polarity = f'atypical (axis={p_axis:.0f}deg)'

    return {
        'status': 'ok',
        'method': 'P_axis',
        'polarity': polarity,
        'p_axis_deg': p_axis,
    }


def detect_combined(record_name: str,
                    record_dir: str | None = None,
                    aecg_filepath: str | None = None) -> dict:
    """Combined polarity detection using all available methods.

    Returns consensus polarity and per-method results.
    """
    if record_dir is None:
        record_dir = os.path.join(DATA_DIR, record_name)
    if aecg_filepath is None:
        aecg_filepath = os.path.join(AECG_DIR, record_name + '.aECG')

    results = {}

    # Method 1: P-wave from HSMM
    r1 = detect_by_p_wave(record_dir)
    results['p_wave'] = r1

    # Method 2: Lead comparison
    if os.path.exists(aecg_filepath):
        r2 = detect_by_qrs_lead_i_vs_ii(aecg_filepath)
        results['lead_comparison'] = r2

        # Method 3: P-axis
        r3 = detect_by_p_axis(aecg_filepath)
        results['p_axis'] = r3

    # Consensus
    votes_reversed = 0
    votes_normal = 0
    for r in results.values():
        if r.get('polarity') == 'reversed':
            votes_reversed += 1
        elif r.get('polarity') == 'normal':
            votes_normal += 1
        elif 'likely_reversed' in str(r.get('polarity', '')):
            votes_reversed += 0.5

    total = votes_reversed + votes_normal
    if total == 0:
        consensus = 'uncertain'
        conf = 0
    else:
        consensus = 'reversed' if votes_reversed > votes_normal else 'normal'
        conf = round(max(votes_reversed, votes_normal) / total, 2)

    return {
        'record': record_name,
        'consensus': consensus,
        'confidence': conf,
        'votes': {'reversed': votes_reversed, 'normal': votes_normal},
        'methods': results,
    }


# =====================================================================
# Batch evaluation on RA-LA dataset
# =====================================================================
def batch_detect(n_records: int | None = None):
    """Run polarity detection on all (or first N) processed records."""
    records = sorted([d for d in os.listdir(DATA_DIR)
                      if os.path.isdir(os.path.join(DATA_DIR, d))
                      and os.path.exists(os.path.join(DATA_DIR, d, 'summary.json'))])

    if n_records:
        records = records[:n_records]

    print(f"Detecting polarity on {len(records)} records...")
    print(f"Method 1: P-wave polarity (Lead II)")
    print(f"Method 2: QRS Lead I vs Lead II comparison")
    print(f"Method 3: P-axis angle\n")

    summary = {'total': len(records), 'normal': 0, 'reversed': 0,
               'uncertain': 0, 'per_record': []}

    for idx, rec in enumerate(records):
        res = detect_combined(rec)
        summary['per_record'].append(res)

        if res['consensus'] == 'normal':
            summary['normal'] += 1
        elif res['consensus'] == 'reversed':
            summary['reversed'] += 1
        else:
            summary['uncertain'] += 1

        if (idx + 1) % 100 == 0:
            pct = (idx + 1) / len(records) * 100
            print(f"  [{idx+1}/{len(records)}] "
                  f"Normal={summary['normal']} Reversed={summary['reversed']} "
                  f"Uncertain={summary['uncertain']} ({pct:.0f}%)")

    # Save
    out_path = os.path.join(DATA_DIR, '_polarity_detection.json')
    class NpEnc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return super().default(o)

    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, cls=NpEnc)

    print(f"\n{'='*55}")
    print(f"  POLARITY DETECTION RESULTS")
    print(f"{'='*55}")
    print(f"  Total: {summary['total']}")
    print(f"  Normal:    {summary['normal']} ({summary['normal']/summary['total']*100:.1f}%)")
    print(f"  Reversed:  {summary['reversed']} ({summary['reversed']/summary['total']*100:.1f}%)")
    print(f"  Uncertain: {summary['uncertain']} ({summary['uncertain']/summary['total']*100:.1f}%)")
    print(f"  Saved: {out_path}")
    print(f"{'='*55}")

    return summary


# =====================================================================
if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'single':
        # Test on first record
        rec = '1805185J6U'
        print(f"=== Single test: {rec} ===\n")
        res = detect_combined(rec)
        print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    else:
        # Batch
        n = int(sys.argv[1]) if len(sys.argv) > 1 else None
        batch_detect(n)
