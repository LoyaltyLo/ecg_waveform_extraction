"""HSMM vs Pan-Tompkins QRS comparison: R-peak detection + polarity agreement.

Uses neurokit2 for Pan-Tompkins (Pantompkins 1985) and compares against
the existing HSMM output. For each beat detected by both methods:
  - R-peak position agreement (ms)
  - QRS polarity v2 agreement (5-criterion voting)
  - QRS duration, energy ratio, R/S ratio correlation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os, json, time, gc, re
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import neurokit2 as nk

from ecg_waveform_extraction.preprocessing import ECGPreprocessor
from ecg_waveform_extraction.features import FeatureExtractor
from ecg_waveform_extraction.hsmm import HSMMModel, smart_initialize_gmms
from ecg_waveform_extraction.segmentation import ECGSegmenter
from ecg_waveform_extraction.extraction.qrs_refiner import (
    refine_qrs_boundaries, compute_qrs_polarity_v2,
)

# =====================================================================
# Config
# =====================================================================
AECG_DIR = 'C:/LoyaltyLo/datasets/RA-LA_Reversal/aECG'
OUT_DIR = str(Path(__file__).resolve().parent / 'output_rala_full/_hsmm_vs_pt')
os.makedirs(OUT_DIR, exist_ok=True)
N_FILES = 50
MAX_SAMPLES = 4000
RPEAK_TOL_MS = 100  # tolerance for matching R-peaks (ms)

POLARITY_COLORS = {'positive': '#4caf50', 'negative': '#f44336', 'biphasic': '#ff9800', 'uncertain': '#9e9e9e'}


# =====================================================================
# Parse aECG (lightweight, single-lead version)
# =====================================================================
def parse_signal(filepath):
    with open(filepath, 'rb') as f:
        raw = f.read()
    content = raw.decode('utf-8', errors='replace')
    fs = 1000.0
    m = re.search(rb'<increment[^>]*value="([^"]+)"[^>]*unit="s"', raw)
    if m:
        fs = 1.0 / float(m.group(1))
    ss = content.find('<sequenceSet')
    se = content.find('</sequenceSet>', ss)
    digits = re.findall(r'<digits[^>]*>([^<]+)</digits>', content[ss:se])
    signals = {}
    for i, name in enumerate(['I', 'II']):
        if i < len(digits):
            sig = np.array([float(x) for x in digits[i].split()], dtype=np.float64)
            signals[name] = sig[:MAX_SAMPLES]
    return signals, fs


# =====================================================================
# Run Pan-Tompkins via neurokit2 (on PRE-FILTERED signal)
# =====================================================================
def run_pantompkins(signal_raw, fs):
    """Run neurokit2 Pan-Tompkins 1985 on RAW ECG (mV).

    PT has its own internal bandpass filter — must receive raw signal.
    Returns its own filtered signal for fair polarity calculation.

    Returns:
        rpeaks: list of R-peak sample indices
        qrs_onsets: delineated Q-onset per beat (-1 if missing)
        qrs_offsets: delineated S-offset per beat (-1 if missing)
        clean: PT's own filtered ECG
    """
    try:
        signals, info = nk.ecg_process(signal_raw, sampling_rate=int(fs))
        clean = signals['ECG_Clean'].values
        rpeaks = info['ECG_R_Peaks']

        try:
            waves, _ = nk.ecg_delineate(
                clean, rpeaks, sampling_rate=int(fs), method='peak', show=False,
            )
        except Exception:
            waves = {}

        qrs_onsets = _extract_delineation_list(waves, 'ECG_Q_Peaks', rpeaks)
        qrs_offsets = _extract_delineation_list(waves, 'ECG_S_Peaks', rpeaks)

    except Exception:
        clean = signal_raw.astype(np.float64)
        rpeaks = np.array([], dtype=int)
        qrs_onsets = []
        qrs_offsets = []

    return rpeaks, qrs_onsets, qrs_offsets, clean


def _extract_delineation_list(waves_dict, key, rpeaks):
    """Extract delineation points from neurokit2's waves dict.

    Returns list of int (one per beat), or -1 if missing.
    """
    items = waves_dict.get(key, [])
    if items is None:
        return [-1] * len(rpeaks)
    result = []
    for i in range(len(rpeaks)):
        try:
            val = int(items[i]) if not np.isnan(items[i]) else -1
        except (IndexError, TypeError):
            val = -1
        result.append(val)
    return result


# =====================================================================
# Run HSMM
# =====================================================================
def run_hsmm(signal_clean, fs):
    """Run HSMM segmentation, returning beats with refined boundaries."""
    fe = FeatureExtractor(fs=fs)
    features = fe.extract(signal_clean)
    model = HSMMModel(fs=fs)
    model.initialize_with_priors()
    model.set_left_right_topology()
    smart_initialize_gmms(model, features)
    seg = ECGSegmenter(
        preprocessor=ECGPreprocessor(fs=fs),
        feature_extractor=fe, model=model, fs=fs,
    )
    result = seg.segment(signal_clean)
    return result


# =====================================================================
# Match beats between methods
# =====================================================================
def match_beats(hsmm_r_peaks, pt_rpeaks, fs):
    """Match HSMM and Pan-Tompkins R-peaks within tolerance.

    Returns list of (hsmm_idx, pt_idx, time_diff_ms).

    hsmm_r_peaks: list of (beat_id, r_peak_sample)
    pt_rpeaks: np.array of R-peak sample indices
    """
    tol = int(RPEAK_TOL_MS / 1000 * fs)
    matches = []
    pt_used = set()

    for bid, hr in hsmm_r_peaks:
        best_dist = tol + 1
        best_idx = -1
        for pi, pr in enumerate(pt_rpeaks):
            if pi in pt_used:
                continue
            dist = abs(hr - int(pr))
            if dist <= tol and dist < best_dist:
                best_dist = dist
                best_idx = pi
        if best_idx >= 0:
            pt_used.add(best_idx)
            time_diff = best_dist / fs * 1000.0
            matches.append((bid, best_idx, round(time_diff, 1)))

    n_hsmm = len(hsmm_r_peaks)
    n_pt = len(pt_rpeaks)
    n_matched = len(matches)

    sensitivity = round(n_matched / max(n_pt, 1) * 100, 1)
    ppv = round(n_matched / max(n_hsmm, 1) * 100, 1)
    f1 = round(2 * sensitivity * ppv / max(sensitivity + ppv, 0.1), 1)

    return matches, {
        'n_hsmm_beats': n_hsmm,
        'n_pt_beats': n_pt,
        'n_matched': n_matched,
        'sensitivity_%': sensitivity,
        'ppv_%': ppv,
        'f1': f1,
        'mean_time_diff_ms': round(np.mean([m[2] for m in matches]), 1) if matches else None,
        'max_time_diff_ms': round(np.max([m[2] for m in matches]), 1) if matches else None,
    }


# =====================================================================
# Process one record
# =====================================================================
def process_record(fname):
    fpath = os.path.join(AECG_DIR, fname)
    rec_name = fname.replace('.aECG', '')
    signals, fs = parse_signal(fpath)

    rec_dir = os.path.join(OUT_DIR, rec_name)
    os.makedirs(rec_dir, exist_ok=True)

    result = {'record': rec_name, 'fs': fs, 'leads': {}}

    for lead_name in ['I', 'II']:
        sig_raw = signals.get(lead_name)
        if sig_raw is None:
            result['leads'][lead_name] = None
            continue
        sig_raw = sig_raw[:MAX_SAMPLES].astype(np.float64)

        # ---- Preprocess (HSMM pipeline) ----
        prep = ECGPreprocessor(fs=fs)
        clean_ecg = prep.preprocess(sig_raw)

        # ---- Pan-Tompkins (on RAW ECG — its own built-in filter) ----
        pt_rpeaks, pt_qons, pt_soffs, pt_clean = run_pantompkins(sig_raw, fs)

        # ---- HSMM (on our filtered ECG) ----
        hsmm_result = run_hsmm(clean_ecg, fs)

        # Collect HSMM R-peaks and refined boundaries
        hsmm_beats = []  # (beat_id, q_on, r_pk, s_off)
        for b in hsmm_result.beats:
            if b.q_onset <= 0 or b.r_peak <= 0 or b.s_offset <= 0:
                continue
            q_on, r_pk, s_off = refine_qrs_boundaries(
                clean_ecg, b.q_onset, b.r_peak, b.s_offset, fs,
            )
            hsmm_beats.append((b.beat_id, q_on, r_pk, s_off))

        hsmm_r_peaks_list = [(bid, r_pk) for bid, _, r_pk, _ in hsmm_beats]
        hsmm_map = {bid: (q_on, r_pk, s_off) for bid, q_on, r_pk, s_off in hsmm_beats}

        # ---- Match R-peaks ----
        matches, match_stats = match_beats(hsmm_r_peaks_list, pt_rpeaks, fs)

        # ---- Polarity comparison (each method on its OWN filtered signal) ----
        polarity_comparison = []
        for bid, pt_idx, time_diff in matches:
            # HSMM polarity (v2 on HSMM-filtered ECG)
            hsmm_q, hsmm_r, hsmm_s = hsmm_map[bid]
            hsmm_pol = compute_qrs_polarity_v2(
                clean_ecg, hsmm_q, hsmm_r, hsmm_s, fs, lead_name=lead_name,
            )

            # PT polarity (v2 on PT's own filtered ECG)
            if pt_clean is not None and pt_idx < len(pt_clean):
                pt_r = int(pt_rpeaks[pt_idx])
                pt_q = pt_qons[pt_idx] if pt_idx < len(pt_qons) and pt_qons[pt_idx] > 0 else max(0, pt_r - int(0.03 * fs))
                pt_s = pt_soffs[pt_idx] if pt_idx < len(pt_soffs) and pt_soffs[pt_idx] > 0 else min(len(pt_clean) - 1, pt_r + int(0.04 * fs))

                if pt_s > pt_q:
                    pt_q_r, pt_r_r, pt_s_r = refine_qrs_boundaries(
                        pt_clean, pt_q, pt_r, pt_s, fs,
                    )
                    pt_pol = compute_qrs_polarity_v2(
                        pt_clean, pt_q_r, pt_r_r, pt_s_r, fs, lead_name=lead_name,
                    )
                else:
                    pt_pol = {'polarity': 'N/A', 'confidence': 0, 'polarity_score': 0}
            else:
                pt_pol = {'polarity': 'N/A', 'confidence': 0, 'polarity_score': 0}

            agree = 1 if hsmm_pol['polarity'] == pt_pol['polarity'] else 0

            polarity_comparison.append({
                'beat_id': bid,
                'pt_idx': int(pt_idx),
                'time_diff_ms': time_diff,
                'hsmm': {
                    'polarity': hsmm_pol['polarity'],
                    'confidence': hsmm_pol['confidence'],
                    'polarity_score': hsmm_pol['polarity_score'],
                    'energy_ratio': hsmm_pol['energy_ratio'],
                    'rs_ratio': hsmm_pol['rs_ratio'],
                },
                'pt': {
                    'polarity': pt_pol['polarity'],
                    'confidence': pt_pol['confidence'],
                    'polarity_score': pt_pol['polarity_score'],
                    'energy_ratio': pt_pol['energy_ratio'],
                    'rs_ratio': pt_pol['rs_ratio'],
                },
                'polarity_agree': agree,
            })

        # Aggregate
        hsmm_pol_counts = Counter(c['hsmm']['polarity'] for c in polarity_comparison)
        pt_pol_counts = Counter(c['pt']['polarity'] for c in polarity_comparison)
        n_agree = sum(c['polarity_agree'] for c in polarity_comparison)
        n_total = max(len(polarity_comparison), 1)
        agreement_pct = round(n_agree / n_total * 100, 1)

        # ---- Save overview plot (two columns: HSMM left, PT right, SAME RAW ECG ref) ----
        _save_comparison_plot(sig_raw, clean_ecg, pt_clean, fs, rec_name, lead_name,
                             hsmm_beats, pt_rpeaks, pt_qons, pt_soffs,
                             polarity_comparison, rec_dir)

        result['leads'][lead_name] = {
            'match_stats': match_stats,
            'n_matched_beats': len(polarity_comparison),
            'polarity_agreement_%': agreement_pct,
            'hsmm_polarity_counts': dict(hsmm_pol_counts),
            'pt_polarity_counts': dict(pt_pol_counts),
            'matched_beats': polarity_comparison,
        }

    # Save per-record JSON
    with open(os.path.join(rec_dir, 'comparison.json'), 'w') as f:
        json.dump(result, f, indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else float(o))

    return result


# =====================================================================
# Comparison plot
# =====================================================================
def _save_comparison_plot(raw_ecg, hsmm_clean, pt_clean, fs, rec_name, lead_name,
                          hsmm_beats, pt_rpeaks, pt_qons, pt_soffs,
                          polarity_comparison, rec_dir):
    """Overview plot: SAME raw ECG as reference in both subplots.

    Two subplots share the identical raw ECG waveform underneath.
    Top: HSMM colored QRS regions + R peaks
    Bottom: PT colored QRS regions + R peaks
    This makes the boundary comparison fair and visually consistent.
    """
    T = len(raw_ecg)
    plot_sec = min(T / fs, 4.0)
    n_plot = int(plot_sec * fs)
    t_plot = np.arange(n_plot) / fs
    e_raw = raw_ecg[:n_plot].astype(np.float64)

    # ---- Light detrend + normalize for display (visual only, not for computation) ----
    from scipy.signal import medfilt
    if len(e_raw) > 200:
        win = min(int(0.2 * fs), len(e_raw) // 3)
        if win % 2 == 0:
            win += 1
        baseline = medfilt(e_raw, kernel_size=win)
        e_plot = e_raw - baseline
    else:
        e_plot = e_raw - np.median(e_raw)
    e_plot = e_plot / (np.std(e_plot) + 1e-8)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 7), sharex=True,
                                    gridspec_kw={'hspace': 0.35})

    pc = {c['beat_id']: c for c in polarity_comparison}

    # ---- HSMM (top) ----
    ax1.plot(t_plot, e_plot, 'k-', linewidth=0.5)
    for bid, q_on, r_pk, s_off in hsmm_beats:
        if q_on < n_plot and s_off < n_plot:
            pol = pc.get(bid, {}).get('hsmm', {}).get('polarity', 'N/A')
            col = POLARITY_COLORS.get(pol, '#9e9e9e')
            ax1.fill_between(t_plot[q_on:s_off + 1], e_plot[q_on:s_off + 1],
                            alpha=0.30, color=col, linewidth=0)
            if r_pk < n_plot:
                ax1.plot(r_pk / fs, e_plot[r_pk], 'rv', markersize=4, alpha=0.8)

    n_match = len(polarity_comparison)
    n_agree = sum(c['polarity_agree'] for c in polarity_comparison)
    ax1.set_title(f'HSMM (9-state Viterbi + boundary refinement) — '
                  f'{len(hsmm_beats)} beats', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Amplitude (norm)')
    ax1.grid(True, alpha=0.15)

    # ---- Pan-Tompkins (bottom, SAME raw reference waveform) ----
    ax2.plot(t_plot, e_plot, 'k-', linewidth=0.5)
    for pi, pr in enumerate(pt_rpeaks):
        if pr < n_plot:
            pt_pol = 'N/A'
            for c in polarity_comparison:
                if c.get('pt_idx') == pi:
                    pt_pol = c.get('pt', {}).get('polarity', 'N/A')
                    break
            col = POLARITY_COLORS.get(pt_pol, '#9e9e9e')
            q_on_v = pt_qons[pi] if pi < len(pt_qons) and pt_qons[pi] > 0 else max(0, int(pr) - int(0.03 * fs))
            s_off_v = pt_soffs[pi] if pi < len(pt_soffs) and pt_soffs[pi] > 0 else min(n_plot - 1, int(pr) + int(0.04 * fs))
            if q_on_v < n_plot and s_off_v < n_plot:
                ax2.fill_between(t_plot[q_on_v:s_off_v + 1], e_plot[q_on_v:s_off_v + 1],
                                alpha=0.30, color=col, linewidth=0)
            ax2.plot(int(pr) / fs, e_plot[int(pr)], 'rv', markersize=4, alpha=0.8)

    ax2.set_title(f'Pan-Tompkins 1985 (neurokit2) — {len(pt_rpeaks)} beats  |  '
                  f'Matched: {n_match}  |  Polarity agree: {n_agree}/{n_match} '
                  f'({n_agree*100//max(n_match,1)}%)',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude (norm)')
    ax2.grid(True, alpha=0.15)

    fig.tight_layout()
    fig.savefig(os.path.join(rec_dir, f'comparison_{lead_name}.png'),
                dpi=130, bbox_inches='tight')
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================
def main():
    files = sorted([f for f in os.listdir(AECG_DIR) if f.endswith('.aECG')])[:N_FILES]

    print(f"{'='*65}")
    print(f"  HSMM vs PAN-TOMPKINS QRS Comparison")
    print(f"  {N_FILES} records (Lead I + Lead II)")
    print(f"  Method: neurokit2 ecg_process (Pantompkins 1985)")
    print(f"  R-peak tolerance: {RPEAK_TOL_MS}ms")
    print(f"{'='*65}\n")

    all_results = []
    t_start = time.time()

    # Global accumulators
    global_hsmm_pol = Counter()
    global_pt_pol = Counter()
    global_agree = 0
    global_total = 0
    all_sensitivities = []
    all_ppvs = []
    all_time_diffs = []

    for idx, fname in enumerate(files):
        rec_name = fname.replace('.aECG', '')
        print(f"[{idx+1:2d}/{N_FILES}] {rec_name}...", end=" ", flush=True)
        t0 = time.time()
        r = process_record(fname)
        dt = time.time() - t0
        all_results.append(r)

        n_matched = 0
        n_agree = 0
        for lead_key, lead_data in r['leads'].items():
            if lead_data is None:
                continue
            ms = lead_data['match_stats']
            all_sensitivities.append(ms['sensitivity_%'])
            all_ppvs.append(ms['ppv_%'])
            if ms.get('mean_time_diff_ms'):
                all_time_diffs.append(ms['mean_time_diff_ms'])

            n_matched += lead_data['n_matched_beats']
            n_agree += sum(c['polarity_agree'] for c in lead_data['matched_beats'])
            global_total += len(lead_data['matched_beats'])

            for c in lead_data['matched_beats']:
                global_hsmm_pol[c['hsmm']['polarity']] += 1
                global_pt_pol[c['pt']['polarity']] += 1
                if c['polarity_agree']:
                    global_agree += 1

        print(f"OK  matched={n_matched} agree={n_agree} "
              f"I:Se={r['leads'].get('I',{}).get('match_stats',{}).get('sensitivity_%','NA')}% "
              f"II:Se={r['leads'].get('II',{}).get('match_stats',{}).get('sensitivity_%','NA')}% "
              f"({dt:.0f}s)")
        gc.collect()

    total_time = time.time() - t_start

    # ---- Global Summary ----
    se_mean = round(np.mean(all_sensitivities), 1) if all_sensitivities else 0
    se_std = round(np.std(all_sensitivities), 1) if all_sensitivities else 0
    ppv_mean = round(np.mean(all_ppvs), 1) if all_ppvs else 0
    td_mean = round(np.mean(all_time_diffs), 1) if all_time_diffs else 0

    agree_pct = round(global_agree / max(global_total, 1) * 100, 1)

    global_summary = {
        'methods': {'HSMM': '9-state Viterbi + refine_qrs_boundaries',
                    'PT': 'neurokit2 ecg_process (Pantompkins 1985) + delineation'},
        'n_records': N_FILES,
        'total_matched_beats': global_total,
        'r_peak_sensitivity_mean_%': se_mean,
        'r_peak_sensitivity_std_%': se_std,
        'r_peak_ppv_mean_%': ppv_mean,
        'mean_r_peak_time_diff_ms': td_mean,
        'polarity_agreement': {
            'n_agree': global_agree,
            'n_total': global_total,
            'agreement_%': agree_pct,
        },
        'hsmm_polarity_distribution': dict(global_hsmm_pol),
        'pt_polarity_distribution': dict(global_pt_pol),
        'per_record': [{
            'record': r['record'],
            'lead_I': r['leads'].get('I'),
            'lead_II': r['leads'].get('II'),
        } for r in all_results],
        'total_time_sec': round(total_time, 1),
    }

    with open(os.path.join(OUT_DIR, 'global_comparison.json'), 'w') as f:
        json.dump(global_summary, f, indent=2)

    # ---- Report ----
    print(f"\n{'='*65}")
    print(f"  HSMM vs PAN-TOMPKINS — Final Report")
    print(f"{'='*65}")
    print(f"  R-PEAK DETECTION")
    print(f"    Sensitivity:               {se_mean}% ± {se_std}%")
    print(f"    PPV:                        {ppv_mean}%")
    print(f"    Mean time diff:             {td_mean} ms")
    print(f"  QRS POLARITY (on {global_total} matched beats)")
    print(f"    Agreement:                  {agree_pct}%")
    print(f"  HSMM Polarity:               {dict(global_hsmm_pol)}")
    print(f"  PT   Polarity:                {dict(global_pt_pol)}")
    print(f"  Time: {total_time:.0f}s")
    print(f"  Output: {OUT_DIR}/")
    print(f"{'='*65}")
    print(f"\n  Per-record:")
    print(f"    comparison.json, comparison_I.png, comparison_II.png")


if __name__ == '__main__':
    main()
