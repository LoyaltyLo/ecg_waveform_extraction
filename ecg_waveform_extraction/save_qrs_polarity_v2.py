"""Post-process existing HSMM output with 5-criterion QRS polarity v2 + crop images.

Reads filtered_ecg.npy + p_qrs_t_wave.json from previous runs, applies:
  - refine_qrs_boundaries() + compute_qrs_polarity_v2()
  - Saves per-beat QRS crop PNGs + per-record overview + JSON summary.

No HSMM re-run needed — reads pre-computed .npy and boundary JSON.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os, json, time, gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ecg_waveform_extraction.extraction.qrs_refiner import (
    refine_qrs_boundaries, compute_qrs_polarity_v2,
)
from ecg_waveform_extraction.utils.aecg_parser import parse_aecg

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_DIR = str(Path(__file__).resolve().parent / 'output_rala_full/_p_qrs_t_wave')
OUT_DIR = str(Path(__file__).resolve().parent / 'output_rala_full/_qrs_polarity_v2')
AECG_DIR = 'C:/LoyaltyLo/datasets/RA-LA_Reversal/aECG'
os.makedirs(OUT_DIR, exist_ok=True)

POLARITY_COLORS = {
    'positive': '#4caf50',
    'negative': '#f44336',
    'biphasic': '#ff9800',
    'uncertain': '#9e9e9e',
}


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------
def process_record(rec_name: str) -> dict | None:
    """Process one record: read existing output, run v2 classifier, save results."""
    rec_in_dir = os.path.join(INPUT_DIR, rec_name)
    rec_out_dir = os.path.join(OUT_DIR, rec_name)
    beats_out_dir = os.path.join(rec_out_dir, 'beats')
    os.makedirs(beats_out_dir, exist_ok=True)

    results = {'record': rec_name, 'leads': {}}

    for lead_name in ['lead_I', 'lead_II']:
        lead_in = os.path.join(rec_in_dir, lead_name)
        lead_out = os.path.join(rec_out_dir, lead_name)
        beats_out = os.path.join(lead_out, 'beats')
        os.makedirs(beats_out, exist_ok=True)

        ecg_path = os.path.join(lead_in, 'filtered_ecg.npy')
        json_path = os.path.join(lead_in, 'p_qrs_t_wave.json')

        if not os.path.exists(ecg_path) or not os.path.exists(json_path):
            results['leads'][lead_name] = None
            continue

        ecg = np.load(ecg_path)
        with open(json_path) as f:
            beats_json = json.load(f)

        # Get fs from aECG
        fs = 1000.0
        aecg_path = os.path.join(AECG_DIR, rec_name + '.aECG')
        if os.path.exists(aecg_path):
            try:
                aecg = parse_aecg(aecg_path)
                fs = aecg['fs']
            except Exception:
                pass

        short_lead = lead_name.replace('lead_', '')  # 'I' or 'II'

        lead_results = []
        for entry in beats_json:
            q_on = entry.get('q_onset', -1)
            r_pk = entry.get('r_peak', -1)
            s_off = entry.get('s_offset', -1)
            if q_on <= 0 or r_pk <= 0 or s_off <= 0:
                continue

            # ---- Refine boundaries ----
            q_on_r, r_pk_r, s_off_r = refine_qrs_boundaries(ecg, q_on, r_pk, s_off, fs)

            # ---- V2 polarity ----
            pol_result = compute_qrs_polarity_v2(
                ecg, q_on_r, r_pk_r, s_off_r, fs,
                lead_name=short_lead,
            )

            # ---- Save QRS crop image ----
            _save_qrs_crop(ecg, q_on_r, r_pk_r, s_off_r, fs,
                          rec_name, entry['beat_id'], short_lead,
                          pol_result, beats_out)

            lead_results.append({
                'beat_id': entry['beat_id'],
                'q_onset': int(q_on_r),
                'r_peak': int(r_pk_r),
                's_offset': int(s_off_r),
                'polarity': pol_result['polarity'],
                'confidence': pol_result['confidence'],
                'polarity_score': pol_result['polarity_score'],
                'energy_ratio': pol_result['energy_ratio'],
                'peak_count': pol_result['peak_count'],
                'rs_ratio': pol_result['rs_ratio'],
                'qrs_net_area': pol_result['qrs_net_area'],
                'criteria': {k: {'vote': v['vote'], 'strength': round(v['strength'], 3)}
                            for k, v in pol_result['criteria'].items()},
            })

        # ---- Lead overview plot ----
        _save_lead_overview(ecg, lead_results, fs, rec_name, short_lead, lead_out)

        # ---- Lead JSON ----
        with open(os.path.join(lead_out, 'qrs_polarity_v2.json'), 'w') as f:
            json.dump(lead_results, f, indent=2)

        # Summary counts
        from collections import Counter
        pol_counts = Counter(r['polarity'] for r in lead_results)
        confs = [r['confidence'] for r in lead_results]
        scores = [r['polarity_score'] for r in lead_results]

        results['leads'][lead_name] = {
            'n_beats': len(lead_results),
            'polarity_counts': dict(pol_counts),
            'mean_confidence': round(float(np.mean(confs)), 3) if confs else 0,
            'mean_polarity_score': round(float(np.mean(scores)), 3) if scores else 0,
        }

    # ---- Record summary ----
    summary = {
        'record': rec_name,
        'lead_I': results['leads'].get('lead_I'),
        'lead_II': results['leads'].get('lead_II'),
    }
    with open(os.path.join(rec_out_dir, 'qrs_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# QRS crop image
# ---------------------------------------------------------------------------
def _save_qrs_crop(ecg, q_on, r_pk, s_off, fs, rec_name, beat_id, lead_name,
                   pol_result, beats_dir):
    """Save a zoomed-in QRS waveform crop with polarity annotation."""
    T = len(ecg)
    margin = int(0.10 * fs)  # 100ms context each side
    ws = max(0, q_on - margin)
    we = min(T - 1, s_off + margin)

    fig, ax = plt.subplots(figsize=(8, 3))
    t_win = np.arange(ws, we + 1) / fs
    e_win = ecg[ws:we + 1]

    ax.plot(t_win, e_win, 'k-', linewidth=1.0)

    # Color the QRS region
    qrs_t = np.arange(q_on, s_off + 1) / fs
    qrs_v = ecg[q_on:s_off + 1]
    c = POLARITY_COLORS.get(pol_result['polarity'], '#9e9e9e')
    ax.fill_between(qrs_t, qrs_v, alpha=0.30, color=c,
                    label=f'QRS ({pol_result["polarity"]})')

    # Markers
    ax.plot(q_on / fs, ecg[q_on], 'g<', markersize=8, label='Q onset')
    ax.plot(r_pk / fs, ecg[r_pk], 'rv', markersize=10, label='R peak')
    ax.plot(s_off / fs, ecg[s_off], 'b>', markersize=8, label='S offset')

    # Baseline
    bl = float(np.mean(ecg[max(0, q_on - 30):q_on])) if q_on >= 30 else float(np.median(e_win))
    ax.axhline(bl, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)

    # Info box
    crit_agree = sum(1 for v in pol_result['criteria'].values()
                     if v['vote'] == (+1 if pol_result['polarity'] == 'positive'
                                      else (-1 if pol_result['polarity'] == 'negative' else 0)))
    info = (f'Beat {beat_id} — Lead {lead_name}\n'
            f'Polarity: {pol_result["polarity"].upper()}  |  '
            f'Conf: {pol_result["confidence"]:.2f}  |  '
            f'Score: {pol_result["polarity_score"]:+.2f}\n'
            f'E-Ratio: {pol_result["energy_ratio"]:.3f}  |  '
            f'R/S: {pol_result["rs_ratio"]:.2f}  |  '
            f'Peaks: {pol_result["peak_count"]}\n'
            f'Criteria agree: {crit_agree}/5')
    ax.text(0.98, 0.97, info, transform=ax.transAxes, fontsize=8.5,
            va='top', ha='right', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))

    ax.set_xlim(t_win[0], t_win[-1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'{rec_name} — Beat {beat_id} — Lead {lead_name} QRS Polarity')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(beats_dir, f'beat_{beat_id:03d}_qrs_crop.png'),
                dpi=130, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Lead overview plot
# ---------------------------------------------------------------------------
def _save_lead_overview(ecg, lead_results, fs, rec_name, lead_name, lead_dir):
    """Overview plot: first 4s of ECG with colored QRS regions."""
    T = len(ecg)
    plot_sec = min(T / fs, 4.0)
    n_plot = int(plot_sec * fs)
    t_plot = np.arange(n_plot) / fs
    e_plot = ecg[:n_plot]

    fig, ax = plt.subplots(figsize=(18, 4))
    ax.plot(t_plot, e_plot, 'k-', linewidth=0.5)

    for r in lead_results:
        q_on = r['q_onset']
        s_off = r['s_offset']
        if q_on < n_plot and s_off < n_plot and s_off > q_on:
            c = POLARITY_COLORS.get(r['polarity'], '#9e9e9e')
            ax.fill_between(t_plot[q_on:s_off + 1], e_plot[q_on:s_off + 1],
                            alpha=0.30, color=c, linewidth=0)
            # Label
            mid = (q_on + s_off) // 2
            if mid < n_plot:
                label = f'{r["polarity"][0].upper()}({r["confidence"]:.1f})'
                y_off = 8 if r['polarity_score'] > 0 else -14
                ax.annotate(label, (t_plot[mid], e_plot[mid]),
                           textcoords='offset points', xytext=(0, y_off),
                           fontsize=6.5, ha='center', color=c, fontweight='bold')

    from collections import Counter
    counts = Counter(r['polarity'] for r in lead_results)
    ax.set_xlim(t_plot[0], t_plot[-1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'{rec_name} — Lead {lead_name} QRS V2  |  '
                 f'+:{counts.get("positive",0)}  -:{counts.get("negative",0)}  '
                 f'±:{counts.get("biphasic",0)}')
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(lead_dir, 'qrs_overview.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    records = sorted([d for d in os.listdir(INPUT_DIR)
                      if os.path.isdir(os.path.join(INPUT_DIR, d))])

    print(f"{'='*60}")
    print(f"  QRS POLARITY V2 — 5-Criterion Weighted Voting")
    print(f"  Post-processing {len(records)} records")
    print(f"  Output: {OUT_DIR}/")
    print(f"{'='*60}\n")

    summaries = []
    t_start = time.time()

    from collections import Counter
    global_pol = Counter()
    global_lead_pol = {'lead_I': Counter(), 'lead_II': Counter()}

    for idx, rec_name in enumerate(records):
        print(f"[{idx+1:2d}/{len(records)}] {rec_name}...", end=" ", flush=True)
        t0 = time.time()
        r = process_record(rec_name)
        dt = time.time() - t0

        if r and r['leads']:
            summaries.append(r)
            li = (r['leads'].get('lead_I') or {})
            lii = (r['leads'].get('lead_II') or {})
            li_pol = li.get('polarity_counts', {})
            lii_pol = lii.get('polarity_counts', {})
            for k, v in li_pol.items():
                global_pol[k] += v
                global_lead_pol['lead_I'][k] += v
            for k, v in lii_pol.items():
                global_pol[k] += v
                global_lead_pol['lead_II'][k] += v
            print(f"OK  I:{li.get('n_beats',0)}b {_fmt_pol(li_pol)} | "
                  f"II:{lii.get('n_beats',0)}b {_fmt_pol(lii_pol)} ({dt:.0f}s)")
        else:
            print("SKIP")
        gc.collect()

    total_time = time.time() - t_start

    # ---- Global summary ----
    n_records = len(summaries)
    total_beats = sum(global_pol.values())
    global_summary = {
        'method': '5-criterion weighted voting (v2)',
        'n_records': n_records,
        'total_beats': total_beats,
        'overall': dict(global_pol),
        'by_lead': {k: dict(v) for k, v in global_lead_pol.items()},
        'per_record': [
            {'record': s['record'],
             'lead_I': s['leads'].get('lead_I'),
             'lead_II': s['leads'].get('lead_II')}
            for s in summaries
        ],
        'total_time_sec': round(total_time, 1),
    }
    with open(os.path.join(OUT_DIR, 'global_summary.json'), 'w') as f:
        json.dump(global_summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  QRS POLARITY V2 COMPLETE")
    print(f"{'='*60}")
    print(f"  Records: {n_records}  |  Total beats: {total_beats}")
    for pol in ['positive', 'negative', 'biphasic']:
        cnt = global_pol.get(pol, 0)
        pct = cnt / max(total_beats, 1) * 100
        print(f"  {pol:<10}: {cnt:>4} ({pct:>5.1f}%)")
    print(f"  Time: {total_time:.1f}s")
    print(f"  Output: {OUT_DIR}/")
    print(f"{'='*60}")
    print(f"\n  Per-record structure:")
    print(f"    {OUT_DIR}/<record>/")
    print(f"      qrs_summary.json")
    print(f"      lead_I/")
    print(f"        qrs_polarity_v2.json, qrs_overview.png")
    print(f"        beats/beat_###_qrs_crop.png")
    print(f"      lead_II/  (same)")


def _fmt_pol(counts: dict) -> str:
    """Format polarity counts compactly."""
    parts = []
    for p in ['positive', 'negative', 'biphasic', 'uncertain']:
        if p in counts:
            parts.append(f'{p[0].upper()}:{counts[p]}')
    return ' '.join(parts)


if __name__ == '__main__':
    main()
