#!/usr/bin/env python3
"""Batch 6-Lead Limb HSMM Processing.

Processes all .aECG records through the 6-lead HSMM pipeline, producing
per-lead QRS polarity, P-wave metrics, and cross-lead summaries.

This is the data foundation for limb lead reversal detection.

Usage:
    python -m ecg_waveform_extraction.src.batch_limb_leads
    python -m ecg_waveform_extraction.src.batch_limb_leads --n 20
    python -m ecg_waveform_extraction.src.batch_limb_leads --n 10 --lead I,II,AVR
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os, json, time, gc, argparse
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ecg_waveform_extraction.src.limb_lead_processor import (
    LimbLeadProcessor, LIMB_LEADS, LEAD_PLOT_ORDER,
    result_to_dict, build_summary_table, compare_polarity_across_leads,
)
from ecg_waveform_extraction.src.utils.aecg_parser import parse_aecg

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AECG_DIR = 'C:/LoyaltyLo/datasets/RA-LA_Reversal/aECG'
OUT_DIR = str(Path(__file__).resolve().parent.parent / 'output/rala_full/_limb_leads')
MAX_SAMPLES = 16000

POLARITY_COLORS = {
    'positive': '#4caf50', 'negative': '#f44336',
    'biphasic': '#ff9800', 'uncertain': '#9e9e9e',
}
LEAD_COLORS = {
    'I':   '#2196f3', 'II':  '#4caf50', 'III': '#ff9800',
    'AVR': '#f44336', 'AVL': '#9c27b0', 'AVF': '#00bcd4',
}


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------
def process_record(fpath: str, processor: LimbLeadProcessor,
                   save_plots: bool = True) -> dict | None:
    """Process one aECG file: parse → HSMM 6-lead → save → return summary."""
    fname = os.path.basename(fpath)
    rec_name = fname.replace('.aECG', '')
    rec_dir = os.path.join(OUT_DIR, rec_name)
    os.makedirs(rec_dir, exist_ok=True)

    # Parse
    aecg = parse_aecg(fpath, max_samples=MAX_SAMPLES)
    if not aecg.get('signals'):
        print(f"    SKIP: no signals in {fname}")
        return None

    # Process all 6 leads
    result, seg_data = processor.process_record(aecg, record_name=rec_name)

    # ---- Save segment data (.npy) + segmentation plots ----
    _save_segmentation_data(seg_data, rec_dir, rec_name, save_plots=save_plots)
    for lead_name in LIMB_LEADS:
        lr = result.leads.get(lead_name)
        if lr is None:
            continue
        lead_dir = os.path.join(rec_dir, f'lead_{lead_name}')
        os.makedirs(lead_dir, exist_ok=True)

        # QRS polarity JSON
        with open(os.path.join(lead_dir, 'qrs_polarity.json'), 'w', encoding='utf-8') as f:
            json.dump(lr.beats, f, indent=2)

        # P-wave JSON
        with open(os.path.join(lead_dir, 'p_waves.json'), 'w', encoding='utf-8') as f:
            json.dump(lr.p_waves, f, indent=2)

        # T-wave JSON
        with open(os.path.join(lead_dir, 't_waves.json'), 'w', encoding='utf-8') as f:
            json.dump(lr.t_waves, f, indent=2)

        # Full BeatBoundary set: every boundary field incl. ISO/PR/ST/TP
        # segment starts and P/T provenance (hsmm vs prominence).
        sd = seg_data.get(lead_name)
        if sd is not None and sd.get('beats'):
            from dataclasses import asdict
            with open(os.path.join(lead_dir, 'beats.json'), 'w', encoding='utf-8') as f:
                json.dump([asdict(b) for b in sd['beats']], f, indent=2)

    # ---- Save record summary ----
    summary = result_to_dict(result)
    with open(os.path.join(rec_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- Overview plot ----
    if save_plots:
        _save_record_plot(result, rec_dir)

    # ---- Cross-lead polarity comparison ----
    xlead = compare_polarity_across_leads(result)
    with open(os.path.join(rec_dir, 'cross_lead.json'), 'w', encoding='utf-8') as f:
        json.dump(xlead, f, indent=2, ensure_ascii=False)

    return summary


# ---------------------------------------------------------------------------
# Segment data saving + plotting
# ---------------------------------------------------------------------------
def _save_segmentation_data(seg_data: dict, rec_dir: str, rec_name: str,
                            save_plots: bool = True):
    """Save HSMM state labels and generate waveform segmentation plots.

    Parameters
    ----------
    seg_data : dict
        lead_name -> dict with filtered_ecg, state_labels, state_names, fs, beats.
    rec_dir : str
    rec_name : str
    save_plots : bool
        When False, only the .npy caches are written (per-beat PNGs can be
        regenerated later from the cache via plot_segmentation).
    """
    # Save numpy arrays per lead
    for lead_name, sd in seg_data.items():
        if sd is None or sd.get('filtered_ecg') is None:
            continue

        lead_dir = os.path.join(rec_dir, f'lead_{lead_name}')
        os.makedirs(lead_dir, exist_ok=True)

        np.save(os.path.join(lead_dir, 'filtered_ecg.npy'), sd['filtered_ecg'])
        np.save(os.path.join(lead_dir, 'state_labels.npy'), sd['state_labels'])

    if not save_plots:
        return

    from ecg_waveform_extraction.src.plot_segmentation import save_all_segmentation_plots
    # Generate all segmentation plots
    try:
        save_all_segmentation_plots(seg_data, rec_name, rec_dir,
                                    max_sec=4.0, max_beats_per_lead=12, dpi=150)
    except Exception as e:
        print(f"    (plot warning: {e})")


# ---------------------------------------------------------------------------
# Per-record overview plot (6-lead subplot grid)
# ---------------------------------------------------------------------------
def _save_record_plot(result, rec_dir):
    """3×2 subplot grid showing first ~4s of each limb lead with QRS coloring."""
    from ecg_waveform_extraction.src.preprocessing import ECGPreprocessor

    fig, axes = plt.subplots(3, 2, figsize=(18, 10))
    fig.suptitle(f'{result.record} — 6-Lead Limb HSMM QRS', fontsize=13, fontweight='bold')

    for idx, lead_name in enumerate(LEAD_PLOT_ORDER):
        ax = axes[idx // 2, idx % 2]
        lr = result.leads.get(lead_name)

        if lr is None or lr.n_beats == 0:
            ax.text(0.5, 0.5, f'Lead {lead_name}\n(no data)', transform=ax.transAxes,
                    ha='center', va='center', fontsize=11, color='gray')
            ax.set_title(f'Lead {lead_name}', fontsize=11)
            continue

        # We need the actual signal. Re-parse minimally.
        # For now, use beats to reconstruct a visualization from saved data.
        # Actually, we already have the beats with q_onset/s_offset.
        # We need the clean signal — let's load a small segment from the
        # aECG raw data instead. But for the batch runner, re-loading is
        # wasteful. We'll store a small segment of the preprocessed signal.

        # Fallback: show a placeholder. The detailed plot will be generated
        # separately from saved numpy arrays.
        ax.text(0.5, 0.5,
                f'Lead {lead_name}\n'
                f'{lr.n_beats} beats\n'
                f'QRS: {lr.polarity_counts}\n'
                f'P: {lr.p_polarity_counts}',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=9, fontfamily='monospace')
        ax.set_title(f'Lead {lead_name}', fontsize=11, fontweight='bold',
                     color=LEAD_COLORS.get(lead_name, 'black'))

    fig.tight_layout()
    fig.savefig(os.path.join(rec_dir, 'overview_6lead.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Global dashboard
# ---------------------------------------------------------------------------
def generate_dashboard(summaries: list[dict]):
    """Generate aggregate dashboard: polarity distributions, cross-lead stats."""
    n_records = len(summaries)

    # ---- Aggregate per-lead stats ----
    lead_stats = {ln: {'qrs': Counter(), 'p': Counter(),
                       'durs': [], 'rs_ratios': [], 'n_beats': [],
                       'qrs_nets': [], 'p_nets': []}
                  for ln in LIMB_LEADS}

    for s in summaries:
        for ln in LIMB_LEADS:
            ld = s['leads'].get(ln)
            if ld is None:
                continue
            lead_stats[ln]['n_beats'].append(ld['n_beats'])
            lead_stats[ln]['durs'].append(ld['mean_qrs_dur_ms'])
            lead_stats[ln]['rs_ratios'].append(ld['mean_rs_ratio'])
            lead_stats[ln]['qrs_nets'].append(ld['mean_qrs_net'])
            lead_stats[ln]['p_nets'].append(ld['mean_p_net'])
            for pol, cnt in ld['polarity_counts'].items():
                lead_stats[ln]['qrs'][pol] += cnt
            for pol, cnt in ld['p_polarity_counts'].items():
                lead_stats[ln]['p'][pol] += cnt

    # ---- Figure ----
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('6-Lead Limb HSMM — Batch Summary', fontsize=16, fontweight='bold')

    # (0, 0-2): Per-lead QRS polarity stacked bars (row 0, spanning cols 0-2)
    ax = fig.add_subplot(2, 3, 1)
    leads_present = [ln for ln in LIMB_LEADS if lead_stats[ln]['qrs']]
    pol_types = ['positive', 'negative', 'biphasic', 'uncertain']
    x = np.arange(len(leads_present))
    width = 0.55
    bottom = np.zeros(len(leads_present))
    for pol in pol_types:
        vals = [lead_stats[ln]['qrs'].get(pol, 0) for ln in leads_present]
        ax.bar(x, vals, width, bottom=bottom, label=pol,
               color=POLARITY_COLORS.get(pol, '#9e9e9e'), edgecolor='white')
        bottom += np.array(vals, dtype=float)
    ax.set_xticks(x); ax.set_xticklabels(leads_present, fontsize=10)
    ax.set_ylabel('Beat Count'); ax.set_title('QRS Polarity by Lead', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')

    # (0, 3): QRS duration boxplot
    ax = fig.add_subplot(2, 3, 2)
    dur_data = [lead_stats[ln]['durs'] for ln in leads_present if lead_stats[ln]['durs']]
    dur_labels = [ln for ln in leads_present if lead_stats[ln]['durs']]
    bp = ax.boxplot(dur_data, tick_labels=dur_labels, patch_artist=True)
    for patch, ln in zip(bp['boxes'], dur_labels):
        patch.set_facecolor(LEAD_COLORS.get(ln, '#cccccc'))
        patch.set_alpha(0.5)
    ax.set_ylabel('QRS Duration (ms)'); ax.set_title('QRS Duration by Lead', fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')

    # (0, 4): R/S ratio boxplot
    ax = fig.add_subplot(2, 3, 3)
    rs_data = [lead_stats[ln]['rs_ratios'] for ln in leads_present if lead_stats[ln]['rs_ratios']]
    rs_labels = [ln for ln in leads_present if lead_stats[ln]['rs_ratios']]
    bp = ax.boxplot(rs_data, tick_labels=rs_labels, patch_artist=True)
    for patch, ln in zip(bp['boxes'], rs_labels):
        patch.set_facecolor(LEAD_COLORS.get(ln, '#cccccc'))
        patch.set_alpha(0.5)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.6)
    ax.set_ylabel('R/S Ratio'); ax.set_title('R/S Ratio by Lead', fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')

    # (1, 0): P polarity stacked bars
    ax = fig.add_subplot(2, 3, 4)
    bottom = np.zeros(len(leads_present))
    for pol in ['positive', 'negative', 'biphasic']:
        vals = [lead_stats[ln]['p'].get(pol, 0) for ln in leads_present]
        ax.bar(x, vals, width, bottom=bottom, label=pol,
               color=POLARITY_COLORS.get(pol, '#9e9e9e'), edgecolor='white')
        bottom += np.array(vals, dtype=float)
    ax.set_xticks(x); ax.set_xticklabels(leads_present, fontsize=10)
    ax.set_ylabel('P-wave Count'); ax.set_title('P-Wave Polarity by Lead', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')

    # (1, 1): Mean P net area by lead
    ax = fig.add_subplot(2, 3, 5)
    p_net_means = [np.mean(lead_stats[ln]['p_nets']) if lead_stats[ln]['p_nets'] else 0
                   for ln in leads_present]
    colors = [LEAD_COLORS.get(ln, '#cccccc') for ln in leads_present]
    ax.bar(leads_present, p_net_means, color=colors, edgecolor='white', alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Mean P Net Area'); ax.set_title('P-Wave Net Area by Lead', fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')

    # (1, 2): Summary text
    ax = fig.add_subplot(2, 3, 6)
    ax.axis('off')
    total_beats = sum(sum(lead_stats[ln]['n_beats']) for ln in LIMB_LEADS)
    lines = [
        f"Records processed: {n_records}",
        f"Total beats: {total_beats}",
        f"",
        f"QRS Polarity Totals:",
    ]
    for ln in leads_present:
        qrs_total = sum(lead_stats[ln]['qrs'].values())
        pos = lead_stats[ln]['qrs'].get('positive', 0)
        neg = lead_stats[ln]['qrs'].get('negative', 0)
        pos_pct = pos / max(qrs_total, 1) * 100
        neg_pct = neg / max(qrs_total, 1) * 100
        lines.append(f"  {ln:<4}: +{pos:>4} ({pos_pct:>4.0f}%)  "
                     f"-{neg:>4} ({neg_pct:>4.0f}%)  "
                     f"dur={np.mean(lead_stats[ln]['durs']):.0f}ms"
                     if lead_stats[ln]['durs'] else f"  {ln:<4}: (no data)")

    lines.append(f"")
    lines.append(f"RA-LA Reversal Quick Check:")
    # Quick heuristic: if Lead I QRS is dominantly negative AND aVR QRS is
    # dominantly positive, that's a strong RA-LA reversal signal
    for s in summaries:
        rec = s['record']
        i_qrs = s['leads'].get('I', {})
        avr_qrs = s['leads'].get('AVR', {})
        if i_qrs and avr_qrs:
            i_neg = i_qrs.get('polarity_counts', {}).get('negative', 0)
            i_pos = i_qrs.get('polarity_counts', {}).get('positive', 0)
            avr_pos = avr_qrs.get('polarity_counts', {}).get('positive', 0)
            avr_neg = avr_qrs.get('polarity_counts', {}).get('negative', 0)
            i_total = i_neg + i_pos
            avr_total = avr_pos + avr_neg
            if i_total > 0 and avr_total > 0:
                i_neg_pct = i_neg / i_total * 100
                avr_pos_pct = avr_pos / avr_total * 100
                if i_neg_pct > 50 and avr_pos_pct > 50:
                    lines.append(f"  ⚠ {rec}: I={i_neg_pct:.0f}%neg, "
                                 f"aVR={avr_pos_pct:.0f}%pos  ← REVERSAL?")

    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
            fontsize=8.5, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.9))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '_dashboard_6lead.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Batch 6-lead limb HSMM processing')
    parser.add_argument('--n', type=int, default=None,
                        help='Number of records to process (default: all)')
    parser.add_argument('--lead', type=str, default=None,
                        help='Comma-separated leads to process (default: all 6)')
    parser.add_argument('--start', type=int, default=0,
                        help='Start index (for resuming)')
    parser.add_argument('--out', type=str, default=None,
                        help='Output directory override')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip per-beat PNG generation (data caches '
                             'still written; plots regenerable from cache)')
    args = parser.parse_args()

    # Output dir
    global OUT_DIR
    if args.out:
        OUT_DIR = args.out
    os.makedirs(OUT_DIR, exist_ok=True)

    # Leads
    leads_to_process = LIMB_LEADS
    if args.lead:
        leads_to_process = [x.strip() for x in args.lead.split(',')]
        # Validate
        for ln in leads_to_process:
            if ln not in LIMB_LEADS:
                print(f"ERROR: unknown lead '{ln}'. Valid: {LIMB_LEADS}")
                return

    # Files
    files = sorted([f for f in os.listdir(AECG_DIR) if f.endswith('.aECG')])
    if args.start > 0:
        files = files[args.start:]
    if args.n is not None:
        files = files[:args.n]
    n_total = len(files)

    # Monkey-patch LIMB_LEADS if subset requested
    import ecg_waveform_extraction.src.limb_lead_processor as llp
    original_limb_leads = llp.LIMB_LEADS
    if args.lead:
        llp.LIMB_LEADS = leads_to_process

    processor = LimbLeadProcessor(max_samples=MAX_SAMPLES)

    print(f"\n{'='*65}")
    print(f"  BATCH 6-LEAD LIMB HSMM PROCESSING")
    print(f"  {n_total} records  |  Leads: {leads_to_process}")
    print(f"  Max samples: {MAX_SAMPLES}  |  Output: {OUT_DIR}")
    print(f"{'='*65}\n")

    t_start = time.time()
    summaries = []
    ok_count = 0

    for idx, fname in enumerate(files):
        fpath = os.path.join(AECG_DIR, fname)
        rec_name = fname.replace('.aECG', '')
        print(f"[{idx+1:3d}/{n_total}] {rec_name}...", end=" ", flush=True)
        t0 = time.time()

        try:
            summary = process_record(fpath, processor, save_plots=not args.no_plots)
        except Exception as e:
            print(f"ERROR: {e}")
            gc.collect()
            continue

        dt = time.time() - t0

        if summary:
            summaries.append(summary)
            ok_count += 1
            n_beats = summary['n_total_beats']
            # Quick polarity summary for key leads
            i_info = summary['leads'].get('I', {}) or {}
            ii_info = summary['leads'].get('II', {}) or {}
            avr_info = summary['leads'].get('AVR', {}) or {}
            i_qrs = i_info.get('polarity_counts', {})
            avr_qrs = avr_info.get('polarity_counts', {})

            print(f"OK  beats={n_beats}  "
                  f"I:{i_qrs}  "
                  f"II:{ii_info.get('polarity_counts', {})}  "
                  f"aVR:{avr_qrs}  "
                  f"({dt:.0f}s)")
        else:
            print("SKIP")

        gc.collect()

    total_time = time.time() - t_start

    # Restore LIMB_LEADS
    llp.LIMB_LEADS = original_limb_leads

    if not summaries:
        print("No records processed.")
        return

    # ---- Global Summary ----
    global_summary = _build_global_summary(summaries, leads_to_process, total_time)
    with open(os.path.join(OUT_DIR, 'global_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)

    # ---- Dashboard ----
    generate_dashboard(summaries)

    # ---- Final Report ----
    print(f"\n{'='*65}")
    print(f"  6-LEAD LIMB HSMM — COMPLETE")
    print(f"{'='*65}")
    print(f"  Records: {ok_count}/{n_total}")
    print(f"  Total time: {total_time/60:.1f} min  "
          f"({total_time/ok_count:.1f}s/record)" if ok_count else "")
    print(f"  Output: {OUT_DIR}/")
    print(f"  Dashboard: {OUT_DIR}/_dashboard_6lead.png")
    print(f"  Summary:   {OUT_DIR}/global_summary.json")
    print(f"{'='*65}")

    # Per-lead quick stats
    for ln in leads_to_process:
        has = sum(1 for s in summaries if s['leads'].get(ln))
        print(f"  {ln:<4}: {has}/{ok_count} records with data")

    print(f"{'='*65}")


def _build_global_summary(summaries, leads_processed, total_time):
    """Build the global summary JSON."""
    lead_agg = {}
    for ln in leads_processed:
        qrs_counter = Counter()
        p_counter = Counter()
        durs = []
        rs_ratios = []
        p_nets = []
        n_records_with_data = 0
        total_beats = 0

        for s in summaries:
            ld = s['leads'].get(ln)
            if ld is None:
                continue
            n_records_with_data += 1
            total_beats += ld['n_beats']
            durs.append(ld['mean_qrs_dur_ms'])
            rs_ratios.append(ld['mean_rs_ratio'])
            p_nets.append(ld['mean_p_net'])
            for pol, cnt in ld['polarity_counts'].items():
                qrs_counter[pol] += cnt
            for pol, cnt in ld['p_polarity_counts'].items():
                p_counter[pol] += cnt

        lead_agg[ln] = {
            'n_records': n_records_with_data,
            'total_beats': total_beats,
            'qrs_polarity': dict(qrs_counter),
            'p_polarity': dict(p_counter),
            'mean_qrs_dur_ms': round(float(np.mean(durs)), 1) if durs else None,
            'mean_rs_ratio': round(float(np.mean(rs_ratios)), 4) if rs_ratios else None,
            'mean_p_net': round(float(np.mean(p_nets)), 4) if p_nets else None,
        }

    return {
        'method': 'HSMM 9-state Viterbi + refine_qrs_boundaries + 5-criterion v2',
        'dataset': 'RA-LA Reversal aECG',
        'n_records': len(summaries),
        'leads_processed': leads_processed,
        'max_samples': MAX_SAMPLES,
        'total_time_sec': round(total_time, 1),
        'per_lead_summary': lead_agg,
    }


if __name__ == '__main__':
    main()
