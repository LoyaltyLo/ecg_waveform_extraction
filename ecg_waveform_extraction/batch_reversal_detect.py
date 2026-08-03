#!/usr/bin/env python3
"""Batch Limb Lead Reversal Detection.

Runs the LimbLeadReversalDetector on all .aECG records (using pre-computed
6-lead HSMM data when available, or computing it on-the-fly).

Usage:
    python -m ecg_waveform_extraction.batch_reversal_detect --n 50
    python -m ecg_waveform_extraction.batch_reversal_detect --n 10 --from-cache
    python -m ecg_waveform_extraction.batch_reversal_detect --all --out results.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os, json, time, gc, argparse
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ecg_waveform_extraction.limb_lead_processor import (
    LimbLeadProcessor, LimbLeadResult,
)
from ecg_waveform_extraction.limb_lead_reversal import (
    LimbLeadReversalDetector, ReversalResult, reversal_result_to_dict,
    REVERSAL_TYPES, REVERSAL_NAMES,
)
from ecg_waveform_extraction.utils.aecg_parser import parse_aecg

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AECG_DIR = 'C:/LoyaltyLo/datasets/RA-LA_Reversal/aECG'
CACHE_DIR = str(Path(__file__).resolve().parent / 'output_rala_full/_limb_leads')
OUT_DIR = str(Path(__file__).resolve().parent / 'output_rala_full/_reversal')
MAX_SAMPLES = 4000

COLORS = {
    'ra_la': '#f44336', 'ra_ll': '#ff9800', 'la_ll': '#9c27b0',
    'normal': '#4caf50', 'uncertain': '#9e9e9e',
}


# ---------------------------------------------------------------------------
# Core: detect one record
# ---------------------------------------------------------------------------
def detect_record(fpath: str, detector: LimbLeadReversalDetector,
                  processor: LimbLeadProcessor | None = None,
                  use_cache: bool = False) -> ReversalResult | None:
    """Run reversal detection on one aECG file.

    Parameters
    ----------
    fpath : str
        Path to .aECG file.
    detector : LimbLeadReversalDetector
    processor : LimbLeadProcessor or None
        Required if use_cache is False.
    use_cache : bool
        If True, load pre-computed LimbLeadResult from CACHE_DIR.

    Returns
    -------
    ReversalResult or None
    """
    rec_name = os.path.basename(fpath).replace('.aECG', '')

    if use_cache:
        # Load pre-computed 6-lead HSMM results
        cache_path = os.path.join(CACHE_DIR, rec_name, 'summary.json')
        if not os.path.exists(cache_path):
            print(f"    (no cache for {rec_name}, computing...)")
            # Fall back to computing with a fresh processor
            if processor is None:
                processor = LimbLeadProcessor(max_samples=MAX_SAMPLES)
            return detect_record(fpath, detector, processor, use_cache=False)

        with open(cache_path, encoding='utf-8') as f:
            cache_data = json.load(f)

        # Reconstruct LimbLeadResult from cache
        from ecg_waveform_extraction.limb_lead_processor import LeadResult
        leads = {}
        for ln, ld in cache_data['leads'].items():
            if ld is None:
                leads[ln] = None
            else:
                leads[ln] = LeadResult(
                    lead_name=ld['lead_name'],
                    n_beats=ld['n_beats'],
                    polarity_counts=ld['polarity_counts'],
                    p_polarity_counts=ld['p_polarity_counts'],
                    t_polarity_counts=ld.get('t_polarity_counts', {}),
                    mean_qrs_dur_ms=ld['mean_qrs_dur_ms'],
                    mean_rs_ratio=ld['mean_rs_ratio'],
                    mean_qrs_net=ld['mean_qrs_net'],
                    mean_p_net=ld['mean_p_net'],
                    mean_t_net=ld.get('mean_t_net', 0.0),
                    mean_r_amplitude=ld.get('mean_r_amplitude', 0),
                    mean_s_amplitude=ld.get('mean_s_amplitude', 0),
                )

        ll_result = LimbLeadResult(
            record=cache_data['record'],
            fs=cache_data['fs'],
            n_samples=cache_data['n_samples'],
            leads=leads,
            measurements=cache_data.get('measurements', {}),
            interpretation=cache_data.get('interpretation', ''),
            n_total_beats=cache_data['n_total_beats'],
        )
    else:
        if processor is None:
            raise ValueError("processor required when use_cache=False")
        aecg = parse_aecg(fpath, max_samples=MAX_SAMPLES)
        if not aecg.get('signals'):
            return None
        ll_result, _ = processor.process_record(aecg, record_name=rec_name)

    return detector.detect(ll_result)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
def generate_dashboard(results: list[ReversalResult], out_dir: str):
    """Generate aggregate dashboard for batch reversal detection."""
    n = len(results)

    # Count verdicts
    verdict_counts = Counter(r.reversal_type for r in results)
    ra_la_count = verdict_counts.get('ra_la', 0)
    ra_ll_count = verdict_counts.get('ra_ll', 0)
    la_ll_count = verdict_counts.get('la_ll', 0)
    normal_count = verdict_counts.get('normal', 0)
    uncertain_count = verdict_counts.get('uncertain', 0)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Limb Lead Reversal Detection — Batch Results',
                 fontsize=15, fontweight='bold')

    # (0, 0): Verdict pie
    ax = axes[0, 0]
    labels = []; sizes = []; pie_colors = []
    for key, label, color in [
        ('ra_la', 'RA-LA', COLORS['ra_la']),
        ('ra_ll', 'RA-LL', COLORS['ra_ll']),
        ('la_ll', 'LA-LL', COLORS['la_ll']),
        ('normal', 'Normal', COLORS['normal']),
        ('uncertain', 'Uncertain', COLORS['uncertain']),
    ]:
        cnt = verdict_counts.get(key, 0)
        if cnt > 0:
            labels.append(f'{label}\n({cnt})')
            sizes.append(cnt)
            pie_colors.append(color)
    if sizes:
        ax.pie(sizes, labels=labels, colors=pie_colors, autopct='%1.0f%%',
               startangle=90, labeldistance=1.08)
    ax.set_title('Reversal Verdicts', fontweight='bold')

    # (0, 1): Confidence histogram by type
    ax = axes[0, 1]
    for tname, color in [('ra_la', COLORS['ra_la']), ('normal', COLORS['normal'])]:
        confs = [r.confidence for r in results if r.reversal_type == tname]
        if confs:
            ax.hist(confs, bins=20, alpha=0.6, color=color, label=REVERSAL_NAMES.get(tname, tname))
    ax.set_xlabel('Confidence'); ax.set_ylabel('Records')
    ax.set_title('Confidence Distribution', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2, axis='y')

    # (0, 2): Key Metrics
    ax = axes[0, 2]
    ax.axis('off')
    total_reversed = ra_la_count + ra_ll_count + la_ll_count
    lines = [
        f"Total records: {n}",
        f"",
        f"NORMAL:     {normal_count} ({normal_count/max(n,1)*100:.1f}%)",
        f"REVERSED:   {total_reversed} ({total_reversed/max(n,1)*100:.1f}%)",
        f"  RA-LA:    {ra_la_count}",
        f"  RA-LL:    {ra_ll_count}",
        f"  LA-LL:    {la_ll_count}",
        f"UNCERTAIN:  {uncertain_count} ({uncertain_count/max(n,1)*100:.1f}%)",
    ]
    ax.text(0.1, 0.95, '\n'.join(lines), transform=ax.transAxes,
            fontsize=9.5, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.9))

    # (1, 0): Per-criterion trigger rates for RA-LA
    ax = axes[1, 0]
    crit_counts = Counter()
    crit_total = 0
    for r in results:
        tr = r.types.get('ra_la')
        if tr is None:
            continue
        crit_total += 1
        for c in tr.criteria:
            if c.verdict == 'reversed':
                crit_counts[c.name] += 1

    if crit_counts:
        names = list(crit_counts.keys())
        counts = [crit_counts[n] for n in names]
        pcts = [c / crit_total * 100 for c in counts]
        bars = ax.barh(names, pcts, color=COLORS['ra_la'], alpha=0.7, edgecolor='white')
        ax.set_xlabel('% of Records Triggered')
        ax.set_title('RA-LA Criteria Trigger Rates', fontweight='bold')
        ax.grid(True, alpha=0.2, axis='x')
        for bar, pct in zip(bars, pcts):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{pct:.0f}%', va='center', fontsize=8)

    # (1, 1): Score scatter: RA-LA normal vs reversed
    ax = axes[1, 1]
    ra_la_normals = []; ra_la_reverseds = []
    for r in results:
        tr = r.types.get('ra_la')
        if tr is None:
            continue
        ra_la_normals.append(tr.score_normal)
        ra_la_reverseds.append(tr.score_reversed)

    if ra_la_normals:
        ax.scatter(ra_la_normals, ra_la_reverseds, c=COLORS['ra_la'],
                   alpha=0.4, s=15, edgecolors='none')
        max_val = max(max(ra_la_normals), max(ra_la_reverseds)) * 1.1
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=0.8, alpha=0.3)
        ax.set_xlabel('Score Normal'); ax.set_ylabel('Score Reversed')
        ax.set_title('RA-LA: Normal vs Reversed Scores', fontweight='bold')
        ax.grid(True, alpha=0.15)

    # (1, 2): Top reversed records
    ax = axes[1, 2]
    ax.axis('off')
    top_reversed = sorted(
        [r for r in results if r.is_reversed()],
        key=lambda r: -r.confidence)[:15]
    lines = ['TOP REVERSED RECORDS:']
    for r in top_reversed:
        lines.append(f"  {r.record}  {r.reversal_type}  conf={r.confidence:.2f}")
    if not top_reversed:
        lines.append('  (none)')
    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
            fontsize=8, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#fff3e0', alpha=0.9))

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '_reversal_dashboard.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Batch limb lead reversal detection')
    parser.add_argument('--n', type=int, default=None,
                        help='Number of records (default: all)')
    parser.add_argument('--start', type=int, default=0,
                        help='Start index')
    parser.add_argument('--from-cache', action='store_true',
                        help='Use pre-computed 6-lead HSMM summaries')
    parser.add_argument('--out', type=str, default=None,
                        help='Output directory override')
    args = parser.parse_args()

    global OUT_DIR
    if args.out:
        OUT_DIR = args.out
    os.makedirs(OUT_DIR, exist_ok=True)

    # Files
    files = sorted([f for f in os.listdir(AECG_DIR) if f.endswith('.aECG')])
    if args.start > 0:
        files = files[args.start:]
    if args.n is not None:
        files = files[:args.n]
    n_total = len(files)

    # Setup
    detector = LimbLeadReversalDetector()
    processor = None if args.from_cache else LimbLeadProcessor(max_samples=MAX_SAMPLES)

    print(f"\n{'='*65}")
    print(f"  BATCH LIMB LEAD REVERSAL DETECTION")
    print(f"  {n_total} records  |  From cache: {args.from_cache}")
    print(f"  Output: {OUT_DIR}")
    print(f"{'='*65}\n")

    t_start = time.time()
    results = []
    ok = 0

    for idx, fname in enumerate(files):
        fpath = os.path.join(AECG_DIR, fname)
        rec_name = fname.replace('.aECG', '')
        print(f"[{idx+1:3d}/{n_total}] {rec_name}...", end=" ", flush=True)
        t0 = time.time()

        try:
            result = detect_record(fpath, detector, processor,
                                   use_cache=args.from_cache)
        except Exception as e:
            print(f"ERROR: {e}")
            gc.collect()
            continue

        dt = time.time() - t0

        if result is None:
            print("SKIP (no signals)")
            continue

        results.append(result)
        ok += 1

        # Quick status line
        marker = '!' if result.is_reversed() else '~'
        meas = result.measurements
        p_axis = meas.get('P_axis', '?')
        qrs_axis = meas.get('QRS_axis', '?')
        print(f"{marker} {result.reversal_type:<8} conf={result.confidence:.2f}  "
              f"P={p_axis}° QRS={qrs_axis}°  "
              f"({dt:.0f}s)")

        # Save per-record
        rec_dir = os.path.join(OUT_DIR, rec_name)
        os.makedirs(rec_dir, exist_ok=True)
        with open(os.path.join(rec_dir, 'reversal.json'), 'w', encoding='utf-8') as f:
            json.dump(reversal_result_to_dict(result), f, indent=2,
                     ensure_ascii=False)

        gc.collect()

    total_time = time.time() - t_start

    if not results:
        print("No results.")
        return

    # ---- Global summary ----
    global_summary = _build_global_summary(results, total_time)
    with open(os.path.join(OUT_DIR, 'reversal_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)

    # ---- Save all results ----
    with open(os.path.join(OUT_DIR, 'all_reversals.json'), 'w', encoding='utf-8') as f:
        json.dump([reversal_result_to_dict(r) for r in results], f,
                  indent=2, ensure_ascii=False)

    # ---- Dashboard ----
    generate_dashboard(results, OUT_DIR)

    # ---- Final report ----
    vc = Counter(r.reversal_type for r in results)
    print(f"\n{'='*65}")
    print(f"  LIMB LEAD REVERSAL DETECTION — COMPLETE")
    print(f"{'='*65}")
    print(f"  Records processed: {ok}/{n_total}")
    print(f"  Total time: {total_time/60:.1f} min"
          f" ({total_time/ok:.1f}s/record)" if ok else "")
    print(f"")
    print(f"  NORMAL:      {vc.get('normal', 0):>4} ({vc.get('normal',0)/ok*100:.1f}%)")
    print(f"  RA-LA:       {vc.get('ra_la', 0):>4} ({vc.get('ra_la',0)/ok*100:.1f}%)")
    print(f"  RA-LL:       {vc.get('ra_ll', 0):>4} ({vc.get('ra_ll',0)/ok*100:.1f}%)")
    print(f"  LA-LL:       {vc.get('la_ll', 0):>4} ({vc.get('la_ll',0)/ok*100:.1f}%)")
    print(f"  UNCERTAIN:   {vc.get('uncertain', 0):>4} ({vc.get('uncertain',0)/ok*100:.1f}%)")
    print(f"")
    print(f"  Output: {OUT_DIR}/")
    print(f"  Dashboard: {OUT_DIR}/_reversal_dashboard.png")
    print(f"{'='*65}")

    # Per-type detail
    for tname in REVERSAL_TYPES:
        matching = [r for r in results if r.reversal_type == tname]
        if matching:
            avg_conf = np.mean([r.confidence for r in matching])
            print(f"  {tname}: {len(matching)} records, "
                  f"mean conf={avg_conf:.2f}")
    print(f"{'='*65}")


def _build_global_summary(results, total_time):
    """Build global summary JSON."""
    vc = Counter(r.reversal_type for r in results)

    # Per-criterion stats for each reversal type
    type_crit_stats = {}
    for tname in REVERSAL_TYPES:
        crit_counts = Counter()
        n_evaluated = 0
        for r in results:
            tr = r.types.get(tname)
            if tr is None:
                continue
            n_evaluated += 1
            for c in tr.criteria:
                if c.verdict == 'reversed':
                    crit_counts[c.name] += 1
        type_crit_stats[tname] = {
            'n_evaluated': n_evaluated,
            'criteria_triggered': {
                name: {'count': cnt,
                       'pct': round(cnt / max(n_evaluated, 1) * 100, 1)}
                for name, cnt in crit_counts.most_common()
            },
        }

    return {
        'method': '6-lead HSMM + multi-criteria weighted voting',
        'dataset': 'RA-LA Reversal aECG',
        'n_records': len(results),
        'total_time_sec': round(total_time, 1),
        'verdict_counts': dict(vc),
        'verdict_pct': {
            k: round(v / len(results) * 100, 1)
            for k, v in vc.items()
        },
        'type_criteria_stats': type_crit_stats,
    }


if __name__ == '__main__':
    main()
