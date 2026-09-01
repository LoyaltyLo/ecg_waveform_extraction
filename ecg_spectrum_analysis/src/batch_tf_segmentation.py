#!/usr/bin/env python3
"""Batch Time-Frequency P/QRS/T Segmentation.

Segments each lead of each record into P/QRS/T directly from the CWT
scalogram (see tf_segmentation.py) and saves per-lead figures, per-beat
boundaries JSON, and optional sample-level comparison against the main
package's cached HSMM labels.

Output structure:
    output/tf_segmentation/
    └── <record_id>/
        ├── lead_I.png                    # 3-panel TF segmentation figure
        ├── lead_I_bounds.json            # per-beat P/QRS/T boundaries (ms)
        ├── lead_I_comparison.png         # (with --compare-hsmm) vs HSMM strips
        ├── lead_I_agreement.json         # (with --compare-hsmm) sample-level stats
        └── ...
    + summary.json, summary_agreement.png at the top level

Usage:
    python -m ecg_spectrum_analysis.src.batch_tf_segmentation --n 10
    python -m ecg_spectrum_analysis.src.batch_tf_segmentation --n 5 --compare-hsmm
    python -m ecg_spectrum_analysis.src.batch_tf_segmentation --n 3 --leads I,II
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os, json, time, gc, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')

from ecg_waveform_extraction.src.utils.aecg_parser import parse_aecg
from ecg_waveform_extraction.src.preprocessing.filters import ECGPreprocessor

from ecg_spectrum_analysis.src.tf_segmentation import (
    segment_tf, compare_with_hsmm, segmentation_to_dict,
)
from ecg_spectrum_analysis.src.plot_tf_segmentation import (
    plot_tf_segmentation, plot_hsmm_comparison, save_figure,
)

# ---------------------------------------------------------------------------
PKG_ROOT = Path(__file__).resolve().parent.parent
AECG_DIR = 'C:/LoyaltyLo/datasets/RA-LA_Reversal/aECG'
OUT_DIR = str(PKG_ROOT / 'output' / 'tf_segmentation')
HSMM_CACHE_DIR = str(PKG_ROOT.parent / 'ecg_waveform_extraction' / 'output'
                     / 'rala_full' / '_limb_leads')
MAX_SAMPLES = 8000  # 8 s at the dataset's 1000 Hz (~8 beats)
DEFAULT_LEADS = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF']  # 12-lead via --leads all
ALL_LEADS = DEFAULT_LEADS + ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def find_aecg_files(aecg_dir: str, n: int | None = None) -> list[str]:
    """Find .aECG files, optionally limiting to `n`."""
    files = sorted(
        os.path.join(aecg_dir, f)
        for f in os.listdir(aecg_dir)
        if f.lower().endswith('.aecg')
    )
    if n is not None and n > 0:
        files = files[:n]
    return files


def load_hsmm_labels(record_id: str, lead_name: str) -> np.ndarray | None:
    """Load cached 9-state HSMM labels for (record, lead), if present."""
    path = os.path.join(HSMM_CACHE_DIR, record_id, f'lead_{lead_name}',
                        'state_labels.npy')
    if not os.path.exists(path):
        return None
    return np.load(path)


def process_one_record(
    filepath: str,
    out_dir: str,
    leads: list[str],
    max_samples: int = MAX_SAMPLES,
    compare_hsmm: bool = False,
    dpi: int = 150,
) -> dict:
    """Segment all requested leads of one record and save results."""
    record_id = Path(filepath).stem
    record_out = Path(out_dir) / record_id
    record_out.mkdir(parents=True, exist_ok=True)

    aecg = parse_aecg(filepath, max_samples=max_samples)
    fs = aecg['fs']
    signals = aecg['signals']
    if not signals:
        return {'record': record_id, 'error': 'No signals found'}

    prep = ECGPreprocessor(fs=fs)
    t0 = time.perf_counter()

    lead_stats = []
    for lead_name in leads:
        raw = signals.get(lead_name)
        if raw is None:
            continue
        clean = prep.preprocess(raw[:max_samples].astype(np.float64))
        seg = segment_tf(clean, fs)

        save_figure(
            plot_tf_segmentation(seg, clean, lead_name=lead_name,
                                 record_name=record_id, dpi=dpi),
            record_out / f'lead_{lead_name}.png',
        )
        with open(record_out / f'lead_{lead_name}_bounds.json', 'w',
                  encoding='utf-8') as f:
            json.dump(segmentation_to_dict(seg), f, indent=2, ensure_ascii=False)

        entry = {'lead': lead_name, 'n_beats': seg.n_beats}

        # ---- Optional comparison with the main package's HSMM cache ----
        if compare_hsmm:
            hsmm = load_hsmm_labels(record_id, lead_name)
            if hsmm is not None:
                report = compare_with_hsmm(seg.labels, hsmm)
                save_figure(
                    plot_hsmm_comparison(seg, clean, hsmm, report,
                                         lead_name=lead_name,
                                         record_name=record_id, dpi=dpi),
                    record_out / f'lead_{lead_name}_comparison.png',
                )
                with open(record_out / f'lead_{lead_name}_agreement.json', 'w',
                          encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                entry['hsmm_agreement'] = report['agreement']
            else:
                entry['hsmm_agreement'] = None

        lead_stats.append(entry)

    dt = time.perf_counter() - t0
    return {
        'record': record_id, 'fs': fs, 'n_samples': aecg['n_samples'],
        'leads': lead_stats, 'time_sec': round(dt, 2),
    }


def plot_agreement_summary(stats: list[dict], out_dir: str):
    """Histogram of per-lead TF-vs-HSMM agreement across the batch."""
    values = [
        e['hsmm_agreement']
        for s in stats for e in s.get('leads', [])
        if e.get('hsmm_agreement') is not None
    ]
    if not values:
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.hist(np.array(values) * 100.0, bins=20, color='#4C72B0', alpha=0.85,
            edgecolor='white')
    ax.axvline(np.mean(values) * 100.0, color='#C44E52', linewidth=1.5,
               label=f'mean = {np.mean(values)*100:.1f}%')
    ax.set_xlabel('Sample-level agreement TF vs HSMM (%)', fontsize=9)
    ax.set_ylabel('# leads', fontsize=9)
    ax.set_title('TF segmentation vs HSMM cache — batch agreement',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, linestyle='--')
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, 'summary_agreement.png'))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Batch TF P/QRS/T Segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ecg_spectrum_analysis.src.batch_tf_segmentation --n 10
  python -m ecg_spectrum_analysis.src.batch_tf_segmentation --n 5 --compare-hsmm
  python -m ecg_spectrum_analysis.src.batch_tf_segmentation --n 3 --leads I,II --max-samples 4000
        """,
    )
    parser.add_argument('--n', type=int, default=10,
                        help='Number of records to process (default: 10).')
    parser.add_argument('--leads', type=str, default=','.join(DEFAULT_LEADS),
                        help="Comma-separated lead list, or 'all' for 12 leads.")
    parser.add_argument('--max-samples', type=int, default=MAX_SAMPLES,
                        help=f'Samples per record (default: {MAX_SAMPLES}).')
    parser.add_argument('--compare-hsmm', action='store_true',
                        help='Compare against cached HSMM labels when present.')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Output image DPI (default: 150).')
    parser.add_argument('--aecg-dir', type=str, default=AECG_DIR,
                        help='Path to aECG files directory.')
    parser.add_argument('--out-dir', type=str, default=OUT_DIR,
                        help='Output directory.')
    args = parser.parse_args()

    leads = ALL_LEADS if args.leads == 'all' else [s.strip().upper() for s in args.leads.split(',')]
    os.makedirs(args.out_dir, exist_ok=True)

    files = find_aecg_files(args.aecg_dir, n=args.n)
    n_total = len(files)
    print(f'Found {n_total} aECG files in {args.aecg_dir}')
    print(f'Leads: {leads} | max_samples: {args.max_samples} | compare HSMM: {args.compare_hsmm}')
    print(f'Output: {args.out_dir}')
    print('=' * 60)

    t_start = time.perf_counter()
    stats, errors = [], []

    for i, fp in enumerate(files, 1):
        rid = Path(fp).stem
        print(f'[{i:4d}/{n_total}] {rid} ...', end=' ', flush=True)
        try:
            stat = process_one_record(
                fp, out_dir=args.out_dir, leads=leads,
                max_samples=args.max_samples, compare_hsmm=args.compare_hsmm,
                dpi=args.dpi,
            )
            stats.append(stat)
            agreements = [e['hsmm_agreement'] for e in stat['leads']
                          if e.get('hsmm_agreement') is not None]
            aggr = f' | HSMM {np.mean(agreements)*100:.1f}%' if agreements else ''
            print(f"OK ({stat['time_sec']}s, {len(stat['leads'])} leads{aggr})")
        except Exception as e:
            errors.append({'record': rid, 'error': str(e)})
            print(f'ERROR: {e}')
        if i % 10 == 0:
            gc.collect()

    t_total = time.perf_counter() - t_start
    print('\n' + '=' * 60)
    print(f'Done. {len(stats)}/{n_total} successful, {len(errors)} errors.')
    print(f'Total time: {t_total:.1f}s ({t_total / max(n_total, 1):.1f}s/record)')

    with open(os.path.join(args.out_dir, 'summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump({
            'total_processed': len(stats),
            'total_errors': len(errors),
            'leads': leads,
            'max_samples': args.max_samples,
            'total_time_sec': round(t_total, 1),
            'stats': stats,
            'errors': errors,
        }, f, indent=2, ensure_ascii=False)
    print(f'Summary saved to {os.path.join(args.out_dir, "summary.json")}')

    if args.compare_hsmm:
        plot_agreement_summary(stats, args.out_dir)


if __name__ == '__main__':
    main()
