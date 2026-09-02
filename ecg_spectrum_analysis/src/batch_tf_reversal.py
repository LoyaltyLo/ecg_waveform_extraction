#!/usr/bin/env python3
"""Batch spectrogram-based RA-LA reversal detection.

Runs detect_tf_reversal (cross-wavelet phase + QRS-correlation gate) on
each record and saves per-record figures/JSON, a global summary, a
phase-vs-correlation scatter, and — with --validate-synthetic — reports
the flip rates of two synthetic reversal variants:

  flip-I       : I' = -I only (pure anti-phase signature; the detection
                 machinery test — verdicts must invert)
  full swap    : I' = -I, II' = III, III' = II (real anatomy; verdicts
                 must invert, but the II<->III morphology confound makes
                 some records undetectable — see tf_reversal.py docstring)

With --v10-csv, the main package's aECG_v10_triple.csv verdicts
(swap_detected) are loaded and the agreement rate is reported. NOTE: the
cardiologist-confirmed labels (反接/左右手反接 xlsx, 26 records) have zero
ID overlap with this working set, so no true-label accuracy is computable.

Output structure:
    output/tf_reversal/
    ├── <record_id>.png        # per-record diagnosis figure
    ├── <record_id>.json       # verdict + features (+ synthetic variants)
    ├── summary.json
    └── score_distribution.png # phase vs corr scatter, colored by verdict

Usage:
    python -m ecg_spectrum_analysis.src.batch_tf_reversal --n 20
    python -m ecg_spectrum_analysis.src.batch_tf_reversal --n 20 --validate-synthetic
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os, json, time, gc, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')

from ecg_waveform_extraction.src.utils.aecg_parser import parse_aecg

from ecg_spectrum_analysis.src.tf_reversal import (
    detect_record_reversal, detect_tf_reversal,
    synthetic_swap, flip_lead_i,
)
from ecg_spectrum_analysis.src.plot_tf_reversal import (
    plot_tf_reversal, reversal_result_to_dict, save_figure,
)
from ecg_spectrum_analysis.src.plot_spectrogram import FIG_DPI

# ---------------------------------------------------------------------------
PKG_ROOT = Path(__file__).resolve().parent.parent
AECG_DIR = 'C:/LoyaltyLo/datasets/RA-LA_Reversal/aECG'
OUT_DIR = str(PKG_ROOT / 'output' / 'tf_reversal')
V10_CSV = 'C:/LoyaltyLo/datasets/RA-LA_Reversal/aECG_v10_triple.csv'
MAX_SAMPLES = 8000  # 8 s at the dataset's 1000 Hz (~8-13 beats)


def find_aecg_files(aecg_dir: str, n: int | None = None) -> list[str]:
    files = sorted(
        os.path.join(aecg_dir, f)
        for f in os.listdir(aecg_dir)
        if f.lower().endswith('.aecg')
    )
    return files[:n] if n else files


def load_v10_verdicts(csv_path: str) -> dict[str, bool]:
    """Main package v10 verdicts: record id -> swap_detected (True=reversed)."""
    if not os.path.exists(csv_path):
        return {}
    import pandas as pd
    df = pd.read_csv(csv_path)
    return {str(r['file']).removesuffix('.aECG'): bool(r['swap_detected'])
            for _, r in df.iterrows()}


def analyze_one(filepath: str, fs: float | None = None,
                max_samples: int = MAX_SAMPLES):
    """Detect on one record.

    Returns (record_id, result, extras, signals, clean, fs); `signals` is
    the raw dict (for the synthetic variants), `clean` the preprocessed
    one (for plotting).
    """
    aecg = parse_aecg(filepath, max_samples=max_samples)
    fs = aecg['fs']
    signals = aecg['signals']
    record_id = Path(filepath).stem

    from ecg_waveform_extraction.src.preprocessing.filters import ECGPreprocessor
    prep = ECGPreprocessor(fs=fs)
    clean = {name: prep.preprocess(np.asarray(sig, dtype=np.float64))
             for name, sig in signals.items() if sig is not None}
    if 'I' not in clean or 'II' not in clean:
        return record_id, None, {}, signals, clean, fs
    result, extras = detect_tf_reversal(
        clean['I'], clean['II'], fs, clean_iii=clean.get('III'),
        return_extras=True,
    )
    return record_id, result, extras, signals, clean, fs


def plot_score_distribution(entries: list[dict], out_dir: str):
    """Scatter of circular-mean phase vs QRS correlation, colored by verdict."""
    import matplotlib.pyplot as plt
    from .plot_tf_reversal import VERDICT_COLORS

    pts = [(e['phase_qrs_deg'], e['corr_qrs'], e['verdict'], e['record'])
           for e in entries
           if e.get('phase_qrs_deg') is not None and e.get('corr_qrs') is not None]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(8, 6), dpi=FIG_DPI)
    for verdict in ('normal', 'reversed', 'uncertain'):
        xs = [p[0] for p in pts if p[2] == verdict]
        ys = [p[1] for p in pts if p[2] == verdict]
        ax.scatter(xs, ys, s=42, alpha=0.85, label=verdict,
                   color=VERDICT_COLORS[verdict],
                   edgecolor='white', linewidth=0.5, zorder=3)
    ax.axvspan(-180, -90, color='#C44E52', alpha=0.06, lw=0)
    ax.axvspan(90, 180, color='#C44E52', alpha=0.06, lw=0)
    ax.axhline(0.0, color='#C44E52', linestyle='--', linewidth=1.0, zorder=2)
    for x, y, v, rid in pts:
        ax.annotate(rid[-4:], (x, y), fontsize=6, ha='center', va='bottom',
                    xytext=(0, 4), textcoords='offset points', alpha=0.8)
    ax.set_xlabel('circular-mean cross-phase φ_qrs (deg)')
    ax.set_ylabel('median QRS correlation corr(I, II)')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title('TF RA-LA reversal — decision space '
                 '(shaded = anti-phase |φ|≥90°, dashed = corr 0)',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, linestyle='--')
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, 'score_distribution.png'))


def main():
    parser = argparse.ArgumentParser(
        description='Batch TF RA-LA Reversal Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ecg_spectrum_analysis.src.batch_tf_reversal --n 20
  python -m ecg_spectrum_analysis.src.batch_tf_reversal --n 20 --validate-synthetic
        """,
    )
    parser.add_argument('--n', type=int, default=20,
                        help='Number of records to process (default: 20).')
    parser.add_argument('--max-samples', type=int, default=MAX_SAMPLES,
                        help=f'Samples per record (default: {MAX_SAMPLES}).')
    parser.add_argument('--validate-synthetic', action='store_true',
                        help='Also run flip-I and full-swap variants and '
                             'report flip rates.')
    parser.add_argument('--v10-csv', type=str, default=V10_CSV,
                        help='Main package verdict CSV for agreement stats.')
    parser.add_argument('--dpi', type=int, default=FIG_DPI)
    parser.add_argument('--aecg-dir', type=str, default=AECG_DIR)
    parser.add_argument('--out-dir', type=str, default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = find_aecg_files(args.aecg_dir, n=args.n)
    n_total = len(files)
    v10 = load_v10_verdicts(args.v10_csv) if args.v10_csv else {}

    print(f'Found {n_total} aECG files in {args.aecg_dir}')
    print(f'max_samples: {args.max_samples} | synthetic validation: '
          f'{args.validate_synthetic} | v10 baseline: {len(v10)} records')
    print(f'Output: {args.out_dir}')
    print('=' * 60)

    t_start = time.perf_counter()
    entries, errors = [], []

    for i, fp in enumerate(files, 1):
        rid = Path(fp).stem
        print(f'[{i:4d}/{n_total}] {rid} ...', end=' ', flush=True)
        try:
            record_id, result, extras, signals, clean, fs = analyze_one(
                fp, max_samples=args.max_samples)
            if result is None:
                print('SKIP (missing lead I/II)')
                continue

            entry = {'record': record_id, **reversal_result_to_dict(result)}
            entry['v10'] = v10.get(record_id)

            # ---- Per-record figure (preprocessed limb leads) ----
            clean_needed = {L: clean.get(L) for L in ('I', 'II', 'III')}
            save_figure(
                plot_tf_reversal(
                    clean_needed, result, fs, record_name=record_id,
                    r_peaks=extras.get('r_peaks'),
                    cross_phase=extras.get('cross_phase'),
                    weight=extras.get('weight'),
                    freqs=extras.get('freqs'),
                    beat_corr=extras.get('beat_corr'),
                    dpi=args.dpi,
                ),
                os.path.join(args.out_dir, f'{record_id}.png'),
            )

            # ---- Synthetic validation variants ----
            if args.validate_synthetic:
                prep_variants = {}
                for tag, sigs in (('flip_i', flip_lead_i(signals)),
                                  ('swap', synthetic_swap(signals))):
                    r = detect_record_reversal(sigs, fs)
                    prep_variants[tag] = reversal_result_to_dict(r)
                entry['synthetic'] = prep_variants

            with open(os.path.join(args.out_dir, f'{record_id}.json'), 'w',
                      encoding='utf-8') as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)
            entries.append(entry)

            syn = ''
            if args.validate_synthetic and 'synthetic' in entry:
                syn = (f" | flipI={entry['synthetic']['flip_i']['verdict']}"
                       f" swap={entry['synthetic']['swap']['verdict']}")
            print(f"{result.verdict:9s} (conf {result.confidence:.2f}){syn}")
        except Exception as e:
            errors.append({'record': rid, 'error': str(e)})
            print(f'ERROR: {e}')
        if i % 10 == 0:
            gc.collect()

    t_total = time.perf_counter() - t_start

    # ---- Summary ----
    counts = {}
    for e in entries:
        counts[e['verdict']] = counts.get(e['verdict'], 0) + 1

    v10_pairs = [(e['v10'], e['verdict']) for e in entries
                 if e.get('v10') is not None and e['verdict'] != 'uncertain']
    v10_agree = sum(
        1 for v, tf in v10_pairs
        if tf == ('reversed' if v else 'normal'))

    report = {
        'total_processed': len(entries),
        'total_errors': len(errors),
        'verdict_counts': counts,
        'v10_baseline': {
            'n_comparable': len(v10_pairs),
            'agreement': v10_agree,
            'note': 'v10 = main package aECG_v10_triple.csv swap_detected; '
                    'uncertain excluded. Confirmed-label xlsx has zero ID '
                    'overlap with this working set.',
        },
        'total_time_sec': round(t_total, 1),
        'entries': entries,
        'errors': errors,
    }

    if args.validate_synthetic:
        flip_stats = {'flip_i': {'to_reversed': 0, 'confident': 0},
                      'swap': {'to_reversed': 0, 'confident': 0}}
        inversions = {'flip_i': [0, 0], 'swap': [0, 0]}  # [ok, n_confident_orig]
        for e in entries:
            orig = e['verdict']
            for tag in flip_stats:
                syn = e.get('synthetic', {}).get(tag)
                if not syn or syn['verdict'] == 'uncertain':
                    continue
                flip_stats[tag]['confident'] += 1
                flip_stats[tag]['to_reversed'] += syn['verdict'] == 'reversed'
                if orig in ('normal', 'reversed'):
                    inversions[tag][1] += 1
                    inversions[tag][0] += syn['verdict'] != orig
        report['synthetic_validation'] = {
            'flip_i': {
                'verdict_inversion': f'{inversions["flip_i"][0]}/'
                                     f'{inversions["flip_i"][1]}',
                'to_reversed': f'{flip_stats["flip_i"]["to_reversed"]}/'
                               f'{flip_stats["flip_i"]["confident"]}',
                'note': 'pure anti-phase signature — machinery test',
            },
            'swap': {
                'verdict_inversion': f'{inversions["swap"][0]}/'
                                     f'{inversions["swap"][1]}',
                'to_reversed': f'{flip_stats["swap"]["to_reversed"]}/'
                               f'{flip_stats["swap"]["confident"]}',
                'note': 'full anatomy (II<->III confound) — see tf_reversal docstring',
            },
        }

    with open(os.path.join(args.out_dir, 'summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    plot_score_distribution(entries, args.out_dir)

    print('\n' + '=' * 60)
    print(f'Done. {len(entries)}/{n_total} processed, {len(errors)} errors, '
          f'{t_total:.1f}s ({t_total / max(n_total, 1):.1f}s/record)')
    print(f'Verdicts: {counts}')
    if v10_pairs:
        print(f'v10 agreement (confident): {v10_agree}/{len(v10_pairs)}')
    if args.validate_synthetic and 'synthetic_validation' in report:
        sv = report['synthetic_validation']
        print(f"Synthetic flip-I inversion:   {sv['flip_i']['verdict_inversion']}"
              f"  (to_reversed {sv['flip_i']['to_reversed']})")
        print(f"Synthetic full-swap inversion: {sv['swap']['verdict_inversion']}"
              f"  (to_reversed {sv['swap']['to_reversed']})")
    print(f'Summary: {os.path.join(args.out_dir, "summary.json")}')


if __name__ == '__main__':
    main()
