#!/usr/bin/env python3
"""Batch ECG Spectrogram Generator.

Generates STFT spectrograms, PSD plots, and CWT scalograms for all 12 leads
from aECG XML files.

Output structure per record:
    output/spectrograms/
    └── <record_id>/
        ├── I/
        │   ├── <record>_I_waveform_stft.png   # waveform + STFT spectrogram
        │   ├── <record>_I_waveform_cwt.png    # waveform + CWT scalogram
        │   └── <record>_I_waveform_psd.png    # waveform + PSD
        ├── II/ ...
        ├── multi_lead_<record>_stft.png        # 12-lead STFT grid overview
        ├── multi_lead_<record>_cwt.png         # 12-lead CWT grid overview
        └── psd_comparison_<record>.png         # 12-lead PSD overlay

Usage:
    python -m ecg_spectrum_analysis.src.batch_spectrogram --n 10
    python -m ecg_spectrum_analysis.src.batch_spectrogram --n 50 --method stft
    python -m ecg_spectrum_analysis.src.batch_spectrogram --n 20 --no-waveform
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

from ecg_spectrum_analysis.src.spectrogram import (
    compute_spectrogram, compute_psd, compute_scalogram,
    ECG_Spectrogram, band_power,
)
from ecg_spectrum_analysis.src.plot_spectrogram import (
    plot_waveform_with_spectrum, plot_multi_lead_spectrograms,
    plot_psd_comparison, save_figure,
)

# ---------------------------------------------------------------------------
AECG_DIR = 'C:/LoyaltyLo/datasets/RA-LA_Reversal/aECG'
OUT_DIR = str(Path(__file__).resolve().parent.parent / 'output' / 'spectrograms')
MAX_SAMPLES = 16000  # ~16 s at 1000 Hz (actual dataset fs; auto-clipped if record is shorter)
ALL_LEADS = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


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


def process_one_record(
    filepath: str,
    out_dir: str,
    method: str = 'all',
    per_lead_waveform: bool = True,
    multi_lead_grid: bool = True,
    psd_comparison: bool = True,
    dpi: int = 150,
) -> dict:
    """Process a single aECG record: compute and save spectrograms.

    Parameters
    ----------
    filepath : str
        Path to .aECG file.
    out_dir : str
        Output directory for spectrogram images.
    method : str
        'stft', 'cwt', 'psd', or 'all'.
    per_lead_waveform : bool
        Save per-lead waveform+spectrum stacked images (each lead in its own subdirectory).
    multi_lead_grid : bool
        Save 12-lead grid overviews.
    psd_comparison : bool
        Save overlaid PSD comparison.
    dpi : int
        Output image resolution.

    Returns
    -------
    dict with processing stats.
    """
    # ---- Parse ----
    record_id = Path(filepath).stem
    record_out = Path(out_dir) / record_id

    aecg = parse_aecg(filepath, max_samples=MAX_SAMPLES)
    fs = aecg['fs']
    signals = aecg['signals']

    if not signals:
        return {'record': record_id, 'error': 'No signals found'}

    prep = ECGPreprocessor(fs=fs)

    t0 = time.perf_counter()
    methods_to_run = ['stft', 'cwt', 'psd'] if method == 'all' else [method]

    # ---- Process each lead ----
    stft_specs = []
    cwt_specs = []
    psd_specs = []

    for lead_name in ALL_LEADS:
        raw = signals.get(lead_name)
        if raw is None:
            continue

        clean = prep.preprocess(raw[:MAX_SAMPLES].astype(np.float64))

        # Per-lead output directory
        lead_out = record_out / lead_name
        if per_lead_waveform:
            lead_out.mkdir(parents=True, exist_ok=True)

        for m in methods_to_run:
            if m == 'stft':
                spec = compute_spectrogram(
                    clean, fs=fs, nperseg=256, freq_limit=60.0,
                    scale='power', lead_name=lead_name, record_name=record_id,
                )
                stft_specs.append(spec)

                if per_lead_waveform:
                    fig = plot_waveform_with_spectrum(clean, spec, dpi=dpi)
                    save_figure(fig, lead_out / f'{record_id}_{lead_name}_waveform_stft.png')

            elif m == 'cwt':
                spec = compute_scalogram(
                    clean, fs=fs, freq_range=(0.5, 60.0), n_voices=64,
                    scale='magnitude', lead_name=lead_name, record_name=record_id,
                )
                cwt_specs.append(spec)

                if per_lead_waveform:
                    fig = plot_waveform_with_spectrum(clean, spec, dpi=dpi)
                    save_figure(fig, lead_out / f'{record_id}_{lead_name}_waveform_cwt.png')

            elif m == 'psd':
                spec = compute_psd(
                    clean, fs=fs, freq_limit=60.0, scale='power',
                    lead_name=lead_name, record_name=record_id,
                )
                psd_specs.append(spec)

                if per_lead_waveform:
                    fig = plot_waveform_with_spectrum(clean, spec, dpi=dpi)
                    save_figure(fig, lead_out / f'{record_id}_{lead_name}_waveform_psd.png')

    # ---- Multi-lead overviews ----
    if multi_lead_grid and stft_specs:
        fig = plot_multi_lead_spectrograms(
            stft_specs, ncols=4, suptitle=f'{record_id} — STFT Spectrograms (12-lead)',
        )
        save_figure(fig, record_out / f'multi_lead_{record_id}_stft.png')

    if multi_lead_grid and cwt_specs:
        fig = plot_multi_lead_spectrograms(
            cwt_specs, ncols=4, suptitle=f'{record_id} — CWT Scalograms (12-lead)',
        )
        save_figure(fig, record_out / f'multi_lead_{record_id}_cwt.png')

    if psd_comparison and psd_specs:
        fig = plot_psd_comparison(
            psd_specs, title=f'PSD Comparison — {record_id}',
        )
        save_figure(fig, record_out / f'psd_comparison_{record_id}.png')

    dt = time.perf_counter() - t0
    return {
        'record': record_id, 'fs': fs, 'n_samples': aecg['n_samples'],
        'leads_processed': len(signals), 'time_sec': round(dt, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Batch ECG Spectrogram Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ecg_spectrum_analysis.src.batch_spectrogram --n 10
  python -m ecg_spectrum_analysis.src.batch_spectrogram --n 50 --method stft --no-per-lead
  python -m ecg_spectrum_analysis.src.batch_spectrogram --n 5 --method all --dpi 200
        """,
    )
    parser.add_argument('--n', type=int, default=10,
                        help='Number of records to process (default: 10).')
    parser.add_argument('--method', choices=['stft', 'cwt', 'psd', 'all'],
                        default='all', help='Spectrogram method (default: all).')
    parser.add_argument('--no-waveform', action='store_true',
                        help='Skip per-lead waveform+spectrum images.')
    parser.add_argument('--no-grid', action='store_true',
                        help='Skip multi-lead grid overviews.')
    parser.add_argument('--no-psd-comp', action='store_true',
                        help='Skip PSD comparison plots.')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Output image DPI (default: 150).')
    parser.add_argument('--aecg-dir', type=str, default=AECG_DIR,
                        help='Path to aECG files directory.')
    parser.add_argument('--out-dir', type=str, default=OUT_DIR,
                        help='Output directory for spectrogram images.')
    args = parser.parse_args()

    aecg_dir = args.aecg_dir
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)

    # Find files
    files = find_aecg_files(aecg_dir, n=args.n)
    n_total = len(files)
    print(f'Found {n_total} aECG files in {aecg_dir}')
    print(f'Method: {args.method} | DPI: {args.dpi}')
    print(f'Output: {out_dir}')
    print(f'Waveform+spectrum: {not args.no_waveform} | Grid: {not args.no_grid} | PSD comp: {not args.no_psd_comp}')
    print('=' * 60)

    t_start = time.perf_counter()
    stats = []
    errors = []

    for i, fp in enumerate(files, 1):
        rid = Path(fp).stem
        print(f'[{i:4d}/{n_total}] {rid} ...', end=' ', flush=True)

        try:
            stat = process_one_record(
                fp,
                out_dir=out_dir,
                method=args.method,
                per_lead_waveform=not args.no_waveform,
                multi_lead_grid=not args.no_grid,
                psd_comparison=not args.no_psd_comp,
                dpi=args.dpi,
            )
            stats.append(stat)
            print(f'OK ({stat["time_sec"]}s, {stat["leads_processed"]} leads)')
        except Exception as e:
            errors.append({'record': rid, 'error': str(e)})
            print(f'ERROR: {e}')

        # Periodic GC
        if i % 10 == 0:
            gc.collect()

    # ---- Summary ----
    t_total = time.perf_counter() - t_start
    print('\n' + '=' * 60)
    print(f'Done. {len(stats)}/{n_total} successful, {len(errors)} errors.')
    print(f'Total time: {t_total:.1f}s ({t_total/n_total:.1f}s/record)')

    if stats:
        avg_fs = np.mean([s.get('fs', 0) for s in stats])
        avg_samples = np.mean([s.get('n_samples', 0) for s in stats])
        print(f'Avg fs: {avg_fs:.1f} Hz | Avg samples: {avg_samples:.0f}')

    if errors:
        print('\nErrors:')
        for e in errors:
            print(f'  {e["record"]}: {e["error"]}')

    # Save summary JSON
    summary_path = os.path.join(out_dir, '_batch_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_processed': len(stats),
            'total_errors': len(errors),
            'method': args.method,
            'total_time_sec': round(t_total, 1),
            'stats': stats,
            'errors': errors,
        }, f, indent=2, ensure_ascii=False)
    print(f'\nSummary saved to {summary_path}')


if __name__ == '__main__':
    main()
