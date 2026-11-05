#!/usr/bin/env python
"""Public calling interface for the ECG delineation + reversal pipeline.

Wraps the seven stage modules (each stays a runnable CLI via
``python -m ecg_waveform_extraction.src.batch_*``) into one importable
API.  Stages run IN PROCESS by calling their ``main(argv)`` — no shell
out, no subprocess overhead.

    from ecg_waveform_extraction.src.api import run_pipeline, analyze_record

    # batch: a directory of .aECG files -> verdicts + CSVs + figures
    res = run_pipeline(data_dir=r'C:\...\aECG', out_dir=r'C:\...\out')

    # single record -> one plain dict (temp workspace, auto-cleaned)
    r = analyze_record(r'C:\...\aECG\1805185J6U.aECG')
    r['verdict'], r['reversal_type'], r['polarity']['I']

Stage chain and its canonical output subdirectories under ``out_dir``:

    1 rpeaks    pan_tompkins_rpeaks_ref/  (descriptive audit table)
    2 qrs       qrs_wavedet_reflead/      (re-detects reflead+PT in process)
    3 twave     twave_wavedet_reflead/
    4 pwave     pwave_reflead/            (reads stage-3 dir)
    5 polarity  polarity_reflead/         (joins stage 3/4/QRS CSVs)
    6 overview  record_overview/
    7 reversal  reversal_reflead/         (reads polarity + overview)

CLI (installed console script, or ``python -m ecg_waveform_extraction.src.api``):

    ecg-pipeline run    --data-dir DIR --out-dir DIR [--records a,b] [--no-plots] [--stages ...]
    ecg-pipeline analyze RECORD.aECG [--work-dir DIR] [--keep] [--json]
    ecg-pipeline results--out-dir DIR

Notes
-----
* Wave detection itself is library code (sleepecg / wavedet /
  prominence-delineator / neurokit2); this module only orchestrates.
* Stages 2-4 always write their own per-record figures (no flag in
  those CLIs); ``plots=False`` only disables the stage-7 figures.
* A record filter ``records=...`` is implemented by copying the
  selected files into ``out_dir/_data_subset`` — that way stage 1
  (which has no --records flag) is filtered too, and every stage sees
  exactly the requested cohort.
* CSVs are read with the ``.new.csv`` fallback used when a stage found
  the target locked (file open in Excel).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

# canonical execution order; subset runs keep this order
STAGES = ('rpeaks', 'qrs', 'twave', 'pwave', 'polarity', 'overview', 'reversal')
SUBDIRS = {
    'rpeaks': 'pan_tompkins_rpeaks_ref',
    'qrs': 'qrs_wavedet_reflead',
    'twave': 'twave_wavedet_reflead',
    'pwave': 'pwave_reflead',
    'polarity': 'polarity_reflead',
    'overview': 'record_overview',
    'reversal': 'reversal_reflead',
}
_MODULE = {
    'rpeaks': 'batch_rpeaks_pt_reflead',
    'qrs': 'batch_qrs_wavedet_reflead',
    'twave': 'batch_twave_wavedet_reflead',
    'pwave': 'batch_pwave_reflead',
    'polarity': 'batch_polarity_reflead',
    'overview': 'batch_record_overview',
    'reversal': 'batch_reversal_reflead',
}
LEADS = ('I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')


# ---------------------------------------------------------------- helpers

def _records_csv(records) -> str | None:
    """'a,b' | ['a','b'] | {'a'} | None -> 'a,b' | None."""
    if records is None:
        return None
    if isinstance(records, str):
        records = [r.strip() for r in records.split(',') if r.strip()]
    else:
        records = [str(r).strip().removesuffix('.aECG')
                   for r in records if str(r).strip()]
    return ','.join(records) if records else None


def _ordered_stages(stages: Iterable[str] | None = None) -> list[str]:
    """Validate a stage subset and return it in canonical order."""
    want = list(stages) if stages else list(STAGES)
    unknown = [s for s in want if s not in STAGES]
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; choose from {list(STAGES)}")
    return [s for s in STAGES if s in set(want)]


def _stage_argv(stage: str, data_dir: str, out_root: Path,
                records_csv: str | None, plots: bool) -> list[str]:
    """Argv for one stage's main() — pure function (unit-tested)."""
    out = out_root / SUBDIRS[stage]
    argv = ['--data-dir', str(data_dir), '--out-dir', str(out)]
    if stage == 'pwave':
        argv += ['--t-stage-dir', str(out_root / SUBDIRS['twave'])]
    elif stage == 'polarity':
        argv += ['--p-dir', str(out_root / SUBDIRS['pwave']),
                 '--t-dir', str(out_root / SUBDIRS['twave']),
                 '--q-dir', str(out_root / SUBDIRS['qrs'])]
    elif stage == 'overview':
        argv = ['--qrs-dir', str(out_root / SUBDIRS['qrs']),
                '--t-dir', str(out_root / SUBDIRS['twave']),
                '--p-dir', str(out_root / SUBDIRS['pwave']),
                '--pol-dir', str(out_root / SUBDIRS['polarity']),
                '--out-dir', str(out)]
    elif stage == 'reversal':
        argv += ['--pol-dir', str(out_root / SUBDIRS['polarity']),
                 '--overview',
                 str(out_root / SUBDIRS['overview'] / 'record_overview.csv')]
        if not plots:
            argv.append('--no-plots')
    if records_csv and stage != 'rpeaks':  # stage 1 has no --records flag
        argv += ['--records', records_csv]
    return argv


def _read_csv(out_dir: Path | str, name: str):
    """read a stage CSV with the locked-file (.new.csv) fallback."""
    import pandas as pd
    base = Path(out_dir)
    main = base / name
    if main.exists():
        return pd.read_csv(main, encoding='utf-8-sig')
    alt = base / (name[:-len('.csv')] + '.new.csv')
    if alt.exists():
        return pd.read_csv(alt, encoding='utf-8-sig')
    raise FileNotFoundError(f"neither {main} nor {alt} exists — run the stage first")


def _j(v):
    """numpy/pandas scalar -> JSON-friendly python scalar (NaN -> None)."""
    try:
        if v != v:  # NaN
            return None
    except (TypeError, ValueError):
        pass
    return v.item() if hasattr(v, 'item') else v


# ---------------------------------------------------------------- pipeline

def run_pipeline(data_dir: str | None = None, out_dir: str | Path | None = None,
                 records: Iterable[str] | str | None = None, plots: bool = True,
                 stages: Iterable[str] | None = None, verbose: bool = True) -> dict:
    """Run the delineation + reversal chain over a directory of .aECG files.

    Parameters
    ----------
    data_dir : directory containing .aECG XML files.  None = the stage
        default (this machine's RA-LA_Reversal dataset path).
    out_dir : REQUIRED.  Root for all stage outputs; canonical
        subdirectories (see module docstring) are created under it.
    records : optional cohort filter — record names ('1805185J6U') or
        file names, single string 'a,b' or an iterable.  The selected
        files are copied to ``out_dir/_data_subset`` and every stage
        runs on that subset.
    plots : False disables the stage-7 figures (--no-plots).
    stages : subset to run, e.g. ('polarity', 'overview', 'reversal')
        to redo the decision layers over existing stage outputs.
        Always executed in canonical order.
    verbose : print a per-stage banner.

    Returns
    -------
    dict with the paths that exist after the run:
        {'out_dir', 'stages_run', 'reversal_csv', 'reversal_simple_csv',
         'overview_csv', 'cohort_summary', 'elapsed_sec'}
    """
    if out_dir is None:
        raise ValueError("out_dir is required (stage defaults point into "
                         "the repo's output/ tree — pass an explicit root)")
    import importlib
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    order = _ordered_stages(stages)
    records_csv = _records_csv(records)

    if records_csv:
        src_dir = Path(data_dir) if data_dir else _default_data_dir()
        subset = out_root / '_data_subset'
        subset.mkdir(parents=True, exist_ok=True)
        wanted = set(records_csv.split(','))
        copied = []
        for f in sorted(src_dir.glob('*.aECG')):
            if f.stem in wanted:
                shutil.copy2(f, subset / f.name)
                copied.append(f.stem)
        missing = wanted - set(copied)
        if missing:
            raise FileNotFoundError(
                f"record(s) {sorted(missing)} not found in {src_dir}")
        data_dir = str(subset)
        if verbose:
            print(f"[api] cohort subset: {len(copied)} file(s) -> {subset}")

    t0 = time.time()
    for stage in order:
        mod = importlib.import_module(f'ecg_waveform_extraction.src.{_MODULE[stage]}')
        argv = _stage_argv(stage, str(data_dir or _default_data_dir()),
                           out_root, records_csv, plots)
        if verbose:
            print(f"[api] stage {STAGES.index(stage) + 1}/7 {stage}: "
                  f"python -m ecg_waveform_extraction.src.{_MODULE[stage]} "
                  + ' '.join(argv), flush=True)
        rc = mod.main(argv)
        if rc not in (0, None):
            raise RuntimeError(f"stage '{stage}' exited with {rc}")

    res = {'out_dir': str(out_root), 'stages_run': order,
           'elapsed_sec': round(time.time() - t0, 1)}
    for key, p in (
            ('reversal_csv', out_root / SUBDIRS['reversal'] / 'reversal_by_record.csv'),
            ('reversal_simple_csv', out_root / SUBDIRS['reversal'] / 'reversal_simple.csv'),
            ('overview_csv', out_root / SUBDIRS['overview'] / 'record_overview.csv'),
            ('cohort_summary', out_root / SUBDIRS['reversal'] / 'cohort_summary.txt')):
        if p.exists():
            res[key] = str(p)
    return res


def _default_data_dir() -> str:
    from ecg_waveform_extraction.src.batch_reversal_reflead import DEFAULT_DATA_DIR
    return DEFAULT_DATA_DIR


# ---------------------------------------------------------- single record

def analyze_record(aecg_path: str | Path, work_dir: str | Path | None = None,
                   keep_workdir: bool = False, plots: bool = True) -> dict:
    """Full-chain analysis of ONE .aECG file -> one plain dict.

    Copies the file into a workspace (temp dir, deleted afterwards
    unless ``keep_workdir``), runs all seven stages there, and merges
    the reversal verdict with the record-overview row.

    Returns (JSON-friendly, NaN -> None)
    ------------------------------------
    record, verdict, reversal_type, score, fired, notes,
    signatures{S1..S3, T1/T2, U1/U2},
    p_frontal_axis_deg, p_amplitude_mv, p_pass_rate_pct, hr_median_bpm,
    polarity{LEAD: {P, QRS, T}},     # majority polarity, 12 leads
    device{P_axis_deg, QRS_axis_deg, T_axis_deg, flagged_reversal,
           interpretation}            # comparison columns only — the
                                       verdict never reads device fields
    """
    src = Path(aecg_path)
    if not src.exists():
        raise FileNotFoundError(src)
    if src.suffix != '.aECG':
        raise ValueError(f"expected a .aECG file, got {src.name}")
    tmp = None
    if work_dir is None:
        tmp = tempfile.mkdtemp(prefix='ecg_analyze_')
        work_dir = tmp
    work = Path(work_dir)
    data, out = work / 'data', work / 'out'
    data.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, data / src.name)
    try:
        run_pipeline(data_dir=str(data), out_dir=out,
                     records=[src.stem], plots=plots)
        row = _read_csv(out / SUBDIRS['reversal'], 'reversal_by_record.csv')
        row = row[row['record'] == src.stem]
        if row.empty:
            raise RuntimeError(f"record {src.stem} missing from stage-7 output")
        r = row.iloc[0]
        ov = _read_csv(out / SUBDIRS['overview'], 'record_overview.csv')
        ov = ov[ov['record'] == src.stem]
        o = ov.iloc[0] if not ov.empty else {}
        polarity = {lead: {w: _j(o.get(f'{lead}_{w}')) for w in ('P', 'QRS', 'T')}
                    for lead in LEADS}
        return {
            'record': src.stem,
            'verdict': r['verdict'], 'reversal_type': r['reversal_type'],
            'score': _j(r['score']), 'fired': r['fired'],
            'notes': '' if r['notes'] != r['notes'] else r['notes'],
            'signatures': {k: bool(r[k]) for k in (
                'S1_P_axis_flip', 'S2_lead_I_inverted', 'S3_aVR_all_positive',
                'T1_II_inverted', 'T2_aVR_aVF_swap', 'U1_III_inverted',
                'U2_aVL_aVF_swap')},
            'p_frontal_axis_deg': _j(r['p_ax_deg']),
            'p_amplitude_mv': _j(r['p_amp_mv']),
            'p_pass_rate_pct': _j(r['p_pass_rate_pct']),
            'hr_median_bpm': _j(o.get('hr_median_bpm')),
            'polarity': polarity,
            'device': {
                'P_axis_deg': _j(r['device_P_axis']),
                'flagged_reversal': bool(r['device_flag']),
                'interpretation': ('' if r['interpretation'] != r['interpretation']
                                   else str(r['interpretation'])),
            },
            'work_dir': str(work),
        }
    finally:
        if tmp is not None and not keep_workdir:
            shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------- read-backs

def read_reversal(out_dir: str | Path, simple: bool = False):
    """Stage-7 verdict table (reversal_by_record.csv or _simple.csv)."""
    d = Path(out_dir) / SUBDIRS['reversal']
    return _read_csv(d, 'reversal_simple.csv' if simple
                     else 'reversal_by_record.csv')


def read_overview(out_dir: str | Path):
    """Stage-6 per-record overview table (666-row style)."""
    return _read_csv(Path(out_dir) / SUBDIRS['overview'], 'record_overview.csv')


# -------------------------------------------------------------------- cli

def _cli() -> int:
    ap = argparse.ArgumentParser(
        prog='ecg-pipeline',
        description='ECG delineation + limb-lead reversal pipeline')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p_run = sub.add_parser('run', help='batch: directory of .aECG files')
    p_run.add_argument('--data-dir', default=None,
                       help='directory of .aECG files (default: stage default)')
    p_run.add_argument('--out-dir', required=True,
                       help='root for all stage outputs')
    p_run.add_argument('--records', default=None,
                       help='comma-separated record filter')
    p_run.add_argument('--no-plots', action='store_true')
    p_run.add_argument('--stages', default=None,
                       help=f"comma-separated subset of {list(STAGES)}")

    p_an = sub.add_parser('analyze', help='single .aECG file -> one dict')
    p_an.add_argument('aecg')
    p_an.add_argument('--work-dir', default=None,
                      help='keep the workspace here (also enables --keep)')
    p_an.add_argument('--keep', action='store_true',
                      help='do not delete the temp workspace')
    p_an.add_argument('--json', action='store_true', help='dump full JSON')

    p_res = sub.add_parser('results', help='verdict counts of a finished run')
    p_res.add_argument('--out-dir', required=True)

    args = ap.parse_args()

    if args.cmd == 'run':
        res = run_pipeline(data_dir=args.data_dir, out_dir=args.out_dir,
                           records=args.records, plots=not args.no_plots,
                           stages=args.stages.split(',') if args.stages else None)
        print(json.dumps({k: v for k, v in res.items()}, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == 'analyze':
        r = analyze_record(args.aecg, work_dir=args.work_dir,
                           keep_workdir=bool(args.work_dir) or args.keep)
        if args.json:
            print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
        else:
            sig = {k: v for k, v in r['signatures'].items() if v}
            print(f"record      {r['record']}")
            print(f"verdict     {r['verdict']}  type={r['reversal_type'] or '-'}"
                  f"  score={r['score']}")
            print(f"P axis      {r['p_frontal_axis_deg']} deg, "
                  f"amp {r['p_amplitude_mv']} mV, pass {r['p_pass_rate_pct']}%")
            print(f"HR          {r['hr_median_bpm']} bpm")
            print(f"signatures  {sig or '-'}")
            print(f"notes       {r['notes'] or '-'}")
            pol = r['polarity']
            print("polarity    " + '  '.join(
                f"{l}:{(pol[l]['P'] or '?')[0]}"
                f"{(pol[l]['QRS'] or '?')[0]}{(pol[l]['T'] or '?')[0]}"
                for l in LEADS[:6]))
            print(f"            " + '  '.join(
                f"{l}:{(pol[l]['P'] or '?')[0]}"
                f"{(pol[l]['QRS'] or '?')[0]}{(pol[l]['T'] or '?')[0]}"
                for l in LEADS[6:]))
        return 0
    # results
    df = read_reversal(args.out_dir)
    print(df['verdict'].value_counts().to_string())
    if 'reversal_type' in df:
        det = df[df['verdict'].isin(('definite', 'probable'))]
        print("\nreversal types among detected:")
        print(det['reversal_type'].value_counts().to_string())
    return 0


# installed console script target
cli = _cli

if __name__ == '__main__':
    raise SystemExit(_cli())
