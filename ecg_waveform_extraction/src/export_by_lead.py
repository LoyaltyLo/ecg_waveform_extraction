#!/usr/bin/env python3
"""Export the per-record limb-lead cache into a lead-first layout.

Creates output/rala_full/_limb_leads_by_lead/lead_<X>/<record>/ containing
the same per-lead artifacts as the record-first cache (beats.json,
qrs_polarity.json, p_waves.json, t_waves.json, filtered_ecg.npy,
state_labels.npy, plus the record summary.json for context), organized so
all records of one lead sit under one directory. Files are HARDLINKED by
default (no data duplication, instant; pass --copy for real copies, e.g.
across drives).

Also writes per-lead aggregate summaries (lead_<X>/_lead_summary.json):
record count, beat count, and polarity totals aggregated from each
record's summary.json.

The record-first cache at output/rala_full/_limb_leads/ stays untouched —
audit_spectral_consistency.py and audit_pt_delineation.py read that layout.

Usage:
    python -m ecg_waveform_extraction.src.export_by_lead
    python -m ecg_waveform_extraction.src.export_by_lead --copy --out D:/by_lead
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import os
from collections import Counter

BASE = Path(__file__).resolve().parent.parent / 'output/rala_full/_limb_leads'
OUT = Path(__file__).resolve().parent.parent / 'output/rala_full/_limb_leads_by_lead'
LEADS = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF']
LEAD_FILES = ['beats.json', 'qrs_polarity.json', 'p_waves.json', 't_waves.json',
              'filtered_ecg.npy', 'state_labels.npy']
RECORD_FILES = ['summary.json', 'cross_lead.json']


def link_or_copy(src: Path, dst: Path, copy: bool):
    if copy:
        dst.link_to(src)
    else:
        try:
            os.link(src, dst)
        except OSError:
            # cross-device or link unsupported -> fall back to copy
            dst.write_bytes(src.read_bytes())


def main():
    ap = argparse.ArgumentParser(description='Export cache lead-first')
    ap.add_argument('--base', default=str(BASE), help='record-first cache dir')
    ap.add_argument('--out', default=str(OUT), help='lead-first output dir')
    ap.add_argument('--copy', action='store_true',
                    help='copy files instead of hardlinking')
    args = ap.parse_args()

    base, out = Path(args.base), Path(args.out)
    records = sorted(d.name for d in base.iterdir()
                     if d.is_dir() and (d / 'summary.json').exists())
    if not records:
        print(f'no records found under {base}')
        return
    print(f'{len(records)} records from {base}')
    print(f'exporting to {out} ({"copy" if args.copy else "hardlink"})')

    n_links = 0
    lead_agg = {ln: {'records': 0, 'beats': 0,
                     'qrs_pol': Counter(), 'p_pol': Counter(),
                     't_pol': Counter()} for ln in LEADS}

    for i, rec in enumerate(records):
        rec_dir = base / rec
        summary = json.loads((rec_dir / 'summary.json').read_text(encoding='utf-8'))
        for ln in LEADS:
            src_lead = rec_dir / f'lead_{ln}'
            if not src_lead.is_dir():
                continue
            dst_lead = out / f'lead_{ln}' / rec
            dst_lead.mkdir(parents=True, exist_ok=True)
            for f in LEAD_FILES:
                src = src_lead / f
                if src.exists():
                    link_or_copy(src, dst_lead / f, args.copy)
                    n_links += 1
            for f in RECORD_FILES:
                src = rec_dir / f
                if src.exists():
                    link_or_copy(src, dst_lead / f, args.copy)
                    n_links += 1

            agg = lead_agg[ln]
            ld = summary['leads'].get(ln)
            if ld is None:
                continue
            agg['records'] += 1
            agg['beats'] += ld['n_beats']
            for pol, cnt in ld.get('polarity_counts', {}).items():
                agg['qrs_pol'][pol] += cnt
            for pol, cnt in ld.get('p_polarity_counts', {}).items():
                agg['p_pol'][pol] += cnt
            for pol, cnt in ld.get('t_polarity_counts', {}).items():
                agg['t_pol'][pol] += cnt
        if (i + 1) % 100 == 0:
            print(f'  {i + 1}/{len(records)} records...')

    for ln in LEADS:
        agg = lead_agg[ln]
        payload = {
            'lead': ln,
            'n_records': agg['records'],
            'total_beats': agg['beats'],
            'qrs_polarity_totals': dict(agg['qrs_pol']),
            'p_polarity_totals': dict(agg['p_pol']),
            't_polarity_totals': dict(agg['t_pol']),
            'source_cache': str(base),
        }
        p = out / f'lead_{ln}' / '_lead_summary.json'
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                     encoding='utf-8')

    print(f'done: {n_links} files, {len(LEADS)} lead dirs + per-lead summaries')


if __name__ == '__main__':
    main()
