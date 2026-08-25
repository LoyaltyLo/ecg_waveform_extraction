#!/usr/bin/env python3
"""Simple RA-LA Reversal Detection — QRS C1 Only.

Rule:
  1. Lead I QRS C1: if positive > negative → NORMAL
  2. Otherwise: check aVR QRS C1
     - aVR negative → NORMAL
     - aVR positive → RA-LA REVERSAL

Usage:
    python -m ecg_waveform_extraction.src.batch_ra_la_simple --n 50
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os, json, time, gc, argparse
from collections import Counter
import numpy as np

from ecg_waveform_extraction.src.limb_lead_processor import LimbLeadProcessor
from ecg_waveform_extraction.src.utils.aecg_parser import parse_aecg
from ecg_waveform_extraction.src.export_reversal_xlsx import get_patient_name

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

# ---------------------------------------------------------------------------
AECG_DIR = 'C:/LoyaltyLo/datasets/RA-LA_Reversal/aECG'
OUT_DIR = str(Path(__file__).resolve().parent.parent / 'output_rala_full/_ra_la_simple')
MAX_SAMPLES = 4000

HEADER_FONT = Font(name='Consolas', size=10, bold=True, color='FFFFFF')
HEADER_FILL = PatternFill(start_color='2F5496', end_color='2F5496', fill_type='solid')
HEADER_ALIGN = Alignment(horizontal='center', vertical='center', wrap_text=True)
DATA_FONT = Font(name='Consolas', size=9)
DATA_ALIGN = Alignment(horizontal='center', vertical='center')
TEXT_ALIGN = Alignment(horizontal='left', vertical='center')
FILL_REV = PatternFill(start_color='FFCDD2', end_color='FFCDD2', fill_type='solid')
FILL_NORM = PatternFill(start_color='C8E6C9', end_color='C8E6C9', fill_type='solid')
FILL_UNC = PatternFill(start_color='FFF9C4', end_color='FFF9C4', fill_type='solid')
THIN_BORDER = Border(
    left=Side(style='thin'), right=Side(style='thin'),
    top=Side(style='thin'), bottom=Side(style='thin'),
)

# ---------------------------------------------------------------------------
def process_record(fpath, processor):
    """Run HSMM, apply simple QRS C1 reversal rule."""
    rec_name = os.path.basename(fpath).replace('.aECG', '')
    pname = get_patient_name(fpath)

    aecg = parse_aecg(fpath, max_samples=MAX_SAMPLES)
    ll_result, seg_data = processor.process_record(aecg, record_name=rec_name)

    # Lead I QRS C1
    lr_I = ll_result.leads.get('I')
    if lr_I is None or lr_I.n_beats == 0:
        return None

    i_pos = lr_I.polarity_counts.get('positive', 0)
    i_neg = lr_I.polarity_counts.get('negative', 0)

    # aVR QRS C1
    lr_avr = ll_result.leads.get('AVR')
    if lr_avr is None or lr_avr.n_beats == 0:
        return None

    avr_pos = lr_avr.polarity_counts.get('positive', 0)
    avr_neg = lr_avr.polarity_counts.get('negative', 0)

    # ---- Rule ----
    if i_pos > i_neg:
        verdict = 'normal'
        reason = f'Lead I QRS +{i_pos} > -{i_neg}'
    else:
        # Lead I: negative >= positive
        if avr_neg > avr_pos:
            verdict = 'normal'
            reason = f'Lead I -{i_neg} >= +{i_pos}, aVR -{avr_neg} > +{avr_pos} → normal'
        elif avr_pos > avr_neg:
            verdict = 'reversed'
            reason = f'Lead I -{i_neg} >= +{i_pos}, aVR +{avr_pos} > -{avr_neg} → RA-LA'
        else:
            verdict = 'uncertain'
            reason = f'Lead I -{i_neg} vs +{i_pos}, aVR +{avr_pos} vs -{avr_neg} → tie'

    meas = ll_result.measurements
    return {
        'record': rec_name,
        'patient_name': pname,
        'verdict': verdict,
        'reason': reason,
        'I_pos': i_pos, 'I_neg': i_neg,
        'avr_pos': avr_pos, 'avr_neg': avr_neg,
        'I_total': lr_I.n_beats,
        'avr_total': lr_avr.n_beats,
        'I_mean_conf': round(float(np.mean([b['confidence'] for b in lr_I.beats
                                            if b['polarity'] in ('positive','negative')])), 3),
        'avr_mean_conf': round(float(np.mean([b['confidence'] for b in lr_avr.beats
                                              if b['polarity'] in ('positive','negative')])), 3),
        'P_axis': meas.get('P_axis', ''),
        'QRS_axis': meas.get('QRS_axis', ''),
        'HR': meas.get('HR', ''),
        'interpretation': ll_result.interpretation or '',
    }

# ---------------------------------------------------------------------------
def export_xlsx(results, path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'RA-LA Simple'

    headers = [
        'Record', 'Patient Name', 'Verdict', 'Reason',
        'I +', 'I -', 'I Beats', 'I C1 Conf',
        'aVR +', 'aVR -', 'aVR Beats', 'aVR C1 Conf',
        'P_axis', 'QRS_axis', 'HR', 'Interpretation',
    ]
    for col, h in enumerate(headers, 1):
        c = ws.cell(row=1, column=col, value=h)
        c.font = HEADER_FONT; c.fill = HEADER_FILL
        c.alignment = HEADER_ALIGN; c.border = THIN_BORDER

    for i, r in enumerate(results):
        row = i + 2
        vals = [
            r['record'], r['patient_name'], r['verdict'], r['reason'],
            r['I_pos'], r['I_neg'], r['I_total'], r['I_mean_conf'],
            r['avr_pos'], r['avr_neg'], r['avr_total'], r['avr_mean_conf'],
            r['P_axis'], r['QRS_axis'], r['HR'],
            (r['interpretation'] or '')[:80],
        ]
        for col, v in enumerate(vals, 1):
            c = ws.cell(row=row, column=col, value=v)
            c.font = DATA_FONT
            c.alignment = DATA_ALIGN if not isinstance(v, str) else TEXT_ALIGN
            c.border = THIN_BORDER

        vc = ws.cell(row=row, column=3)
        if r['verdict'] == 'reversed':
            vc.fill = FILL_REV
            vc.font = Font(name='Consolas', size=9, bold=True, color='B71C1C')
        elif r['verdict'] == 'normal':
            vc.fill = FILL_NORM
        else:
            vc.fill = FILL_UNC

    widths = [14, 10, 10, 50, 5, 5, 7, 8, 5, 5, 7, 8, 8, 9, 6, 40]
    for col, w in enumerate(widths, 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(col)].width = w
    ws.freeze_panes = 'A2'
    ws.auto_filter.ref = f'A1:{openpyxl.utils.get_column_letter(len(headers))}{len(results)+1}'

    wb.save(path)
    print(f'Saved: {path}')

# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=None)
    args = parser.parse_args()

    files = sorted([f for f in os.listdir(AECG_DIR) if f.endswith('.aECG')])
    if args.n:
        files = files[:args.n]
    os.makedirs(OUT_DIR, exist_ok=True)

    processor = LimbLeadProcessor(max_samples=MAX_SAMPLES)
    results = []

    print(f'RA-LA Simple Rule: {len(files)} records')
    t0 = time.time()

    for idx, fname in enumerate(files):
        fpath = os.path.join(AECG_DIR, fname)
        rec = fname.replace('.aECG', '')
        print(f'[{idx+1:3d}/{len(files)}] {rec}...', end=' ', flush=True)
        t1 = time.time()

        try:
            r = process_record(fpath, processor)
            if r:
                results.append(r)
                print(f'{r["verdict"]:<10} I:+{r["I_pos"]}/-{r["I_neg"]} '
                      f'aVR:+{r["avr_pos"]}/-{r["avr_neg"]} '
                      f'({time.time()-t1:.0f}s)')
            else:
                print('SKIP')
        except Exception as e:
            print(f'ERROR: {e}')
        gc.collect()

    dt = time.time() - t0
    print(f'\nCompleted {len(results)} records in {dt:.0f}s ({dt/len(results):.1f}s/rec)')

    # XLSX
    xlsx_path = os.path.join(OUT_DIR, 'ra_la_simple.xlsx')
    export_xlsx(results, xlsx_path)

    # Summary
    vc = Counter(r['verdict'] for r in results)
    print(f'\nResults:')
    for k in ['normal', 'reversed', 'uncertain']:
        cnt = vc.get(k, 0)
        print(f'  {k:<12}: {cnt:>4} ({cnt/len(results)*100:>5.1f}%)')

    # Gold standard check: I neg + aVR pos pairs
    i_neg_avr_pos = [r for r in results if r['I_neg'] >= r['I_pos'] and r['avr_pos'] > r['avr_neg']]
    print(f'  I neg + aVR pos : {len(i_neg_avr_pos)}')


if __name__ == '__main__':
    main()
