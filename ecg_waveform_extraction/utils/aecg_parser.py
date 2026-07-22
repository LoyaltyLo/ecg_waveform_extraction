"""Unified HL7 aECG XML parser for RA-LA Reversal dataset.

Extracts multi-lead signals, global measurements, P/QRS/T annotations,
and interpretation statements from aECG XML files.

Usage:
    from ecg_waveform_extraction.utils.aecg_parser import parse_aecg

    result = parse_aecg('path/to/file.aECG')
    # result['signals']  -> {'I': array, 'II': array, ...}
    # result['measurements'] -> {'HR': 72.0, 'QRS_dur': 88.0, ...}
    # result['annotations']  -> {'P_on_ms': 20.0, 'QRS_off_ms': 120.0, ...}
"""

import os
import re
import numpy as np

LEAD_NAMES = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
LIMB_LEADS = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF']


def parse_aecg(filepath: str, max_samples: int | None = None) -> dict:
    """Parse a single .aECG (HL7 XML) file.

    Parameters
    ----------
    filepath : str
        Path to .aECG file.
    max_samples : int or None
        Truncate signals to this many samples (None = keep all).

    Returns
    -------
    dict with keys:
        filename    : str — basename without extension
        filepath    : str — full path
        fs          : float — sampling rate (Hz)
        n_samples   : int — signal length in samples
        signals     : dict[str, np.ndarray] — lead_name -> 1-D float64 array
        annotations : dict — P/QRS/T onset/offset in ms
        measurements: dict — HR, PR, QRS, QT, P/QRS/T axis, etc.
        interpretation : str — clinical interpretation text
    """
    # ---- Read raw bytes (handle mixed encodings) ----
    with open(filepath, 'rb') as f:
        raw = f.read()

    for enc in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
        try:
            content = raw.decode(enc)
            if '<?xml' in content[:100] or '<' in content[:100]:
                break
        except (UnicodeDecodeError, UnicodeError):
            continue

    result = {
        'filename': os.path.basename(filepath).replace('.aECG', ''),
        'filepath': filepath,
        'fs': _parse_sampling_rate(raw),
        'n_samples': 0,
        'signals': {},
        'annotations': {},
        'measurements': {},
        'interpretation': '',
    }

    fs = result['fs']

    # ---- Extract lead signals ----
    ss = content.find('<sequenceSet')
    se = content.find('</sequenceSet>', ss) if ss >= 0 else -1
    if ss >= 0 and se > ss:
        digits_matches = re.findall(r'<digits[^>]*>([^<]+)</digits>', content[ss:se])
        for i, name in enumerate(LEAD_NAMES):
            if i < len(digits_matches):
                sig = np.array([float(x) for x in digits_matches[i].split()], dtype=np.float64)
                if max_samples is not None:
                    sig = sig[:max_samples]
                result['signals'][name] = sig
        if result['signals']:
            result['n_samples'] = len(next(iter(result['signals'].values())))

    # ---- Waveform annotations (P / QRS / T boundaries, in ms) ----
    for wave, code in [('P', 'PWAVE'), ('QRS', 'QRSWAVE'), ('T', 'TWAVE')]:
        m = re.search(
            rf'MDC_ECG_WAVC_{code}.*?<low value="([^"]+)" unit="ms".*?<high value="([^"]+)" unit="ms"',
            content, re.DOTALL,
        )
        if m:
            result['annotations'][f'{wave}_on_ms'] = float(m.group(1))
            result['annotations'][f'{wave}_off_ms'] = float(m.group(2))

    # ---- Global measurements ----
    _extract_measurements(result, content, raw)

    return result


def _parse_sampling_rate(raw: bytes) -> float:
    """Extract sampling rate from increment element. Falls back to raw bytes regex."""
    m = re.search(rb'<increment[^>]*value="([^"]+)"[^>]*unit="s"', raw)
    if m:
        return 1.0 / float(m.group(1))
    # Fallback: search decoded
    for enc in ['utf-8', 'gbk', 'latin-1']:
        try:
            content = raw.decode(enc)
            m = re.search(r'<increment[^>]*value="([^"]+)"[^>]*unit="s"', content)
            if m:
                return 1.0 / float(m.group(1))
        except Exception:
            continue
    return 1000.0  # default


def _extract_measurements(result: dict, content: str, raw: bytes):
    """Parse all global measurements from the aECG content."""
    # Regex-based measurements (searched in raw bytes for reliability)
    meas_patterns = {
        'HR':        rb'HEART_RATE.*?value="([^"]+)"',
        'QRS_dur':   rb'TIME_PD_QRS\b(?!c).*?value="([^"]+)"',
        'P_dur':     rb'TIME_PD_P\b(?!R).*?value="([^"]+)"',
        'PR_ms':     rb'TIME_PD_PR.*?value="([^"]+)"',
        'QT_ms':     rb'TIME_PD_QT\b(?!c).*?value="([^"]+)"',
        'QTc_ms':    rb'TIME_PD_QTc.*?value="([^"]+)"',
        'P_axis':    rb'ANGLE_P_FRONT.*?value="([^"]+)"',
        'QRS_axis':  rb'ANGLE_QRS_FRONT.*?value="([^"]+)"',
        'T_axis':    rb'ANGLE_T_FRONT.*?value="([^"]+)"',
    }
    for key, pat in meas_patterns.items():
        m = re.search(pat, raw, re.DOTALL)
        if m:
            result['measurements'][key] = float(m.group(1))

    # String-based measurements (searched in decoded content for flexibility)
    str_patterns = {
        'HR_bpm': r'MDC_ECG_HEART_RATE.*?<value[^>]*value="([^"]+)"[^>]*unit="bpm"',
        'PR_ms_str': r'MDC_ECG_TIME_PD_PR.*?<value[^>]*value="([^"]+)"[^>]*unit="ms"',
    }
    for key, pat in str_patterns.items():
        if key.replace('_str', '') not in result['measurements']:
            m = re.search(pat, content, re.DOTALL)
            if m:
                result['measurements'][key.replace('_str', '')] = float(m.group(1))

    # ---- Interpretation ----
    interp = re.search(
        rb'INTERPRETATION_STATEMENT.*?xsi:type="ST"[^>]*>([^<]+)</value>',
        raw, re.DOTALL,
    )
    if interp:
        result['interpretation'] = (
            interp.group(1).decode('utf-8', errors='replace')
            .strip().replace('\n', '; ').replace('\r', '')
        )


def get_default_leads(result: dict, leads: list[str] | None = None) -> dict[str, np.ndarray]:
    """Convenience: extract specific leads from a parsed aECG result.

    Parameters
    ----------
    result : dict
        Output from parse_aecg().
    leads : list[str] or None
        Lead names to extract. None defaults to ['I', 'II'].

    Returns
    -------
    dict[str, np.ndarray] — only leads that exist in the record.
    """
    if leads is None:
        leads = ['I', 'II']
    return {ln: result['signals'][ln] for ln in leads if ln in result['signals']}
