"""5-method polarity detection: P-axis, QRS-axis, P-wave(HSMM), Lead I, aVR.

RA-LA Reversal key signs (ranked by reliability):
  1. Lead I P-wave INVERTED  ← most reliable single-lead sign
  2. aVR P-wave UPRIGHT      ← pathognomonic (normally always negative)
  3. P-axis extreme right    ← machine measurement, can have errors
  4. QRS axis extreme right  ← supportive
  5. Lead II P-wave inverted ← HSMM detected

All 5 methods use ANNOTATION-GUIDED windows when available.
"""

import sys
sys.path.insert(0, 'c:/LoyaltyLo/PythonProjects/ECG_engineering')

import os, json, re
import numpy as np

AECG_DIR = 'C:/LoyaltyLo/datasets/RA-LA_Reversal/aECG'
OUT_DIR = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/output_rala_full'
RESULT_JSON = os.path.join(OUT_DIR, '_polarity_v2.json')


def parse_aecg(filepath):
    """Parse aECG XML: signals + measurements + annotations."""
    for enc in ['utf-8','gbk','latin-1']:
        try:
            with open(filepath,'r',encoding=enc) as f:
                content = f.read()
            if '<?xml' in content[:100]: break
        except: continue

    r = {'fs': 1000}

    inc = re.search(r'<increment[^>]*value="([^"]+)"[^>]*unit="s"', content)
    if inc: r['fs'] = 1.0/float(inc.group(1))

    # Rhythm waveform leads (first sequenceSet)
    ss = content.find('<sequenceSet')
    se = content.find('</sequenceSet>', ss)
    rhythm = content[ss:se]
    lead_order = re.findall(r'MDC_ECG_LEAD_(\w+)', rhythm)[:12]
    digits = re.findall(r'<digits[^>]*>([^<]+)</digits>', rhythm)
    for i, name in enumerate(lead_order):
        if i < len(digits):
            r[name] = np.array([float(x) for x in digits[i].split()], dtype=np.float64)

    # Measurements
    for key, pat in {
        'p_axis':   r'MDC_ECG_ANGLE_P_FRONT.*?<value[^>]*value="([^"]+)"',
        'qrs_axis': r'MDC_ECG_ANGLE_QRS_FRONT.*?<value[^>]*value="([^"]+)"',
        't_axis':   r'MDC_ECG_ANGLE_T_FRONT.*?<value[^>]*value="([^"]+)"',
        'hr':  r'MDC_ECG_HEART_RATE.*?<value[^>]*value="([^"]+)"[^>]*unit="bpm"',
        'pr':  r'MDC_ECG_TIME_PD_PR.*?<value[^>]*value="([^"]+)"[^>]*unit="ms"',
        'qrs': r'MDC_ECG_TIME_PD_QRS\b(?!c).*?<value[^>]*value="([^"]+)"[^>]*unit="ms"',
        'qt':  r'MDC_ECG_TIME_PD_QT\b(?!c).*?<value[^>]*value="([^"]+)"[^>]*unit="ms"',
    }.items():
        m = re.search(pat, content, re.DOTALL)
        r[key] = float(m.group(1)) if m else None

    # Annotations (representative beat P/QRS/T boundaries in ms)
    for key, wave, edge in [
        ('p_on','PWAVE','low'),('p_off','PWAVE','high'),
        ('qrs_on','QRSWAVE','low'),('qrs_off','QRSWAVE','high'),
        ('t_on','TWAVE','low'),('t_off','TWAVE','high'),
    ]:
        m = re.search(rf'MDC_ECG_WAVC_{wave}.*?<{edge} value="([^"]+)" unit="ms"', content, re.DOTALL)
        r[key] = float(m.group(1)) if m else None

    interp = re.search(r'MDC_ECG_INTERPRETATION_STATEMENT.*?xsi:type="ST"[^>]*>([^<]+)</value>', content, re.DOTALL)
    r['interpretation'] = interp.group(1).strip().replace('\n','; ') if interp else ''

    return r


def p_wave_window(raw, lead_name, margin_ms=20):
    """Get precise P-wave sample window using annotations or estimation."""
    fs = raw.get('fs', 1000)
    sig = raw.get(lead_name)
    if sig is None:
        return None, None, None

    p_on = raw.get('p_on')
    p_off = raw.get('p_off')

    if p_on and p_off:
        # Annotation-guided: use exact boundaries + small margin
        start = int((p_on - margin_ms) / 1000 * fs)
        end = int((p_off + margin_ms) / 1000 * fs)
    else:
        # Estimate: find QRS region via max derivative, then look backward
        d1 = np.abs(np.diff(sig))
        w = int(0.05 * fs)
        d1_smooth = np.convolve(d1, np.ones(w)/w, mode='same')
        qrs_peak = np.argmax(d1_smooth)
        qrs_on = max(0, qrs_peak - int(0.08 * fs))
        start = max(0, qrs_on - int(0.25 * fs))
        end = max(0, qrs_on - int(0.02 * fs))

    start = max(0, min(start, len(sig) - 1))
    end = max(start + 5, min(end, len(sig) - 1))

    bl = np.median(sig[max(0, start-50):start]) if start >= 50 else np.median(sig[:end//2])
    region = sig[start:end + 1] - bl
    return start, end, region


def qrs_window(raw, lead_name, margin_ms=30):
    """Get QRS region around max derivative."""
    fs = raw.get('fs', 1000)
    sig = raw.get(lead_name)
    if sig is None:
        return None, None, None

    qrs_on = raw.get('qrs_on')
    qrs_off = raw.get('qrs_off')

    if qrs_on and qrs_off:
        start = int((qrs_on - margin_ms) / 1000 * fs)
        end = int((qrs_off + margin_ms) / 1000 * fs)
    else:
        d1 = np.abs(np.diff(sig))
        w = int(0.05 * fs)
        d1_smooth = np.convolve(d1, np.ones(w)/w, mode='same')
        qrs_peak = np.argmax(d1_smooth)
        start = max(0, qrs_peak - int(0.06 * fs))
        end = min(len(sig) - 1, qrs_peak + int(0.08 * fs))

    start = max(0, min(start, len(sig) - 1))
    end = max(start + 5, min(end, len(sig) - 1))
    bl = np.median(sig[max(0, start-50):start]) if start >= 50 else np.median(sig)
    region = sig[start:end + 1] - bl
    return start, end, region


# =====================================================================
# 5 Methods
# =====================================================================

def method_p_axis(raw):
    """P-wave axis."""
    a = raw.get('p_axis')
    if a is None: return {'polarity': None, 'conf': 0, 'detail': 'N/A'}
    if 0 <= a <= 75:
        return {'polarity': 'normal', 'conf': 0.95, 'detail': f'{a:.0f}° (0-75° normal)'}
    if a > 100 or a < -30:
        return {'polarity': 'reversed', 'conf': 0.90, 'detail': f'{a:.0f}° (>100° or <-30° → reversed)'}
    if 75 < a <= 100:
        return {'polarity': 'borderline', 'conf': 0.40, 'detail': f'{a:.0f}° (75-100° borderline right)'}
    return {'polarity': 'borderline', 'conf': 0.30, 'detail': f'{a:.0f}°'}


def method_qrs_axis(raw):
    """QRS axis."""
    a = raw.get('qrs_axis')
    if a is None: return {'polarity': None, 'conf': 0, 'detail': 'N/A'}
    if -30 <= a <= 90:
        return {'polarity': 'normal', 'conf': 0.90, 'detail': f'{a:.0f}° (-30~+90° normal)'}
    if a > 120 or a < -90:
        return {'polarity': 'reversed', 'conf': 0.85, 'detail': f'{a:.0f}° (>120° extreme right → reversed)'}
    if 90 < a <= 120:
        return {'polarity': 'borderline', 'conf': 0.35, 'detail': f'{a:.0f}° (90-120° right deviation)'}
    if -90 <= a < -30:
        return {'polarity': 'borderline', 'conf': 0.40, 'detail': f'{a:.0f}° (-90~-30° left deviation)'}
    return {'polarity': 'borderline', 'conf': 0.30, 'detail': f'{a:.0f}°'}


def method_p_wave_hsmm(rec_name):
    """HSMM Lead II P-wave polarity."""
    pw_path = os.path.join(OUT_DIR, rec_name, 'p_waves.json')
    ecg_path = os.path.join(OUT_DIR, rec_name, 'filtered_ecg.npy')
    if not os.path.exists(pw_path) or not os.path.exists(ecg_path):
        return {'polarity': None, 'conf': 0, 'detail': 'HSMM output missing'}

    with open(pw_path) as f: pws = json.load(f)
    ecg = np.load(ecg_path)

    areas = []
    for pw in pws:
        o, off = pw['onset_sample'], pw['offset_sample']
        if o < 0 or off <= o: continue
        seg = ecg[o:off+1]
        bl = np.mean(ecg[max(0,o-50):o]) if o >= 50 else np.mean(seg[:10])
        areas.append(np.sum(seg - bl))

    if not areas: return {'polarity': None, 'conf': 0, 'detail': 'No P-waves'}

    mean_a = np.mean(areas)
    n_pos = sum(1 for a in areas if a > 0)
    n_neg = sum(1 for a in areas if a < 0)
    agree = max(n_pos, n_neg) / len(areas)
    pol = 'normal' if mean_a > 0 else 'reversed'
    conf = round(agree, 2) if agree >= 0.7 else 0.5
    return {
        'polarity': pol, 'conf': conf,
        'detail': f'{n_pos}+/{n_neg}- P-waves, net={mean_a:.1f}',
        'n_pos': n_pos, 'n_neg': n_neg, 'n_total': len(areas),
        'mean_area': round(float(mean_a), 2),
    }


def method_lead_I(raw):
    """Lead I: P-wave + QRS polarity.

    The single most reliable sign: Lead I P-wave inverted = RA-LA reversal.
    Uses annotation-guided P-window and QRS-window for precise measurement.
    """
    if 'I' not in raw:
        return {'polarity': None, 'conf': 0, 'detail': 'No Lead I'}

    # P-wave
    p_on, p_off, p_region = p_wave_window(raw, 'I')
    if p_region is not None and len(p_region) > 3:
        p_net = float(np.sum(p_region))
        p_pos = p_net > 0
        p_peak = float(np.max(np.abs(p_region)))
        p_dominant_sign = '+' if np.max(p_region) > abs(np.min(p_region)) else '-'
    else:
        p_net, p_pos, p_peak, p_dominant_sign = 0, None, 0, '?'

    # QRS
    q_on, q_off, q_region = qrs_window(raw, 'I')
    if q_region is not None and len(q_region) > 3:
        qrs_net = float(np.sum(q_region))
        qrs_neg = qrs_net < 0
        qrs_peak = float(np.max(np.abs(q_region)))
    else:
        qrs_net, qrs_neg, qrs_peak = 0, None, 0

    # Decision logic (weighted toward P-wave as it's most specific)
    p_inv = (p_net < 0 and p_peak > 5) if p_region is not None else None
    q_inv = qrs_neg if q_region is not None else None

    if p_inv is True and q_inv is True:
        pol, conf = 'reversed', 0.95
    elif p_inv is True:
        pol, conf = 'reversed', 0.90
    elif p_inv is False and q_inv is False:
        pol, conf = 'normal', 0.90
    elif p_inv is False and q_inv is True:
        pol, conf = 'borderline', 0.50  # QRS inverted but P normal → could be BBB
    elif p_inv is None and q_inv is not None:
        pol, conf = ('reversed' if q_inv else 'normal'), 0.60
    else:
        pol, conf = 'uncertain', 0.30

    return {
        'polarity': pol, 'conf': conf,
        'detail': f'P={p_dominant_sign}(net={p_net:.0f}) QRS={"neg" if qrs_net<0 else "pos"}(net={qrs_net:.0f})',
        'p_inverted': bool(p_inv) if p_inv is not None else None,
        'p_net': round(p_net, 1),
        'qrs_inverted': bool(q_inv) if q_inv is not None else None,
        'qrs_net': round(qrs_net, 1),
    }


def method_avr(raw):
    """aVR lead: P-wave + QRS polarity.

    PATHOGNOMONIC for RA-LA reversal:
    - Normal: aVR P-wave is ALWAYS negative (downward)
    - Reversed: aVR P-wave becomes POSITIVE (upward) ← almost never happens otherwise

    Uses exact annotation-guided P-wave window. QRS becoming positive
    is also highly specific for reversal.
    """
    if 'AVR' not in raw:
        return {'polarity': None, 'conf': 0, 'detail': 'No aVR'}

    # P-wave with annotation-guided window
    p_on, p_off, p_region = p_wave_window(raw, 'AVR', margin_ms=10)
    if p_region is not None and len(p_region) > 3:
        p_net = float(np.sum(p_region))
        p_pos = p_net > 0
        p_peak_abs = float(np.max(np.abs(p_region)))
        # Check if dominant peak is positive
        p_peak_pos = float(np.max(p_region))
        p_peak_neg = float(np.min(p_region))
        p_dominant_up = p_peak_pos > abs(p_peak_neg)
    else:
        p_net, p_pos, p_peak_abs, p_dominant_up = 0, None, 0, None

    # QRS window
    q_on, q_off, q_region = qrs_window(raw, 'AVR')
    if q_region is not None and len(q_region) > 3:
        qrs_net = float(np.sum(q_region))
        qrs_pos = qrs_net > 0
        qrs_peak = float(np.max(np.abs(q_region)))
    else:
        qrs_net, qrs_pos, qrs_peak = 0, None, 0

    # Decision: aVR P-wave positive = PATHOGNOMONIC for RA-LA reversal
    if p_pos is True and p_peak_abs > 3:
        pol, conf = 'reversed', 0.95
        detail = f'aVR P↑(net={p_net:.0f}) => PATHOGNOMONIC reversal'
    elif p_pos is True:
        pol, conf = 'reversed', 0.80
        detail = f'aVR P slightly↑(net={p_net:.0f}) => likely reversed'
    elif qrs_pos is True and qrs_peak > 5:
        pol, conf = 'likely_reversed', 0.70
        detail = f'aVR P↓ QRS↑(net={qrs_net:.0f}) => possible reversed'
    elif p_pos is False and qrs_pos is False:
        pol, conf = 'normal', 0.90
        detail = f'aVR P↓(net={p_net:.0f}) QRS↓(net={qrs_net:.0f}) => normal'
    elif p_peak_abs < 3:
        pol, conf = 'uncertain', 0.30
        detail = 'aVR low amplitude, uncertain'
    else:
        pol, conf = 'borderline', 0.50
        detail = f'aVR P_net={p_net:.0f} QRS_net={qrs_net:.0f}'

    return {
        'polarity': pol, 'conf': conf, 'detail': detail,
        'p_positive': bool(p_pos) if p_pos is not None else None,
        'qrs_positive': bool(qrs_pos) if qrs_pos is not None else None,
        'p_net': round(p_net, 1),
        'qrs_net': round(qrs_net, 1),
    }


# =====================================================================
def detect_one(rec_name):
    aecg_path = os.path.join(AECG_DIR, rec_name + '.aECG')
    r = {
        'record': rec_name, 'methods': {},
        'votes': {'normal': 0., 'reversed': 0.},
        'consensus': 'uncertain', 'confidence': 0.,
        'hr': None, 'pr': None, 'qrs': None, 'qt': None,
        'p_axis': None, 'qrs_axis': None, 'interpretation': '',
    }

    if not os.path.exists(aecg_path):
        r['consensus'] = 'file_missing'
        return r

    raw = parse_aecg(aecg_path)
    for k in ['hr','pr','qrs','qt','p_axis','qrs_axis','interpretation']:
        r[k] = raw.get(k)

    # Run all 5 methods
    methods = {
        'p_axis':       method_p_axis(raw),
        'qrs_axis':     method_qrs_axis(raw),
        'p_wave_hsmm':  method_p_wave_hsmm(rec_name),
        'lead_I':       method_lead_I(raw),
        'avr_lead':     method_avr(raw),
    }
    r['methods'] = methods

    # Weighted voting: Lead I P-wave and aVR P-wave have highest weight
    weights = {
        'p_axis':       0.9,
        'qrs_axis':     0.7,
        'p_wave_hsmm':  1.0,
        'lead_I':       1.3,   # Lead I P-wave inversion = most reliable
        'avr_lead':     1.3,   # aVR P-wave upright = pathognomonic
    }

    for mname, w in weights.items():
        m = methods.get(mname, {})
        pol = m.get('polarity')
        conf = m.get('conf', 0)
        if pol == 'normal':
            r['votes']['normal'] += w * conf
        elif pol == 'reversed':
            r['votes']['reversed'] += w * conf
        elif pol == 'likely_reversed':
            r['votes']['reversed'] += w * conf * 0.7
        elif pol == 'borderline':
            r['votes']['reversed'] += w * conf * 0.5
            r['votes']['normal'] += w * conf * 0.5

    vn, vr = r['votes']['normal'], r['votes']['reversed']
    total = vn + vr
    if total > 0.5:
        r['consensus'] = 'reversed' if vr > vn else 'normal'
        r['confidence'] = round(max(vn, vr) / total, 2)
    else:
        r['consensus'] = 'uncertain'
        r['confidence'] = 0.0

    return r


# =====================================================================
if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        for rec in ['1805185J6U', '180518ZG06', '180519IS5Q']:
            r = detect_one(rec)
            print(f"\n{'='*65}")
            print(f"  {rec}  |  Consensus: {r['consensus']} (conf={r['confidence']})")
            print(f"  P-axis={r['p_axis']}°  QRS-axis={r['qrs_axis']}°")
            print(f"  Votes: N={r['votes']['normal']:.2f}  R={r['votes']['reversed']:.2f}")
            for mname, m in r['methods'].items():
                print(f"    [{mname:15s}] {m.get('polarity','?'):20s} conf={m.get('conf',0):.2f} | {m.get('detail','')}")
    else:
        records = sorted([d for d in os.listdir(OUT_DIR)
                         if os.path.isdir(os.path.join(OUT_DIR,d)) and d[:1].isdigit()])
        print(f"5-method detection on {len(records)} records")
        print(f"Weighted: Lead_I×1.3, aVR×1.3, P_axis×0.9, QRS_axis×0.7, P_wave×1.0\n")

        all_r = []
        for i, rec in enumerate(records):
            r = detect_one(rec)
            all_r.append(r)
            if (i+1) % 100 == 0:
                nn = sum(1 for x in all_r if x['consensus']=='normal')
                nr = sum(1 for x in all_r if x['consensus']=='reversed')
                nu = sum(1 for x in all_r if x['consensus']=='uncertain')
                print(f"  [{i+1}/{len(records)}] N={nn} R={nr} U={nu}")

        class NpEnc(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, (np.integer,)): return int(o)
                if isinstance(o, (np.floating,)): return float(o)
                if isinstance(o, np.ndarray): return o.tolist()
                if isinstance(o, (np.bool_,bool)): return bool(o)
                return super().default(o)

        with open(RESULT_JSON, 'w') as f:
            json.dump(all_r, f, indent=2, cls=NpEnc)

        nn = sum(1 for x in all_r if x['consensus']=='normal')
        nr = sum(1 for x in all_r if x['consensus']=='reversed')
        nu = sum(1 for x in all_r if x['consensus']=='uncertain')
        t = len(all_r)
        print(f"\n{'='*55}")
        print(f"  5-METHOD RESULT (with QRS axis + aVR)")
        print(f"{'='*55}")
        print(f"  Normal:    {nn} ({nn/t*100:.1f}%)")
        print(f"  Reversed:  {nr} ({nr/t*100:.1f}%)")
        print(f"  Uncertain: {nu} ({nu/t*100:.1f}%)")
        print(f"  Saved: {RESULT_JSON}")
        print(f"{'='*55}")
