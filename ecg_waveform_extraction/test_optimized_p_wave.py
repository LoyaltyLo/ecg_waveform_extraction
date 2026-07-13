"""Test optimized P-wave extraction on first 50 RA-LA aECG records.

Reports new metrics: SNR, symmetry, consistency, morphology, quality flags.
Compares with aECG annotations where available.
"""

import sys
sys.path.insert(0, 'c:/LoyaltyLo/PythonProjects/ECG_engineering')

import os, json, re, time, gc
from collections import Counter
import numpy as np

from ecg_waveform_extraction.preprocessing import ECGPreprocessor
from ecg_waveform_extraction.features import FeatureExtractor
from ecg_waveform_extraction.hsmm import HSMMModel, smart_initialize_gmms
from ecg_waveform_extraction.segmentation import ECGSegmenter
from ecg_waveform_extraction.extraction import PWaveExtractor, PWaveAnalyzer

AECG_DIR = 'C:/LoyaltyLo/datasets/RA-LA_Reversal/aECG'
N_RECORDS = 50
MAX_SAMPLES = 4000  # 4s @ 1000Hz

# =====================================================================
def parse_aecg_xml(filepath):
    """Quick XML parse: get Lead II signal, fs, annotations, measurements."""
    for enc in ['utf-8', 'gbk', 'latin-1']:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                content = f.read()
            if '<?xml' in content[:100]:
                break
        except:
            continue

    r = {'fs': 1000.0, 'interpretation': ''}

    inc = re.search(r'<increment[^>]*value="([^"]+)"[^>]*unit="s"', content)
    if inc: r['fs'] = 1.0 / float(inc.group(1))

    # Lead II signal (2nd digits block in rhythm sequenceSet)
    ss = content.find('<sequenceSet')
    se = content.find('</sequenceSet>', ss)
    rhythm = content[ss:se]
    digits = re.findall(r'<digits[^>]*>([^<]+)</digits>', rhythm)
    if len(digits) >= 2:
        r['lead_II'] = np.array([float(x) for x in digits[1].split()], dtype=np.float64)

    # Annotations
    for key, wave, edge in [
        ('P_on_ms','PWAVE','low'),('P_off_ms','PWAVE','high'),
        ('QRS_on_ms','QRSWAVE','low'),('QRS_off_ms','QRSWAVE','high'),
        ('T_on_ms','TWAVE','low'),('T_off_ms','TWAVE','high'),
    ]:
        m = re.search(rf'MDC_ECG_WAVC_{wave}.*?<{edge} value="([^"]+)" unit="ms"', content, re.DOTALL)
        r[key] = float(m.group(1)) if m else None

    # Measurements
    for key, pat in {
        'HR': r'MDC_ECG_HEART_RATE.*?<value[^>]*value="([^"]+)"[^>]*unit="bpm"',
        'PR': r'MDC_ECG_TIME_PD_PR.*?<value[^>]*value="([^"]+)"[^>]*unit="ms"',
        'QRS_dur': r'MDC_ECG_TIME_PD_QRS\b(?!c).*?<value[^>]*value="([^"]+)"[^>]*unit="ms"',
        'QT': r'MDC_ECG_TIME_PD_QT\b(?!c).*?<value[^>]*value="([^"]+)"[^>]*unit="ms"',
        'P_axis': r'MDC_ECG_ANGLE_P_FRONT.*?<value[^>]*value="([^"]+)"',
    }.items():
        m = re.search(pat, content, re.DOTALL)
        r[key] = float(m.group(1)) if m else None

    interp = re.search(r'MDC_ECG_INTERPRETATION_STATEMENT.*?xsi:type="ST"[^>]*>([^<]+)</value>', content, re.DOTALL)
    if interp: r['interpretation'] = interp.group(1).strip().replace('\n','; ')

    return r

# =====================================================================
print(f"{'='*75}")
print(f"  OPTIMIZED P-WAVE EXTRACTION — RA-LA aECG (First {N_RECORDS} records)")
print(f"{'='*75}")
print(f"  New metrics: SNR, Symmetry, Consistency, Morphology, Quality Flag")
print()

files = sorted([f for f in os.listdir(AECG_DIR) if f.endswith('.aECG')])[:N_RECORDS]

all_results = []
total_time = 0
ok = 0
fail = 0

for idx, fname in enumerate(files):
    fpath = os.path.join(AECG_DIR, fname)
    rec_name = fname.replace('.aECG', '')

    try:
        aecg = parse_aecg_xml(fpath)
        fs = aecg['fs']
        sig = aecg.get('lead_II')
        if sig is None:
            print(f"[{idx+1:2d}/{N_RECORDS}] {rec_name} SKIP (no Lead II)")
            fail += 1
            continue

        n = min(len(sig), MAX_SAMPLES)
        sig = sig[:n].astype(np.float64)

        t0 = time.time()
        prep = ECGPreprocessor(fs=fs)
        clean = prep.preprocess(sig)
        fe = FeatureExtractor(fs=fs)
        features = fe.extract(clean)

        model = HSMMModel(fs=fs)
        model.initialize_with_priors()
        model.set_left_right_topology()
        smart_initialize_gmms(model, features)

        seg = ECGSegmenter(preprocessor=prep, feature_extractor=fe, model=model, fs=fs)
        seg_result = seg.segment(sig)

        # Optimized P-wave extraction
        hr = aecg.get('HR', None)
        p_ext = PWaveExtractor(fs=fs, refine_boundaries=True, enable_template_fallback=True)
        p_waves = p_ext.extract(seg_result, heart_rate=hr)

        analyzer = PWaveAnalyzer(fs=fs)
        p_feats = analyzer.analyze(p_waves, clean, seg_result.beats)
        p_summary = analyzer.summarize(p_feats)
        dt = time.time() - t0
        total_time += dt

        # Per-beat metrics
        beat_metrics = []
        for pw in p_waves:
            bm = {
                'beat_id': pw.beat_id,
                'onset': pw.onset_sample, 'offset': pw.offset_sample,
                'dur_ms': pw.duration_ms,
                'conf': pw.confidence, 'snr': pw.snr_db,
                'sym': pw.symmetry, 'cons': pw.consistency,
                'morph': pw.morphology, 'absent': pw.absence_type,
            }
            if pw.onset_sample > 0 and pw.peak_sample > 0:
                bm['peak'] = pw.peak_sample
            beat_metrics.append(bm)

        # Compare with annotation
        ann_p_dur = None
        if aecg.get('P_on_ms') and aecg.get('P_off_ms'):
            ann_p_dur = aecg['P_off_ms'] - aecg['P_on_ms']

        # Select the best HSMM beat by confidence
        best_pw = max(p_waves, key=lambda p: p.confidence) if p_waves else None

        rec_result = {
            'record': rec_name,
            'n_beats': p_summary.n_beats,
            'n_absent': p_summary.n_absent,
            'n_total': seg_result.n_beats if hasattr(seg_result, 'n_beats') else len(seg_result.beats),
            'ann_P_dur_ms': ann_p_dur,
            'ann_PR_ms': aecg.get('PR'),
            'ann_HR': aecg.get('HR'),
            'ann_P_axis': aecg.get('P_axis'),
            'hsmm_P_dur': p_summary.duration_mean_ms,
            'hsmm_P_std': p_summary.duration_std_ms,
            'hsmm_PR': p_summary.pr_mean_ms,
            'hsmm_SNR': p_summary.mean_snr_db,
            'hsmm_Symmetry': p_summary.mean_symmetry,
            'hsmm_Consistency': p_summary.mean_consistency,
            'morphology': p_summary.morphology_distribution,
            'quality': p_summary.quality_distribution,
            'flagged': len(p_summary.flagged_beats),
            'best_beat_conf': round(best_pw.confidence, 3) if best_pw else None,
            'best_beat_morph': best_pw.morphology if best_pw else None,
            'interpretation': aecg.get('interpretation', '')[:80],
            'time_sec': round(dt, 1),
            'beat_details': beat_metrics,
        }
        all_results.append(rec_result)
        ok += 1

        # Print summary line
        morph_str = ' '.join(f'{k}:{v}' for k, v in sorted(p_summary.morphology_distribution.items()))
        qual_str = ' '.join(f'{k}:{v}' for k, v in sorted(p_summary.quality_distribution.items()))
        interp = aecg.get('interpretation', '')[:50].split('\n')[0]

        print(f"[{idx+1:2d}] {rec_name}: {p_summary.n_beats}valid+{p_summary.n_absent}absent "
              f"SNR={p_summary.mean_snr_db:.1f}dB Sym={p_summary.mean_symmetry:.3f} "
              f"Cons={p_summary.mean_consistency:.3f} "
              f"Morph=[{morph_str}] Q=[{qual_str}] "
              f"({dt:.0f}s)")

    except Exception as e:
        import traceback
        traceback.print_exc()
        fail += 1
        print(f"[{idx+1:2d}] {rec_name} FAIL: {e}")

    gc.collect()

# =====================================================================
# Aggregate report
# =====================================================================
print(f"\n{'='*75}")
print(f"  AGGREGATE RESULTS")
print(f"{'='*75}")
print(f"  Processed: {ok} OK, {fail} failed")
print(f"  Total time: {total_time:.0f}s ({total_time/ok:.1f}s avg)")

# Numeric aggregates
valid_recs = [r for r in all_results if r['n_beats'] > 0]
n_valid = len(valid_recs)

snrs = [r['hsmm_SNR'] for r in valid_recs if r['hsmm_SNR'] > 0]
syms = [r['hsmm_Symmetry'] for r in valid_recs if r['hsmm_Symmetry'] > 0]
conss = [r['hsmm_Consistency'] for r in valid_recs if r['hsmm_Consistency'] > 0]
p_durs = [r['hsmm_P_dur'] for r in valid_recs if r['hsmm_P_dur']]
p_disps = [r['hsmm_P_std'] for r in valid_recs if r['hsmm_P_std']]

# Comparison with annotations
ann_dur_match = [(r['ann_P_dur_ms'], r['hsmm_P_dur']) for r in valid_recs
                 if r['ann_P_dur_ms'] and r['hsmm_P_dur']]
if ann_dur_match:
    dur_errors = [abs(a - h) for a, h in ann_dur_match]
else:
    dur_errors = []

print()
print(f"  {'Metric':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print(f"  {'-'*57}")
for name, vals in [
    ('SNR (dB)', snrs),
    ('Symmetry (0-1)', syms),
    ('Consistency (0-1)', conss),
    ('P-wave Duration (ms)', p_durs),
    ('P-wave Dispersion (ms)', p_disps),
]:
    if vals:
        print(f"  {name:<25} {np.mean(vals):>8.1f} {np.std(vals):>8.1f} {np.min(vals):>8.1f} {np.max(vals):>8.1f}")

if dur_errors:
    print(f"  {'P-dur vs ANN MAE (ms)':<25} {np.mean(dur_errors):>8.1f}")
    print(f"  P-dur errors <= 10ms: {sum(1 for e in dur_errors if e <= 10)}/{len(dur_errors)} ({sum(1 for e in dur_errors if e <= 10)/len(dur_errors)*100:.0f}%)")

# Morphology distribution
all_morph = Counter()
all_qual = Counter()
all_absent = 0
for r in all_results:
    for m, c in r.get('morphology', {}).items():
        all_morph[m] += c
    for q, c in r.get('quality', {}).items():
        all_qual[q] += c
    all_absent += r.get('n_absent', 0)

print()
print(f"  Morphology Distribution (across all beats):")
for m, c in all_morph.most_common():
    print(f"    {m:<20s}: {c:>5d} ({c/sum(all_morph.values())*100:.1f}%)")

print(f"\n  Quality Distribution:")
for q, c in all_qual.most_common():
    print(f"    {q:<20s}: {c:>5d} ({c/sum(all_qual.values())*100:.1f}%)")

print(f"\n  Total beats: {sum(r['n_beats'] for r in all_results)}")
print(f"  Total absent P-waves: {all_absent}")
print(f"  Total flagged beats: {sum(r['flagged'] for r in all_results)}")

# Top/bottom records by confidence
print(f"\n  Top 5 records (best P-wave confidence):")
sorted_by_conf = sorted(all_results, key=lambda r: r['best_beat_conf'] or 0, reverse=True)
for r in sorted_by_conf[:5]:
    print(f"    {r['record']}: conf={r['best_beat_conf']:.3f} {r['best_beat_morph']} SNR={r['hsmm_SNR']:.1f}dB")

print(f"\n  Bottom 5 records (worst P-wave confidence):")
for r in sorted_by_conf[-5:]:
    print(f"    {r['record']}: conf={r['best_beat_conf']:.3f} {r['best_beat_morph']} SNR={r['hsmm_SNR']:.1f}dB")

print(f"{'='*75}")
