"""Re-run P-wave test and save all results to JSON."""
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
OUT_DIR = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/output_rala_full'

files = sorted([f for f in os.listdir(AECG_DIR) if f.endswith('.aECG')])[:50]
results = []

for i, fname in enumerate(files):
    with open(os.path.join(AECG_DIR, fname), 'rb') as f:
        raw = f.read()
    content = raw.decode('utf-8', errors='replace')
    fs = 1000.0
    m = re.search(rb'<increment[^>]*value="([^"]+)"[^>]*unit="s"', raw)
    if m: fs = 1.0 / float(m.group(1))
    ss = content.find('<sequenceSet')
    se = content.find('</sequenceSet>', ss)
    digits = re.findall(r'<digits[^>]*>([^<]+)</digits>', content[ss:se])
    sig = np.array([float(x) for x in digits[1].split()], dtype=np.float64) if len(digits) >= 2 else None
    if sig is None: continue
    n = min(len(sig), 4000); sig = sig[:n].astype(np.float64)

    t0 = time.time()
    clean = ECGPreprocessor(fs=fs).preprocess(sig)
    feats = FeatureExtractor(fs=fs).extract(clean)
    model = HSMMModel(fs=fs); model.initialize_with_priors(); model.set_left_right_topology()
    smart_initialize_gmms(model, feats)
    seg = ECGSegmenter(preprocessor=ECGPreprocessor(fs=fs), feature_extractor=FeatureExtractor(fs=fs),
                       model=model, fs=fs)
    seg_result = seg.segment(sig)

    p_ext = PWaveExtractor(fs=fs, refine_boundaries=True, enable_template_fallback=True)
    p_waves = p_ext.extract(seg_result)
    dt = time.time() - t0

    beats = [{'beat_id': pw.beat_id, 'onset': pw.onset_sample, 'offset': pw.offset_sample,
              'dur_ms': pw.duration_ms, 'conf': pw.confidence, 'snr': pw.snr_db,
              'sym': pw.symmetry, 'cons': pw.consistency, 'morph': pw.morphology,
              'absent': pw.absence_type} for pw in p_waves]

    analyzer = PWaveAnalyzer(fs=fs)
    p_feats = analyzer.analyze(p_waves, clean, seg_result.beats)
    p_summary = analyzer.summarize(p_feats)

    results.append({'record': fname.replace('.aECG',''), 'n_beats': p_summary.n_beats,
        'n_absent': p_summary.n_absent, 'n_total': p_summary.n_total,
        'P_dur_ms': p_summary.duration_mean_ms, 'SNR_dB': p_summary.mean_snr_db,
        'Symmetry': p_summary.mean_symmetry, 'consistency': p_summary.mean_consistency,
        'morphology': dict(p_summary.morphology_distribution),
        'quality': dict(p_summary.quality_distribution), 'flagged': len(p_summary.flagged_beats),
        'time_sec': round(dt,1), 'beats': beats})
    print(f"[{i+1:2d}/50] {fname[:12]}: {p_summary.n_beats}valid+{p_summary.n_absent}abs SNR={p_summary.mean_snr_db:.1f}dB ({dt:.0f}s)", flush=True)
    gc.collect()

class NpEnc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.bool_, bool)): return bool(o)
        return super().default(o)

out_path = os.path.join(OUT_DIR, 'optimized_p_wave_test_50.json')
full_data = {'test_config': {'dataset': 'RA-LA Reversal aECG', 'n_records': 50, 'max_samples': 4000},
             'aggregate': {
                 'total_beats': sum(r['n_total'] for r in results),
                 'valid_p_waves': sum(r['n_beats'] for r in results),
                 'absent_p_waves': sum(r['n_absent'] for r in results)},
             'per_record': results}
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(full_data, f, indent=2, ensure_ascii=False, cls=NpEnc)
print(f"\nSaved: {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")
