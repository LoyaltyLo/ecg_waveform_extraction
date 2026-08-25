"""Batch QRS polarity v2 — ALL RA-LA records.  Minimal, self-contained."""
import sys
sys.path.insert(0, 'c:/LoyaltyLo/PythonProjects/ECG_engineering')

import os, json, time, gc
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ecg_waveform_extraction.preprocessing import ECGPreprocessor
from ecg_waveform_extraction.features import FeatureExtractor
from ecg_waveform_extraction.hsmm import HSMMModel, smart_initialize_gmms
from ecg_waveform_extraction.segmentation import ECGSegmenter
from ecg_waveform_extraction.extraction.qrs_refiner import (
    refine_qrs_boundaries, compute_qrs_polarity_v2,
)
from ecg_waveform_extraction.utils.aecg_parser import parse_aecg

AECG_DIR = 'C:/LoyaltyLo/datasets/RA-LA_Reversal/aECG'
OUT_DIR = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/output_rala_full/_qrs_polarity_v2_all'
MAX_SAMPLES = 4000
COLORS = {'positive': '#4caf50', 'negative': '#f44336', 'biphasic': '#ff9800', 'uncertain': '#9e9e9e'}

os.makedirs(OUT_DIR, exist_ok=True)
files = sorted([f for f in os.listdir(AECG_DIR) if f.endswith('.aECG')])
n_total = len(files)

print(f"{'='*60}")
print(f"  BATCH QRS POLARITY V2 — {n_total} records")
print(f"{'='*60}")

t_start = time.time()
global_pol = Counter()
ok_count = 0

for idx, fname in enumerate(files):
    fpath = os.path.join(AECG_DIR, fname)
    rec_name = fname.replace('.aECG', '')
    rec_dir = os.path.join(OUT_DIR, rec_name)
    os.makedirs(rec_dir, exist_ok=True)

    aecg = parse_aecg(fpath, max_samples=MAX_SAMPLES)
    fs = aecg['fs']
    leads_result = {}

    for lead_name in ['I', 'II']:
        sig_raw = aecg['signals'].get(lead_name)
        if sig_raw is None:
            leads_result[lead_name] = None
            continue

        sig = sig_raw[:MAX_SAMPLES].astype(np.float64)
        clean = ECGPreprocessor(fs=fs).preprocess(sig)
        features = FeatureExtractor(fs=fs).extract(clean)
        model = HSMMModel(fs=fs)
        model.initialize_with_priors()
        model.set_left_right_topology()
        smart_initialize_gmms(model, features)
        seg_result = ECGSegmenter(model=model, fs=fs).segment(sig)

        beat_results = []
        for b in seg_result.beats:
            if b.q_onset <= 0 or b.r_peak <= 0 or b.s_offset <= 0:
                continue
            q_on, r_pk, s_off = refine_qrs_boundaries(clean, b.q_onset, b.r_peak, b.s_offset, fs)
            pol = compute_qrs_polarity_v2(clean, q_on, r_pk, s_off, fs, lead_name=lead_name)
            beat_results.append({
                'beat_id': b.beat_id,
                'q_onset': int(q_on), 'r_peak': int(r_pk), 's_offset': int(s_off),
                'polarity': pol['polarity'],
                'confidence': pol['confidence'],
                'polarity_score': pol['polarity_score'],
                'energy_ratio': pol['energy_ratio'],
                'peak_count': pol['peak_count'],
                'rs_ratio': pol['rs_ratio'],
            })

        # Save lead JSON
        lead_dir = os.path.join(rec_dir, f'lead_{lead_name}')
        os.makedirs(lead_dir, exist_ok=True)
        with open(os.path.join(lead_dir, 'qrs_polarity.json'), 'w') as f:
            json.dump(beat_results, f, indent=2)

        # Quick overview plot
        T = len(clean)
        plot_sec = min(T / fs, 4.0)
        n_plot = int(plot_sec * fs)
        t_plot = np.arange(n_plot) / fs
        e_plot = clean[:n_plot]
        fig, ax = plt.subplots(figsize=(12, 2.5))
        ax.plot(t_plot, e_plot, 'k-', linewidth=0.3)
        for r in beat_results:
            q, s = r['q_onset'], r['s_offset']
            if q < n_plot and s < n_plot and s > q:
                ax.fill_between(t_plot[q:s + 1], e_plot[q:s + 1],
                               alpha=0.25, color=COLORS.get(r['polarity'], '#9e9e9e'), linewidth=0)
        ax.set_title(f'{rec_name} Lead {lead_name}')
        fig.tight_layout()
        fig.savefig(os.path.join(lead_dir, 'overview.png'), dpi=80, bbox_inches='tight')
        plt.close(fig)

        pc = Counter(r['polarity'] for r in beat_results)
        leads_result[lead_name] = {'n_beats': len(beat_results), 'polarity_counts': dict(pc)}
        for k, v in pc.items():
            global_pol[k] += v

    # Save summary
    with open(os.path.join(rec_dir, 'summary.json'), 'w') as f:
        json.dump({'record': rec_name, 'lead_I': leads_result.get('I'), 'lead_II': leads_result.get('II')}, f, indent=2)

    ok_count += 1
    li = leads_result.get('I') or {}
    lii = leads_result.get('II') or {}
    print(f"[{idx+1:3d}/{n_total}] {rec_name}  I:{li.get('n_beats',0)}b  II:{lii.get('n_beats',0)}b  ({time.time()-t_start:.0f}s)", flush=True)
    gc.collect()

total_time = time.time() - t_start
total_beats = sum(global_pol.values())

summary = {
    'method': 'HSMM + 5-criterion v2 + uncertain gating',
    'n_records': ok_count, 'total_beats': total_beats,
    'overall': dict(global_pol),
    'total_time_sec': round(total_time, 1),
}
with open(os.path.join(OUT_DIR, 'global_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*60}")
print(f"  COMPLETE: {ok_count} records, {total_beats} beats, {total_time:.0f}s")
for pol in ['positive', 'negative', 'biphasic', 'uncertain']:
    cnt = global_pol.get(pol, 0)
    print(f"  {pol:<12}: {cnt:>5} ({cnt/max(total_beats,1)*100:>5.1f}%)")
print(f"{'='*60}")
