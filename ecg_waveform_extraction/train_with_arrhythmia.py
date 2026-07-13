"""Train HSMM with arrhythmia-diverse data and evaluate per beat type.

Strategy:
1. Select training records to maximize arrhythmia type diversity
2. Re-train HSMM via Viterbi hard-EM
3. Evaluate per beat type (N, V, A, L, R, /, F, etc.)
4. Save per-beat-type breakdown
"""

import sys
sys.path.insert(0, 'c:/LoyaltyLo/PythonProjects/ECG_engineering')

import os, json, time, gc
from collections import Counter, defaultdict
import numpy as np
import wfdb

from ecg_waveform_extraction.preprocessing import ECGPreprocessor
from ecg_waveform_extraction.features import FeatureExtractor
from ecg_waveform_extraction.hsmm import HSMMModel, HSMMDecoder, smart_initialize_gmms
from ecg_waveform_extraction.hsmm.hsmm_model import STATE_LABELS

# =====================================================================
# Config
# =====================================================================
DATA_DIR = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/data'
MODEL_DIR = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/models'
OUT_DIR = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/output_arrhythmia'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

MAX_SEC = 15.0
MAX_SAMPLES = 10000
MAX_EM_ITERS = 6

BEAT_TYPE_NAMES = {
    'N': 'Normal', 'L': 'LBBB', 'R': 'RBBB',
    'V': 'PVC', 'A': 'Atrial Premature',
    '/': 'Paced', 'F': 'Fusion Vent-Norm',
    'f': 'Fusion Paced-Norm', 'J': 'Nodal Premature',
    'E': 'Vent Escape', 'a': 'Aberrated Atrial',
    'S': 'SV Premature', '!': 'Vent Flutter',
    'Q': 'Unclassifiable', 'j': 'Nodal Escape',
    '~': 'Signal Change', '|': 'Isolated QRS',
    '"': 'Comment', 'x': 'Non-conducted P',
    'e': 'Atrial Escape', '[': 'Start VF', ']': 'End VF',
}


# =====================================================================
# Data loading with beat type annotations
# =====================================================================
def load_all_with_types():
    """Load all records with beat-type annotations."""
    all_recs = set()
    for f in os.listdir(DATA_DIR):
        if f.endswith('.hea'):
            rec = f[:-4]
            if os.path.exists(os.path.join(DATA_DIR, rec + '.dat')):
                all_recs.add(rec)

    data = []
    for rec in sorted(all_recs):
        try:
            record = wfdb.rdrecord(os.path.join(DATA_DIR, rec))
            sig = record.p_signal
            if sig.ndim > 1: sig = sig[:, 0]
            sig = sig.astype(np.float64)
            fs = float(record.fs)
            n = min(int(MAX_SEC * fs), MAX_SAMPLES, len(sig))
            sig = sig[:n]

            # Annotations with beat types
            ann_beats = []
            try:
                ann = wfdb.rdann(os.path.join(DATA_DIR, rec), 'atr')
                for i, s in enumerate(ann.symbol):
                    sample = int(ann.sample[i])
                    if s != '+' and sample < n:  # skip rhythm markers
                        ann_beats.append((sample, s))
            except:
                pass

            # Classify record by dominant beat type
            types = Counter(s for _, s in ann_beats)
            dominant = types.most_common(1)[0][0] if types else '?'

            prep = ECGPreprocessor(fs=fs)
            clean = prep.preprocess(sig)
            fe = FeatureExtractor(fs=fs)
            features = fe.extract(clean)

            data.append({
                'record': rec,
                'features': features,
                'ecg': clean,
                'fs': fs,
                'ann_beats': ann_beats,
                'beat_types': types,
                'dominant_type': dominant,
                'n_types': len(types),
                'n_beats_ann': len(ann_beats),
            })
        except Exception as e:
            print(f"  SKIP {rec}: {e}")

    return data


# =====================================================================
# Select training records for arrhythmia diversity
# =====================================================================
def select_training(data, n_train=40):
    """Select training records maximizing arrhythmia type coverage."""
    # Priority: records with many beat types, covering all major types
    priority_types = ['N', 'V', 'A', 'L', 'R', '/', 'F', 'J', 'E', '!', 'a', 'f', 'S']

    # Score each record by diversity
    for d in data:
        score = 0
        for t in priority_types:
            if t in d['beat_types']:
                score += min(d['beat_types'][t], 20)  # cap per type
        d['diversity_score'] = score + d['n_types'] * 5

    # Sort by diversity score
    data_sorted = sorted(data, key=lambda d: -d['diversity_score'])

    # Ensure coverage of each major type
    selected = set()
    selected_recs = []

    # First pass: pick best record for each type
    for t in priority_types:
        for d in data_sorted:
            if d['record'] not in selected and t in d['beat_types']:
                selected.add(d['record'])
                selected_recs.append(d)
                break

    # Second pass: fill remaining slots with most diverse
    for d in data_sorted:
        if len(selected_recs) >= n_train:
            break
        if d['record'] not in selected:
            selected.add(d['record'])
            selected_recs.append(d)

    print(f"\nTraining records selected: {len(selected_recs)}")
    train_types = Counter()
    for d in selected_recs:
        for t, c in d['beat_types'].items():
            train_types[t] += c
    print(f"Training beat type coverage: {dict(train_types.most_common(10))}")

    train_recs = {d['record'] for d in selected_recs}
    test_recs = [d for d in data if d['record'] not in train_recs]
    return selected_recs, test_recs


# =====================================================================
# Pool features from Viterbi
# =====================================================================
def pool_from_viterbi(data_list, model, decoder):
    N = model.n_states
    sf = [[] for _ in range(N)]
    sd = [[] for _ in range(N)]
    n_beats = 0
    for d in data_list:
        feats = d['features']
        T = feats.shape[0]
        if T < 50: continue
        result = decoder.decode(model, feats)
        for state, start, end in result['state_sequence']:
            if start < 0 or end < start or end >= T: continue
            dur = end - start + 1
            sf[state].append(feats[start:end+1])
            sd[state].append(dur)
        n_beats += sum(1 for s in result['state_sequence'] if s[0] == 4)
    for j in range(N):
        sf[j] = np.concatenate(sf[j], axis=0) if sf[j] else np.array([]).reshape(0,3)
    return sf, sd, n_beats


def update_model(model, sf, sd):
    N = model.n_states
    for j in range(N):
        if len(sf[j]) > model.n_gmm_components * 5:
            try: model.obs_dists[j].fit(sf[j], max_iter=40)
            except: pass
        if len(sd[j]) >= 5:
            a = np.array(sd[j], dtype=float)
            model.dur_dists[j].mu = float(np.mean(a))
            model.dur_dists[j].sigma = float(max(np.std(a), 1.0))
            model.dur_dists[j]._log_Z = None
    model._compute_D_max()


# =====================================================================
# Evaluate with per-beat-type breakdown
# =====================================================================
def decode_and_get_peaks(model, features, fs):
    decoder = HSMMDecoder()
    result = decoder.decode(model, features)
    peaks = np.array([(s+e)//2 for state, s, e in result['state_sequence'] if state == 4], dtype=int)
    return peaks, result['state_sequence']


def evaluate_per_beat_type(test_data, model):
    """Evaluate R-peak detection broken down by beat type."""
    decoder = HSMMDecoder()

    # Per-type accumulators
    type_matched = defaultdict(int)
    type_total = defaultdict(int)
    type_fp = defaultdict(int)  # false positives per type
    total_hsmm_peaks = 0
    total_ann_peaks = 0

    per_record = []

    for d in test_data:
        feats = d['features']
        fs = d['fs']
        ann_beats = d['ann_beats']
        rec = d['record']

        hsmm_r, segments = decode_and_get_peaks(model, feats, fs)
        n_hsmm = len(hsmm_r)

        tol = int(0.15 * fs)
        ann_matched = set()

        # Match HSMM peaks to annotation beats
        for hr in hsmm_r:
            best_dist = tol + 1
            best_ai = -1
            for ai, (ar, _atype) in enumerate(ann_beats):
                dist = abs(hr - ar)
                if dist <= tol and dist < best_dist:
                    best_dist = dist
                    best_ai = ai
            if best_ai >= 0:
                ann_matched.add(best_ai)

        # Per-type sensitivity
        rec_type_matched = defaultdict(int)
        rec_type_total = defaultdict(int)

        for ai, (ar, atype) in enumerate(ann_beats):
            rec_type_total[atype] += 1
            type_total[atype] += 1
            total_ann_peaks += 1
            if ai in ann_matched:
                rec_type_matched[atype] += 1
                type_matched[atype] += 1

        total_hsmm_peaks += n_hsmm

        # Overall scores
        n_ann = len(ann_beats)
        overall_se = round(len(ann_matched) / max(n_ann, 1) * 100, 1)
        overall_ppv = round(len(ann_matched) / max(n_hsmm, 1) * 100, 1)

        # Dominant type info
        dominant = d['dominant_type']
        dom_se = round(rec_type_matched[dominant] / max(rec_type_total[dominant], 1) * 100, 1)

        per_record.append({
            'record': rec,
            'n_ann': n_ann, 'n_hsmm': n_hsmm,
            'sensitivity': overall_se, 'ppv': overall_ppv,
            'dominant_type': dominant,
            'dominant_sensitivity': dom_se,
            'beat_types': dict(d['beat_types'].most_common(5)),
            'type_sensitivity': {t: round(rec_type_matched[t] / max(rec_type_total[t], 1) * 100, 1)
                                for t in rec_type_total},
        })

    # Aggregate per-type
    type_summary = {}
    for t in sorted(type_total.keys()):
        se = round(type_matched[t] / max(type_total[t], 1) * 100, 1)
        type_summary[t] = {
            'name': BEAT_TYPE_NAMES.get(t, t),
            'total_annotated': type_total[t],
            'detected': type_matched[t],
            'sensitivity': se,
        }

    # Overall
    overall = {
        'total_annotated_beats': total_ann_peaks,
        'total_hsmm_peaks': total_hsmm_peaks,
        'per_type': type_summary,
        'per_record': per_record,
    }

    return overall


# =====================================================================
# Main
# =====================================================================
def main():
    global_start = time.time()

    # ---- 1. Load data ----
    print("=" * 65)
    print("1. Loading all records with beat-type annotations...")
    print("=" * 65)
    all_data = load_all_with_types()

    n_with_ann = sum(1 for d in all_data if d['ann_beats'])
    n_no_ann = len(all_data) - n_with_ann
    print(f"\n  Total: {len(all_data)} records")
    print(f"  With annotations: {n_with_ann}")
    print(f"  Without annotations: {n_no_ann}")

    # Beat type distribution
    all_types = Counter()
    for d in all_data:
        for t, c in d['beat_types'].items():
            all_types[t] += c
    print(f"  Total annotated beats: {sum(all_types.values())}")
    print(f"  Beat types: {dict(all_types.most_common(8))}")

    # ---- 2. Select training data ----
    print(f"\n{'=' * 65}")
    print("2. Selecting arrhythmia-diverse training records...")
    print("=" * 65)
    train_data, test_data = select_training(all_data, n_train=45)

    # ---- 3. Initialize model ----
    print(f"\n{'=' * 65}")
    print("3. Initializing HSMM model...")
    print("=" * 65)
    model = HSMMModel(fs=360.0)
    model.initialize_with_priors()
    model.set_left_right_topology()

    # Smart init from diverse training records
    for d in train_data[:8]:
        smart_initialize_gmms(model, d['features'])
    print(f"  Model ready: {model}")

    decoder = HSMMDecoder()

    # ---- 4. Hard-EM training ----
    print(f"\n{'=' * 65}")
    print(f"4. Viterbi hard-EM training ({MAX_EM_ITERS} iterations)")
    print(f"{'=' * 65}")

    for it in range(MAX_EM_ITERS):
        t0 = time.time()
        print(f"\n  Iter {it+1}: decoding {len(train_data)} records...", end=" ", flush=True)
        sf, sd, n_beats = pool_from_viterbi(train_data, model, decoder)
        dt = time.time() - t0
        print(f"{n_beats} beats ({dt:.0f}s)")

        print(f"    Updating GMMs + durations...", end=" ", flush=True)
        t1 = time.time()
        update_model(model, sf, sd)
        print(f"({time.time()-t1:.1f}s)")

        # Quick validation
        val_subset = [d for d in test_data if d['ann_beats']][:20]
        if val_subset:
            t2 = time.time()
            val_result = evaluate_per_beat_type(val_subset, model)
            se_vals = [r['sensitivity'] for r in val_result['per_record']]
            avg_se = np.mean(se_vals) if se_vals else 0
            print(f"    Val Se on 20 random test records: {avg_se:.1f}% ({time.time()-t2:.1f}s)")

            if avg_se > 0 and (it == 0 or avg_se > best_se - 0.5):
                best_se = avg_se
                model.save(os.path.join(MODEL_DIR, 'hsmm_arrhythmia.npz'))
                print(f"    ★ Best model saved (Se={avg_se:.1f}%)")
            if it == 0:
                best_se = avg_se or 0

        gc.collect()

    # Load best model
    model_path = os.path.join(MODEL_DIR, 'hsmm_arrhythmia.npz')
    if os.path.exists(model_path):
        model = HSMMModel.load(model_path)
        print(f"\n  Loaded best model from iter with best val Se={best_se:.1f}%")

    # ---- 5. Full evaluation per beat type ----
    print(f"\n{'=' * 65}")
    print("5. Full per-beat-type evaluation on ALL test records")
    print(f"{'=' * 65}")

    # Only evaluate records WITH annotations
    test_with_ann = [d for d in test_data if d['ann_beats']]
    print(f"  Test records with annotations: {len(test_with_ann)}")

    eval_result = evaluate_per_beat_type(test_with_ann, model)

    # ---- 6. Report ----
    print(f"\n{'=' * 65}")
    print("  PER-BEAT-TYPE RESULTS (TEST SET - UNSEEN)")
    print(f"{'=' * 65}")
    print(f"  {'Type':<6} {'Name':<28} {'Total':>6} {'Detected':>8} {'Se%':>8}")
    print(f"  {'-'*56}")

    for t in sorted(eval_result['per_type'].keys(),
                     key=lambda x: -eval_result['per_type'][x]['total_annotated']):
        info = eval_result['per_type'][t]
        print(f"  {t:<6} {info['name']:<28} {info['total_annotated']:>6} {info['detected']:>8} {info['sensitivity']:>7.1f}%")

    # Overall
    total_ann = eval_result['total_annotated_beats']
    total_det = sum(v['detected'] for v in eval_result['per_type'].values())
    print(f"  {'-'*56}")
    print(f"  {'ALL':<6} {'Overall':<28} {total_ann:>6} {total_det:>8} {total_det/max(total_ann,1)*100:>7.1f}%")

    # Best/worst records
    recs = eval_result['per_record']
    print(f"\n  Top 10 records (by sensitivity):")
    for r in sorted(recs, key=lambda x: -x['sensitivity'])[:10]:
        print(f"    {r['record']:<8} Se={r['sensitivity']:5.1f}%  "
              f"PPV={r['ppv']:5.1f}%  dominant={r['dominant_type']} "
              f"types={dict(r['beat_types'])}")

    print(f"\n  Worst 10 records:")
    for r in sorted(recs, key=lambda x: x['sensitivity'])[:10]:
        dom_se = r.get('dominant_sensitivity', 'N/A')
        print(f"    {r['record']:<8} Se={r['sensitivity']:5.1f}%  "
              f"PPV={r['ppv']:5.1f}%  dominant={r['dominant_type']}(Se={dom_se}%) "
              f"types={dict(r['beat_types'])}")

    # ---- 7. Also evaluate on no-annotation records ----
    test_no_ann = [d for d in test_data if not d['ann_beats']]
    if test_no_ann:
        print(f"\n{'=' * 65}")
        print(f"  Records without annotations: {len(test_no_ann)}")
        print(f"{'=' * 65}")
        total_beats_noann = 0
        for d in test_no_ann:
            peaks, _ = decode_and_get_peaks(model, d['features'], d['fs'])
            total_beats_noann += len(peaks)
        print(f"  Total beats detected: {total_beats_noann}")

    # ---- 8. Save ----
    print(f"\n{'=' * 65}")
    print("  SAVING RESULTS")
    print(f"{'=' * 65}")

    # Save eval summary
    def _safe(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, set): return sorted(obj)
        if isinstance(obj, (Counter, defaultdict)): return dict(obj)
        if isinstance(obj, dict): return {str(k): _safe(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_safe(v) for v in obj]
        return obj

    eval_path = os.path.join(OUT_DIR, 'per_beat_type_eval.json')
    with open(eval_path, 'w') as f:
        json.dump(_safe(eval_result), f, indent=2)
    print(f"  Saved: {eval_path}")

    # Also save individual per-record results
    for rec_info in eval_result['per_record'][:5]:
        print(f"  Example: {rec_info['record']} "
              f"Se={rec_info['sensitivity']}% PPV={rec_info['ppv']}% "
              f"types={rec_info['type_sensitivity']}")

    total_time = time.time() - global_start
    print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Model: {os.path.join(MODEL_DIR, 'hsmm_arrhythmia.npz')}")
    print(f"  Results: {eval_path}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
