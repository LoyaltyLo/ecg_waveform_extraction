"""Fast HSMM training via Viterbi hard-EM across all records.

Algorithm (per iteration):
  1. Viterbi decode each record → state labels + segment durations
  2. Pool labeled features across all records → fit GMMs per state
  3. Pool durations across all records → fit duration distributions per state
  4. Re-decode with updated model
  5. Repeat until convergence or max iters

Much faster than full Baum-Welch (no backward pass, no soft counts).
"""

import sys
sys.path.insert(0, 'c:/LoyaltyLo/PythonProjects/ECG_engineering')

import os
import gc
import json
import time
import numpy as np
import wfdb

from ecg_waveform_extraction.preprocessing import ECGPreprocessor
from ecg_waveform_extraction.features import FeatureExtractor
from ecg_waveform_extraction.hsmm import (
    HSMMModel, HSMMDecoder, smart_initialize_gmms,
)
from ecg_waveform_extraction.hsmm.hsmm_model import STATE_LABELS

# =====================================================================
# Config
# =====================================================================
DATA_DIR = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/data'
MODEL_DIR = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/models'
os.makedirs(MODEL_DIR, exist_ok=True)

MAX_SEC_PER_RECORD = 12.0
MAX_SAMPLES = 6000
MAX_TRAINING_ITERS = 5      # Hard-EM iterations
CONVERGENCE_TOL = 0.5        # Stop if sensitivity change < 0.5%


# =====================================================================
# Gather all training data
# =====================================================================
def gather_data(data_dir, max_sec, max_samples):
    """Load and preprocess all records. Returns list of dicts."""
    record_names = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith('.hea'):
            rec = fname[:-4]
            if os.path.exists(os.path.join(data_dir, rec + '.dat')):
                record_names.append(rec)

    print(f"Found {len(record_names)} records with signal data")
    data = []

    for idx, rec in enumerate(record_names):
        try:
            record = wfdb.rdrecord(os.path.join(data_dir, rec))
            sig = record.p_signal
            if sig.ndim > 1:
                sig = sig[:, 0]
            sig = sig.astype(np.float64)
            fs = float(record.fs)

            max_n = min(int(max_sec * fs), max_samples)
            if len(sig) > max_n:
                sig = sig[:max_n]

            prep = ECGPreprocessor(fs=fs)
            clean = prep.preprocess(sig)
            fe = FeatureExtractor(fs=fs)
            features = fe.extract(clean)

            # Annotations
            ann_r = None
            try:
                ann = wfdb.rdann(os.path.join(data_dir, rec), 'atr')
                a_s = np.asarray(ann.sample)[np.asarray(ann.symbol) != '+']
                ann_r = a_s[a_s < len(sig)]
            except Exception:
                pass

            data.append({
                "record": rec,
                "features": features,
                "ecg": clean,
                "fs": fs,
                "ann_r": ann_r,
                "n_samples": len(sig),
            })
        except Exception as e:
            print(f"  SKIP {rec}: {e}")

    print(f"  Loaded {len(data)} records")
    return data


# =====================================================================
# Pool features by state from Viterbi decode results
# =====================================================================
def pool_from_viterbi(data_list, model, decoder):
    """Run Viterbi on each record, pool features + durations per state.

    Returns:
        state_features: list of np.ndarray, one per state (N_states,)
        state_durations: list of list of int, one per state
    """
    N = model.n_states
    state_features = [[] for _ in range(N)]
    state_durations = [[] for _ in range(N)]
    total_beats = 0
    total_segments = 0

    for d in data_list:
        features = d["features"]
        T = features.shape[0]
        if T < 50:
            continue

        result = decoder.decode(model, features)
        segments = result["state_sequence"]
        total_segments += len(segments)

        for state, start, end in segments:
            if start < 0 or end < start or end >= T:
                continue
            dur = end - start + 1
            seg_features = features[start:end + 1]
            state_features[state].append(seg_features)
            state_durations[state].append(dur)

        # Count beats (R state = 4)
        n_beats = sum(1 for s in segments if s[0] == 4)
        total_beats += n_beats

    # Concatenate feature arrays per state
    for j in range(N):
        if state_features[j]:
            state_features[j] = np.concatenate(state_features[j], axis=0)
        else:
            state_features[j] = np.array([]).reshape(0, 3)

    return state_features, state_durations, total_beats, total_segments


# =====================================================================
# Update model from pooled data
# =====================================================================
def update_model(model, state_features, state_durations):
    """M-step: update GMMs and durations from pooled labeled data."""
    N = model.n_states

    for j in range(N):
        # Update GMM
        feats = state_features[j]
        if len(feats) > model.n_gmm_components * 5:
            try:
                model.obs_dists[j].fit(feats, max_iter=40, tol=1e-3)
            except Exception:
                pass

        # Update duration distribution
        durs = state_durations[j]
        if len(durs) >= 5:
            durs_arr = np.array(durs, dtype=np.float64)
            model.dur_dists[j].mu = float(np.mean(durs_arr))
            model.dur_dists[j].sigma = float(max(np.std(durs_arr), 1.0))
            model.dur_dists[j]._log_Z = None

    # Update D_max
    model._compute_D_max()


# =====================================================================
# Evaluate model
# =====================================================================
def evaluate_model(data_list, model, decoder):
    """Compute sensitivity/PPV on annotated records."""
    results = []
    for d in data_list:
        features = d["features"]
        ann_r = d["ann_r"]
        fs = d["fs"]
        rec = d["record"]

        if features.shape[0] < 50:
            continue

        result = decoder.decode(model, features)
        segments = result["state_sequence"]

        hsmm_r = np.array(
            [(s + e) // 2 for state, s, e in segments if state == 4],
            dtype=int,
        )

        n_beats = len(hsmm_r)
        se, ppv = None, None

        if ann_r is not None and len(ann_r) > 0 and n_beats > 0:
            tol = int(0.15 * fs)
            matched_se = sum(1 for ar in ann_r
                             if np.any(np.abs(hsmm_r - int(ar)) <= tol))
            se = round(matched_se / len(ann_r) * 100, 1)
            matched_ppv = sum(1 for hr in hsmm_r
                              if np.any(np.abs(ann_r - hr) <= tol))
            ppv = round(matched_ppv / n_beats * 100, 1)

        results.append({
            "record": rec, "n_beats": n_beats,
            "sensitivity": se, "ppv": ppv,
        })

    return results


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 60)
    print("  HSMM Viterbi Hard-EM Training")
    print("=" * 60)
    print(f"  Data: {DATA_DIR}")
    print(f"  Max: {MAX_SEC_PER_RECORD}s/record, {MAX_TRAINING_ITERS} EM iters")
    print()

    # ---- 1. Load data ----
    print("1. Loading training data...")
    all_data = gather_data(DATA_DIR, MAX_SEC_PER_RECORD, MAX_SAMPLES)
    if not all_data:
        print("ERROR: No data!")
        return

    # Use first 30 records for training, rest for validation
    np.random.seed(42)
    indices = np.random.permutation(len(all_data))
    n_train = min(30, len(all_data))
    train_data = [all_data[i] for i in indices[:n_train]]
    val_data = [all_data[i] for i in indices[n_train:]]
    print(f"   Train: {len(train_data)}, Val: {len(val_data)}")

    # ---- 2. Initialize model ----
    print("\n2. Initializing HSMM model...")
    model = HSMMModel(fs=360.0)
    model.initialize_with_priors()
    model.set_left_right_topology()

    # Smart init from first few training records
    for i in range(min(5, len(train_data))):
        smart_initialize_gmms(model, train_data[i]["features"])

    decoder = HSMMDecoder()
    print(f"   Model ready: {model}")

    # ---- 3. Viterbi Hard-EM iterations ----
    print(f"\n3. Hard-EM training ({MAX_TRAINING_ITERS} iterations)...")
    best_model = None
    best_sensitivity = 0
    best_iter = 0
    eval_history = []

    for it in range(MAX_TRAINING_ITERS):
        t0 = time.time()

        # ---- Decode all records ----
        print(f"\n  Iter {it + 1}/{MAX_TRAINING_ITERS}:")
        print(f"    Decoding {len(train_data)} records...", end=" ", flush=True)

        st_feats, st_durs, n_beats, n_segs = pool_from_viterbi(
            train_data, model, decoder
        )
        print(f"{n_beats} beats in {n_segs} segments "
              f"({time.time() - t0:.1f}s)")

        # ---- Update model ----
        print(f"    Updating GMMs & durations...", end=" ", flush=True)
        t1 = time.time()
        update_model(model, st_feats, st_durs)
        print(f"({time.time() - t1:.1f}s)")

        # Per-state info
        for j in range(model.n_states):
            n_feats = len(st_feats[j])
            n_durs = len(st_durs[j])
            if n_durs > 0:
                durs_arr = np.array(st_durs[j])
                print(f"    {STATE_LABELS[j]:4s}: {n_durs:4d} segs, "
                      f"dur={durs_arr.mean():.0f}±{durs_arr.std():.0f} samples, "
                      f"feats={n_feats}")

        # ---- Evaluate on validation set ----
        print(f"    Evaluating on {len(val_data)} validation records...", end=" ", flush=True)
        t2 = time.time()
        val_results = evaluate_model(val_data, model, decoder)
        val_se = [r["sensitivity"] for r in val_results
                   if r["sensitivity"] is not None]
        avg_se = np.mean(val_se) if val_se else 0.0
        print(f"Se={avg_se:.1f}% ({time.time() - t2:.1f}s)")

        eval_history.append({
            "iter": it + 1,
            "n_beats": n_beats,
            "n_segments": n_segs,
            "val_sensitivity": round(float(avg_se), 1),
            "time_sec": round(time.time() - t0, 1),
        })

        # Track best model
        if avg_se > best_sensitivity:
            best_sensitivity = avg_se
            best_model = model
            best_iter = it + 1
            # Save best model immediately
            bp = os.path.join(MODEL_DIR, "hsmm_trained.npz")
            model.save(bp)
            print(f"    ★ New best model saved (Se={avg_se:.1f}%)")

        # Convergence check
        if it >= 1:
            prev_se = eval_history[-2]["val_sensitivity"]
            if abs(avg_se - prev_se) < CONVERGENCE_TOL:
                print(f"    Converged (ΔSe < {CONVERGENCE_TOL}%)")
                break

        gc.collect()

    # ---- 4. Final evaluation on all records ----
    print(f"\n4. Final evaluation on all {len(all_data)} records...")
    final_results = evaluate_model(all_data, model, decoder)
    final_se = [r["sensitivity"] for r in final_results
                if r["sensitivity"] is not None]
    final_ppv = [r["ppv"] for r in final_results
                 if r["ppv"] is not None]
    total_beats = sum(r["n_beats"] for r in final_results)

    print(f"   Mean Sensitivity: {np.mean(final_se):.1f}% ± {np.std(final_se):.1f}%")
    print(f"   Mean PPV: {np.mean(final_ppv):.1f}% ± {np.std(final_ppv):.1f}%")
    print(f"   Total beats: {total_beats}")

    # Per-record detail
    sorted_r = sorted(
        [r for r in final_results if r["sensitivity"] is not None],
        key=lambda x: x["sensitivity"],
    )
    print(f"   Worst 5: {[(r['record'], r['sensitivity']) for r in sorted_r[:5]]}")
    print(f"   Best 5: {[(r['record'], r['sensitivity']) for r in sorted_r[-5:]]}")

    # ---- 5. Save final model and results ----
    model_path = os.path.join(MODEL_DIR, "hsmm_trained.npz")
    if best_model is not None and best_iter == MAX_TRAINING_ITERS:
        best_model.save(model_path)
    elif best_model is not None:
        # Already saved at best iter
        pass
    else:
        model.save(model_path)

    print(f"\n5. Model saved: {model_path}")

    # Save training history
    results_path = os.path.join(MODEL_DIR, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "n_train_records": len(train_data),
            "n_val_records": len(val_data),
            "n_total_records": len(all_data),
            "max_sec_per_record": MAX_SEC_PER_RECORD,
            "best_iteration": best_iter,
            "best_val_sensitivity": round(best_sensitivity, 1),
            "final_sensitivity_mean": round(float(np.mean(final_se)), 1) if final_se else None,
            "final_sensitivity_std": round(float(np.std(final_se)), 1) if final_se else None,
            "final_ppv_mean": round(float(np.mean(final_ppv)), 1) if final_ppv else None,
            "final_ppv_std": round(float(np.std(final_ppv)), 1) if final_ppv else None,
            "total_beats": total_beats,
            "iteration_history": eval_history,
            "per_record": final_results,
        }, f, indent=2)
    print(f"   Results saved: {results_path}")

    # Learnings
    print(f"\n{'=' * 60}")
    print(f"  Training complete!")
    print(f"  Best iter: {best_iter}, Val Se: {best_sensitivity:.1f}%")
    print(f"  Final: Se={np.mean(final_se):.1f}%, PPV={np.mean(final_ppv):.1f}%")
    print(f"  Model: {model_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
