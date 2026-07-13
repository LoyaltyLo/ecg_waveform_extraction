"""Generate per-beat P-QRS-T waveform + P-wave images from saved batch outputs.

Reads existing output/{record}/ data and produces:
  output/{record}/beats/
    beat_{id}_waveform.png   — full P-QRS-T waveform with state colors
    beat_{id}_p_wave.png     — zoomed P-wave with onset/offset/peak markers
"""

import sys
sys.path.insert(0, 'c:/LoyaltyLo/PythonProjects/ECG_engineering')

import os
import json
import gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# State color palette (consistent with utils/vis.py)
STATE_COLORS = {
    "ISO": "#e0e0e0", "P": "#4caf50", "PR": "#c8e6c9",
    "Q": "#f44336",   "R": "#d32f2f", "S":  "#ff7043",
    "ST": "#fff176",  "T": "#2196f3", "TP": "#b0bec5",
    "UNKNOWN": "#9e9e9e",
}
STATE_LABELS = ["ISO", "P", "PR", "Q", "R", "S", "ST", "T", "TP"]


def process_record(rec_name: str, output_base: str):
    """Generate per-beat plots for one record."""
    rec_dir = os.path.join(output_base, rec_name)
    beats_dir = os.path.join(rec_dir, "beats")
    os.makedirs(beats_dir, exist_ok=True)

    # ---- Load data ----
    filtered_path = os.path.join(rec_dir, "filtered_ecg.npy")
    labels_path = os.path.join(rec_dir, "state_labels.npy")
    seg_path = os.path.join(rec_dir, "segmentation.json")
    pw_path = os.path.join(rec_dir, "p_waves.json")

    if not all(os.path.exists(p) for p in [filtered_path, labels_path, seg_path]):
        print(f"  SKIP: missing data files")
        return {"record": rec_name, "beats_plotted": 0, "skipped": True}

    ecg = np.load(filtered_path)
    state_labels = np.load(labels_path)
    with open(seg_path) as f:
        segments = json.load(f)
    with open(pw_path) as f:
        p_waves = json.load(f)

    summary_path = os.path.join(rec_dir, "summary.json")
    fs = 360.0
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            s = json.load(f)
            fs_val = s.get("fs")
            if fs_val:
                fs = float(fs_val)
    t_total = len(ecg)

    # Build time axis
    time = np.arange(t_total) / fs

    # ---- Check what exists already ----
    existing = set()
    if os.path.exists(beats_dir):
        for fname in os.listdir(beats_dir):
            if fname.endswith('.png'):
                existing.add(fname)

    # ---- Plot each beat ----
    n_plotted = 0
    beat_ids = []

    for seg in segments:
        beat_id = seg["beat_id"]
        b_qrs_on = seg["q_onset"]
        b_s_off = seg["s_offset"]
        b_p_on = seg["p_onset"]
        b_p_off = seg["p_offset"]
        b_t_off = seg["t_offset"]

        if b_qrs_on <= 0 or b_s_off <= 0:
            continue
        if b_p_on <= 0 or b_p_off <= 0:
            continue
        if b_t_off <= 0:
            continue

        # Define context window: from 150ms before P to 150ms after T
        margin = int(0.15 * fs)
        win_start = max(0, b_p_on - margin)
        win_end = min(t_total - 1, b_t_off + margin)

        if win_end - win_start < 30:
            continue

        beat_ids.append(beat_id)

        # ---- (A) Full P-QRS-T waveform plot ----
        wf_name = f"beat_{beat_id:03d}_waveform.png"
        wf_path = os.path.join(beats_dir, wf_name)

        if wf_name not in existing:
            fig, ax = plt.subplots(figsize=(12, 4))

            t_win = time[win_start:win_end + 1]
            ecg_win = ecg[win_start:win_end + 1]
            lbl_win = state_labels[win_start:win_end + 1]

            # Draw state-colored background bands
            if len(lbl_win) > 0:
                prev = lbl_win[0]
                seg_start = 0
                for i in range(1, len(lbl_win)):
                    if lbl_win[i] != prev:
                        c = STATE_COLORS.get(
                            STATE_LABELS[prev] if 0 <= prev < 9 else "UNKNOWN",
                            "#9e9e9e")
                        ax.axvspan(t_win[seg_start], t_win[i], alpha=0.25, color=c)
                        seg_start = i
                        prev = lbl_win[i]
                c = STATE_COLORS.get(
                    STATE_LABELS[prev] if 0 <= prev < 9 else "UNKNOWN", "#9e9e9e")
                ax.axvspan(t_win[seg_start], t_win[-1], alpha=0.25, color=c)

            # ECG trace
            ax.plot(t_win, ecg_win, 'k-', linewidth=0.8)

            # Mark boundaries
            y_min, y_max = ecg_win.min(), ecg_win.max()
            y_range = y_max - y_min
            y_lo = y_min - 0.1 * y_range
            y_hi = y_max + 0.1 * y_range

            # P onset/offset
            if b_p_on > 0:
                t_p_on = b_p_on / fs
                ax.axvline(t_p_on, color='green', linestyle='--', linewidth=0.8, alpha=0.7)
                ax.text(t_p_on, y_hi, 'P on', fontsize=7, color='green', ha='center')
            if b_p_off > 0:
                t_p_off = b_p_off / fs
                ax.axvline(t_p_off, color='green', linestyle='--', linewidth=0.8, alpha=0.7)
                ax.text(t_p_off, y_hi, 'P off', fontsize=7, color='green', ha='center')

            # QRS onset/offset
            if b_qrs_on > 0:
                t_q_on = b_qrs_on / fs
                ax.axvline(t_q_on, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
                ax.text(t_q_on, y_lo, 'QRS on', fontsize=7, color='red', ha='center')
            if b_s_off > 0:
                t_s_off = b_s_off / fs
                ax.axvline(t_s_off, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
                ax.text(t_s_off, y_lo, 'QRS off', fontsize=7, color='red', ha='center')

            # T offset
            if b_t_off > 0:
                t_t_off = b_t_off / fs
                ax.axvline(t_t_off, color='blue', linestyle='--', linewidth=0.8, alpha=0.7)
                ax.text(t_t_off, y_hi, 'T off', fontsize=7, color='blue', ha='center')

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (norm)")
            ax.set_title(f"Record {rec_name} — Beat {beat_id} — P-QRS-T Waveform")
            ax.set_xlim(t_win[0], t_win[-1])
            ax.grid(True, alpha=0.2)

            # Legend
            legend_handles = [
                Rectangle((0, 0), 1, 1, facecolor=STATE_COLORS[s], alpha=0.25, label=s)
                for s in STATE_LABELS
            ]
            ax.legend(handles=legend_handles, loc='upper right', ncol=9, fontsize=6)

            fig.tight_layout()
            fig.savefig(wf_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            n_plotted += 1

        # ---- (B) P-wave detail plot ----
        pw_name = f"beat_{beat_id:03d}_p_wave.png"
        pw_fpath = os.path.join(beats_dir, pw_name)

        if pw_name not in existing:
            pw_match = None
            for pw in p_waves:
                if pw["beat_id"] == beat_id:
                    pw_match = pw
                    break

            if pw_match and pw_match["onset_sample"] > 0 and pw_match["offset_sample"] > pw_match["onset_sample"]:
                pw_on = pw_match["onset_sample"]
                pw_off = pw_match["offset_sample"]
                pw_peak = pw_match.get("peak_sample", (pw_on + pw_off) // 2)
                pw_dur = pw_match.get("duration_ms", (pw_off - pw_on) / fs * 1000)

                # Wider context for P-wave
                pw_margin = int(0.1 * fs)
                pw_win_start = max(0, pw_on - pw_margin)
                pw_win_end = min(t_total - 1, pw_off + pw_margin)

                fig, ax = plt.subplots(figsize=(7, 3))
                t_pw = time[pw_win_start:pw_win_end + 1]
                ecg_pw = ecg[pw_win_start:pw_win_end + 1]

                ax.plot(t_pw, ecg_pw, 'k-', linewidth=1.0)

                # P-wave shaded region
                p_idx = np.arange(pw_on, pw_off + 1)
                ax.fill_between(time[p_idx], ecg[p_idx] if len(p_idx) <= len(ecg) else 0,
                                alpha=0.3, color='#4caf50', label='P wave')

                # Markers
                ax.axvline(time[pw_on], color='green', linestyle='--', linewidth=1.0, alpha=0.8)
                ax.axvline(time[pw_off], color='red', linestyle='--', linewidth=1.0, alpha=0.8)
                if 0 <= pw_peak < t_total:
                    ax.axvline(time[pw_peak], color='blue', linestyle=':', linewidth=0.8, alpha=0.6)

                # Annotations
                y_mid = np.median(ecg_pw)
                ax.annotate('onset', (time[pw_on], y_mid),
                            textcoords="offset points", xytext=(-5, -15),
                            fontsize=8, color='green', ha='right')
                ax.annotate('offset', (time[pw_off], y_mid),
                            textcoords="offset points", xytext=(5, -15),
                            fontsize=8, color='red', ha='left')
                ax.annotate(f'{pw_dur:.0f}ms', (time[(pw_on + pw_off) // 2], ecg[(pw_on + pw_off) // 2]),
                            textcoords="offset points", xytext=(0, 10),
                            fontsize=9, ha='center', fontweight='bold')

                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                ax.set_title(f"Record {rec_name} — Beat {beat_id} — P-Wave Detail")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.2)

                fig.tight_layout()
                fig.savefig(pw_fpath, dpi=120, bbox_inches='tight')
                plt.close(fig)
                n_plotted += 1

    return {
        "record": rec_name,
        "beats_plotted": n_plotted,
        "total_beats": len(beat_ids),
        "skipped": False,
    }


# =====================================================================
# Main
# =====================================================================
def main():
    output_base = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/output_trained'
    os.makedirs(output_base, exist_ok=True)

    # Find all record directories
    record_dirs = []
    for name in os.listdir(output_base):
        d = os.path.join(output_base, name)
        if os.path.isdir(d) and name.isdigit() and os.path.exists(os.path.join(d, "segmentation.json")):
            record_dirs.append(name)
    record_dirs.sort()

    print(f"Records to process: {len(record_dirs)}")
    print(f"Output: {output_base}/<record>/beats/beat_###_waveform.png")
    print(f"Output: {output_base}/<record>/beats/beat_###_p_wave.png")
    print()

    total_plots = 0
    total_beats = 0
    results = []

    for idx, rec in enumerate(record_dirs):
        print(f"[{idx+1}/{len(record_dirs)}] Record {rec}...", end=" ", flush=True)
        res = process_record(rec, output_base)
        results.append(res)
        if res["skipped"]:
            print("SKIP")
        else:
            print(f"{res['beats_plotted']} plots saved ({res['total_beats']} beats)")
            total_plots += res["beats_plotted"]
            total_beats += res["total_beats"]

        gc.collect()

    print(f"\n{'='*60}")
    print(f"  Total: {total_plots} images across {total_beats} beats")
    print(f"  Records: {len(record_dirs)}")
    print(f"  Location: {output_base}/<record>/beats/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
