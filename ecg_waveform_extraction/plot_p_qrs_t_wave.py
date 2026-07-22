"""Generate complete P-QRS-T waveform plots for Lead I + Lead II.

For each beat saves:
  beats/beat_###_waveform.png — P-QRS-T with state colors, boundary markers, info boxes
  per lead: lead_I/ and lead_II/ subdirectories
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os, json, re, time, gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ecg_waveform_extraction.preprocessing import ECGPreprocessor
from ecg_waveform_extraction.features import FeatureExtractor
from ecg_waveform_extraction.hsmm import HSMMModel, smart_initialize_gmms
from ecg_waveform_extraction.segmentation import ECGSegmenter
from ecg_waveform_extraction.extraction import PWaveExtractor, refine_qrs_boundaries
from ecg_waveform_extraction.hsmm.hsmm_model import STATE_LABELS
from ecg_waveform_extraction.utils.vis import STATE_COLORS
from ecg_waveform_extraction.utils.aecg_parser import parse_aecg

AECG_DIR = 'C:/LoyaltyLo/datasets/RA-LA_Reversal/aECG'
OUT_DIR = str(Path(__file__).resolve().parent / 'output/rala_full/_p_qrs_t_wave')
os.makedirs(OUT_DIR, exist_ok=True)
N_FILES = 50
MAX_SAMPLES = 4000
MAX_BEATS_PER_RECORD = 6
LEADS_TO_PROCESS = ['I', 'II']




def process_one_lead(sig, fs, lead_name):
    """Run HSMM + P-wave extraction + QRS refinement on one lead.

    Returns (seg_result, clean, p_waves)
    """
    n = min(len(sig), MAX_SAMPLES)
    sig = sig[:n].astype(np.float64)

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

    # P-wave extraction
    p_ext = PWaveExtractor(fs=fs, refine_boundaries=True, enable_template_fallback=True)
    p_waves = p_ext.extract(seg_result)

    return seg_result, clean, p_waves


def save_lead_output(rec_name, rec_dir, lead_name, seg_result, clean, p_waves, fs, sig_raw):
    """Save all outputs for one lead."""
    lead_dir = os.path.join(rec_dir, f'lead_{lead_name}')
    beats_dir = os.path.join(lead_dir, 'beats')
    os.makedirs(beats_dir, exist_ok=True)

    pw_map = {pw.beat_id: pw for pw in p_waves}
    T = len(clean)

    # ---- numpy data ----
    np.save(os.path.join(lead_dir, 'raw_ecg.npy'), sig_raw[:T])
    np.save(os.path.join(lead_dir, 'filtered_ecg.npy'), clean)
    np.save(os.path.join(lead_dir, 'state_labels.npy'), seg_result.state_labels)

    # ---- Segmentation overview ----
    fig, ax = plt.subplots(figsize=(18, 4))
    plot_sec = min(T / fs, 3.8)
    n_plot = int(plot_sec * fs)
    t_plot = np.arange(n_plot) / fs
    e_plot = clean[:n_plot]
    lbl_plot = seg_result.state_labels[:n_plot]

    if len(lbl_plot) > 0:
        prev = lbl_plot[0]; seg_start = 0
        for i in range(1, len(lbl_plot)):
            if lbl_plot[i] != prev:
                c = STATE_COLORS.get(STATE_LABELS[prev] if 0<=prev<9 else 'UNKNOWN','#e0e0e0')
                ax.axvspan(t_plot[seg_start], t_plot[i], alpha=0.20, color=c)
                seg_start = i; prev = lbl_plot[i]
        c = STATE_COLORS.get(STATE_LABELS[prev] if 0<=prev<9 else 'UNKNOWN','#e0e0e0')
        ax.axvspan(t_plot[seg_start], t_plot[-1], alpha=0.20, color=c)

    ax.plot(t_plot, e_plot, 'k-', linewidth=0.5)
    ax.set_xlim(t_plot[0], t_plot[-1]); ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude')
    ax.set_title(f'{rec_name} — Lead {lead_name} — HSMM P-QRS-T Segmentation')
    handles = [Rectangle((0,0),1,1,facecolor=STATE_COLORS[s],alpha=0.25,label=s) for s in STATE_LABELS]
    ax.legend(handles=handles, loc='upper right', ncol=9, fontsize=5)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(lead_dir, 'segmentation.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ---- Per-beat plots + JSON ----
    beats_json = []
    n_plotted = 0

    for b in seg_result.beats:
        if b.p_onset <= 0 or b.t_offset <= 0: continue

        bid = b.beat_id
        pw = pw_map.get(bid)

        # Refine QRS boundaries
        if b.q_onset > 0 and b.s_offset > 0 and b.r_peak > 0:
            q_on, r_pk, s_off = refine_qrs_boundaries(clean, b.q_onset, b.r_peak, b.s_offset, fs)
        else:
            q_on, r_pk, s_off = b.q_onset, b.r_peak, b.s_offset

        # JSON entry
        entry = {'beat_id': int(bid)}
        for name, val in [('p_onset', b.p_onset), ('p_offset', b.p_offset),
                           ('q_onset', q_on), ('r_peak', r_pk), ('s_offset', s_off),
                           ('t_onset', b.t_onset), ('t_offset', b.t_offset)]:
            entry[name] = int(val) if val is not None and val > 0 else -1
        if pw and pw.onset_sample > 0:
            entry['p_wave'] = {
                'onset_sample': pw.onset_sample, 'offset_sample': pw.offset_sample,
                'peak_sample': pw.peak_sample, 'duration_ms': pw.duration_ms,
                'confidence': pw.confidence, 'morphology': pw.morphology,
            }
        beats_json.append(entry)

        # ---- Waveform plot (up to MAX_BEATS) ----
        if n_plotted >= MAX_BEATS_PER_RECORD: continue

        margin = int(0.15 * fs)
        ws = max(0, b.p_onset - margin)
        we = min(T - 1, b.t_offset + margin)
        if we - ws < 30: continue

        fig, ax = plt.subplots(figsize=(12, 4))
        t_win = np.arange(ws, we + 1) / fs
        e_win = clean[ws:we + 1]
        l_win = seg_result.state_labels[ws:we + 1]

        # 9-state color bands
        if len(l_win) > 0:
            prev = l_win[0]; seg_start = 0
            for i in range(1, len(l_win)):
                if l_win[i] != prev:
                    c = STATE_COLORS.get(STATE_LABELS[prev] if 0<=prev<9 else 'UNKNOWN','#e0e0e0')
                    ax.axvspan(t_win[seg_start], t_win[i], alpha=0.22, color=c)
                    seg_start = i; prev = l_win[i]
            c = STATE_COLORS.get(STATE_LABELS[prev] if 0<=prev<9 else 'UNKNOWN','#e0e0e0')
            ax.axvspan(t_win[seg_start], t_win[-1], alpha=0.22, color=c)

        ax.plot(t_win, e_win, 'k-', linewidth=0.8)
        ylo, yhi = e_win.min(), e_win.max()
        yr = max(yhi - ylo, 0.01)

        # ---- P-wave markers (refined) ----
        if pw and pw.onset_sample > 0:
            ax.axvline(pw.onset_sample / fs, color='green', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.axvline(pw.offset_sample / fs, color='green', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.text(pw.onset_sample / fs, ylo - 0.04*yr, 'P up', fontsize=7, color='green', ha='center')
            ax.text(pw.offset_sample / fs, ylo - 0.04*yr, 'P down', fontsize=7, color='green', ha='center')
        else:
            if b.p_onset > 0:
                ax.axvline(b.p_onset / fs, color='green', linestyle='--', linewidth=0.6, alpha=0.5)
            if b.p_offset > 0:
                ax.axvline(b.p_offset / fs, color='green', linestyle='--', linewidth=0.6, alpha=0.5)

        # ---- QRS markers (refined) ----
        if q_on > 0:
            ax.axvline(q_on / fs, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.text(q_on / fs, ylo - 0.10*yr, 'QRS up', fontsize=7, color='red', ha='center')
        if s_off > 0:
            ax.axvline(s_off / fs, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.text(s_off / fs, ylo - 0.10*yr, 'QRS down', fontsize=7, color='red', ha='center')

        # ---- T-wave marker ----
        if b.t_offset > 0:
            ax.axvline(b.t_offset / fs, color='blue', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.text(b.t_offset / fs, yhi + 0.04*yr, 'T down', fontsize=7, color='blue', ha='center')

        # ---- P-wave info box ----
        if pw and pw.onset_sample > 0:
            pw_mid = (pw.onset_sample + pw.offset_sample) // 2
            if ws <= pw_mid <= we:
                info = f'P: {pw.duration_ms:.0f}ms | conf={pw.confidence:.2f} | {pw.morphology} | SNR={pw.snr_db:.1f}dB'
                ax.text(pw_mid / fs, yhi + 0.10*yr, info, fontsize=7, ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5e9', alpha=0.85))

        # ---- QRS info box ----
        if q_on > 0 and s_off > 0:
            qrs_seg = clean[q_on:s_off + 1]
            bl = float(np.mean(clean[max(0,q_on-30):q_on])) if q_on >= 30 else float(np.median(qrs_seg[:5]))
            qrs_net = float(np.sum(qrs_seg - bl))
            r_val = float(clean[r_pk] - bl) if 0 <= r_pk < T else 0.0
            s_val = float(np.min(qrs_seg[r_pk - q_on:] - bl)) if r_pk > q_on and r_pk - q_on < len(qrs_seg) else 0.0
            rs = r_val / max(abs(s_val), 0.001) if r_val > 0 else -1
            dur = (s_off - q_on) / fs * 1000
            pol = 'up' if qrs_net > 0 else ('down' if qrs_net < 0 else 'bip')
            info2 = f'QRS: {dur:.0f}ms | R/S={rs:.1f} | net={qrs_net:.0f} | {pol}'
            ax.text(r_pk / fs, ylo - 0.16*yr, info2, fontsize=7, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#fce4ec', alpha=0.85))

        ax.set_xlim(t_win[0], t_win[-1]); ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude')
        ax.set_title(f'{rec_name} — Beat {bid} — Lead {lead_name} P-QRS-T Waveform')
        handles = [Rectangle((0,0),1,1,facecolor=STATE_COLORS[s],alpha=0.25,label=s) for s in STATE_LABELS]
        ax.legend(handles=handles, loc='upper right', ncol=9, fontsize=5)
        ax.grid(True, alpha=0.15)
        fig.tight_layout()
        fig.savefig(os.path.join(beats_dir, f'beat_{bid:03d}_waveform.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)
        n_plotted += 1

    # Save per-beat JSON
    with open(os.path.join(lead_dir, 'p_qrs_t_wave.json'), 'w') as f:
        json.dump(beats_json, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, (np.integer,)) else float(o))

    return {'n_beats': len(seg_result.beats), 'n_plotted': n_plotted, 'n_p_waves': len(p_waves)}


def process_record(fname):
    """Process one aECG file: extract Lead I + Lead II, generate separate outputs."""
    fpath = os.path.join(AECG_DIR, fname)
    aecg = parse_aecg(fpath, max_samples=MAX_SAMPLES)
    rec_name = aecg['filename']
    rec_dir = os.path.join(OUT_DIR, rec_name)
    os.makedirs(rec_dir, exist_ok=True)

    signals = aecg['signals']
    fs = aecg['fs']
    meas = aecg['measurements']
    meas['interpretation'] = aecg['interpretation']

    results = {}
    for lead_name in LEADS_TO_PROCESS:
        sig = signals.get(lead_name)
        if sig is None:
            results[lead_name] = None
            continue
        seg_result, clean, p_waves = process_one_lead(sig, fs, lead_name)
        r = save_lead_output(rec_name, rec_dir, lead_name, seg_result, clean, p_waves, fs, sig)
        results[lead_name] = r

    # Record-level summary
    summary = {'record': rec_name}
    for ln in LEADS_TO_PROCESS:
        summary[f'lead_{ln}'] = results.get(ln)
    summary['measurements'] = meas

    with open(os.path.join(rec_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, (np.integer,)) else float(o))

    return summary


def main():
    files = sorted([f for f in os.listdir(AECG_DIR) if f.endswith('.aECG')])[:N_FILES]
    print(f"{'='*62}")
    print(f"  P-QRS-T WAVEFORM PLOTS — Lead I + Lead II (refined boundaries)")
    print(f"  {N_FILES} records from RA-LA Reversal aECG")
    print(f"{'='*62}\n")

    summaries = []; t_start = time.time()

    for idx, fname in enumerate(files):
        print(f"[{idx+1:2d}/{N_FILES}] {fname[:14]}...", end=" ", flush=True)
        t0 = time.time(); s = process_record(fname); dt = time.time() - t0
        if s:
            summaries.append(s)
            li = s.get('lead_I', {}) or {}
            lii = s.get('lead_II', {}) or {}
            print(f"OK  I: beats={li.get('n_beats','?')} plots={li.get('n_plotted','?')} | "
                  f"II: beats={lii.get('n_beats','?')} plots={lii.get('n_plotted','?')} ({dt:.0f}s)")
        else:
            print("SKIP")
        gc.collect()

    total_time = time.time() - t_start

    total_I_beats = sum((s.get('lead_I') or {}).get('n_beats', 0) for s in summaries)
    total_II_beats = sum((s.get('lead_II') or {}).get('n_beats', 0) for s in summaries)
    total_I_plots = sum((s.get('lead_I') or {}).get('n_plotted', 0) for s in summaries)
    total_II_plots = sum((s.get('lead_II') or {}).get('n_plotted', 0) for s in summaries)

    print(f"\n{'='*62}")
    print(f"  COMPLETE — Lead I + Lead II")
    print(f"{'='*62}")
    print(f"  Records: {len(summaries)}/{N_FILES}")
    print(f"  Lead I:  {total_I_beats} beats, {total_I_plots} waveform plots")
    print(f"  Lead II: {total_II_beats} beats, {total_II_plots} waveform plots")
    print(f"  Time: {total_time/60:.1f} min")
    print(f"  Output: {OUT_DIR}/")
    print(f"{'='*62}")
    print(f"\n  Per-record structure:")
    print(f"    {OUT_DIR}/<record>/")
    print(f"      summary.json")
    print(f"      lead_I/")
    print(f"        segmentation.png, p_qrs_t_wave.json")
    print(f"        raw_ecg.npy, filtered_ecg.npy, state_labels.npy")
    print(f"        beats/beat_###_waveform.png")
    print(f"      lead_II/")
    print(f"        (same structure)")


if __name__ == '__main__':
    main()
