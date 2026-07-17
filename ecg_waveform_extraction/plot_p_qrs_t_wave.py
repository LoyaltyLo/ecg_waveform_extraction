"""Generate complete P-QRS-T waveform plots for all beats in first 50 RA-LA records.

Uses:
  - HSMM 9-state Viterbi for waveform segmentation
  - Optimized P-wave extraction (refine_boundaries=True)
  - Optimized QRS extraction (refine_qrs_boundaries=True)

Output matches output_test_only format:
  output_rala_full/_p_qrs_t_wave/
    {record}/
      segmentation.png       — full waveform with state colors
      beats/
        beat_###_waveform.png — P-QRS-T complete waveform with boundary markers
"""

import sys
sys.path.insert(0, 'c:/LoyaltyLo/PythonProjects/ECG_engineering')

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
from ecg_waveform_extraction.extraction import PWaveExtractor
from ecg_waveform_extraction.hsmm.hsmm_model import STATE_LABELS
from ecg_waveform_extraction.utils.vis import STATE_COLORS
from ecg_waveform_extraction.qrs_polarity import refine_qrs_boundaries

AECG_DIR = 'C:/LoyaltyLo/datasets/RA-LA_Reversal/aECG'
OUT_DIR = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/output_rala_full/_p_qrs_t_wave'
os.makedirs(OUT_DIR, exist_ok=True)
N_FILES = 50
MAX_SAMPLES = 4000
MAX_BEATS_PER_RECORD = 8  # max per-beat plots per record


def parse_signal(filepath):
    with open(filepath, 'rb') as f: raw = f.read()
    content = raw.decode('utf-8', errors='replace')
    fs = 1000.0
    m = re.search(rb'<increment[^>]*value="([^"]+)"[^>]*unit="s"', raw)
    if m: fs = 1.0 / float(m.group(1))
    ss = content.find('<sequenceSet'); se = content.find('</sequenceSet>', ss)
    digits = re.findall(r'<digits[^>]*>([^<]+)</digits>', content[ss:se])
    lead_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF']
    signals = {}
    for i, name in enumerate(lead_names):
        if i < len(digits):
            sig = np.array([float(x) for x in digits[i].split()], dtype=np.float64)
            signals[name] = sig[:MAX_SAMPLES]
    # Measurements
    meas = {}
    for key, pat in {
        'HR': 'HEART_RATE.*?value="([^"]+)"',
        'QRS_dur': 'TIME_PD_QRS\b(?!c).*?value="([^"]+)"',
        'P_dur': 'TIME_PD_P\b(?!R).*?value="([^"]+)"',
    }.items():
        m = re.search(pat.encode(), raw, re.DOTALL)
        meas[key] = float(m.group(1)) if m else None
    interp = re.search(rb'INTERPRETATION_STATEMENT.*?xsi:type="ST"[^>]*>([^<]+)</value>', raw, re.DOTALL)
    meas['interpretation'] = (interp.group(1).decode('utf-8',errors='replace').strip().replace('\n','; ')
                              if interp else '')
    return signals, fs, meas


def process_record(fname):
    fpath = os.path.join(AECG_DIR, fname)
    rec_name = fname.replace('.aECG', '')
    rec_dir = os.path.join(OUT_DIR, rec_name)
    beats_dir = os.path.join(rec_dir, 'beats')
    os.makedirs(beats_dir, exist_ok=True)

    signals, fs, meas = parse_signal(fpath)
    sig_II = signals.get('II')
    if sig_II is None: return None

    n = min(len(sig_II), MAX_SAMPLES)
    sig = sig_II[:n].astype(np.float64)

    # ---- Full HSMM pipeline ----
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

    # ---- P-wave extraction ----
    p_ext = PWaveExtractor(fs=fs, refine_boundaries=True, enable_template_fallback=True)
    p_waves = p_ext.extract(seg_result)
    pw_map = {pw.beat_id: pw for pw in p_waves}

    # ---- Save numpy data ----
    np.save(os.path.join(rec_dir, 'raw_ecg.npy'), sig)
    np.save(os.path.join(rec_dir, 'filtered_ecg.npy'), clean)
    np.save(os.path.join(rec_dir, 'state_labels.npy'), seg_result.state_labels)

    # ---- Per-beat JSON ----
    beats_json = []
    for b in seg_result.beats:
        pw = pw_map.get(b.beat_id)
        # QRS refinement
        if b.q_onset > 0 and b.s_offset > 0 and b.r_peak > 0:
            q_on, r_pk, s_off = refine_qrs_boundaries(clean, b.q_onset, b.r_peak, b.s_offset, fs)
        else:
            q_on, r_pk, s_off = b.q_onset, b.r_peak, b.s_offset

        entry = {'beat_id': b.beat_id}
        for name, val in [('p_onset', b.p_onset), ('p_offset', b.p_offset),
                           ('q_onset', q_on), ('r_peak', r_pk), ('s_offset', s_off),
                           ('t_onset', b.t_onset), ('t_offset', b.t_offset)]:
            entry[name] = int(val) if isinstance(val, (int, np.integer)) else (int(val) if val and val > 0 else -1)

        if pw and pw.onset_sample > 0:
            entry['p_wave'] = {
                'onset_sample': pw.onset_sample, 'offset_sample': pw.offset_sample,
                'peak_sample': pw.peak_sample, 'duration_ms': pw.duration_ms,
                'confidence': pw.confidence, 'morphology': pw.morphology,
            }
        beats_json.append(entry)
    with open(os.path.join(rec_dir, 'p_qrs_t_wave.json'), 'w') as f:
        json.dump(beats_json, f, indent=2, default=lambda o: int(o) if isinstance(o, (np.integer,)) else float(o))

    # ---- Segmentation overview plot ----
    T = len(clean)
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
    ax.set_title(f'{rec_name} — HSMM P-QRS-T Segmentation (Lead II)')
    handles = [Rectangle((0,0),1,1,facecolor=STATE_COLORS[s],alpha=0.25,label=s) for s in STATE_LABELS]
    ax.legend(handles=handles, loc='upper right', ncol=9, fontsize=5)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(rec_dir, 'segmentation.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ---- Per-beat P-QRS-T waveform plots ----
    n_plotted = 0
    for b in seg_result.beats:
        if n_plotted >= MAX_BEATS_PER_RECORD: break
        if b.p_onset <= 0 or b.t_offset <= 0: continue

        bid = b.beat_id
        pw = pw_map.get(bid)

        # Refined QRS
        if b.q_onset > 0 and b.s_offset > 0 and b.r_peak > 0:
            q_on, r_pk, s_off = refine_qrs_boundaries(clean, b.q_onset, b.r_peak, b.s_offset, fs)
        else:
            q_on, r_pk, s_off = b.q_onset, b.r_peak, b.s_offset

        margin = int(0.15 * fs)
        ws = max(0, b.p_onset - margin)
        we = min(T - 1, b.t_offset + margin)
        if we - ws < 30: continue

        fig, ax = plt.subplots(figsize=(12, 4))
        t_win = np.arange(ws, we + 1) / fs
        e_win = clean[ws:we + 1]
        l_win = seg_result.state_labels[ws:we + 1]

        # State color bands
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

        # P-wave refined boundaries
        if pw and pw.onset_sample > 0:
            ax.axvline(pw.onset_sample / fs, color='green', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.axvline(pw.offset_sample / fs, color='green', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.text(pw.onset_sample / fs, ylo - 0.04 * yr, 'P↑', fontsize=7, color='green', ha='center')
            ax.text(pw.offset_sample / fs, ylo - 0.04 * yr, 'P↓', fontsize=7, color='green', ha='center')
        else:
            # Use Stage 1 boundaries
            if b.p_onset > 0:
                ax.axvline(b.p_onset / fs, color='green', linestyle='--', linewidth=0.6, alpha=0.5)
            if b.p_offset > 0:
                ax.axvline(b.p_offset / fs, color='green', linestyle='--', linewidth=0.6, alpha=0.5)

        # QRS refined boundaries
        if q_on > 0:
            ax.axvline(q_on / fs, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.text(q_on / fs, ylo - 0.10 * yr, 'QRS↑', fontsize=7, color='red', ha='center')
        if s_off > 0:
            ax.axvline(s_off / fs, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.text(s_off / fs, ylo - 0.10 * yr, 'QRS↓', fontsize=7, color='red', ha='center')

        # T-wave boundary
        if b.t_offset > 0:
            ax.axvline(b.t_offset / fs, color='blue', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.text(b.t_offset / fs, yhi + 0.04 * yr, 'T↓', fontsize=7, color='blue', ha='center')

        # P-wave info box
        if pw and pw.onset_sample > 0:
            pw_mid = (pw.onset_sample + pw.offset_sample) // 2
            if ws <= pw_mid <= we:
                info = (f'P: {pw.duration_ms:.0f}ms | conf={pw.confidence:.2f} | {pw.morphology} | SNR={pw.snr_db:.1f}dB')
                ax.text(pw_mid / fs, yhi + 0.10 * yr, info, fontsize=7, ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5e9', alpha=0.85))

        # QRS info box
        if q_on > 0 and s_off > 0:
            qrs_seg = clean[q_on:s_off + 1]
            bl = float(np.mean(clean[max(0,q_on-30):q_on])) if q_on >= 30 else float(np.median(qrs_seg[:5]))
            qrs_net = float(np.sum(qrs_seg - bl))
            r_val = float(clean[r_pk] - bl) if 0 <= r_pk < T else 0.0
            s_val = float(np.min(qrs_seg[r_pk-q_on:] - bl)) if r_pk > q_on else 0.0
            rs = r_val / max(abs(s_val), 0.001) if r_val > 0 else -1
            dur = (s_off - q_on) / fs * 1000
            pol = '↑' if qrs_net > 0 else ('↓' if qrs_net < 0 else '±')
            info2 = f'QRS: {dur:.0f}ms | R/S={rs:.1f} | net={qrs_net:.0f} | {pol}'
            ax.text(r_pk / fs, ylo - 0.16 * yr, info2, fontsize=7, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#fce4ec', alpha=0.85))

        ax.set_xlim(t_win[0], t_win[-1]); ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude')
        ax.set_title(f'{rec_name} — Beat {bid} — P-QRS-T Waveform (Lead II)')
        handles = [Rectangle((0,0),1,1,facecolor=STATE_COLORS[s],alpha=0.25,label=s) for s in STATE_LABELS]
        ax.legend(handles=handles, loc='upper right', ncol=9, fontsize=5)
        ax.grid(True, alpha=0.15)
        fig.tight_layout()
        fig.savefig(os.path.join(beats_dir, f'beat_{bid:03d}_waveform.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)
        n_plotted += 1

    return {
        'record': rec_name, 'n_beats': len(seg_result.beats),
        'n_plotted': n_plotted, 'n_p_waves': len(p_waves),
    }


def main():
    files = sorted([f for f in os.listdir(AECG_DIR) if f.endswith('.aECG')])[:N_FILES]
    print(f"{'='*60}")
    print(f"  P-QRS-T WAVEFORM PLOTS (Lead II, refined boundaries)")
    print(f"  {N_FILES} records from RA-LA Reversal aECG")
    print(f"{'='*60}\n")

    summaries = []; t_start = time.time()
    total_beats = 0; total_plotted = 0

    for idx, fname in enumerate(files):
        print(f"[{idx+1:2d}/{N_FILES}] {fname[:14]}...", end=" ", flush=True)
        t0 = time.time(); s = process_record(fname); dt = time.time() - t0
        if s:
            summaries.append(s); total_beats += s['n_beats']; total_plotted += s['n_plotted']
            print(f"OK beats={s['n_beats']} plotted={s['n_plotted']} P-waves={s['n_p_waves']} ({dt:.0f}s)")
        else: print("SKIP")
        gc.collect()

    total_time = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"  COMPLETE")
    print(f"{'='*60}")
    print(f"  Records: {len(summaries)}/{N_FILES}")
    print(f"  Total beats: {total_beats}")
    print(f"  Total waveform plots: {total_plotted}")
    print(f"  Time: {total_time/60:.1f} min")
    print(f"  Output: {OUT_DIR}/")
    print(f"{'='*60}")
    print(f"\n  Per-record format:")
    print(f"    segmentation.png — full waveform with 9-state color bands")
    print(f"    p_qrs_t_wave.json — per-beat boundary data + P-wave metrics")
    print(f"    raw_ecg.npy, filtered_ecg.npy, state_labels.npy")
    print(f"    beats/beat_###_waveform.png — complete P-QRS-T waveform\n"
          f"      with color bands, P(↑↓)/QRS(↑↓)/T(↓) markers, P-wave info box, QRS info box")


if __name__ == '__main__':
    main()
