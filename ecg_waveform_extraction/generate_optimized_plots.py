"""Generate complete optimized P-wave result plots for first 50 RA-LA aECG records.

For each record:
  segmentation.png       — full waveform with HSMM state colors
  beats/beat_###_waveform.png  — per-beat P-QRS-T with boundary markers
  beats/beat_###_p_wave.png    — per-beat P-wave zoom with metrics overlay

Also generates:
  _summary/overview.png  — 5×5 grid of segmentation overviews
  _summary/dashboard.png — aggregate metrics dashboard
"""

import sys
sys.path.insert(0, 'c:/LoyaltyLo/PythonProjects/ECG_engineering')

import os, re, json, time, gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import Counter

from ecg_waveform_extraction.preprocessing import ECGPreprocessor
from ecg_waveform_extraction.features import FeatureExtractor
from ecg_waveform_extraction.hsmm import HSMMModel, smart_initialize_gmms
from ecg_waveform_extraction.segmentation import ECGSegmenter
from ecg_waveform_extraction.extraction import PWaveExtractor, PWaveAnalyzer
from ecg_waveform_extraction.hsmm.hsmm_model import STATE_LABELS
from ecg_waveform_extraction.utils.vis import STATE_COLORS

# ---- Config ----
AECG_DIR = 'C:/LoyaltyLo/datasets/RA-LA_Reversal/aECG'
OUT_DIR = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/output_rala_full'
PLOTS_DIR = os.path.join(OUT_DIR, '_optimized_plots')
os.makedirs(PLOTS_DIR, exist_ok=True)
N_FILES = 50
MAX_SAMPLES = 4000
DETAIL_PLOTS_MAX = 6  # max beats to plot per record

# =====================================================================
def parse_signal(filepath):
    """Extract Lead II + fs from aECG XML."""
    with open(filepath, 'rb') as f:
        raw = f.read()
    content = raw.decode('utf-8', errors='replace')
    fs = 1000.0
    m = re.search(rb'<increment[^>]*value="([^"]+)"[^>]*unit="s"', raw)
    if m: fs = 1.0 / float(m.group(1))
    ss = content.find('<sequenceSet')
    se = content.find('</sequenceSet>', ss)
    digits = re.findall(r'<digits[^>]*>([^<]+)</digits>', content[ss:se])
    sig = np.array([float(x) for x in digits[1].split()], dtype=np.float64) if len(digits)>=2 else None
    if sig is None: return None, None, None
    n = min(len(sig), MAX_SAMPLES)
    # Annotation info
    interp = re.search(r'MDC_ECG_INTERPRETATION_STATEMENT.*?xsi:type="ST"[^>]*>([^<]+)</value>', content, re.DOTALL)
    interp_text = interp.group(1).strip().replace('\n','; ') if interp else ''
    return sig[:n].astype(np.float64), fs, interp_text


# =====================================================================
def process_record(fname, rec_dir):
    """Full pipeline on one record, generate all plots."""
    fpath = os.path.join(AECG_DIR, fname)
    rec_name = fname.replace('.aECG', '')
    beats_dir = os.path.join(rec_dir, 'beats')
    os.makedirs(beats_dir, exist_ok=True)

    sig, fs, interp = parse_signal(fpath)
    if sig is None: return None

    # --- Pipeline ---
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

    # Optimized P-wave
    p_ext = PWaveExtractor(fs=fs, refine_boundaries=True, enable_template_fallback=True)
    p_waves = p_ext.extract(seg_result)

    # Map beat_id→p_wave
    pw_map = {pw.beat_id: pw for pw in p_waves}

    T = len(clean)
    n_beats = len(seg_result.beats)

    # ---- (A) Segmentation overview ----
    fig, ax = plt.subplots(figsize=(18, 4))
    plot_sec = min(T/fs, 3.8)
    t_plot = np.arange(int(plot_sec*fs)) / fs
    ecg_plot = clean[:len(t_plot)]
    lbl_plot = seg_result.state_labels[:len(t_plot)]

    # State color bands
    if len(lbl_plot) > 0:
        prev = lbl_plot[0]; seg_start = 0
        for i in range(1, len(lbl_plot)):
            if lbl_plot[i] != prev:
                c = STATE_COLORS.get(STATE_LABELS[prev] if 0<=prev<9 else 'UNKNOWN','#e0e0e0')
                ax.axvspan(t_plot[seg_start], t_plot[i], alpha=0.20, color=c)
                seg_start = i; prev = lbl_plot[i]
        c = STATE_COLORS.get(STATE_LABELS[prev] if 0<=prev<9 else 'UNKNOWN','#e0e0e0')
        ax.axvspan(t_plot[seg_start], t_plot[-1], alpha=0.20, color=c)

    ax.plot(t_plot, ecg_plot, 'k-', linewidth=0.6)

    # Mark detected R-peaks and P-waves
    for b in seg_result.beats:
        if b.r_peak > 0 and b.r_peak < len(t_plot):
            ax.axvline(b.r_peak/fs, color='red', linewidth=0.4, alpha=0.5)
        pw = pw_map.get(b.beat_id)
        if pw and pw.onset_sample > 0 and pw.morphology:
            mid = (pw.onset_sample + pw.offset_sample)//2
            if mid < len(t_plot):
                color_map = {'normal':'green','inverted':'orange','biphasic':'purple','peaked':'blue','absent':'gray'}
                c = color_map.get(pw.morphology, 'gray')
                ax.plot(mid/fs, clean[mid], 'o', markersize=4, color=c, alpha=0.7)

    ax.set_title(f"{rec_name}  |  {n_beats} beats  |  {len(p_waves)} P-waves  |  {interp[:60]}")
    ax.set_xlim(t_plot[0], t_plot[-1])
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude')
    handles = [Rectangle((0,0),1,1,facecolor=STATE_COLORS[s],alpha=0.25,label=s) for s in STATE_LABELS]
    ax.legend(handles=handles, loc='upper right', ncol=9, fontsize=5)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(rec_dir, 'segmentation.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ---- (B) Per-beat waveform + P-wave plots ----
    n_plotted = 0
    for b in seg_result.beats:
        if n_plotted >= DETAIL_PLOTS_MAX: break
        if b.p_onset <= 0 or b.t_offset <= 0: continue

        pw = pw_map.get(b.beat_id)
        bid = b.beat_id
        margin = int(0.15 * fs)
        ws = max(0, b.p_onset - margin)
        we = min(T-1, b.t_offset + margin)
        if we - ws < 30: continue

        # -- Waveform plot --
        fig, ax = plt.subplots(figsize=(12, 4))
        t_win = np.arange(ws, we+1)/fs
        e_win = clean[ws:we+1]
        l_win = seg_result.state_labels[ws:we+1]

        if len(l_win) > 0:
            prev = l_win[0]; seg_start = 0
            for i in range(1, len(l_win)):
                if l_win[i] != prev:
                    c = STATE_COLORS.get(STATE_LABELS[prev] if 0<=prev<9 else 'UNKNOWN','#e0e0e0')
                    ax.axvspan(t_win[seg_start], t_win[i], alpha=0.22, color=c)
                    seg_start = i; prev = l_win[i]
            c = STATE_COLORS.get(STATE_LABELS[prev] if 0<=prev<9 else 'UNKNOWN','#e0e0e0')
            ax.axvspan(t_win[seg_start], t_win[-1], alpha=0.22, color=c)

        ax.plot(t_win, e_win, 'k-', linewidth=0.7)

        ylo, yhi = e_win.min(), e_win.max()
        yr = max(yhi-ylo, 0.01)

        # Boundary markers
        for lbl, idx, color in [
            ('P↑', b.p_onset,'green'), ('P↓', b.p_offset,'green'),
            ('QRS↑', b.q_onset,'red'), ('QRS↓', b.s_offset,'red'),
            ('T↓', b.t_offset,'blue')]:
            if idx > 0 and ws <= idx <= we:
                tx = idx/fs
                ax.axvline(tx, color=color, linestyle='--', linewidth=0.6, alpha=0.6)
                ax.text(tx, yhi+0.03*yr, lbl, fontsize=6, color=color, ha='center')

        # P-wave metrics on plot
        if pw and pw.onset_sample > 0:
            pw_mid = (pw.onset_sample + pw.offset_sample)//2
            if ws <= pw_mid <= we:
                ax.text(pw_mid/fs, ylo-0.08*yr,
                        f'P:{pw.duration_ms:.0f}ms conf:{pw.confidence:.2f}\\nmorph:{pw.morphology} SNR:{pw.snr_db:.1f}dB',
                        fontsize=7, ha='center', va='top', color='#333333',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffcc', alpha=0.8))

        ax.set_title(f'{rec_name} Beat {bid}  |  P-QRS-T Waveform')
        ax.set_xlim(t_win[0], t_win[-1])
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude')
        handles = [Rectangle((0,0),1,1,facecolor=STATE_COLORS[s],alpha=0.25,label=s) for s in STATE_LABELS]
        ax.legend(handles=handles, loc='upper right', ncol=9, fontsize=5)
        ax.grid(True, alpha=0.15)
        fig.tight_layout()
        fig.savefig(os.path.join(beats_dir, f'beat_{bid:03d}_waveform.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

        # -- P-wave detail --
        if pw and pw.onset_sample > 0 and pw.offset_sample > pw.onset_sample:
            pw_on, pw_off = pw.onset_sample, pw.offset_sample
            pmg = int(0.1*fs)
            pws, pwe = max(0, pw_on-pmg), min(T-1, pw_off+pmg)
            if pws >= pwe: continue

            fig, ax = plt.subplots(figsize=(7, 3))
            t_pw = np.arange(pws, pwe+1)/fs
            ax.plot(t_pw, clean[pws:pwe+1], 'k-', linewidth=1.0)
            p_idx = np.arange(pw_on, pw_off+1)
            if len(p_idx) <= len(clean):
                ax.fill_between(p_idx/fs, clean[p_idx], alpha=0.25, color='#4caf50', label='P wave')
            ax.axvline(pw_on/fs, color='green', linestyle='--', linewidth=1.0)
            ax.axvline(pw_off/fs, color='red', linestyle='--', linewidth=1.0)
            if 0 <= pw.peak_sample < T:
                ax.axvline(pw.peak_sample/fs, color='blue', linestyle=':', linewidth=0.6)

            mid_t = (pw_on+pw_off)//2
            if 0 <= mid_t < T:
                ax.annotate(f'{pw.duration_ms:.0f}ms\\nconf={pw.confidence:.2f}\\n{pw.morphology}\\nSNR={pw.snr_db:.1f}dB\\nsym={pw.symmetry:.3f}',
                            (mid_t/fs, clean[mid_t]), textcoords='offset points', xytext=(0,12),
                            fontsize=8, ha='center', fontweight='normal',
                            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffffcc', alpha=0.85))

            ax.set_title(f'{rec_name} Beat {bid} — P-Wave Detail')
            ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude')
            ax.legend(fontsize=7); ax.grid(True, alpha=0.15)
            fig.tight_layout()
            fig.savefig(os.path.join(beats_dir, f'beat_{bid:03d}_p_wave.png'), dpi=120, bbox_inches='tight')
            plt.close(fig)

        n_plotted += 1

    # Return summary for dashboard
    analyzer = PWaveAnalyzer(fs=fs)
    p_feats = analyzer.analyze(p_waves, clean, seg_result.beats)
    ps = analyzer.summarize(p_feats)

    return {
        'record': rec_name,
        'n_total': ps.n_total, 'n_valid': ps.n_beats, 'n_absent': ps.n_absent,
        'P_dur': ps.duration_mean_ms, 'SNR': ps.mean_snr_db,
        'Sym': ps.mean_symmetry, 'Cons': ps.mean_consistency,
        'morph': dict(ps.morphology_distribution), 'qual': dict(ps.quality_distribution),
        'interp': interp,
    }


# =====================================================================
# Main
# =====================================================================
def main():
    files = sorted([f for f in os.listdir(AECG_DIR) if f.endswith('.aECG')])[:N_FILES]
    print(f"Generating complete plots for {N_FILES} records...")
    print(f"Output: {OUT_DIR}/_optimized_plots/<record>/")
    print()

    summaries = []
    t_start = time.time()

    for idx, fname in enumerate(files):
        rec_name = fname.replace('.aECG', '')
        rec_dir = os.path.join(PLOTS_DIR, rec_name)
        os.makedirs(rec_dir, exist_ok=True)

        print(f"[{idx+1:2d}/{N_FILES}] {rec_name}...", end=" ", flush=True)
        t0 = time.time()
        summary = process_record(fname, rec_dir)
        dt = time.time() - t0

        if summary:
            summaries.append(summary)
            print(f"OK beats={summary['n_total']} P_valid={summary['n_valid']} P_abs={summary['n_absent']} SNR={summary['SNR']:.1f}dB ({dt:.0f}s)")
        else:
            print(f"SKIP")

        gc.collect()

    total_time = time.time() - t_start

    # ---- Generate dashboard ----
    if summaries:
        generate_dashboard(summaries, total_time)

    print(f"\n{'='*60}")
    print(f"  Complete! {len(summaries)} records plotted")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"  Plots: {PLOTS_DIR}/")
    print(f"{'='*60}")


def generate_dashboard(summaries, total_time):
    """Generate aggregate dashboard and 5x5 overview grid."""
    n = len(summaries)
    if n == 0: return

    # ---- Dashboard ----
    snr_vals = [s['SNR'] for s in summaries if s['SNR'] > 0]
    sym_vals = [s['Sym'] for s in summaries if s['Sym'] > 0]
    dur_vals = [s['P_dur'] for s in summaries if s['P_dur']]
    all_morph = Counter()
    all_qual = Counter()
    for s in summaries:
        for k, v in s.get('morph', {}).items(): all_morph[k] += v
        for k, v in s.get('qual', {}).items(): all_qual[k] += v

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)

    # (0,0) Key metrics
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis('off')
    table_data = [
        ['Metric', 'Mean ± Std', 'Min', 'Max'],
        ['SNR (dB)', f'{np.mean(snr_vals):.1f}±{np.std(snr_vals):.1f}', f'{np.min(snr_vals):.1f}', f'{np.max(snr_vals):.1f}'],
        ['Symmetry', f'{np.mean(sym_vals):.3f}±{np.std(sym_vals):.3f}', f'{np.min(sym_vals):.3f}', f'{np.max(sym_vals):.3f}'],
        ['P dur (ms)', f'{np.mean(dur_vals):.1f}±{np.std(dur_vals):.1f}', f'{np.min(dur_vals):.1f}', f'{np.max(dur_vals):.1f}'],
    ]
    tbl = ax0.table(cellText=table_data, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1.15, 2.0)
    for i in range(4):
        for j in range(4):
            cell = tbl[i, j]
            if i == 0: cell.set_facecolor('#2F5496'); cell.set_text_props(color='white', fontweight='bold')
    ax0.set_title('Key Metrics', fontsize=12, fontweight='bold', pad=25)

    # (0,1) Morphology pie
    ax1 = fig.add_subplot(gs[0, 1])
    labels=[]; sizes=[]
    colors=['#4caf50','#f44336','#9c27b0','#2196f3','#9e9e9e','#ff9800']
    for (morph, cnt), c in zip(all_morph.most_common(), colors):
        labels.append(f'{morph}\\n({cnt})')
        sizes.append(cnt)
    wedges, texts = ax1.pie(sizes, labels=labels, colors=colors[:len(labels)], startangle=90)
    for t in texts: t.set_fontsize(8)
    ax1.set_title('Morphology Distribution', fontsize=11, fontweight='bold')

    # (0,2) Quality pie
    ax2 = fig.add_subplot(gs[0, 2])
    ql = []; qs = []
    for q, c in sorted(all_qual.items()):
        ql.append(f'{q}\\n({c})'); qs.append(c)
    qcols = ['#4caf50' if 'good' in q else '#ff9800' if 'fair' in q else '#f44336' for q, _ in sorted(all_qual.items())]
    ax2.pie(qs, labels=ql, colors=qcols, startangle=90)
    ax2.set_title('Quality Distribution', fontsize=11, fontweight='bold')

    # (1,:) SNR + Symmetry histogram side by side
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.hist(snr_vals, bins=20, color='#2196f3', edgecolor='white', alpha=0.8)
    ax3.axvline(np.mean(snr_vals), color='red', linestyle='-', linewidth=1.5, label=f'Mean={np.mean(snr_vals):.1f}dB')
    ax3.set_xlabel('SNR (dB)'); ax3.set_ylabel('Records')
    ax3.set_title('SNR Distribution'); ax3.legend(); ax3.grid(True, alpha=0.2, axis='y')

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(sym_vals, bins=20, color='#ff9800', edgecolor='white', alpha=0.8)
    ax4.axvline(np.mean(sym_vals), color='red', linestyle='-', linewidth=1.5, label=f'Mean={np.mean(sym_vals):.3f}')
    ax4.set_xlabel('Symmetry (0-1)'); ax4.set_ylabel('Records')
    ax4.set_title('Symmetry Distribution'); ax4.legend(); ax4.grid(True, alpha=0.2, axis='y')

    # (2,:) P-duration histogram + best/worst
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.hist(dur_vals, bins=20, color='#4caf50', edgecolor='white', alpha=0.8)
    ax5.axvline(np.mean(dur_vals), color='red', linestyle='-', linewidth=1.5, label=f'Mean={np.mean(dur_vals):.1f}ms')
    ax5.set_xlabel('P-wave Duration (ms)'); ax5.set_ylabel('Records')
    ax5.set_title('P-wave Duration Distribution'); ax5.legend(); ax5.grid(True, alpha=0.2, axis='y')

    # (2,2) Top 10 records
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    sorted_by_snr = sorted(summaries, key=lambda s: s['SNR'] if s['SNR'] else 0, reverse=True)
    text_lines = ['Best SNR:'] + [f'{s["record"][:12]} SNR={s["SNR"]:.1f}dB' for s in sorted_by_snr[:5]]
    text_lines += ['\\nWorst SNR:'] + [f'{s["record"][:12]} SNR={s["SNR"]:.1f}dB' for s in sorted_by_snr[-5:]]
    ax6.text(0.05, 0.95, '\\n'.join(text_lines), transform=ax6.transAxes, fontsize=8,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))

    fig.suptitle(f'Optimized P-Wave Extraction — RA-LA aECG (n={n} records)', fontsize=14, fontweight='bold', y=1.01)
    fig.savefig(os.path.join(PLOTS_DIR, '_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
