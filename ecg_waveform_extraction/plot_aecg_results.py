"""Generate comprehensive result plots from RA-LA aECG dataset processing.

Reads all completed result.json files from output_rala/ and generates:
  1. Bland-Altman plots (QRS dur, P dur, PR interval)
  2. Scatter plots: HSMM vs ANN with correlation
  3. Error distribution histograms
  4. Per-interpretation-category breakdown
  5. Best/worst example waveforms
"""

import sys
sys.path.insert(0, 'c:/LoyaltyLo/PythonProjects/ECG_engineering')

import os, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---- Config ----
OUT = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/output_rala'
PLOTS_DIR = os.path.join(OUT, '_summary_plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---- Load data ----
results = []
for dname in sorted(os.listdir(OUT)):
    rpath = os.path.join(OUT, dname, 'result.json')
    if not os.path.exists(rpath): continue
    try:
        with open(rpath) as f: r = json.load(f)
        if 'error' not in r: results.append(r)
    except: pass

print(f"Loaded {len(results)} complete results")

# Extract metrics
hsmm_p = []; ann_p = []
hsmm_qrs = []; ann_qrs = []; qrs_errs = []
hsmm_pr = []; ann_pr = []
hr_bpm = []
interps = defaultdict(list)

for r in results:
    hm = r.get('hsmm_metrics', {})
    gm = r.get('global_measurements', {})

    hp = hm.get('hsmm_mean_p_dur_ms')
    ap = r.get('ann_P_dur_ms')
    if hp and ap: hsmm_p.append(hp); ann_p.append(ap)

    hq = hm.get('hsmm_mean_qrs_dur_ms')
    aq = r.get('ann_QRS_dur_ms')
    if hq and aq:
        hsmm_qrs.append(hq); ann_qrs.append(aq)
        e = hm.get('best_match_qrs_dur_err_ms')
        if e is not None: qrs_errs.append(e)

    hpr = hm.get('hsmm_mean_pr_ms')
    apr = r.get('ann_PR_ms')
    if hpr and apr: hsmm_pr.append(hpr); ann_pr.append(apr)

    hr = gm.get('HR_bpm')
    if hr: hr_bpm.append(hr)

    interp = r.get('interpretation', '')
    interps[interp].append(r)

hsmm_p = np.array(hsmm_p); ann_p = np.array(ann_p)
hsmm_qrs = np.array(hsmm_qrs); ann_qrs = np.array(ann_qrs)
qrs_errs = np.array(qrs_errs)
hsmm_pr = np.array(hsmm_pr); ann_pr = np.array(ann_pr)

n = len(qrs_errs)
print(f"Matched records: {n}")
print(f"QRS err median: {np.median(qrs_errs):.1f}ms, mean: {np.mean(qrs_errs):.1f}ms")


# =====================================================================
# Figure 1: Bland-Altman + Scatter (3x2 grid)
# =====================================================================
def bland_altman(ax, x, y, label, unit='ms'):
    """Bland-Altman plot: difference vs mean."""
    diff = x - y
    mean_ = (x + y) / 2
    md = np.mean(diff)
    sd = np.std(diff)
    ax.scatter(mean_, diff, s=8, alpha=0.4, c='#2196f3', edgecolors='none')
    ax.axhline(md, color='red', linestyle='-', linewidth=1.5, label=f'Bias={md:.1f}{unit}')
    ax.axhline(md + 1.96*sd, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(md - 1.96*sd, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xlabel(f'Mean {label} ({unit})')
    ax.set_ylabel(f'HSMM - ANN ({unit})')
    ax.set_title(f'Bland-Altman: {label}')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)
    return md, sd

def scatter_plot(ax, x, y, label, unit='ms'):
    """Scatter + identity line + correlation."""
    ax.scatter(x, y, s=8, alpha=0.4, c='#4caf50', edgecolors='none')
    mn = min(x.min(), y.min()) * 0.9
    mx = max(x.max(), y.max()) * 1.1
    ax.plot([mn, mx], [mn, mx], 'k--', linewidth=0.8, alpha=0.5)
    r = np.corrcoef(x, y)[0, 1]
    mae = np.mean(np.abs(x - y))
    ax.set_xlabel(f'ANN {label} ({unit})')
    ax.set_ylabel(f'HSMM {label} ({unit})')
    ax.set_title(f'{label}: r={r:.3f}, MAE={mae:.1f}{unit}')
    ax.grid(True, alpha=0.2)

fig, axes = plt.subplots(3, 2, figsize=(14, 16))
fig.suptitle(f'RA-LA Reversal Dataset — HSMM vs Annotated Measurements (n={n})',
             fontsize=14, fontweight='bold', y=0.995)

# Row 1: QRS duration
bland_altman(axes[0, 0], hsmm_qrs, ann_qrs, 'QRS Duration')
scatter_plot(axes[0, 1], ann_qrs, hsmm_qrs, 'QRS Duration')

# Row 2: P-wave duration
bland_altman(axes[1, 0], hsmm_p, ann_p, 'P-wave Duration')
scatter_plot(axes[1, 1], ann_p, hsmm_p, 'P-wave Duration')

# Row 3: PR interval
bland_altman(axes[2, 0], hsmm_pr, ann_pr, 'PR Interval')
scatter_plot(axes[2, 1], ann_pr, hsmm_pr, 'PR Interval')

fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, '01_bland_altman_scatter.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: 01_bland_altman_scatter.png")


# =====================================================================
# Figure 2: Error distributions (3 histograms)
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Error Distributions', fontsize=13, fontweight='bold')

# QRS error
ax = axes[0]
qrs_err = hsmm_qrs - ann_qrs
ax.hist(qrs_err, bins=40, color='#2196f3', edgecolor='white', alpha=0.8)
ax.axvline(0, color='red', linestyle='--', linewidth=1)
ax.axvline(np.mean(qrs_err), color='darkred', linestyle='-', linewidth=1.5,
           label=f'Mean={np.mean(qrs_err):.1f}ms')
ax.set_xlabel('Error (ms)')
ax.set_ylabel('Count')
ax.set_title(f'QRS Duration Error')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, axis='y')

# P-wave error
ax = axes[1]
p_err = hsmm_p - ann_p
ax.hist(p_err, bins=40, color='#4caf50', edgecolor='white', alpha=0.8)
ax.axvline(0, color='red', linestyle='--', linewidth=1)
ax.axvline(np.mean(p_err), color='darkgreen', linestyle='-', linewidth=1.5,
           label=f'Mean={np.mean(p_err):.1f}ms')
ax.set_xlabel('Error (ms)')
ax.set_title(f'P-wave Duration Error')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, axis='y')

# PR error
ax = axes[2]
pr_err = hsmm_pr - ann_pr
ax.hist(pr_err, bins=40, color='#ff9800', edgecolor='white', alpha=0.8)
ax.axvline(0, color='red', linestyle='--', linewidth=1)
ax.axvline(np.mean(pr_err), color='darkorange', linestyle='-', linewidth=1.5,
           label=f'Mean={np.mean(pr_err):.1f}ms')
ax.set_xlabel('Error (ms)')
ax.set_title(f'PR Interval Error')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, axis='y')

fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, '02_error_histograms.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: 02_error_histograms.png")


# =====================================================================
# Figure 3: Metrics summary bar chart
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('HSMM vs Annotated — Mean ± Std', fontsize=13, fontweight='bold')

metrics = [
    ('QRS Duration (ms)', hsmm_qrs, ann_qrs),
    ('P-wave Duration (ms)', hsmm_p, ann_p),
    ('PR Interval (ms)', hsmm_pr, ann_pr),
]
colors = ['#2196f3', '#4caf50', '#ff9800']

for idx, (name, h_vals, a_vals) in enumerate(metrics):
    ax = axes[idx]
    x = np.arange(2)
    means = [np.mean(a_vals), np.mean(h_vals)]
    stds = [np.std(a_vals), np.std(h_vals)]
    bars = ax.bar(x, means, 0.5, yerr=stds, capsize=8,
                  color=['#9e9e9e', colors[idx]], edgecolor='white', linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(['Annotated', 'HSMM'])
    ax.set_ylabel('ms')
    ax.set_title(name)
    ax.grid(True, alpha=0.2, axis='y')
    # Add value labels
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stds)*0.3,
                f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')

fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, '03_mean_comparison.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: 03_mean_comparison.png")


# =====================================================================
# Figure 4: QRS error by HR category
# =====================================================================
if hr_bpm:
    hr_arr = np.array(hr_bpm[:len(qrs_errs)])
    qrs_err_arr = np.array(qrs_errs[:len(hr_arr)])

    # Categorize HR
    cats = {'<60': [], '60-80': [], '80-100': [], '>100': []}
    for h, e in zip(hr_arr, qrs_err_arr):
        if h < 60: cats['<60'].append(e)
        elif h < 80: cats['60-80'].append(e)
        elif h < 100: cats['80-100'].append(e)
        else: cats['>100'].append(e)

    fig, ax = plt.subplots(figsize=(10, 5))
    positions = []
    labels = []
    means = []
    for idx, (name, vals) in enumerate(cats.items()):
        if vals:
            positions.append(idx)
            labels.append(f'{name}\n(n={len(vals)})')
            means.append(np.mean(np.abs(vals)))

    bars = ax.bar(positions, means, color=['#2196f3', '#4caf50', '#ff9800', '#f44336'],
                  edgecolor='white', linewidth=1.2)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('|QRS Error| (ms)')
    ax.set_title(f'QRS Duration MAE by Heart Rate Category')
    ax.grid(True, alpha=0.2, axis='y')
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, '04_error_by_hr.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved: 04_error_by_hr.png")


# =====================================================================
# Figure 5: Cumulative error distribution
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 5))
sorted_errs = np.sort(np.abs(qrs_err))
cum_frac = np.arange(1, len(sorted_errs) + 1) / len(sorted_errs)

ax.plot(sorted_errs, cum_frac * 100, 'b-', linewidth=2)
ax.fill_between(sorted_errs, cum_frac * 100, alpha=0.15, color='#2196f3')

# Mark percentiles
for pct in [50, 75, 90, 95]:
    idx = int(pct / 100 * len(sorted_errs)) - 1
    if idx >= 0:
        ax.axhline(pct, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
        ax.axvline(sorted_errs[idx], color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
        ax.annotate(f'P{pct}={sorted_errs[idx]:.0f}ms',
                    (sorted_errs[idx], pct), textcoords='offset points',
                    xytext=(5, -12), fontsize=7, color='gray')

ax.set_xlabel('Absolute QRS Duration Error (ms)')
ax.set_ylabel('Cumulative Fraction (%)')
ax.set_title(f'Cumulative Error Distribution (n={n})')
ax.grid(True, alpha=0.2)
ax.set_xlim(0, min(150, sorted_errs[-1]))

fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, '05_cumulative_error.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: 05_cumulative_error.png")


# =====================================================================
# Figure 6: Best/Worst example waveforms
# =====================================================================
# Find best and worst QRS match records
qrs_err_records = []
for r in results:
    hm = r.get('hsmm_metrics', {})
    e = hm.get('best_match_qrs_dur_err_ms')
    if e is not None:
        qrs_err_records.append((e, r))

qrs_err_records.sort(key=lambda x: x[0])

best_5 = qrs_err_records[:5]
worst_5 = qrs_err_records[-5:]

# Plot 5 best + 5 worst
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
fig.suptitle('Best vs Worst QRS Duration Matching Examples', fontsize=14, fontweight='bold')

# Need to regenerate segmentation plots for these specific records
from ecg_waveform_extraction.preprocessing import ECGPreprocessor
from ecg_waveform_extraction.features import FeatureExtractor
from ecg_waveform_extraction.hsmm import HSMMModel, smart_initialize_gmms
from ecg_waveform_extraction.segmentation import ECGSegmenter
from ecg_waveform_extraction.utils.vis import plot_segmentation
import gc

AECG_DIR = 'C:/LoyaltyLo/datasets/RA-LA_Reversal/aECG'
all_files = {f.replace('.aECG',''): os.path.join(AECG_DIR, f)
             for f in os.listdir(AECG_DIR) if f.endswith('.aECG')}

for row_idx, examples in enumerate([best_5, worst_5]):
    row_label = "BEST (lowest error)" if row_idx == 0 else "WORST (highest error)"
    for col_idx, (qrs_err, rec) in enumerate(examples):
        ax = axes[row_idx, col_idx]
        rec_name = rec['record']

        fpath = all_files.get(rec_name)
        if not fpath:
            ax.text(0.5, 0.5, 'File not found', ha='center', va='center', transform=ax.transAxes)
            continue

        try:
            # Reload and re-segment
            from ecg_waveform_extraction.process_aecg_dataset import parse_aecg
            aecg = parse_aecg(fpath)
            sig = aecg['signals'].get('II')
            if sig is None:
                sig = next(iter(aecg['signals'].values()))
            fs = aecg['fs'] or 1000.0
            n_samp = min(len(sig), 4000)
            sig = sig[:n_samp].astype(np.float64)

            prep = ECGPreprocessor(fs=fs)
            clean = prep.preprocess(sig)
            fe = FeatureExtractor(fs=fs)
            features = fe.extract(clean)

            model = HSMMModel(fs=fs)
            model.initialize_with_priors()
            model.set_left_right_topology()
            smart_initialize_gmms(model, features)

            segmenter = ECGSegmenter(preprocessor=prep, feature_extractor=fe,
                                     model=model, fs=fs)
            seg_result = segmenter.segment(sig)

            # Compact plot
            t = np.arange(len(clean)) / fs
            lbls = seg_result.state_labels
            from ecg_waveform_extraction.hsmm.hsmm_model import STATE_LABELS
            from ecg_waveform_extraction.utils.vis import STATE_COLORS
            if len(lbls) > 0:
                prev = lbls[0]; seg_start = 0
                for i in range(1, len(lbls)):
                    if lbls[i] != prev:
                        c = STATE_COLORS.get(STATE_LABELS[prev] if 0<=prev<9 else 'UNKNOWN','#e0e0e0')
                        ax.axvspan(t[seg_start], t[i], alpha=0.2, color=c)
                        seg_start = i; prev = lbls[i]

            ax.plot(t, clean, 'k-', linewidth=0.5)

            # Mark R-peaks
            for b in seg_result.beats:
                if b.r_peak > 0:
                    ax.axvline(b.r_peak / fs, color='red', linewidth=0.4, alpha=0.5)

            ann = aecg.get('annotations', {})
            ann_qrs_on = ann.get('QRS_on_ms', 0)
            if ann_qrs_on:
                ax.axvline(ann_qrs_on / 1000, color='orange', linewidth=1, linestyle='--', alpha=0.8)

            a_on = ann.get('QRS_on_ms', '?')
            a_off = ann.get('QRS_off_ms', '?')
            ax.set_title(f'{rec_name[:12]}\nerr={qrs_err}ms  ANN_QRS={a_on}-{a_off}ms',
                         fontsize=7, fontweight='bold')
            ax.set_xlabel(''); ax.set_ylabel('')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(t[0], t[-1])

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='red')

        gc.collect()

    axes[row_idx, 0].set_ylabel(row_label, fontsize=10, fontweight='bold')

fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, '06_best_worst_examples.png'), dpi=120, bbox_inches='tight')
plt.close(fig)
print("Saved: 06_best_worst_examples.png")


# =====================================================================
# Figure 7: Final summary dashboard
# =====================================================================
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# (0,0): Key metrics table
ax0 = fig.add_subplot(gs[0, 0])
ax0.axis('off')
table_data = [
    ['Metric', 'ANN', 'HSMM', 'MAE'],
    ['QRS dur', f'{np.mean(ann_qrs):.1f}±{np.std(ann_qrs):.1f}',
     f'{np.mean(hsmm_qrs):.1f}±{np.std(hsmm_qrs):.1f}', f'{np.mean(np.abs(qrs_err)):.1f}'],
    ['P dur', f'{np.mean(ann_p):.1f}±{np.std(ann_p):.1f}',
     f'{np.mean(hsmm_p):.1f}±{np.std(hsmm_p):.1f}', f'{np.mean(np.abs(p_err)):.1f}'],
    ['PR', f'{np.mean(ann_pr):.1f}±{np.std(ann_pr):.1f}',
     f'{np.mean(hsmm_pr):.1f}±{np.std(hsmm_pr):.1f}', f'{np.mean(np.abs(pr_err)):.1f}'],
]
table = ax0.table(cellText=table_data, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.8)
for i in range(4):
    for j in range(4):
        cell = table[i, j]
        if i == 0:
            cell.set_facecolor('#333333')
            cell.set_text_props(color='white', fontweight='bold')
ax0.set_title('Key Metrics (ms)', fontsize=11, fontweight='bold', pad=20)

# (0,1): QRS duration Bland-Altman (compact)
ax1 = fig.add_subplot(gs[0, 1])
bland_altman(ax1, hsmm_qrs, ann_qrs, 'QRS Duration')

# (0,2): QRS error histogram
ax2 = fig.add_subplot(gs[0, 2])
ax2.hist(qrs_err, bins=35, color='#2196f3', edgecolor='white', alpha=0.85)
ax2.axvline(0, color='red', linestyle='--', linewidth=1)
ax2.set_xlabel('Error (ms)')
ax2.set_ylabel('Count')
ax2.set_title('QRS Duration Error Distribution')
ax2.grid(True, alpha=0.2, axis='y')

# (1,0): Scatter QRS
ax3 = fig.add_subplot(gs[1, 0])
scatter_plot(ax3, ann_qrs, hsmm_qrs, 'QRS Duration')

# (1,1): Scatter P-wave
ax4 = fig.add_subplot(gs[1, 1])
scatter_plot(ax4, ann_p, hsmm_p, 'P-wave Duration')

# (1,2): Scatter PR
ax5 = fig.add_subplot(gs[1, 2])
scatter_plot(ax5, ann_pr, hsmm_pr, 'PR Interval')

# (2,0): Cumulative error
ax6 = fig.add_subplot(gs[2, 0])
ax6.plot(sorted_errs, cum_frac * 100, 'b-', linewidth=2)
ax6.fill_between(sorted_errs, cum_frac * 100, alpha=0.12, color='#2196f3')
for pct in [50, 80, 90, 95]:
    idx = int(pct / 100 * len(sorted_errs)) - 1
    if idx >= 0:
        ax6.axhline(pct, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        ax6.axvline(sorted_errs[idx], color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
ax6.set_xlabel('|QRS Error| (ms)'); ax6.set_ylabel('Cumulative %')
ax6.set_title(f'Cumulative Error (P50={np.median(np.abs(qrs_err)):.0f}ms, P90={np.percentile(np.abs(qrs_err),90):.0f}ms)')
ax6.grid(True, alpha=0.2)
ax6.set_xlim(0, min(150, sorted_errs[-1]))

# (2,1): Error by HR
if hr_bpm:
    ax7 = fig.add_subplot(gs[2, 1])
    hr_arr2 = np.array(hr_bpm[:len(qrs_errs)])
    qrs_abs = np.abs(np.array(qrs_errs[:len(hr_arr2)]))
    cats2 = {'<60': [], '60-80': [], '80-100': [], '>100': []}
    for h, e in zip(hr_arr2, qrs_abs):
        if h < 60: cats2['<60'].append(e)
        elif h < 80: cats2['60-80'].append(e)
        elif h < 100: cats2['80-100'].append(e)
        else: cats2['>100'].append(e)
    pos2 = []; labs2 = []; means2 = []
    for name in ['<60', '60-80', '80-100', '>100']:
        if cats2[name]:
            pos2.append(len(pos2)); labs2.append(f'{name}\n(n={len(cats2[name])})')
            means2.append(np.mean(cats2[name]))
    bars = ax7.bar(pos2, means2, color=['#2196f3','#4caf50','#ff9800','#f44336'], edgecolor='white')
    ax7.set_xticks(pos2); ax7.set_xticklabels(labs2)
    ax7.set_ylabel('|QRS Error| (ms)')
    ax7.set_title('Error by Heart Rate')
    ax7.grid(True, alpha=0.2, axis='y')

# (2,2): Dataset summary text
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')
n_total_beats = sum(r['n_beats_detected'] for r in results)
text = (
    f"RA-LA Reversal aECG Dataset\n"
    f"{'─'*30}\n"
    f"Files processed: {len(results)}\n"
    f"Total HSMM beats: {n_total_beats}\n"
    f"Matched records: {n}\n\n"
    f"QRS Duration:\n"
    f"  Mean error: {np.mean(qrs_err):.1f} ms\n"
    f"  Median error: {np.median(qrs_err):.1f} ms\n"
    f"  Std error: {np.std(qrs_err):.1f} ms\n"
    f"  ≤10ms: {sum(1 for e in qrs_errs if e<=10)}/{n} "
    f"({sum(1 for e in qrs_errs if e<=10)/n*100:.1f}%)\n\n"
    f"P-wave Duration:\n"
    f"  MAE: {np.mean(np.abs(hsmm_p-ann_p)):.1f} ms\n\n"
    f"PR Interval:\n"
    f"  MAE: {np.mean(np.abs(hsmm_pr-ann_pr)):.1f} ms"
)
ax8.text(0.05, 0.95, text, transform=ax8.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))

fig.suptitle(f'HSMM ECG Waveform Extraction — RA-LA Reversal Dataset (n={n})',
             fontsize=15, fontweight='bold', y=1.01)
fig.savefig(os.path.join(PLOTS_DIR, '07_dashboard.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: 07_dashboard.png")


# =====================================================================
# Print text summary
# =====================================================================
print(f"\n{'='*60}")
print(f"  RA-LA REVERSAL DATASET — RESULTS SUMMARY")
print(f"{'='*60}")
print(f"  Files: {len(results)}")
print(f"  QRS Duration: MAE={np.mean(np.abs(qrs_err)):.1f}ms, Median={np.median(np.abs(qrs_err)):.0f}ms")
print(f"  QRS ≤10ms: {sum(1 for e in qrs_errs if e<=10)}/{n} ({sum(1 for e in qrs_errs if e<=10)/n*100:.1f}%)")
print(f"  QRS ≤20ms: {sum(1 for e in qrs_errs if e<=20)}/{n} ({sum(1 for e in qrs_errs if e<=20)/n*100:.1f}%)")
print(f"  P-wave MAE: {np.mean(np.abs(hsmm_p-ann_p)):.1f}ms")
print(f"  PR MAE: {np.mean(np.abs(hsmm_pr-ann_pr)):.1f}ms")
print(f"  Plots: {PLOTS_DIR}/")
print(f"{'='*60}")
