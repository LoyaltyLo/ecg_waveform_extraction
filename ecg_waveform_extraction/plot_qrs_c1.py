"""QRS C1 Polarity Plotting — Dominant Deflection with Peak Markers.

For each beat, plots the QRS segment with:
  - Red dot + value at the highest point (R peak / positive deflection)
  - Blue dot + value at the lowest point (S/Q nadir / negative deflection)
  - C1 polarity label: positive (|max| >= |min|) or negative (|min| > |max|)
  - Confidence based on the asymmetry ratio

Generates:
  - Per-lead QRS overview with all beats
  - Per-beat detail zoom with max/min annotations
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Compute C1 polarity for a single QRS segment
# ---------------------------------------------------------------------------
def compute_c1(seg_detrend):
    """Compute C1 (dominant deflection) polarity.

    Returns dict with polarity, confidence, max_val, max_idx, min_val, min_idx.
    """
    if len(seg_detrend) < 3:
        return {'polarity': '?', 'confidence': 0.0,
                'max_val': 0, 'max_idx': 0, 'min_val': 0, 'min_idx': 0}

    pos_max = float(np.max(seg_detrend))
    neg_min = float(np.min(seg_detrend))
    pos_idx = int(np.argmax(seg_detrend))
    neg_idx = int(np.argmin(seg_detrend))

    abs_pos = abs(pos_max)
    abs_neg = abs(neg_min)
    total = abs_pos + abs_neg + 1e-12

    if abs_pos >= abs_neg:
        polarity = 'positive'
        confidence = abs_pos / total
    else:
        polarity = 'negative'
        confidence = abs_neg / total

    return {
        'polarity': polarity,
        'confidence': round(confidence, 3),
        'max_val': round(pos_max, 4),
        'max_idx': pos_idx,
        'min_val': round(neg_min, 4),
        'min_idx': neg_idx,
    }


# ---------------------------------------------------------------------------
# QRS Detail Plot: single beat
# ---------------------------------------------------------------------------
def plot_qrs_c1_beat(ecg_clean, q_on, r_pk, s_off, fs, lead_name,
                     beat_id, rec_name, save_path, dpi=130):
    """Single QRS beat with max/min markers and C1 polarity label.

    Parameters
    ----------
    ecg_clean : np.ndarray — filtered ECG
    q_on, r_pk, s_off : int — QRS boundaries
    fs : float
    lead_name : str
    beat_id : int
    rec_name : str
    save_path : str
    """
    margin = int(0.05 * fs)  # 50ms padding each side
    T = len(ecg_clean)
    ws = max(0, q_on - margin)
    we = min(T - 1, s_off + margin)

    if we - ws < 10:
        return

    t_win = np.arange(ws, we + 1) / fs
    e_win = ecg_clean[ws:we + 1]

    # QRS segment (offset to window)
    qrs_s = q_on - ws
    qrs_e = s_off - ws
    detrend_qrs = ecg_clean[q_on:s_off + 1]

    # Baseline from pre-QRS
    bl = float(np.median(ecg_clean[max(0, q_on - 30):q_on])) if q_on >= 30 \
         else float(np.median(detrend_qrs[:5]))
    detrend = detrend_qrs - bl

    c1 = compute_c1(detrend)

    fig, ax = plt.subplots(figsize=(8, 3.5))

    # Full window trace
    ax.plot(t_win, e_win, 'k-', linewidth=0.8, alpha=0.5)

    # QRS segment highlighted
    t_qrs = np.arange(q_on, s_off + 1) / fs
    e_qrs = ecg_clean[q_on:s_off + 1]
    qrs_color = '#4caf50' if c1['polarity'] == 'positive' else '#f44336'
    ax.plot(t_qrs, e_qrs, color=qrs_color, linewidth=1.2)
    ax.fill_between(t_qrs, e_qrs, alpha=0.12, color=qrs_color, linewidth=0)

    # Max point (R peak or positive peak) — RED
    max_sample = q_on + c1['max_idx']
    if ws <= max_sample <= we:
        ax.plot(max_sample / fs, ecg_clean[max_sample], 'rv', markersize=12,
                markeredgecolor='darkred', markeredgewidth=1.5, zorder=5)
        ax.annotate(f'{c1["max_val"]:.2f}',
                    (max_sample / fs, ecg_clean[max_sample]),
                    textcoords='offset points', xytext=(10, 8),
                    fontsize=9, color='darkred', fontweight='bold')

    # Min point (S/Q nadir) — BLUE
    min_sample = q_on + c1['min_idx']
    if ws <= min_sample <= we:
        ax.plot(min_sample / fs, ecg_clean[min_sample], 'bv', markersize=12,
                markeredgecolor='darkblue', markeredgewidth=1.5, zorder=5)
        ax.annotate(f'{c1["min_val"]:.2f}',
                    (min_sample / fs, ecg_clean[min_sample]),
                    textcoords='offset points', xytext=(10, -14),
                    fontsize=9, color='darkblue', fontweight='bold')

    # C1 polarity info box
    pol_label = 'POSITIVE' if c1['polarity'] == 'positive' else 'NEGATIVE'
    box_color = '#c8e6c9' if c1['polarity'] == 'positive' else '#ffcdd2'
    info = (f'QRS C1: {pol_label}\n'
            f'Confidence: {c1["confidence"]:.2f}\n'
            f'Max: {c1["max_val"]:.3f}  |Min|: {abs(c1["min_val"]):.3f}')
    ax.text(0.02, 0.95, info, transform=ax.transAxes,
            fontsize=9, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=box_color, alpha=0.85))

    ax.set_xlim(t_win[0], t_win[-1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (norm)')
    ax.set_title(f'{rec_name} — Lead {lead_name} Beat {beat_id}  |  '
                 f'QRS C1 Polarity', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.12)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# QRS Overview Plot: all beats in one lead
# ---------------------------------------------------------------------------
def plot_qrs_c1_lead_overview(ecg_clean, beats_data, fs, lead_name,
                               rec_name, save_path, max_sec=4.0, dpi=150):
    """Lead overview with QRS segments colored by C1 polarity.

    Parameters
    ----------
    ecg_clean : np.ndarray
    beats_data : list[dict] — each with q_onset, r_peak, s_offset, polarity
    fs : float
    lead_name : str
    rec_name : str
    save_path : str
    """
    T = len(ecg_clean)
    n_plot = min(int(max_sec * fs), T)
    t_plot = np.arange(n_plot) / fs
    e_plot = ecg_clean[:n_plot]

    fig, ax = plt.subplots(figsize=(14, 3.5))

    ax.plot(t_plot, e_plot, 'k-', linewidth=0.4, alpha=0.8)

    pos_count = 0
    neg_count = 0
    for b in beats_data:
        q, s = b['q_onset'], b['s_offset']
        if q >= n_plot or s >= n_plot:
            continue
        if s <= q:
            continue

        color = '#4caf50' if b['polarity'] == 'positive' else '#f44336'
        ax.fill_between(t_plot[q:s + 1], e_plot[q:s + 1],
                        alpha=0.30, color=color, linewidth=0)

        # R peak marker
        r = b['r_peak']
        if r < n_plot:
            ax.plot(r / fs, e_plot[r], 'v',
                    color='darkred' if b['polarity'] == 'positive' else 'darkblue',
                    markersize=6, alpha=0.9)

        if b['polarity'] == 'positive':
            pos_count += 1
        else:
            neg_count += 1

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4caf50', alpha=0.3, label=f'Positive ({pos_count})'),
        Patch(facecolor='#f44336', alpha=0.3, label=f'Negative ({neg_count})'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    ax.set_xlim(t_plot[0], t_plot[-1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (norm)')
    ax.set_title(f'{rec_name} — Lead {lead_name} QRS C1 Polarity '
                 f'(+{pos_count} / -{neg_count})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.12)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Save all QRS C1 plots for one record
# ---------------------------------------------------------------------------
def save_qrs_c1_plots(seg_data, ll_result, rec_name, rec_dir, dpi=130):
    """Generate all QRS C1 plots for one record.

    For each lead:
      - lead_{name}/qrs_c1_overview.png — all beats
      - lead_{name}/qrs_c1_beats/beat_{###}.png — per-beat detail

    Parameters
    ----------
    seg_data : dict — lead_name -> {filtered_ecg, fs, ...}
    ll_result : LimbLeadResult
    rec_name : str
    rec_dir : str
    """
    from ecg_waveform_extraction.limb_lead_processor import LIMB_LEADS

    for lead_name in LIMB_LEADS:
        sd = seg_data.get(lead_name)
        lr = ll_result.leads.get(lead_name)
        if sd is None or lr is None or lr.n_beats == 0:
            continue

        ecg = sd['filtered_ecg']
        fs = sd.get('fs', 250.0)
        beats = lr.beats

        lead_dir = os.path.join(rec_dir, f'lead_{lead_name}')
        beats_dir = os.path.join(lead_dir, 'qrs_c1_beats')
        os.makedirs(beats_dir, exist_ok=True)

        # Lead overview
        plot_qrs_c1_lead_overview(ecg, beats, fs, lead_name, rec_name,
                                  os.path.join(lead_dir, 'qrs_c1_overview.png'),
                                  dpi=dpi)

        # Per-beat detail (up to 16)
        for i, b in enumerate(beats[:16]):
            q_on = b['q_onset']
            r_pk = b['r_peak']
            s_off = b['s_offset']
            if q_on <= 0 or s_off <= q_on:
                continue
            plot_qrs_c1_beat(ecg, q_on, r_pk, s_off, fs, lead_name,
                             b['beat_id'], rec_name,
                             os.path.join(beats_dir, f'beat_{b["beat_id"]:03d}.png'),
                             dpi=dpi)
