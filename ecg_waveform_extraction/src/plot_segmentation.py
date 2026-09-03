"""HSMM Waveform Segmentation Plotting.

Generates per-record and per-lead detailed waveform segmentation plots
showing P-QRS-T state boundaries annotated by the 9-state HSMM.

Plot Types
----------
  record_overview  — 3x2 grid of all 6 limb leads, state-colored, ~4s
  lead_overview    — Single lead full view with state colors + beat labels
  beat_detail      — Zoomed ~0.6s window per beat with boundary markers
  beat_detail_grid — All beats of one lead in a Nx2 grid

Usage
-----
    from ecg_waveform_extraction.src.plot_segmentation import (
        plot_lead_overview, plot_beat_detail, plot_record_overview,
        save_all_segmentation_plots,
    )
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# State colors (9-state HSMM)
# ---------------------------------------------------------------------------
STATE_COLORS = {
    'ISO': '#d0d0d0',   # grey — isoelectric
    'P':   '#42a5f5',   # blue — P wave
    'PR':  '#b3e5fc',   # light blue — PR segment
    'Q':   '#ff9800',   # orange — Q wave
    'R':   '#f44336',   # red — R wave
    'S':   '#e65100',   # dark orange — S wave
    'ST':  '#fff176',   # yellow — ST segment
    'T':   '#66bb6a',   # green — T wave
    'TP':  '#e0e0e0',   # light grey — TP segment
    'UNKNOWN': '#9e9e9e',
}

STATE_ORDER = ['ISO', 'P', 'PR', 'Q', 'R', 'S', 'ST', 'T', 'TP']

# Boundary marker style
BOUNDARY_STYLES = {
    'p_onset':   dict(color='#1565c0', ls='--', lw=1.2, label='P on'),
    'p_offset':  dict(color='#1565c0', ls='--', lw=1.2, label='P off'),
    'q_onset':   dict(color='#e65100', ls='--', lw=1.2, label='Q on'),
    'r_peak':    dict(color='#c62828', ls='-',  lw=1.8, label='R'),
    's_offset':  dict(color='#e65100', ls='--', lw=1.2, label='S off'),
    't_offset':  dict(color='#2e7d32', ls='--', lw=1.2, label='T off'),
}


# ---------------------------------------------------------------------------
# Record Overview: 3x2 grid of all 6 limb leads
# ---------------------------------------------------------------------------
def plot_record_overview(seg_data: dict, rec_name: str, fs: float,
                         save_path: str, max_sec: float = 4.0,
                         dpi: int = 150):
    """3×2 grid overview of all 6 limb leads with HSMM state coloring.

    Parameters
    ----------
    seg_data : dict
        lead_name -> dict with 'filtered_ecg', 'state_names', 'fs'
    rec_name : str
        Record identifier for the title.
    fs : float
        Sampling frequency.
    save_path : str
        Output PNG path.
    max_sec : float
        Maximum seconds to display.
    """
    from .limb_lead_processor import LEAD_PLOT_ORDER

    fig, axes = plt.subplots(3, 2, figsize=(20, 12))
    fig.suptitle(f'{rec_name} — 6-Lead HSMM Waveform Segmentation',
                 fontsize=14, fontweight='bold')

    for idx, lead_name in enumerate(LEAD_PLOT_ORDER):
        ax = axes[idx // 2, idx % 2]
        sd = seg_data.get(lead_name) if seg_data else None

        if sd is None or sd.get('filtered_ecg') is None:
            ax.text(0.5, 0.5, f'Lead {lead_name}\n(no data)',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, color='gray')
            ax.set_title(f'Lead {lead_name}', fontsize=12)
            ax.set_axis_off()
            continue

        ecg = sd['filtered_ecg']
        state_names = sd.get('state_names', [])
        lead_fs = sd.get('fs', fs)
        T = len(ecg)

        n_plot = min(int(max_sec * lead_fs), T)
        t_plot = np.arange(n_plot) / lead_fs
        e_plot = ecg[:n_plot]

        # Plot ECG trace
        ax.plot(t_plot, e_plot, 'k-', linewidth=0.4, alpha=0.9)

        # State-colored background
        if len(state_names) >= n_plot:
            _fill_states(ax, t_plot, e_plot, state_names[:n_plot])

        # Beat boundary markers
        beats = sd.get('beats', [])
        for b in beats:
            if b is None:
                continue
            _draw_beat_markers(ax, b, n_plot, lead_fs, e_plot)

        ax.set_xlim(t_plot[0], t_plot[-1])
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Amplitude (norm)', fontsize=9)
        ax.set_title(f'Lead {lead_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.15)

    # Shared legend
    legend_elements = [Patch(facecolor=STATE_COLORS[s], alpha=0.35, label=s)
                       for s in STATE_ORDER]
    fig.legend(handles=legend_elements, loc='lower center', ncol=9,
               fontsize=7, framealpha=0.8)

    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Lead Overview: Single lead with full annotation
# ---------------------------------------------------------------------------
def plot_lead_overview(seg_data: dict, lead_name: str, rec_name: str,
                       save_path: str, max_sec: float = 4.0, dpi: int = 150):
    """Single-lead full overview with state colors, beat numbers, and markers.

    Parameters
    ----------
    seg_data : dict
        Should contain 'filtered_ecg', 'state_names', 'beats', 'fs'.
    lead_name : str
    rec_name : str
    save_path : str
    max_sec : float
    """
    fig, ax = plt.subplots(figsize=(18, 4))

    ecg = seg_data['filtered_ecg']
    state_names = seg_data.get('state_names', [])
    beats = seg_data.get('beats', [])
    lead_fs = seg_data.get('fs', 250.0)
    T = len(ecg)

    n_plot = min(int(max_sec * lead_fs), T)
    t_plot = np.arange(n_plot) / lead_fs
    e_plot = ecg[:n_plot]

    # ECG trace
    ax.plot(t_plot, e_plot, 'k-', linewidth=0.5, alpha=0.9, zorder=1)

    # State colors
    if len(state_names) >= n_plot:
        _fill_states(ax, t_plot, e_plot, state_names[:n_plot])

    # Beat boundaries + annotations
    for b in beats:
        if b is None:
            continue
        _draw_beat_markers(ax, b, n_plot, lead_fs, e_plot, label_beat=True)

    # Legend
    legend_elements = [Patch(facecolor=STATE_COLORS[s], alpha=0.35, label=s)
                       for s in STATE_ORDER]
    ax.legend(handles=legend_elements, loc='upper right', ncol=9,
              fontsize=6.5, framealpha=0.7)

    ax.set_xlim(t_plot[0], t_plot[-1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (norm)')
    # Provenance note: P/T beat markers may be prominence-refined on top of
    # the HSMM states (delineation.prominence_stage).
    n_prom = seg_data.get('prominence_refined_beats', 0)
    prom_note = f', P/T prominence-refined: {n_prom}' if n_prom else ''
    ax.set_title(f'{rec_name} — Lead {lead_name} HSMM Segmentation '
                 f'({len(beats)} beats{prom_note})', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.12)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Beat Detail: Single beat zoomed view
# ---------------------------------------------------------------------------
def plot_beat_detail(seg_data: dict, beat, beat_idx: int, lead_name: str,
                     rec_name: str, save_path: str,
                     margin_ms: float = 200.0, dpi: int = 150):
    """Zoomed single-beat view with P-QRS-T boundary markers and state colors.

    Parameters
    ----------
    seg_data : dict
    beat : BeatBoundary
    beat_idx : int — beat number (0-based or beat_id)
    lead_name : str
    rec_name : str
    save_path : str
    margin_ms : float — padding before P-onset and after T-offset
    """
    ecg = seg_data['filtered_ecg']
    state_names = seg_data.get('state_names', [])
    lead_fs = seg_data.get('fs', 250.0)
    T = len(ecg)

    # Determine window: P-onset to T-offset + margin
    margin_samp = int(margin_ms / 1000.0 * lead_fs)
    p_on = beat.p_onset if beat.p_onset > 0 else beat.q_onset - int(0.16 * lead_fs)
    t_off = beat.t_offset if beat.t_offset > 0 else beat.s_offset + int(0.24 * lead_fs)

    ws = max(0, p_on - margin_samp)
    we = min(T - 1, t_off + margin_samp)

    if we - ws < 20:
        # Fallback: use Q-onset to S-offset + margin
        ws = max(0, beat.q_onset - margin_samp)
        we = min(T - 1, beat.s_offset + margin_samp)

    t_win = np.arange(ws, we + 1) / lead_fs
    e_win = ecg[ws:we + 1]

    fig, ax = plt.subplots(figsize=(10, 4))

    # ECG trace
    ax.plot(t_win, e_win, 'k-', linewidth=1.0, zorder=1)

    # State colors
    if len(state_names) > we:
        state_win = state_names[ws:we + 1]
        _fill_states(ax, t_win, e_win, state_win, alpha=0.28)

    # ---- P-wave zone highlight ----
    if beat.p_onset > 0 and beat.p_offset > 0 and beat.p_offset > beat.p_onset:
        p_s, p_e = max(ws, beat.p_onset), min(we, beat.p_offset)
        if p_e > p_s:
            ax.fill_between(t_win[p_s - ws:p_e - ws + 1],
                            e_win[p_s - ws:p_e - ws + 1],
                            alpha=0.20, color='#1565c0', linewidth=0)

    # ---- QRS zone highlight ----
    if beat.q_onset > 0 and beat.s_offset > 0:
        q_s, q_e = max(ws, beat.q_onset), min(we, beat.s_offset)
        if q_e > q_s:
            ax.fill_between(t_win[q_s - ws:q_e - ws + 1],
                            e_win[q_s - ws:q_e - ws + 1],
                            alpha=0.18, color='#e65100', linewidth=0)

    # ---- T-wave zone highlight ----
    if beat.t_onset > 0 and beat.t_offset > 0 and beat.t_offset > beat.t_onset:
        t_s, t_e = max(ws, beat.t_onset), min(we, beat.t_offset)
        if t_e > t_s:
            ax.fill_between(t_win[t_s - ws:t_e - ws + 1],
                            e_win[t_s - ws:t_e - ws + 1],
                            alpha=0.18, color='#2e7d32', linewidth=0)

    # Boundary markers (offset-adjusted to window)
    _draw_beat_markers(ax, beat, we + 1, lead_fs, ecg,
                       offset=ws, detailed=True)

    # Beat info box
    info_lines = [
        f'Beat {beat.beat_id}',
        f'P: {_fmt_boundary(beat.p_onset, beat.p_offset, lead_fs)}',
        f'QRS: {_fmt_boundary(beat.q_onset, beat.s_offset, lead_fs)}',
        f'R peak: {beat.r_peak / lead_fs * 1000:.0f}ms',
        f'T: {_fmt_boundary(beat.t_onset, beat.t_offset, lead_fs)}',
    ]
    ax.text(0.98, 0.95, '\n'.join(info_lines), transform=ax.transAxes,
            fontsize=8.5, va='top', ha='right', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85))

    ax.set_xlim(t_win[0], t_win[-1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (norm)')
    _src = getattr(beat, 'p_source', 'hsmm')
    src_note = f' [P:{_src}]' if _src != 'hsmm' else ''
    ax.set_title(f'{rec_name} — Lead {lead_name} Beat {beat.beat_id} '
                 f'(#{beat_idx + 1}){src_note}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.12)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Beat Detail Grid: All beats of one lead in a N×2 grid
# ---------------------------------------------------------------------------
def plot_beat_detail_grid(seg_data: dict, lead_name: str, rec_name: str,
                          save_path: str, max_beats: int = 12,
                          margin_ms: float = 200.0, dpi: int = 150):
    """All beats of one lead in a grid layout.

    Parameters
    ----------
    seg_data : dict
    lead_name : str
    rec_name : str
    save_path : str
    max_beats : int — maximum beats to plot
    margin_ms : float
    """
    ecg = seg_data['filtered_ecg']
    state_names = seg_data.get('state_names', [])
    beats = seg_data.get('beats', [])
    lead_fs = seg_data.get('fs', 250.0)
    T = len(ecg)

    if not beats:
        return

    n_beats = min(len(beats), max_beats)
    n_cols = 2
    n_rows = (n_beats + n_cols - 1) // n_cols

    margin_samp = int(margin_ms / 1000.0 * lead_fs)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    fig.suptitle(f'{rec_name} — Lead {lead_name} Per-Beat Waveform Segmentation '
                 f'({n_beats} beats)', fontsize=13, fontweight='bold')

    # Flatten axes for easy iteration
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    for i in range(n_beats):
        ax = axes_flat[i]
        b = beats[i]

        p_on = b.p_onset if b.p_onset > 0 else b.q_onset - int(0.16 * lead_fs)
        t_off = b.t_offset if b.t_offset > 0 else b.s_offset + int(0.24 * lead_fs)

        ws = max(0, p_on - margin_samp)
        we = min(T - 1, t_off + margin_samp)

        if we - ws < 20:
            ws = max(0, b.q_onset - margin_samp)
            we = min(T - 1, b.s_offset + margin_samp)

        t_win = np.arange(ws, we + 1) / lead_fs
        e_win = ecg[ws:we + 1]

        ax.plot(t_win, e_win, 'k-', linewidth=0.7)

        if len(state_names) > we:
            _fill_states(ax, t_win, e_win, state_names[ws:we + 1], alpha=0.22)

        _draw_beat_markers(ax, b, we + 1, lead_fs, ecg, offset=ws, detailed=True)

        # Minimal info
        ax.text(0.98, 0.93, f'Beat {b.beat_id}',
                transform=ax.transAxes, fontsize=8, ha='right',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        ax.set_title(f'Beat {b.beat_id}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Amp', fontsize=8)
        ax.grid(True, alpha=0.1)

    # Hide unused subplots
    for i in range(n_beats, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Convenience: generate all plots for one record
# ---------------------------------------------------------------------------
def save_all_segmentation_plots(seg_data: dict, rec_name: str,
                                rec_dir: str, max_sec: float = 4.0,
                                max_beats_per_lead: int = 12,
                                dpi: int = 150):
    """Generate all segmentation plots for one record.

    Creates:
      - {rec_dir}/segmentation_overview.png   — 6-lead grid
      - {rec_dir}/lead_{name}/segmentation_overview.png  — per-lead
      - {rec_dir}/lead_{name}/beats/beat_{###}.png       — per-beat
      - {rec_dir}/lead_{name}/beats/_grid.png           — beat grid

    Parameters
    ----------
    seg_data : dict
        lead_name -> segment data dict.
    rec_name : str
    rec_dir : str
    max_sec : float
    max_beats_per_lead : int
    dpi : int
    """
    fs = 250.0
    # Find actual fs from any lead
    for sd in seg_data.values():
        if sd and sd.get('fs'):
            fs = sd['fs']
            break

    # ---- 1. Record overview: 6-lead grid ----
    plot_record_overview(seg_data, rec_name, fs,
                         os.path.join(rec_dir, 'segmentation_overview.png'),
                         max_sec=max_sec, dpi=dpi)

    # ---- 2. Per-lead plots ----
    from .limb_lead_processor import LIMB_LEADS

    for lead_name in LIMB_LEADS:
        sd = seg_data.get(lead_name)
        if sd is None or sd.get('filtered_ecg') is None:
            continue

        lead_dir = os.path.join(rec_dir, f'lead_{lead_name}')
        beats_dir = os.path.join(lead_dir, 'beats')
        os.makedirs(beats_dir, exist_ok=True)

        # Lead overview
        plot_lead_overview(sd, lead_name, rec_name,
                           os.path.join(lead_dir, 'segmentation_overview.png'),
                           max_sec=max_sec, dpi=dpi)

        # Beat detail grid
        beats = sd.get('beats', [])
        if beats:
            plot_beat_detail_grid(sd, lead_name, rec_name,
                                  os.path.join(beats_dir, '_grid.png'),
                                  max_beats=max_beats_per_lead, dpi=dpi)

        # Individual beat plots (up to 16)
        for i, b in enumerate(beats[:16]):
            if b.q_onset <= 0 or b.s_offset <= 0:
                continue
            plot_beat_detail(sd, b, i, lead_name, rec_name,
                             os.path.join(beats_dir, f'beat_{b.beat_id:03d}.png'),
                             dpi=dpi)


# ===========================================================================
# Internal helpers
# ===========================================================================

def _fill_states(ax, t, ecg, state_names, alpha=0.30):
    """Fill background with state-specific colors."""
    if len(state_names) == 0:
        return

    # Find contiguous segments of the same state
    changes = [0]
    for i in range(1, len(state_names)):
        if state_names[i] != state_names[i - 1]:
            changes.append(i)
    changes.append(len(state_names))

    for c in range(len(changes) - 1):
        s, e = changes[c], changes[c + 1]
        if e <= s:
            continue
        sn = state_names[s]
        color = STATE_COLORS.get(sn, STATE_COLORS['UNKNOWN'])
        if s < len(t) and e <= len(t):
            ax.fill_between(t[s:e], ecg[s:e], alpha=alpha, color=color, linewidth=0)


def _draw_beat_markers(ax, beat, n_plot, fs, ecg, offset=0, detailed=False,
                       label_beat=False):
    """Draw boundary markers for one beat.

    Parameters
    ----------
    ax : Axes
    beat : BeatBoundary
    n_plot : int — sample limit for the current plot window
    fs : float
    ecg : np.ndarray — full signal (for amplitude at marker)
    offset : int — sample offset for windowed plots
    detailed : bool — if True, show all boundaries; else just key ones
    label_beat : bool — if True, add beat number label
    """
    T = len(ecg)

    def _vline(sample, style, y_offset=0):
        """Draw a vertical line if sample is in plot range."""
        if sample < offset or sample >= n_plot:
            return
        t = sample / fs
        y_val = ecg[sample] if sample < T else 0
        ax.axvline(t, color=style['color'], linestyle=style['ls'],
                   linewidth=style['lw'], alpha=0.7)
        if detailed:
            ax.annotate(style['label'],
                        (t, y_val + y_offset),
                        textcoords='offset points',
                        xytext=(0, 8 if y_offset == 0 else y_offset),
                        fontsize=6, color=style['color'],
                        ha='center', fontweight='bold')

    # R peak (always shown)
    if beat.r_peak > 0:
        _vline(beat.r_peak, BOUNDARY_STYLES['r_peak'])

    # Q onset
    if beat.q_onset > 0:
        _vline(beat.q_onset, BOUNDARY_STYLES['q_onset'], -12)

    # S offset
    if beat.s_offset > 0:
        _vline(beat.s_offset, BOUNDARY_STYLES['s_offset'], -12)

    if detailed:
        # P wave
        if beat.p_onset > 0:
            _vline(beat.p_onset, BOUNDARY_STYLES['p_onset'], -8)
        if beat.p_offset > 0:
            _vline(beat.p_offset, BOUNDARY_STYLES['p_offset'], -8)

        # T offset
        if beat.t_offset > 0:
            _vline(beat.t_offset, BOUNDARY_STYLES['t_offset'], -8)

    # Beat number label
    if label_beat and beat.r_peak > 0 and beat.r_peak < n_plot:
        t_r = beat.r_peak / fs
        y_r = ecg[beat.r_peak] if beat.r_peak < T else 0
        ax.annotate(f'B{beat.beat_id}',
                    (t_r, y_r),
                    textcoords='offset points',
                    xytext=(0, 14),
                    fontsize=8, ha='center', fontweight='bold',
                    color='#c62828')


def _fmt_boundary(onset, offset, fs):
    """Format boundary as 'onset-offset ms'."""
    if onset < 0 or offset < 0 or offset <= onset:
        return '—'
    dur = (offset - onset) / fs * 1000.0
    return f'{onset / fs * 1000:.0f}-{offset / fs * 1000:.0f}ms ({dur:.0f}ms)'
