"""Visualization for time-frequency P/QRS/T segmentation.

Each lead produces one 3-panel figure:
  1. waveform with P/QRS/T shading
  2. CWT scalogram with the same shading overlaid
  3. QRS / low-frequency band envelopes with the detection thresholds

An optional comparison figure aligns the TF bands with the main package's
cached HSMM bands (sample-level strips + agreement stats).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from .plot_spectrogram import FIG_DPI, save_figure
from .tf_segmentation import (
    BAND_COLORS, BAND_NAMES,
    LABEL_NONE, LABEL_P, LABEL_QRS, LABEL_T,
    TFSegmentation, hsmm_to_bands,
)

CMAP_SCALOGRAM = 'inferno'
BAND_ALPHA = 0.18


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _label_regions(labels: np.ndarray, code: int) -> list[tuple[int, int]]:
    """Contiguous (start, end) sample regions where labels == code."""
    mask = labels == code
    if not mask.any():
        return []
    edges = np.flatnonzero(np.diff(mask.astype(np.int8)))
    starts = [0] if mask[0] else []
    ends = []
    for i in edges:
        (ends if mask[i] else starts).append(i + 1)
    if mask[-1]:
        ends.append(len(mask))
    return list(zip(starts, ends))


def _shade(ax, labels: np.ndarray, t: np.ndarray, annotate: bool = False):
    """Shade P/QRS/T regions; optionally label the first region of each band."""
    for code in (LABEL_P, LABEL_QRS, LABEL_T):
        first = True
        for s, e in _label_regions(labels, code):
            ax.axvspan(t[s], t[min(e, len(t) - 1)], alpha=BAND_ALPHA,
                       color=BAND_COLORS[code],
                       label=f'{BAND_NAMES[code]}' if first and annotate else None)
            first = False


def _labels_strip(ax, labels: np.ndarray, t_range: tuple[float, float], title: str):
    """Draw a 1-row color strip of band labels on an axes."""
    cmap = ListedColormap(['#FFFFFF', BAND_COLORS[LABEL_P],
                           BAND_COLORS[LABEL_QRS], BAND_COLORS[LABEL_T]])
    ax.imshow(labels[np.newaxis, :], aspect='auto', interpolation='nearest',
              cmap=cmap, vmin=0, vmax=3,
              extent=[t_range[0], t_range[1], 0, 1])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9, fontweight='bold', loc='left')
    ax.set_xlim(*t_range)


# ---------------------------------------------------------------------------
# Per-lead figure
# ---------------------------------------------------------------------------
def plot_tf_segmentation(
    seg: TFSegmentation,
    clean: np.ndarray,
    lead_name: str = '',
    record_name: str = '',
    dpi: int = FIG_DPI,
) -> plt.Figure:
    """3-panel TF segmentation figure for one lead."""
    fs = seg.fs
    N = len(clean)
    t = np.arange(N) / fs

    fig, (ax_wave, ax_spec, ax_env) = plt.subplots(
        3, 1, figsize=(12, 8), dpi=dpi, sharex=True,
        gridspec_kw={'height_ratios': [1.0, 1.2, 0.8]},
    )

    # --- Panel 1: waveform + shading ---
    ax_wave.plot(t, clean, color='#2F5496', linewidth=0.6)
    _shade(ax_wave, seg.labels, t, annotate=True)
    ax_wave.set_ylabel('Amplitude (z)', fontsize=9)
    ax_wave.set_title(
        f'TF Segmentation — {lead_name}'
        + (f' ({record_name})' if record_name else '')
        + f'   [{seg.n_beats} beats]',
        fontsize=10, fontweight='bold',
    )
    ax_wave.grid(True, alpha=0.25, linestyle='--')
    if seg.beats:
        ax_wave.legend(fontsize=7, loc='upper right', ncol=3, framealpha=0.7)

    # --- Panel 2: scalogram + shading ---
    if seg.cwt_mag.size:
        im = ax_spec.imshow(
            seg.cwt_mag, aspect='auto', origin='lower', cmap=CMAP_SCALOGRAM,
            extent=[t[0], t[-1], seg.cwt_freqs[0], seg.cwt_freqs[-1]],
            interpolation='bilinear',
        )
        _shade(ax_spec, seg.labels, t)
        ax_spec.set_ylabel('Frequency (Hz)', fontsize=9)
        cbar = fig.colorbar(im, ax=ax_spec, shrink=0.9, pad=0.01)
        cbar.set_label('|CWT|', fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    ax_spec.set_title('CWT scalogram + detected bands', fontsize=9, fontweight='bold')
    ax_spec.grid(True, alpha=0.25, linestyle='--')

    # --- Panel 3: envelopes + thresholds ---
    ax_env.plot(t, seg.qrs_env, color=BAND_COLORS[LABEL_QRS], linewidth=1.0,
                label=f'QRS env ({BAND_NAMES[LABEL_QRS]} band)')
    ax_env.axhline(seg.qrs_threshold, color=BAND_COLORS[LABEL_QRS],
                   linewidth=0.8, linestyle=':',
                   label=f'QRS thr = {seg.qrs_threshold:.3g}')
    ax_env.plot(t, seg.low_env / max(seg.low_env.max(), 1e-12) * max(seg.qrs_env.max(), 1e-12),
                color=BAND_COLORS[LABEL_T], linewidth=0.8, alpha=0.8,
                label='low-freq env (rescaled)')
    ax_env.set_xlabel('Time (s)', fontsize=9)
    ax_env.set_ylabel('QRS envelope', fontsize=9)
    ax_env.set_xlim(t[0], t[-1])
    ax_env.grid(True, alpha=0.25, linestyle='--')
    ax_env.legend(fontsize=7, loc='upper right', framealpha=0.7)
    ax_env.set_title('Band envelopes', fontsize=9, fontweight='bold')

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Comparison figure (TF vs cached HSMM bands)
# ---------------------------------------------------------------------------
def plot_hsmm_comparison(
    seg: TFSegmentation,
    clean: np.ndarray,
    hsmm_labels: np.ndarray,
    report: dict,
    lead_name: str = '',
    record_name: str = '',
    dpi: int = FIG_DPI,
) -> plt.Figure:
    """Waveform + aligned TF/HSMM band strips + agreement stats."""
    fs = seg.fs
    n = min(len(clean), len(hsmm_labels))
    t = np.arange(n) / fs
    tf = seg.labels[:n]
    ref = hsmm_to_bands(np.asarray(hsmm_labels[:n]))

    fig, (ax_wave, ax_tf, ax_ref) = plt.subplots(
        3, 1, figsize=(12, 5.5), dpi=dpi, sharex=True,
        gridspec_kw={'height_ratios': [2.0, 0.5, 0.5]},
    )

    ax_wave.plot(t, clean[:n], color='#2F5496', linewidth=0.6)
    _shade(ax_wave, tf, t)
    ax_wave.set_ylabel('Amplitude (z)', fontsize=9)
    detail = ', '.join(
        f"{name}: P={v['precision']}/R={v['recall']}"
        for name, v in report['per_band'].items() if v['precision'] is not None
    )
    ax_wave.set_title(
        f'TF vs HSMM — {lead_name}' + (f' ({record_name})' if record_name else '')
        + f"   agreement={report['agreement']*100:.1f}%   ({detail})",
        fontsize=10, fontweight='bold',
    )
    ax_wave.grid(True, alpha=0.25, linestyle='--')

    _labels_strip(ax_tf, tf, (t[0], t[-1]), 'TF segmentation (this package)')
    _labels_strip(ax_ref, ref, (t[0], t[-1]), 'HSMM bands (main package cache)')

    ax_ref.set_xlabel('Time (s)', fontsize=9)
    fig.tight_layout()
    return fig


__all__ = ['plot_tf_segmentation', 'plot_hsmm_comparison', 'save_figure']
