"""Visualization for spectrogram-based RA-LA reversal detection.

One figure per record:
  1. leads I / II / III waveform strips with R-peak anchors and the QRS
     windows used by the detector
  2. cross-wavelet phase (I x conj(II)) in the QRS band, masked by
     coherence weight, with the circular mean phase marked
  3. histogram of the QRS-band cross-phase samples (circles -> degrees)
     with the weighted mean and the 90-deg decision boundary
  4. per-beat I/II QRS-window correlation (the gating feature)
The suptitle carries verdict + confidence + the phase/corr votes.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .plot_spectrogram import FIG_DPI, save_figure
from .tf_reversal import (
    ANTI_PHASE_MIN_DEG, QRS_BAND_REV, QRS_WIN_MS,
)

CMAP_PHASE = 'twilight'  # cyclic — matches angular data
VERDICT_COLORS = {'normal': '#55A868', 'reversed': '#C44E52',
                  'uncertain': '#8C8C8C'}


def plot_tf_reversal(clean: dict, result, fs: float,
                     record_name: str = '',
                     r_peaks: list[int] | None = None,
                     cross_phase: np.ndarray | None = None,
                     weight: np.ndarray | None = None,
                     freqs: np.ndarray | None = None,
                     beat_corr: list[float] | None = None,
                     dpi: int = FIG_DPI) -> plt.Figure:
    """Build the reversal-diagnosis figure for one record.

    Parameters
    ----------
    clean : dict lead name -> preprocessed signal (I and II required,
        III optional).
    result : TFReversalResult from detect_tf_reversal.
    fs : sampling rate.
    r_peaks : R-peak sample indices (anchors); drawn when provided.
    cross_phase, weight : (n_freq, N) cross-wavelet phase (rad) and
        weight |W_I||W_II|; panel 2 is skipped when None.
    freqs : CWT frequency axis matching cross_phase rows.
    beat_corr : per-beat I/II QRS correlation values (panel 4).
    """
    n = len(clean['I'])
    t = np.arange(n) / fs
    color = VERDICT_COLORS.get(result.verdict, '#8C8C8C')

    n_panels = 2 + (cross_phase is not None) + (beat_corr is not None)
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 2.9 * n_panels),
                             dpi=dpi, squeeze=False, layout='constrained')
    axes = axes.ravel()

    # ---- Panel 1: limb-lead waveforms ----
    ax = axes[0]
    offsets = {'I': 2.5, 'II': 0.0, 'III': -2.5, 'AVR': -5.0}
    for lead, off in offsets.items():
        if lead in clean and clean[lead] is not None:
            sig = np.asarray(clean[lead], dtype=np.float64)[:n]
            scale = np.std(sig) or 1.0
            ax.plot(t, sig / scale * 0.9 + off, linewidth=0.7,
                    color='#333333', label=lead)
            ax.text(0.002, off + 0.35, lead, fontsize=8, va='bottom',
                    transform=ax.get_yaxis_transform())
    if r_peaks:
        for r in r_peaks:
            ax.axvline(r / fs, color='#4C72B0', alpha=0.25, linewidth=0.6)
        lo = -QRS_WIN_MS[0] / 1000.0
        hi = QRS_WIN_MS[1] / 1000.0
        for r in r_peaks:
            ax.axvspan((r - lo) / fs, (r + hi) / fs, color='#4C72B0',
                       alpha=0.10, lw=0)
    ax.set_ylim(-6.2, 4.0)
    ax.set_ylabel('lead (norm.)')
    ax.set_title('limb leads — R anchors (blue) + QRS windows (shaded)',
                 fontsize=9, loc='left')
    ax.legend(loc='upper right', fontsize=7, ncol=4, framealpha=0.6)

    # ---- Panel 2: cross-wavelet phase in the QRS band ----
    if cross_phase is not None and freqs is not None:
        ax = axes[1]
        fmask = (freqs >= QRS_BAND_REV[0]) & (freqs <= QRS_BAND_REV[1])
        w = weight[fmask, :] if weight is not None else None
        if w is not None and w.max() > 0:
            alpha = np.clip(w / np.percentile(w, 99), 0.0, 1.0) * 0.85
        else:
            alpha = np.ones_like(cross_phase[fmask, :])
        im = ax.imshow(np.degrees(cross_phase[fmask, :]), aspect='auto',
                       cmap=CMAP_PHASE, vmin=-180, vmax=180,
                       origin='lower', extent=(t[0], t[-1],
                                               freqs[fmask][0],
                                               freqs[fmask][-1]),
                       alpha=alpha if isinstance(alpha, np.ndarray) else None)
        fig.colorbar(im, ax=ax, pad=0.01, label='phase (deg)')
        for r in (r_peaks or []):
            ax.axvline(r / fs, color='white', alpha=0.4, linewidth=0.5)
        ax.set_ylabel('frequency (Hz)')
        ax.set_title('cross-wavelet phase I x II* — QRS band, '
                     'opacity = coherence weight', fontsize=9, loc='left')

    # ---- Panel 3: phase histogram ----
    pi = 2 if cross_phase is not None else 1
    ax = axes[pi]
    if cross_phase is not None and result.phase_qrs_deg is not None:
        fmask = (freqs >= QRS_BAND_REV[0]) & (freqs <= QRS_BAND_REV[1])
        ph = np.degrees(cross_phase[fmask, :]).ravel()
        if weight is not None:
            wsel = np.repeat(weight[fmask, :], 1, axis=0).ravel()
            keep = wsel >= np.percentile(wsel, 50)  # drop weak half
            ph = ph[keep]
        ax.hist(ph, bins=72, range=(-180, 180), color='#4C72B0',
                alpha=0.8, edgecolor='white', linewidth=0.2)
    for k, b in enumerate((ANTI_PHASE_MIN_DEG, -ANTI_PHASE_MIN_DEG)):
        ax.axvline(b, color='#C44E52', linestyle='--', linewidth=1.2,
                   label=f'±{ANTI_PHASE_MIN_DEG:.0f}° boundary' if k == 0 else None)
    if result.phase_qrs_deg is not None:
        ax.axvline(result.phase_qrs_deg, color=color, linewidth=2.0,
                   label=f'circular mean {result.phase_qrs_deg:+.1f}°')
    ax.set_xlim(-180, 180)
    ax.set_xlabel('QRS-band cross-phase (deg)')
    ax.set_ylabel('# samples')
    rbar_txt = f'{result.rbar_qrs:.2f}' if result.rbar_qrs is not None else '—'
    ax.set_title(f'cross-phase distribution — R̄={rbar_txt} (top-weight half)',
                 fontsize=9, loc='left')
    ax.legend(fontsize=7, framealpha=0.6)

    # ---- Panel 4: per-beat correlation ----
    if beat_corr:
        ax = axes[-1]
        ax.plot(range(1, len(beat_corr) + 1), beat_corr, 'o-',
                color='#4C72B0', markersize=4, linewidth=0.9)
        ax.axhline(0.0, color='#C44E52', linestyle='--', linewidth=1.0,
                   label='0 (decision boundary)')
        if result.corr_qrs is not None:
            ax.axhline(result.corr_qrs, color=color, linewidth=1.6,
                       label=f'median {result.corr_qrs:+.2f}')
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel('beat #')
        ax.set_ylabel('corr(I, II)\nQRS window')
        ax.set_title('per-beat I/II QRS correlation (gating feature)',
                     fontsize=9, loc='left')
        ax.legend(fontsize=7, framealpha=0.6)
        ax.grid(True, alpha=0.25, linestyle='--')

    if result.phase_qrs_deg is not None and result.corr_qrs is not None:
        headline = (f'{result.verdict.upper()} '
                    f'(confidence {result.confidence:.2f}, '
                    f'φ_qrs {result.phase_qrs_deg:+.1f}°, '
                    f'corr {result.corr_qrs:+.2f}, n={result.n_beats})')
    else:
        headline = f'{result.verdict.upper()} (no phase measurement)'
    fig.suptitle(f'{record_name} — TF RA-LA reversal: {headline}',
                 fontsize=11, fontweight='bold', color=color)
    return fig


def reversal_result_to_dict(result) -> dict:
    """JSON-safe summary of a TFReversalResult."""
    return {
        'verdict': result.verdict,
        'confidence': result.confidence,
        'phase_qrs_deg': result.phase_qrs_deg,
        'rbar_qrs': result.rbar_qrs,
        'corr_qrs': result.corr_qrs,
        'phase_p_deg': result.phase_p_deg,
        'rbar_p': result.rbar_p,
        'n_beats': result.n_beats,
        'qrs_power_ratio': result.qrs_power_ratio,
    }


__all__ = ['plot_tf_reversal', 'reversal_result_to_dict', 'save_figure']
