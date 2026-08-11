"""ECG Spectrogram Plotting.

High-quality matplotlib-based visualization functions for ECG spectrograms,
PSD, and CWT scalograms. All plots use the Agg backend for file output.
"""

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from .spectrogram import ECG_Spectrogram, band_power


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
FIG_DPI = 150
CMAP = 'inferno'        # perceptually uniform, works for power/magnitude
CMAP_DB = 'viridis'     # good for dB-scaled data
GRID_ALPHA = 0.25
FREQ_LIM_COLOR = '#FF6B6B'


def _setup_axes(ax, xlabel='Time (s)', ylabel='Frequency (Hz)'):
    """Common axis styling."""
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=GRID_ALPHA, linestyle='--')


# ---------------------------------------------------------------------------
# Single spectrogram plot
# ---------------------------------------------------------------------------
def plot_spectrogram(
    spec: ECG_Spectrogram,
    figsize: tuple[float, float] = (12, 5),
    dpi: int = FIG_DPI,
    show_colorbar: bool = True,
    show_band_power: bool = False,
    cmap: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot a single spectrogram (STFT or CWT).

    Parameters
    ----------
    spec : ECG_Spectrogram
        Computed spectrogram from compute_spectrogram() or compute_scalogram().
    figsize : tuple
        Figure size in inches.
    dpi : int
        Output resolution.
    show_colorbar : bool
        Whether to show the color bar.
    show_band_power : bool
        If True, overlay frequency band annotations.
    cmap : str or None
        Colormap name. Auto-selected if None.
    title : str or None
        Plot title. Auto-generated if None.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if cmap is None:
        cmap = CMAP

    # If data is 1-D (PSD), use a line plot instead
    if spec.data.ndim == 1:
        return _plot_psd_line(spec, figsize, dpi, title)

    extent = [spec.times[0], spec.times[-1],
              spec.freqs[0], spec.freqs[-1]]

    im = ax.imshow(
        spec.data, aspect='auto', origin='lower',
        extent=extent, cmap=cmap, interpolation='bilinear',
    )

    _setup_axes(ax)

    if title is None:
        title = f'{spec.method.upper()} Spectrogram — {spec.lead_name}'
        if spec.record_name:
            title += f' ({spec.record_name})'
    ax.set_title(title, fontsize=10, fontweight='bold')

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.02)
        if spec.method == 'cwt':
            cbar.set_label('Magnitude', fontsize=8)
        else:
            cbar.set_label('Power (V²/Hz)', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    if show_band_power:
        _add_band_overlay(ax)

    fig.tight_layout()
    return fig


def _plot_psd_line(
    spec: ECG_Spectrogram,
    figsize: tuple[float, float],
    dpi: int,
    title: str | None,
) -> plt.Figure:
    """Fallback: line plot for 1-D PSD data."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(spec.freqs, spec.data, color='#2F5496', linewidth=1.0)
    ax.fill_between(spec.freqs, spec.data, alpha=0.15, color='#2F5496')
    _setup_axes(ax, xlabel='Frequency (Hz)', ylabel='Power (V²/Hz)')

    if title is None:
        title = f'PSD — {spec.lead_name}'
        if spec.record_name:
            title += f' ({spec.record_name})'
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlim(spec.freqs[0], spec.freqs[-1])

    # Log y-scale for power
    ax.set_yscale('log')
    ax.set_ylabel('Power (V²/Hz) [log]', fontsize=9)

    fig.tight_layout()
    return fig


def _add_band_overlay(ax):
    """Overlay frequency band boundaries."""
    band_edges = [0.5, 4.0, 8.0, 12.0, 30.0, 60.0]
    names = ['δ', 'θ', 'α', 'β', 'γ']
    colors = ['#E8F5E9', '#FFF3E0', '#E3F2FD', '#FCE4EC', '#F3E5F5']

    for i in range(len(band_edges) - 1):
        ax.axhspan(band_edges[i], band_edges[i+1],
                   alpha=0.08, color=colors[i % len(colors)])
        mid = (band_edges[i] + band_edges[i+1]) / 2
        ax.text(ax.get_xlim()[1] * 0.995, mid, names[i],
                ha='right', va='center', fontsize=7, alpha=0.5,
                transform=ax.get_data_transform())


# ---------------------------------------------------------------------------
# Multi-lead spectrogram grid
# ---------------------------------------------------------------------------
def plot_multi_lead_spectrograms(
    specs: list[ECG_Spectrogram],
    ncols: int = 4,
    figsize: tuple[float, float] | None = None,
    dpi: int = FIG_DPI,
    cmap: str | None = None,
    suptitle: str | None = None,
) -> plt.Figure:
    """Plot multiple lead spectrograms in a grid.

    Parameters
    ----------
    specs : list of ECG_Spectrogram
        One spectrogram per lead.
    ncols : int
        Number of columns in the grid.
    figsize : tuple or None
        Auto-sized if None.
    dpi : int
        Output resolution.
    cmap : str or None
        Colormap.
    suptitle : str or None
        Overall figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(specs)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (4.0 * ncols, 3.0 * nrows)

    if cmap is None:
        cmap = CMAP

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    # Ensure axes is always a flat array
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    for i, spec in enumerate(specs):
        ax = axes[i]
        if spec.data.ndim == 1:
            ax.plot(spec.freqs, spec.data, color='#2F5496', linewidth=0.8)
            ax.fill_between(spec.freqs, spec.data, alpha=0.1, color='#2F5496')
            ax.set_xlim(spec.freqs[0], spec.freqs[-1])
            ax.set_yscale('log')
        else:
            extent = [spec.times[0], spec.times[-1],
                      spec.freqs[0], spec.freqs[-1]]
            ax.imshow(spec.data, aspect='auto', origin='lower',
                      extent=extent, cmap=cmap, interpolation='bilinear')
        _setup_axes(ax)
        ax.set_title(spec.lead_name, fontsize=9, fontweight='bold')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight='bold', y=0.98)

    fig.tight_layout()
    if suptitle:
        fig.subplots_adjust(top=0.93)
    return fig


# ---------------------------------------------------------------------------
# Multi-lead PSD comparison overlay
# ---------------------------------------------------------------------------
def plot_psd_comparison(
    specs: list[ECG_Spectrogram],
    figsize: tuple[float, float] = (12, 6),
    dpi: int = FIG_DPI,
    title: str | None = None,
    show_bands: bool = True,
) -> plt.Figure:
    """Overlay PSD curves from multiple leads on the same axes.

    Parameters
    ----------
    specs : list of ECG_Spectrogram
        PSD results (1-D) from different leads.
    figsize : tuple
        Figure size.
    dpi : int
        Resolution.
    title : str or None
        Plot title.
    show_bands : bool
        Overlay frequency band regions.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Color cycle for leads
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(specs), 10)))

    for i, spec in enumerate(specs):
        label = spec.lead_name or f'Lead {i+1}'
        ax.plot(spec.freqs, spec.data, color=colors[i],
                linewidth=1.2, label=label, alpha=0.85)

    _setup_axes(ax, xlabel='Frequency (Hz)', ylabel='Power (V²/Hz)')
    ax.set_yscale('log')
    ax.set_ylabel('Power (V²/Hz) [log]', fontsize=9)

    if show_bands:
        _add_band_overlay(ax)

    ax.legend(fontsize=8, loc='upper right', ncol=2, framealpha=0.7)

    if title is None:
        rec = specs[0].record_name if specs else ''
        title = f'PSD Comparison — {rec}' if rec else 'PSD Comparison Across Leads'
    ax.set_title(title, fontsize=10, fontweight='bold')

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Combined view: signal + spectrogram + PSD
# ---------------------------------------------------------------------------
def plot_signal_and_spectrogram(
    ecg_signal: np.ndarray,
    spec: ECG_Spectrogram,
    psd: ECG_Spectrogram | None = None,
    time_start: float = 0,
    figsize: tuple[float, float] = (14, 8),
    dpi: int = FIG_DPI,
    cmap: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Three-panel figure: time-domain signal, spectrogram, and PSD.

    This is the recommended view for a single lead — it shows the signal
    in its original domain alongside its time-frequency and frequency-domain
    representations.

    Parameters
    ----------
    ecg_signal : np.ndarray
        Raw or preprocessed ECG signal (time domain).
    spec : ECG_Spectrogram
        STFT or CWT spectrogram.
    psd : ECG_Spectrogram or None
        Optional PSD. If None, computed on-the-fly from spec.
    time_start : float
        Start time offset in seconds.
    figsize : tuple
        Figure size.
    dpi : int
        Resolution.
    cmap : str or None
        Colormap for spectrogram.
    title : str or None
        Overall title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if cmap is None:
        cmap = CMAP

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2],
                          hspace=0.35, wspace=0.30)

    # --- Panel A: Time-domain signal ---
    ax_sig = fig.add_subplot(gs[0, :])
    T = len(ecg_signal) / spec.fs
    t_sig = np.linspace(time_start, time_start + T, len(ecg_signal))
    ax_sig.plot(t_sig, ecg_signal, color='#2F5496', linewidth=0.6)
    _setup_axes(ax_sig, xlabel='Time (s)', ylabel='Amplitude')
    ax_sig.set_title('ECG Signal (Time Domain)', fontsize=9, fontweight='bold')
    ax_sig.set_xlim(t_sig[0], t_sig[-1])

    # --- Panel B: Spectrogram ---
    ax_spec = fig.add_subplot(gs[1, 0])
    extent = [spec.times[0] + time_start, spec.times[-1] + time_start,
              spec.freqs[0], spec.freqs[-1]]
    im = ax_spec.imshow(spec.data, aspect='auto', origin='lower',
                        extent=extent, cmap=cmap, interpolation='bilinear')
    _setup_axes(ax_spec)
    ax_spec.set_title(f'{spec.method.upper()} Spectrogram', fontsize=9, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax_spec, shrink=0.85, pad=0.02)
    cbar.ax.tick_params(labelsize=6)

    # --- Panel C: PSD ---
    ax_psd = fig.add_subplot(gs[1, 1])
    if psd is not None:
        psd_data = psd
    else:
        # Use the spectrogram averaged over time
        from .spectrogram import ECG_Spectrogram
        avg_data = np.mean(spec.data, axis=1) if spec.data.ndim == 2 else spec.data
        psd_data = ECG_Spectrogram(
            data=avg_data, freqs=spec.freqs, times=np.array([]),
            fs=spec.fs, method='psd', lead_name=spec.lead_name,
        )

    if psd_data.data.ndim == 1:
        ax_psd.plot(psd_data.freqs, psd_data.data, color='#E74C3C', linewidth=1.0)
        ax_psd.fill_between(psd_data.freqs, psd_data.data, alpha=0.12, color='#E74C3C')
    ax_psd.set_xlim(psd_data.freqs[0], psd_data.freqs[-1])
    ax_psd.set_yscale('log')
    _setup_axes(ax_psd, xlabel='Frequency (Hz)', ylabel='Power [log]')
    ax_psd.set_title('Power Spectral Density', fontsize=9, fontweight='bold')

    if title:
        fig.suptitle(title, fontsize=11, fontweight='bold', y=0.99)
    elif spec.lead_name:
        fig.suptitle(f'{spec.lead_name} — {spec.record_name}',
                     fontsize=11, fontweight='bold', y=0.99)

    fig.tight_layout()
    if title or spec.lead_name:
        fig.subplots_adjust(top=0.94)
    return fig


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------
def save_figure(fig: plt.Figure, filepath: str | Path, dpi: int | None = None,
                close: bool = True):
    """Save a matplotlib figure to disk.

    Parameters
    ----------
    fig : plt.Figure
    filepath : str or Path
        Output path. Directory is auto-created.
    dpi : int or None
        Override DPI. Uses figure's DPI if None.
    close : bool
        Close the figure after saving to free memory.
    """
    fp = Path(filepath)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(fp), dpi=dpi or fig.dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    if close:
        plt.close(fig)
