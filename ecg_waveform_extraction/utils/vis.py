"""Visualization utilities for ECG segmentation and P-wave analysis.

All functions use matplotlib. State color palette is consistent across plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# Consistent color palette for the 9 ECG states
STATE_COLORS = {
    "ISO": "#e0e0e0",
    "P":   "#4caf50",
    "PR":  "#c8e6c9",
    "Q":   "#f44336",
    "R":   "#d32f2f",
    "S":   "#ff7043",
    "ST":  "#fff176",
    "T":   "#2196f3",
    "TP":  "#b0bec5",
    "UNKNOWN": "#9e9e9e",
}


def _get_color(label: str) -> str:
    return STATE_COLORS.get(label, "#9e9e9e")


# ===========================================================================
# Segmentation overlay plot
# ===========================================================================
def plot_segmentation(ecg: np.ndarray, state_labels: np.ndarray,
                      state_names: list[str], fs: float = 250.0,
                      title: str = "ECG Waveform Segmentation",
                      time_range: tuple[float, float] | None = None,
                      ax: plt.Axes | None = None):
    """Plot ECG signal with color-coded state regions as background bands.

    Parameters
    ----------
    ecg : np.ndarray, shape (T,)
        Filtered ECG signal.
    state_labels : np.ndarray, shape (T,), dtype int
        Per-sample state indices.
    state_names : list[str]
        Per-sample state name strings.
    fs : float
        Sampling frequency.
    title : str
        Plot title.
    time_range : tuple[float, float] or None
        (start_sec, end_sec) to zoom in. If None, shows full signal.
    ax : matplotlib Axes or None
        Axis to plot on. Creates new figure if None.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(16, 5))

    T = len(ecg)
    time = np.arange(T) / fs

    # Apply time range if specified
    if time_range is not None:
        start_idx = max(0, int(time_range[0] * fs))
        end_idx = min(T, int(time_range[1] * fs))
        ecg = ecg[start_idx:end_idx]
        state_labels = state_labels[start_idx:end_idx]
        state_names = state_names[start_idx:end_idx]
        time = np.arange(start_idx, end_idx) / fs

    T_plot = len(ecg)

    # Draw color bands for each contiguous state segment
    if len(state_labels) > 0:
        prev_label = state_labels[0]
        seg_start = 0
        for i in range(1, T_plot):
            if state_labels[i] != prev_label:
                color = _get_color(state_names[seg_start])
                ax.axvspan(time[seg_start], time[i], alpha=0.3, color=color)
                seg_start = i
                prev_label = state_labels[i]
        # Last segment
        color = _get_color(state_names[seg_start])
        ax.axvspan(time[seg_start], time[-1], alpha=0.3, color=color)

    # ECG trace
    ax.plot(time, ecg, 'k-', linewidth=0.8, label='ECG')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (normalized)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Legend for state colors
    legend_patches = [
        Rectangle((0, 0), 1, 1, facecolor=_get_color(s), alpha=0.3, label=s)
        for s in ["ISO", "P", "PR", "Q", "R", "S", "ST", "T", "TP"]
    ]
    ax.legend(handles=legend_patches, loc='upper right', ncol=9, fontsize=7)


# ===========================================================================
# P-wave detail plot
# ===========================================================================
def plot_p_wave_detail(ecg: np.ndarray, onset: int, offset: int,
                       fs: float = 250.0, title: str = "P-Wave Detail",
                       ax: plt.Axes | None = None):
    """Zoom-in view of a single P-wave with onset/offset markers.

    Parameters
    ----------
    ecg : np.ndarray, shape (T,)
        Filtered ECG signal.
    onset : int
        P-wave onset sample index.
    offset : int
        P-wave offset sample index.
    fs : float
        Sampling frequency.
    title : str
        Plot title.
    ax : matplotlib Axes or None
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))

    # Show slightly wider context
    margin = int(0.1 * fs)  # 100ms
    start = max(0, onset - margin)
    end = min(len(ecg) - 1, offset + margin)

    time = np.arange(start, end + 1) / fs
    sig = ecg[start:end + 1]

    ax.plot(time, sig, 'k-', linewidth=1.0)

    # P-wave region
    p_time = np.arange(onset, offset + 1) / fs
    p_sig = ecg[onset:offset + 1]
    ax.fill_between(p_time, p_sig, alpha=0.3, color='#4caf50', label='P wave')

    # Onset / offset lines
    ax.axvline(onset / fs, color='green', linestyle='--', linewidth=1.0, label='P onset')
    ax.axvline(offset / fs, color='red', linestyle='--', linewidth=1.0, label='P offset')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ===========================================================================
# Duration distribution comparison
# ===========================================================================
def plot_duration_distributions(model, fs: float = 250.0,
                                ax: plt.Axes | None = None):
    """Bar chart comparing learned vs prior duration distributions per state.

    Parameters
    ----------
    model : HSMMModel
        Trained model.
    fs : float
        Sampling frequency.
    ax : matplotlib Axes or None
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    from ..hsmm.distributions import DurationDistribution

    n_states = model.n_states
    x = np.arange(n_states)
    width = 0.35

    learned_mu = [model.dur_dists[i].mu / fs * 1000.0 for i in range(n_states)]
    prior_mu = []
    for i, label in enumerate(model.state_labels):
        p = DurationDistribution.physiological_prior(label, fs)
        prior_mu.append(p["mu"] / fs * 1000.0)

    ax.bar(x - width / 2, learned_mu, width, label="Learned", color="#2196f3")
    ax.bar(x + width / 2, prior_mu, width, label="Prior", color="#9e9e9e")
    ax.set_xticks(x)
    ax.set_xticklabels(model.state_labels)
    ax.set_ylabel("Mean Duration (ms)")
    ax.set_title("Learned vs. Prior Duration Distributions")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')


# ===========================================================================
# Transition matrix heatmap
# ===========================================================================
def plot_transition_matrix(A: np.ndarray, state_labels: list[str],
                           title: str = "Transition Matrix",
                           ax: plt.Axes | None = None):
    """Heatmap of the HSMM transition probability matrix.

    Parameters
    ----------
    A : np.ndarray, shape (N, N)
        Transition matrix.
    state_labels : list[str]
        State names.
    title : str
    ax : matplotlib Axes or None
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(A, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    n = len(state_labels)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(state_labels, rotation=45, ha="right")
    ax.set_yticklabels(state_labels)
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    ax.set_title(title)

    # Annotate non-zero entries
    for i in range(n):
        for j in range(n):
            if A[i, j] > 0.01:
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if A[i, j] > 0.5 else "black")

    plt.colorbar(im, ax=ax, label="Probability")


# ===========================================================================
# Training progress plot
# ===========================================================================
def plot_training_progress(log_likelihoods: list[float],
                           title: str = "HSMM Training Progress",
                           ax: plt.Axes | None = None):
    """Log-likelihood vs. EM iteration.

    Parameters
    ----------
    log_likelihoods : list[float]
        Per-iteration log-likelihood values.
    title : str
    ax : matplotlib Axes or None
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))

    iters = range(1, len(log_likelihoods) + 1)
    ax.plot(iters, log_likelihoods, 'b-o', markersize=4, linewidth=1.0)
    ax.set_xlabel("EM Iteration")
    ax.set_ylabel("Log-Likelihood")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
