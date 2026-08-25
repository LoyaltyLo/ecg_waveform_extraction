"""Three-Method P-Wave Segmentation Comparison.

Implements and compares three P-wave detection approaches, all anchored by
HSMM Stage 1 (beat/QRS detection):

  Method 1 — HSMM + Correlation-Enhanced Multi-Feature Template
    Focused 3-state HSMM → template-matching fallback → multi-dimensional
    confidence scoring (SNR + symmetry + consistency + duration).

  Method 2 — HSMM + Phasor Transform
    Maps the P-region window to a complex plane y(n)=R_V+j*x(n), detects
    P-wave onset/offset via phase-transition analysis. The arctan non-linearity
    amplifies low-amplitude P-waves relative to baseline noise.

  Method 3 — HSMM + Time-Frequency Analysis (CWT)
    Continuous Wavelet Transform on the P-region window; identifies P-wave
    boundaries via energy concentration in the 5–15 Hz band.

Usage
------
    python -m ecg_waveform_extraction.src.p_wave_three_methods

Output
------
    output/three_methods/<record>/
        overview.png          — 3-row comparison over ~4 s
        beat_###.png           — per-beat zoom with all three methods
        beat_###_phasor.png    — phasor-domain diagnostic plot
        beat_###_cwt.png       — CWT scalogram
        summary.json           — per-beat boundary table
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os, json, glob, textwrap, math
import numpy as np
from scipy.signal import find_peaks, savgol_filter, convolve
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from dataclasses import dataclass, field

from ecg_waveform_extraction.src.preprocessing import ECGPreprocessor
from ecg_waveform_extraction.src.features import FeatureExtractor
from ecg_waveform_extraction.src.hsmm import HSMMModel, HSMMDecoder, smart_initialize_gmms
from ecg_waveform_extraction.src.segmentation import ECGSegmenter
from ecg_waveform_extraction.src.extraction import PWaveExtractor, PWaveResult

# =============================================================================
# Config
# =============================================================================
MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "hsmm_trained.npz")
AECG_DIR = Path("C:/LoyaltyLo/datasets/RA-LA_Reversal/aECG")
OUTPUT_BASE = Path(__file__).resolve().parent.parent / "output" / "three_methods"

DEFAULT_FS = 250.0
PHASOR_R_V = 0.002  # small constant for phasor transform
CWT_WIDTHS = np.arange(1, 31)  # wavelet scales for CWT
MAX_BEATS_PER_RECORD = 20  # plot up to 20 beats per record


# ---------------------------------------------------------------------------
# Manual CWT + Ricker wavelet (scipy >= 1.15 removed cwt/ricker)
# ---------------------------------------------------------------------------
def _ricker(points, a):
    """Ricker (Mexican Hat) wavelet: (2/(√3a·π^(1/4))) · (1 - (x/a)²) · exp(-(x/a)²/2)."""
    A = 2.0 / (np.sqrt(3.0 * a) * (np.pi ** 0.25))
    wsq = a ** 2
    vec = np.arange(0, points) - (points - 1.0) / 2.0
    xsq = vec ** 2
    mod = (1.0 - xsq / wsq)
    gauss = np.exp(-xsq / (2.0 * wsq))
    return A * mod * gauss


def _cwt_manual(data, widths, wavelet_fn=_ricker):
    """Manual continuous wavelet transform using convolution.

    Parameters
    ----------
    data : np.ndarray, shape (N,)
    widths : np.ndarray, shape (n_scales,)
    wavelet_fn : callable — wavelet_fn(points, width) -> np.ndarray

    Returns
    -------
    cwt_mat : np.ndarray, shape (n_scales, N)
    """
    N = len(data)
    n_scales = len(widths)
    out = np.zeros((n_scales, N))
    for i, width in enumerate(widths):
        # Wavelet length: 10 * width, must be odd
        wl = int(10 * width + 1)
        if wl % 2 == 0:
            wl += 1
        if wl > N:
            wl = N if N % 2 == 1 else N - 1
        if wl < 3:
            wl = 3
        wavelet = wavelet_fn(wl, width)
        # Convolve
        conv = convolve(data, wavelet, mode="same")
        out[i, :] = conv
    return out

# =============================================================================
# Data loading — aECG XML parser (lightweight)
# =============================================================================
def parse_aecg(filepath):
    """Parse an aECG XML file into a dict with 'fs', 'signals', 'annotations'."""
    import re
    for enc in ["utf-8", "gbk", "latin-1"]:
        try:
            with open(filepath, "r", encoding=enc) as f:
                content = f.read()
            if "<?xml" in content[:100]:
                break
        except Exception:
            continue

    result = {"fs": DEFAULT_FS, "signals": {}, "annotations": {}}
    inc = re.search(r'<increment[^>]*value="([^"]+)"[^>]*unit="s"', content)
    if inc:
        result["fs"] = 1.0 / float(inc.group(1))

    lead_names = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    digits = list(re.finditer(r"<digits[^>]*>([^<]*)</digits>", content))
    for i, m in enumerate(digits[:12]):
        samples = np.array([float(x) for x in m.group(1).split()], dtype=np.float64)
        result["signals"][lead_names[i]] = samples

    for key, tag in [
        ("P_on_ms", "PWAVE"), ("P_off_ms", "PWAVE"),
        ("QRS_on_ms", "QRSWAVE"), ("QRS_off_ms", "QRSWAVE"),
        ("T_on_ms", "TWAVE"), ("T_off_ms", "TWAVE"),
    ]:
        if "on" in key:
            m = re.search(
                rf'MDC_ECG_WAVC_{tag}.*?<low value="([^"]+)" unit="ms"',
                content, re.DOTALL,
            )
        else:
            m = re.search(
                rf'MDC_ECG_WAVC_{tag}.*?<high value="([^"]+)" unit="ms"',
                content, re.DOTALL,
            )
        if m:
            result["annotations"][key] = float(m.group(1))
    return result


# =============================================================================
# Method 1 helper — thin wrapper for clarity
# =============================================================================
def detect_p_wave_template(p_extractor, segment_result, heart_rate):
    """Run the full HSMM+Template pipeline (existing PWaveExtractor)."""
    return p_extractor.extract(segment_result, heart_rate=heart_rate)


# =============================================================================
# Method 2: HSMM + Phasor Transform
# =============================================================================
@dataclass
class PhasorPWave:
    beat_id: int
    onset_sample: int
    offset_sample: int
    peak_sample: int
    duration_ms: float
    phase_range: float       # max-min phase within P region (rad)
    confidence: float = 1.0


class PhasorPWaveDetector:
    """P-wave detection via phasor transform within HSMM-anchored windows.

    Maps the P-region to the complex plane:
        y(n) = R_V + j * x(n)
        phi(n) = arctan(x(n) / R_V)

    A small R_V (≈0.002) amplifies low-amplitude P-waves. P-wave onset/offset
    are detected from phase-transition points crossing a dynamic threshold.

    Reference
    ---------
    Saclova, L. et al. (2022). "A pathology-aware P-wave detector based on the
    phasor transform." Scientific Reports, 12, 6576.
    """

    def __init__(self, fs: float = DEFAULT_FS, r_v: float = PHASOR_R_V):
        self.fs = fs
        self.r_v = r_v

    def compute_phasor(self, ecg: np.ndarray) -> np.ndarray:
        """Compute phasor phase for an ECG segment."""
        return np.arctan2(ecg, self.r_v)

    def detect(self, ecg: np.ndarray, beat, heart_rate: float) -> PhasorPWave | None:
        """Detect P-wave for one beat using phasor transform.

        Strategy (two-pass):
          1. Phase residual — subtract a wide median-filtered baseline from
             the raw phase φ.  The P-wave appears as a unimodal excursion in
             the residual whose width ≈ true P-wave duration.
          2. |dφ/dt| refinement — tighten onset/offset to the nearest
             derivative peak, avoiding inclusion of flat baseline.

        Parameters
        ----------
        ecg : np.ndarray — full filtered ECG
        beat : BeatBoundary — Stage 1 beat with approximate P/QRS boundaries
        heart_rate : float

        Returns
        -------
        PhasorPWave or None
        """
        T = len(ecg)

        # Define search window: from well before expected P to Q onset
        if beat.p_onset > 0:
            p_expected = beat.p_onset
        else:
            p_expected = beat.q_onset - int(0.20 * self.fs)

        window_before = int(0.30 * self.fs)
        window_after = int(0.05 * self.fs)
        ws = max(0, p_expected - window_before)
        we = min(T - 1, p_expected + window_after)
        if beat.q_onset > 0:
            we = min(we, beat.q_onset - 1)
        if we - ws < 20:
            return None

        ecg_win = ecg[ws : we + 1].copy()
        ecg_win = ecg_win - np.median(ecg_win[: max(5, len(ecg_win) // 4)])

        # ---- Phasor transform ----
        phi = self.compute_phasor(ecg_win)

        # ---- Pass 1: Phase residual (wide baseline removal) ----
        # Use a wide median filter (~120ms) to estimate the local phase baseline
        med_wide = max(int(0.12 * self.fs), 7)
        if med_wide % 2 == 0:
            med_wide += 1
        from scipy.signal import medfilt
        phi_bl = medfilt(phi, med_wide)
        phi_residual = phi - phi_bl  # deviation from baseline

        # Smooth the residual
        win_res = min(7, len(phi_residual) - (len(phi_residual) % 2 == 0) - 1)
        if win_res >= 3:
            phi_res_smooth = savgol_filter(phi_residual, win_res, 2)
        else:
            phi_res_smooth = phi_residual

        # Threshold on |residual|
        bl_end = min(len(phi_res_smooth) // 4, int(0.08 * self.fs))
        if bl_end < 3:
            bl_end = max(3, len(phi_res_smooth) // 4)
        noise_std = np.std(phi_res_smooth[:bl_end])
        thresh = max(2.5 * noise_std, 0.005)

        above = np.abs(phi_res_smooth) > thresh
        if not np.any(above):
            return None

        # Find contiguous above-threshold regions
        transitions = np.diff(above.astype(int))
        starts = np.where(transitions == 1)[0] + 1
        ends = np.where(transitions == -1)[0]
        if above[0]:
            starts = np.concatenate([[0], starts])
        if above[-1]:
            ends = np.concatenate([ends, [len(above) - 1]])

        if len(starts) == 0:
            return None

        # Pick region closest to expected P position with sufficient width
        expected_rel = p_expected - ws
        best_idx = 0
        best_score = -np.inf
        for i in range(len(starts)):
            dur = ends[i] - starts[i] + 1
            dur_ms = dur / self.fs * 1000.0
            # Prefer regions with plausible P-wave duration (60-160ms)
            dur_score = np.exp(-((dur_ms - 100.0) / 40.0) ** 2)
            mid = (starts[i] + ends[i]) / 2
            max_res = np.max(np.abs(phi_res_smooth[starts[i] : ends[i] + 1]))
            dist_penalty = abs(mid - expected_rel) / max(expected_rel, 1)
            score = max_res * 3.0 + dur_score * 2.0 - dist_penalty
            if score > best_score:
                best_score = score
                best_idx = i

        p_on_win = int(starts[best_idx])
        p_off_win = int(ends[best_idx])

        # ---- Pass 2: |dφ/dt| refinement ----
        dphi = np.abs(np.gradient(phi))
        win_d = min(5, len(dphi) - (len(dphi) % 2 == 0) - 1)
        if win_d >= 3:
            dphi_s = savgol_filter(dphi, win_d, 2)
        else:
            dphi_s = dphi

        # Pull onset rightward to the nearest |dφ/dt| peak (start of transition)
        search_range = max(int(0.04 * self.fs), 5)
        for offset_dir in [(p_on_win, 1, min(p_on_win + search_range, len(dphi_s) - 2)),
                            (p_off_win, -1, max(p_off_win - search_range, 2))]:
            cur, direction, limit = offset_dir
            best = cur
            best_val = dphi_s[cur]
            if direction > 0:
                rng = range(cur, limit)
            else:
                rng = range(cur, limit, -1)
            for j in rng:
                if dphi_s[j] > best_val:
                    best_val = dphi_s[j]
                    best = j
            if direction > 0:
                p_off_win = best
            else:
                p_on_win = best

        if p_off_win - p_on_win < 4:
            return None

        # Validate amplitude
        p_amp = np.max(np.abs(ecg_win[p_on_win : p_off_win + 1]))
        if p_amp < 0.02:
            return None

        onset_sample = ws + p_on_win
        offset_sample = ws + p_off_win
        p_ecg_seg = ecg[onset_sample : offset_sample + 1]

        peak_offset = np.argmax(np.abs(p_ecg_seg))
        peak_sample = onset_sample + peak_offset
        duration_ms = len(p_ecg_seg) / self.fs * 1000.0

        # Phase range
        phase_range = float(
            np.max(np.abs(phi_res_smooth[p_on_win : p_off_win + 1]))
        )

        # Confidence
        hr = max(heart_rate, 30)
        rr_ms = 60000.0 / hr
        expected_dur = np.clip(80.0 + (rr_ms - 600.0) * 0.05, 60.0, 140.0)
        dur_ok = np.exp(-((duration_ms - expected_dur) / max(expected_dur, 1)) ** 2)
        phase_ok = np.clip(phase_range / 0.2, 0.0, 1.0)
        confidence = float(np.clip(0.6 * phase_ok + 0.4 * dur_ok, 0.0, 1.0))

        return PhasorPWave(
            beat_id=beat.beat_id,
            onset_sample=onset_sample,
            offset_sample=offset_sample,
            peak_sample=peak_sample,
            duration_ms=round(duration_ms, 2),
            phase_range=round(phase_range, 3),
            confidence=round(confidence, 3),
        )


# =============================================================================
# Method 3: HSMM + Time-Frequency Analysis (CWT)
# =============================================================================
@dataclass
class TFPWave:
    beat_id: int
    onset_sample: int
    offset_sample: int
    peak_sample: int
    duration_ms: float
    peak_freq_hz: float      # dominant frequency in P region
    energy_ratio: float      # P-band energy / total energy
    confidence: float = 1.0


class TimeFrequencyPWaveDetector:
    """P-wave detection via CWT within HSMM-anchored windows.

    Applies Continuous Wavelet Transform (Ricker/Mexican Hat) to the
    P-region window. P-wave energy concentrates in the 5–15 Hz band
    (scales corresponding to 15–40 ms at 250 Hz). Boundaries are
    detected from the time-localised energy envelope in the P-band.

    Reference
    ---------
    Addison, P.S. (2005). "Wavelet transforms and the ECG: a review."
    Physiological Measurement, 26(5), R155.
    """

    def __init__(self, fs: float = DEFAULT_FS,
                 widths: np.ndarray = CWT_WIDTHS):
        self.fs = fs
        self.widths = widths

    def compute_cwt(self, ecg: np.ndarray) -> np.ndarray:
        """Compute CWT scalogram (abs)."""
        if len(ecg) < 4:
            return np.zeros((len(self.widths), len(ecg)))
        try:
            coeffs = _cwt_manual(ecg, self.widths, _ricker)
            return np.abs(coeffs)
        except Exception:
            return np.zeros((len(self.widths), len(ecg)))

    def detect(self, ecg: np.ndarray, beat, heart_rate: float) -> TFPWave | None:
        """Detect P-wave for one beat using CWT time-frequency analysis.

        Parameters
        ----------
        ecg : np.ndarray — full filtered ECG
        beat : BeatBoundary
        heart_rate : float

        Returns
        -------
        TFPWave or None
        """
        T = len(ecg)

        # Search window
        if beat.p_onset > 0:
            p_expected = beat.p_onset
        else:
            p_expected = beat.q_onset - int(0.20 * self.fs)

        window_before = int(0.30 * self.fs)
        window_after = int(0.05 * self.fs)
        ws = max(0, p_expected - window_before)
        we = min(T - 1, p_expected + window_after)
        if beat.q_onset > 0:
            we = min(we, beat.q_onset - 1)
        if we - ws < 20:
            return None

        ecg_win = ecg[ws : we + 1].copy()
        ecg_win = ecg_win - np.median(ecg_win[: max(5, len(ecg_win) // 4)])

        # ---- CWT ----
        cwt_mat = self.compute_cwt(ecg_win)
        if cwt_mat.size == 0:
            return None

        # P-band: scales where wavelet centre frequency ≈ 5–15 Hz
        # Ricker wavelet centre freq ≈ 0.25 / width at 250 Hz
        # scales 4–12 → roughly 5–16 Hz
        p_band_lo, p_band_hi = 3, min(13, len(self.widths) - 1)
        if p_band_hi <= p_band_lo:
            p_band_lo, p_band_hi = 0, len(self.widths) - 1

        # Energy envelope in P-band
        p_energy = np.mean(cwt_mat[p_band_lo : p_band_hi + 1, :], axis=0)
        total_energy = np.mean(cwt_mat, axis=0) + 1e-10

        # Smooth energy envelope
        smooth_len = min(7, len(p_energy) - (len(p_energy) % 2 == 0) - 1)
        if smooth_len >= 3:
            p_energy_smooth = savgol_filter(p_energy, smooth_len, 2)
        else:
            p_energy_smooth = p_energy

        # Threshold
        baseline = np.median(p_energy_smooth[: max(3, len(p_energy_smooth) // 4)])
        noise_std = np.std(p_energy_smooth[: max(3, len(p_energy_smooth) // 4)])
        thresh = baseline + 2.5 * noise_std + 1e-8

        above = p_energy_smooth > thresh
        if not np.any(above):
            return None

        transitions = np.diff(above.astype(int))
        starts = np.where(transitions == 1)[0] + 1
        ends = np.where(transitions == -1)[0]
        if above[0]:
            starts = np.concatenate([[0], starts])
        if above[-1]:
            ends = np.concatenate([ends, [len(above) - 1]])

        if len(starts) == 0:
            return None

        # Best region: highest energy closest to expected P
        expected_rel = p_expected - ws
        best_idx = 0
        best_score = -np.inf
        for i in range(len(starts)):
            mid = (starts[i] + ends[i]) / 2
            peak_e = np.max(p_energy_smooth[starts[i] : ends[i] + 1])
            dist_penalty = abs(mid - expected_rel) / max(expected_rel, 1)
            score = peak_e * 3.0 - dist_penalty
            if score > best_score:
                best_score = score
                best_idx = i

        p_on_win = int(starts[best_idx])
        p_off_win = int(ends[best_idx])

        # Refine
        for i in range(p_on_win, max(2, p_on_win - 10), -1):
            if p_energy_smooth[i] <= baseline + noise_std:
                p_on_win = i
            else:
                break
        for i in range(p_off_win, min(len(p_energy_smooth) - 2, p_off_win + 10)):
            if p_energy_smooth[i] <= baseline + noise_std:
                p_off_win = i
            else:
                break

        if p_off_win - p_on_win < 4:
            return None

        onset_sample = ws + p_on_win
        offset_sample = ws + p_off_win
        p_ecg_seg = ecg[onset_sample : offset_sample + 1]

        peak_offset = np.argmax(np.abs(p_ecg_seg))
        peak_sample = onset_sample + peak_offset
        duration_ms = len(p_ecg_seg) / self.fs * 1000.0

        # Dominant frequency and energy ratio
        cwt_seg = cwt_mat[:, p_on_win : p_off_win + 1] if p_off_win > p_on_win else cwt_mat[:, p_on_win:p_on_win+1]
        scale_profile = np.mean(cwt_seg, axis=1)
        best_scale_idx = int(np.argmax(scale_profile))
        peak_freq = 0.25 / max(self.widths[best_scale_idx], 0.01) * self.fs  # approx
        peak_freq_hz = round(peak_freq, 1)

        e_p_band = np.sum(scale_profile[p_band_lo : p_band_hi + 1])
        e_total = np.sum(scale_profile) + 1e-10
        energy_ratio = round(float(e_p_band / e_total), 3)

        hr = max(heart_rate, 30)
        rr_ms = 60000.0 / hr
        expected_dur = np.clip(80.0 + (rr_ms - 600.0) * 0.05, 60.0, 140.0)
        dur_ok = np.exp(-((duration_ms - expected_dur) / max(expected_dur, 1)) ** 2)
        confidence = float(np.clip(0.5 * energy_ratio + 0.5 * dur_ok, 0.0, 1.0))

        return TFPWave(
            beat_id=beat.beat_id,
            onset_sample=onset_sample,
            offset_sample=offset_sample,
            peak_sample=peak_sample,
            duration_ms=round(duration_ms, 2),
            peak_freq_hz=peak_freq_hz,
            energy_ratio=energy_ratio,
            confidence=round(confidence, 3),
        )


# =============================================================================
# Plotting
# =============================================================================

# Consistent colours
C_TEMPLATE = "#2196F3"   # blue
C_PHASOR   = "#FF9800"   # orange
C_CWT      = "#4CAF50"   # green
C_ECG       = "#212121"
C_P_HIGHLIGHT = "#E3F2FD"
C_QRS       = "#E53935"   # red for QRS

METHOD_STYLES = {
    "template": dict(color=C_TEMPLATE, lw=2.5, ls="-",  label="HSMM+Template"),
    "phasor":   dict(color=C_PHASOR,   lw=2.5, ls="--", label="HSMM+Phasor"),
    "cwt":      dict(color=C_CWT,      lw=2.5, ls="-.", label="HSMM+CWT"),
}


def _plot_method_band(ax, pw, t_offset_sec, style, ecg_seg, y_bottom):
    """Draw a filled band for this method's P-wave and a boundary marker line."""
    if pw is None or pw.onset_sample < 0:
        return
    t0 = pw.onset_sample / DEFAULT_FS - t_offset_sec
    t1 = pw.offset_sample / DEFAULT_FS - t_offset_sec
    t_vals = np.linspace(t0, t1, 50)
    # P region highlight
    ax.fill_between(t_vals, y_bottom, y_bottom + 0.06,
                    alpha=0.22, color=style["color"], linewidth=0)
    # Onset/offset markers
    ax.axvline(t0, color=style["color"], ls=style["ls"], lw=style["lw"], alpha=0.8)
    ax.axvline(t1, color=style["color"], ls=style["ls"], lw=style["lw"], alpha=0.8)


def plot_overview(ecg, beats, results_template, results_phasor, results_cwt,
                  rec_name, save_path, max_sec=4.0, dpi=200):
    """3-row overview: one row per method, showing P-wave detections.

    Each row shows the filtered ECG with P-wave bands overlaid.
    """
    T = len(ecg)
    n_plot = min(int(max_sec * DEFAULT_FS), T)
    t_plot = np.arange(n_plot) / DEFAULT_FS
    e_plot = ecg[:n_plot]

    fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True, sharey=True)
    fig.suptitle(f"{rec_name} — Three-Method P-Wave Segmentation Comparison",
                 fontsize=15, fontweight="bold")

    method_results = [
        ("template", results_template, axes[0],
         "Method 1: HSMM + Correlation-Enhanced Multi-Feature Template"),
        ("phasor", results_phasor, axes[1],
         "Method 2: HSMM + Phasor Transform (R_V=0.002)"),
        ("cwt", results_cwt, axes[2],
         "Method 3: HSMM + Time-Frequency Analysis (CWT, Ricker)"),
    ]

    for method_key, results, ax, title in method_results:
        # ECG trace
        ax.plot(t_plot, e_plot, color=C_ECG, linewidth=0.5, alpha=0.85, zorder=1)

        # ---- QRS region highlighting (same for all methods) ----
        for b in beats:
            q_on = getattr(b, "q_onset", -1)
            s_off = getattr(b, "s_offset", -1)
            r_pk = getattr(b, "r_peak", -1)
            if q_on < 0 or s_off <= q_on:
                continue
            if q_on >= n_plot:
                break
            s_clip = min(s_off, n_plot - 1)
            if s_clip <= q_on:
                continue
            ax.fill_between(
                t_plot[q_on : s_clip + 1],
                e_plot[q_on : s_clip + 1],
                alpha=0.12, color=C_QRS, linewidth=0, zorder=2,
            )
            # R peak marker
            if 0 <= r_pk < n_plot:
                ax.plot(r_pk / DEFAULT_FS, e_plot[r_pk], "v",
                        color=C_QRS, markersize=5, alpha=0.8, zorder=5)

        # ---- P-wave method-specific highlighting ----
        style = METHOD_STYLES[method_key]
        for r in results:
            if r is None:
                continue
            onset = getattr(r, "onset_sample", -1)
            offset = getattr(r, "offset_sample", -1)
            if onset < 0 or offset <= onset:
                continue
            if onset >= n_plot:
                break
            off_clip = min(offset, n_plot - 1)
            if off_clip <= onset:
                continue
            ax.fill_between(
                t_plot[onset : off_clip + 1],
                e_plot[onset : off_clip + 1],
                alpha=0.28, color=style["color"], linewidth=0,
            )
            # Onset/offset markers
            ax.axvline(onset / DEFAULT_FS, color=style["color"],
                       ls=style["ls"], lw=style["lw"], alpha=0.7)
            ax.axvline(off_clip / DEFAULT_FS, color=style["color"],
                       ls=style["ls"], lw=style["lw"], alpha=0.7)

        ax.set_ylabel("Amplitude (norm)", fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold", color=style["color"])
        ax.grid(True, alpha=0.12)

        # Count detections in view
        n_det = sum(
            1 for r in results
            if r is not None and getattr(r, "onset_sample", -1) >= 0
            and getattr(r, "onset_sample", -1) < n_plot
        )
        ax.text(0.99, 0.94, f"{n_det} P-waves detected",
                transform=ax.transAxes, fontsize=9, ha="right",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    axes[-1].set_xlim(t_plot[0], t_plot[-1])

    # Shared legend
    legend_elements = [
        Patch(facecolor=C_TEMPLATE, alpha=0.3, label="M1: HSMM+Template"),
        Patch(facecolor=C_PHASOR, alpha=0.3, label="M2: HSMM+Phasor"),
        Patch(facecolor=C_CWT, alpha=0.3, label="M3: HSMM+CWT"),
        Patch(facecolor=C_QRS, alpha=0.2, label="QRS (HSMM Stage 1)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=10, framealpha=0.9)

    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_beat_detail(ecg, beat, pw_t, pw_p, pw_c, rec_name,
                     save_path, dpi=200):
    """Zoomed single-beat plot comparing all three methods.

    Shows P-wave (3 methods) + QRS (HSMM Stage 1) + T-wave boundaries.
    """
    T = len(ecg)
    fs = DEFAULT_FS

    # ---- Window: P onset → T offset + margins ----
    margin = int(0.15 * fs)
    # P region start
    if beat.p_onset > 0:
        p_centre = (beat.p_onset + beat.p_offset) // 2
    else:
        p_centre = beat.q_onset - int(0.16 * fs)
    ws = max(0, p_centre - int(0.30 * fs))
    # Extend to include QRS + T
    if beat.t_offset > 0:
        we = min(T - 1, beat.t_offset + margin)
    elif beat.s_offset > 0:
        we = min(T - 1, beat.s_offset + int(0.40 * fs))
    else:
        we = min(T - 1, beat.r_peak + int(0.50 * fs) if beat.r_peak > 0 else p_centre + int(0.60 * fs))
    if we - ws < 20:
        return

    t_win = np.arange(ws, we + 1) / fs
    e_win = ecg[ws : we + 1]

    fig, (ax_ecg, ax_phase, ax_cwt) = plt.subplots(3, 1, figsize=(14, 10),
                                                     gridspec_kw={"height_ratios": [2, 1, 1]})
    fig.suptitle(f"{rec_name} — Beat {beat.beat_id}  Three-Method P-Wave + QRS Comparison",
                 fontsize=13, fontweight="bold")

    # ---- Top: ECG with P-wave (3 methods) + QRS + T-wave ----
    ax_ecg.plot(t_win, e_win, color=C_ECG, linewidth=0.9, zorder=1)
    ax_ecg.set_ylabel("Amplitude (norm)", fontsize=9)
    ax_ecg.set_title("ECG: P-Wave (3 methods) + QRS + T-Wave (HSMM Stage 1)", fontsize=11, fontweight="bold")
    ax_ecg.grid(True, alpha=0.12)

    y_min, y_max = e_win.min(), e_win.max()

    # ---- P-wave: three methods ----
    for method_key, pw, style in [
        ("template", pw_t, METHOD_STYLES["template"]),
        ("phasor", pw_p, METHOD_STYLES["phasor"]),
        ("cwt", pw_c, METHOD_STYLES["cwt"]),
    ]:
        if pw is not None and pw.onset_sample >= 0:
            on = pw.onset_sample - ws
            off = pw.offset_sample - ws
            if 0 <= on < len(e_win) and off < len(e_win) and off > on:
                ax_ecg.fill_between(
                    t_win[on : off + 1], e_win[on : off + 1],
                    alpha=0.18, color=style["color"], linewidth=0,
                )
                ax_ecg.axvline(t_win[on], color=style["color"],
                               ls=style["ls"], lw=style["lw"], alpha=0.85)
                ax_ecg.axvline(t_win[off], color=style["color"],
                               ls=style["ls"], lw=style["lw"], alpha=0.85)

    # ---- QRS region ----
    if beat.q_onset > 0 and beat.s_offset > 0:
        q_rel = beat.q_onset - ws
        s_rel = beat.s_offset - ws
        if 0 <= q_rel < len(e_win) and s_rel < len(e_win) and s_rel > q_rel:
            ax_ecg.fill_between(
                t_win[q_rel : s_rel + 1], e_win[q_rel : s_rel + 1],
                alpha=0.15, color=C_QRS, linewidth=0, zorder=2,
            )
            # QRS onset
            ax_ecg.axvline(t_win[q_rel], color=C_QRS, ls="-", lw=2.0, alpha=0.8)
            ax_ecg.annotate("Q on", (t_win[q_rel], y_min),
                            textcoords="offset points", xytext=(-2, -10),
                            fontsize=7, color=C_QRS, ha="right", fontweight="bold")
            # QRS offset
            ax_ecg.axvline(t_win[s_rel], color=C_QRS, ls="-", lw=2.0, alpha=0.8)
            ax_ecg.annotate("S off", (t_win[s_rel], y_min),
                            textcoords="offset points", xytext=(2, -10),
                            fontsize=7, color=C_QRS, ha="left", fontweight="bold")

    # ---- R peak ----
    if beat.r_peak > 0:
        r_rel = beat.r_peak - ws
        if 0 <= r_rel < len(e_win):
            ax_ecg.plot(t_win[r_rel], e_win[r_rel], "v",
                        color=C_QRS, markersize=10, markeredgecolor="darkred",
                        markeredgewidth=1.5, zorder=5)
            ax_ecg.annotate("R", (t_win[r_rel], e_win[r_rel]),
                            textcoords="offset points", xytext=(8, 6),
                            fontsize=8, color=C_QRS, fontweight="bold")

    # ---- T-wave region ----
    if beat.t_onset > 0 and beat.t_offset > 0:
        t_on_rel = beat.t_onset - ws
        t_off_rel = beat.t_offset - ws
        if 0 <= t_on_rel < len(e_win) and t_off_rel < len(e_win) and t_off_rel > t_on_rel:
            ax_ecg.fill_between(
                t_win[t_on_rel : t_off_rel + 1], e_win[t_on_rel : t_off_rel + 1],
                alpha=0.10, color="#2E7D32", linewidth=0, zorder=1,
            )
            ax_ecg.axvline(t_win[t_off_rel], color="#2E7D32", ls=":", lw=1.5, alpha=0.6)
            ax_ecg.annotate("T off", (t_win[t_off_rel], y_max),
                            textcoords="offset points", xytext=(2, 4),
                            fontsize=7, color="#2E7D32", ha="left")

    # QRS duration annotation
    if beat.q_onset > 0 and beat.s_offset > 0:
        qrs_dur = (beat.s_offset - beat.q_onset) / fs * 1000.0
        qrs_mid = (beat.q_onset + beat.s_offset) / 2
        qrs_mid_rel = qrs_mid - ws
        if 0 <= qrs_mid_rel < len(e_win):
            ax_ecg.annotate(f"QRS\n{qrs_dur:.0f}ms",
                            (t_win[int(qrs_mid_rel)], y_min),
                            textcoords="offset points", xytext=(0, -20),
                            fontsize=7, color=C_QRS, ha="center")

    ax_ecg.legend(
        handles=[
            Patch(facecolor=C_TEMPLATE, alpha=0.3, label="M1: Template"),
            Patch(facecolor=C_PHASOR, alpha=0.3, label="M2: Phasor"),
            Patch(facecolor=C_CWT, alpha=0.3, label="M3: CWT"),
            Patch(facecolor=C_QRS, alpha=0.2, label="QRS (HSMM)"),
            Patch(facecolor="#2E7D32", alpha=0.15, label="T-wave (HSMM)"),
        ],
        loc="upper right", fontsize=7.5, framealpha=0.8, ncol=2,
    )

    # ---- Middle: Phasor domain (phase + residual) ----
    ecg_win = ecg[ws : we + 1].copy()
    ecg_win_dt = ecg_win - np.median(ecg_win[: max(5, len(ecg_win) // 4)])
    phi = np.arctan2(ecg_win_dt, PHASOR_R_V)

    # Phase residual (matching the detector logic)
    med_wide = max(int(0.12 * DEFAULT_FS), 7)
    if med_wide % 2 == 0:
        med_wide += 1
    from scipy.signal import medfilt
    phi_bl = medfilt(phi, med_wide)
    phi_residual = phi - phi_bl

    win_phi = min(11, len(phi) - (len(phi) % 2 == 0) - 1)
    if win_phi >= 3:
        phi_smooth = savgol_filter(phi, win_phi, 3)
    else:
        phi_smooth = phi

    # Plot raw phase + baseline
    ax_phase.plot(t_win, phi, color="#BDBDBD", linewidth=0.4, alpha=0.5, label="Raw φ")
    ax_phase.plot(t_win, phi_bl, color="#9E9E9E", linewidth=0.8, ls="--", alpha=0.7, label="Baseline φ_bl")
    ax_phase.plot(t_win, phi_smooth, color=C_PHASOR, linewidth=1.2, label="Smoothed φ")
    ax_phase.plot(t_win, phi_residual, color="#E65100", linewidth=0.8, alpha=0.8, label="φ residual")
    ax_phase.axhline(0, color="gray", ls="--", lw=0.6, alpha=0.5)
    if pw_p is not None and pw_p.onset_sample >= 0:
        on = pw_p.onset_sample - ws
        off = pw_p.offset_sample - ws
        if 0 <= on < len(t_win) and off < len(t_win):
            ax_phase.axvspan(t_win[on], t_win[off], alpha=0.15, color=C_PHASOR)
            ax_phase.axvline(t_win[on], color=C_PHASOR, ls="--", lw=1.5)
            ax_phase.axvline(t_win[off], color=C_PHASOR, ls="--", lw=1.5)
    ax_phase.set_ylabel("Phase φ (rad)", fontsize=9)
    ax_phase.set_title(f"Phasor Transform Domain  (R_V={PHASOR_R_V})", fontsize=10,
                       fontweight="bold", color=C_PHASOR)
    ax_phase.legend(fontsize=7, loc="upper right")
    ax_phase.grid(True, alpha=0.12)

    # ---- Bottom: CWT scalogram ----
    widths = CWT_WIDTHS
    try:
        cwt_mat = np.abs(_cwt_manual(ecg_win_dt, widths, _ricker))
    except Exception:
        cwt_mat = np.zeros((len(widths), len(ecg_win_dt)))

    if cwt_mat.size > 0:
        vmax = np.percentile(cwt_mat, 95)
        im = ax_cwt.pcolormesh(t_win, np.arange(len(widths)), cwt_mat,
                               cmap="viridis", shading="auto",
                               vmin=0, vmax=max(vmax, 0.01))
        # Mark P-band
        ax_cwt.axhspan(3, 13, alpha=0.08, color="red")
        ax_cwt.annotate("P-band\n(5-15 Hz)", (t_win[0] + 0.02, 6),
                        fontsize=7, color="red", va="center")
        if pw_c is not None and pw_c.onset_sample >= 0:
            on = pw_c.onset_sample - ws
            off = pw_c.offset_sample - ws
            if 0 <= on < len(t_win) and off < len(t_win):
                ax_cwt.axvline(t_win[on], color=C_CWT, ls="-.", lw=1.5)
                ax_cwt.axvline(t_win[off], color=C_CWT, ls="-.", lw=1.5)
        plt.colorbar(im, ax=ax_cwt, label="|CWT|", shrink=0.9)
    ax_cwt.set_ylabel("Wavelet Scale", fontsize=9)
    ax_cwt.set_xlabel("Time (s)", fontsize=9)
    ax_cwt.set_title("CWT Scalogram (Ricker wavelet)", fontsize=10,
                     fontweight="bold", color=C_CWT)

    # ---- Info table ----
    lines = [f"Beat {beat.beat_id} — Waveform Boundaries"]
    lines.append("-" * 75)
    # QRS info
    qrs_dur_ms = (beat.s_offset - beat.q_onset) / fs * 1000.0 if beat.q_onset > 0 and beat.s_offset > 0 else 0
    lines.append(f"{'QRS (HSMM):':<22} {'Q on':>8} {beat.q_onset/fs*1000:>8.1f}ms  {'S off':>8} {beat.s_offset/fs*1000:>8.1f}ms  {'Dur':>6} {qrs_dur_ms:>7.1f}ms")
    lines.append(f"{'R peak:':<22} {beat.r_peak/fs*1000:>8.1f}ms")
    if beat.t_offset > 0:
        t_dur = (beat.t_offset - beat.t_onset) / fs * 1000.0 if beat.t_onset > 0 else 0
        lines.append(f"{'T-wave (HSMM):':<22} {'T off':>8} {beat.t_offset/fs*1000:>8.1f}ms  {'Dur':>6} {t_dur:>7.1f}ms")
    lines.append("")
    lines.append(f"{'P-Wave Method':<22} {'Onset(ms)':>10} {'Offset(ms)':>10} {'Dur(ms)':>8} {'Conf':>6}")
    lines.append("-" * 60)
    for label, pw in [("M1: Template", pw_t), ("M2: Phasor", pw_p), ("M3: CWT", pw_c)]:
        if pw is not None and pw.onset_sample >= 0:
            lines.append(
                f"{label:<22} {pw.onset_sample/fs*1000:>10.1f} "
                f"{pw.offset_sample/fs*1000:>10.1f} "
                f"{pw.duration_ms:>8.1f} {pw.confidence:>6.3f}"
            )
        else:
            lines.append(f"{label:<22} {'—':>10} {'—':>10} {'—':>8} {'—':>6}")
    info_text = "\n".join(lines)

    fig.text(0.5, -0.02, info_text, transform=ax_cwt.transAxes,
             fontsize=8, ha="center", va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="white", alpha=0.9,
                       edgecolor="gray"))

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main processing
# =============================================================================
def process_record(rec_path, rec_name, segmenter, p_extractor,
                   phasor_detector, cwt_detector, out_dir):
    """Process one record with all three methods and save plots."""
    print(f"  Parsing {rec_name} ...")
    data = parse_aecg(str(rec_path))
    fs = data["fs"]

    # Use Lead II
    lead = "II"
    if lead not in data["signals"]:
        lead = list(data["signals"].keys())[0]
    raw_ecg = data["signals"][lead]

    # Limit to first 16s for speed (at 360Hz after resampling)
    max_samps = int(16.0 * fs)
    if len(raw_ecg) > max_samps:
        raw_ecg = raw_ecg[:max_samps]

    print(f"    Lead {lead}, {len(raw_ecg)} samples @ {fs:.0f} Hz")

    # Resample to model's sampling rate if different
    model_fs = segmenter.fs
    if abs(fs - model_fs) > 5:
        from scipy.signal import resample_poly
        import math
        gcd = math.gcd(int(fs), int(model_fs))
        up = int(model_fs) // gcd
        down = int(fs) // gcd
        print(f"    Resampling {fs:.0f} → {model_fs:.0f} Hz (ratio {up}/{down}) ...")
        raw_ecg = resample_poly(raw_ecg.astype(np.float64), up, down)
        fs = model_fs

    # Update detectors' fs
    phasor_detector.fs = fs
    cwt_detector.fs = fs
    p_extractor.fs = fs

    # ---- HSMM Stage 1 ----
    try:
        seg_result = segmenter.segment(raw_ecg)
    except Exception as e:
        print(f"    Stage 1 failed: {e}")
        return

    beats = seg_result.beats
    if not beats:
        print("    No beats found.")
        return

    # Estimate heart rate
    r_peaks = [b.r_peak for b in beats if b.r_peak > 0]
    if len(r_peaks) >= 2:
        rr_mean = np.mean(np.diff(r_peaks)) / fs * 1000.0
        heart_rate = 60000.0 / rr_mean if rr_mean > 0 else 60.0
    else:
        heart_rate = 60.0

    ecg_filt = seg_result.filtered_ecg

    # ---- Method 1: HSMM + Template ----
    print(f"    Method 1: HSMM+Template ...")
    pw_template = p_extractor.extract(seg_result, heart_rate=heart_rate)

    # ---- Method 2: HSMM + Phasor Transform ----
    print(f"    Method 2: HSMM+Phasor ...")
    pw_phasor = []
    for b in beats:
        try:
            pw = phasor_detector.detect(ecg_filt, b, heart_rate)
        except Exception:
            pw = None
        pw_phasor.append(pw)

    # ---- Method 3: HSMM + CWT ----
    print(f"    Method 3: HSMM+CWT ...")
    pw_cwt = []
    for b in beats:
        try:
            pw = cwt_detector.detect(ecg_filt, b, heart_rate)
        except Exception:
            pw = None
        pw_cwt.append(pw)

    n_beats = len(beats)
    n_detected = {
        "template": sum(1 for p in pw_template if p.onset_sample > 0),
        "phasor":   sum(1 for p in pw_phasor if p is not None and p.onset_sample >= 0),
        "cwt":      sum(1 for p in pw_cwt if p is not None and p.onset_sample >= 0),
    }
    print(f"    Detected P-waves: Template={n_detected['template']}, "
          f"Phasor={n_detected['phasor']}, CWT={n_detected['cwt']} / {n_beats} beats")

    # ---- Plots ----
    rec_dir = out_dir / rec_name
    os.makedirs(rec_dir, exist_ok=True)

    # Overview
    plot_overview(ecg_filt, beats, pw_template, pw_phasor, pw_cwt,
                  rec_name, str(rec_dir / "overview.png"))

    # Per-beat detail
    max_beats = min(len(beats), MAX_BEATS_PER_RECORD)
    for i in range(max_beats):
        b = beats[i]
        pt = pw_template[i] if i < len(pw_template) else None
        pp = pw_phasor[i] if i < len(pw_phasor) else None
        pc = pw_cwt[i] if i < len(pw_cwt) else None
        plot_beat_detail(ecg_filt, b, pt, pp, pc, rec_name,
                         str(rec_dir / f"beat_{b.beat_id:03d}.png"))

    # ---- Summary JSON ----
    summary = {
        "record": rec_name,
        "lead": lead,
        "fs": fs,
        "heart_rate_bpm": round(heart_rate, 1),
        "n_beats": n_beats,
        "n_detected": n_detected,
        "beats": [],
    }
    for i in range(n_beats):
        b = beats[i]
        pt = pw_template[i] if i < len(pw_template) else None
        pp = pw_phasor[i] if i < len(pw_phasor) else None
        pc = pw_cwt[i] if i < len(pw_cwt) else None
        entry = {
            "beat_id": b.beat_id,
            "r_peak_sample": b.r_peak,
            "template": _pw_to_dict(pt),
            "phasor": _pw_to_dict(pp),
            "cwt": _pw_to_dict(pc),
        }
        summary["beats"].append(entry)

    with open(rec_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"    Saved {max_beats} beat plots + overview → {rec_dir}")
    return summary


def _pw_to_dict(pw):
    if pw is None:
        return None
    if hasattr(pw, "onset_sample"):
        d = {
            "onset_sample": int(pw.onset_sample) if pw.onset_sample >= 0 else -1,
            "offset_sample": int(pw.offset_sample) if pw.offset_sample >= 0 else -1,
            "peak_sample": int(pw.peak_sample) if pw.peak_sample >= 0 else -1,
            "duration_ms": getattr(pw, "duration_ms", 0),
            "confidence": getattr(pw, "confidence", 0),
        }
        # Add method-specific fields
        if hasattr(pw, "phase_range"):
            d["phase_range"] = pw.phase_range
        if hasattr(pw, "peak_freq_hz"):
            d["peak_freq_hz"] = pw.peak_freq_hz
        if hasattr(pw, "energy_ratio"):
            d["energy_ratio"] = pw.energy_ratio
        if hasattr(pw, "morphology"):
            d["morphology"] = pw.morphology
        if hasattr(pw, "snr_db"):
            d["snr_db"] = pw.snr_db
        if hasattr(pw, "symmetry"):
            d["symmetry"] = pw.symmetry
        if hasattr(pw, "consistency"):
            d["consistency"] = pw.consistency
        return d
    return None


# =============================================================================
# Main entry point
# =============================================================================
def main():
    print("=" * 60)
    print("Three-Method P-Wave Segmentation Comparison")
    print("=" * 60)

    # ---- Load model ----
    print(f"\nLoading model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return
    model = HSMMModel.load(MODEL_PATH)
    print(f"  Model: {model}")

    # ---- Init pipeline ----
    fs = model.fs
    preprocessor = ECGPreprocessor(fs=fs)
    feature_extractor = FeatureExtractor(fs=fs)
    decoder = HSMMDecoder()
    segmenter = ECGSegmenter(
        preprocessor=preprocessor,
        feature_extractor=feature_extractor,
        model=model,
        decoder=decoder,
        fs=fs,
    )

    # ---- Init detectors ----
    p_extractor = PWaveExtractor(fs=fs,
                                 enable_template_fallback=True,
                                 refine_boundaries=True)
    phasor_detector = PhasorPWaveDetector(fs=fs)
    cwt_detector = TimeFrequencyPWaveDetector(fs=fs)

    # ---- Find records ----
    aecg_files = sorted(glob.glob(str(AECG_DIR / "*.aECG")) + glob.glob(str(AECG_DIR / "*.xml")))
    if not aecg_files:
        print(f"\nNo aECG files found in {AECG_DIR}")
        print("Looking for alternative data sources...")
        # Try output/trained directory for pre-loaded NPY files
        alt_dir = Path(__file__).resolve().parent.parent / "output/trained"
        if alt_dir.exists():
            rec_dirs = sorted([d for d in alt_dir.iterdir() if d.is_dir()])
            if rec_dirs:
                print(f"Found {len(rec_dirs)} output/trained record dirs")
                # Process from raw NPY files
                return process_from_output_trained(
                    rec_dirs[:5], segmenter, p_extractor,
                    phasor_detector, cwt_detector,
                )
        print("No data found.")
        return

    print(f"\nFound {len(aecg_files)} aECG files. Processing up to 50 ...")

    all_summaries = []
    for rec_path in aecg_files[:50]:
        rec_name = Path(rec_path).stem
        print(f"\n--- {rec_name} ---")
        try:
            summary = process_record(
                rec_path, rec_name, segmenter, p_extractor,
                phasor_detector, cwt_detector,
                OUTPUT_BASE,
            )
            if summary:
                all_summaries.append(summary)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # ---- Aggregate summary ----
    if all_summaries:
        agg = {
            "n_records": len(all_summaries),
            "records": all_summaries,
        }
        with open(OUTPUT_BASE / "aggregate_summary.json", "w") as f:
            json.dump(agg, f, indent=2, default=str)

    print(f"\nDone. Output: {OUTPUT_BASE}")


def process_from_output_trained(rec_dirs, segmenter, p_extractor,
                                phasor_detector, cwt_detector):
    """Fallback: process records already saved as NPY files."""
    all_summaries = []
    for rec_dir in rec_dirs[:5]:
        rec_name = rec_dir.name
        raw_path = rec_dir / "raw_ecg.npy"
        if not raw_path.exists():
            continue
        print(f"\n--- {rec_name} ---")
        try:
            raw_ecg = np.load(str(raw_path))
            # Limit
            max_samps = int(8.0 * DEFAULT_FS)
            if len(raw_ecg) > max_samps:
                raw_ecg = raw_ecg[:max_samps]

            print(f"    {len(raw_ecg)} samples")
            seg_result = segmenter.segment(raw_ecg)
            beats = seg_result.beats
            if not beats:
                print("    No beats found.")
                continue

            r_peaks = [b.r_peak for b in beats if b.r_peak > 0]
            heart_rate = 60.0
            if len(r_peaks) >= 2:
                rr_mean = np.mean(np.diff(r_peaks)) / DEFAULT_FS * 1000.0
                heart_rate = 60000.0 / max(rr_mean, 1)

            ecg_filt = seg_result.filtered_ecg

            # Method 1
            pw_template = p_extractor.extract(seg_result, heart_rate=heart_rate)

            # Method 2
            pw_phasor = []
            for b in beats:
                try:
                    pw_phasor.append(phasor_detector.detect(ecg_filt, b, heart_rate))
                except Exception:
                    pw_phasor.append(None)

            # Method 3
            pw_cwt = []
            for b in beats:
                try:
                    pw_cwt.append(cwt_detector.detect(ecg_filt, b, heart_rate))
                except Exception:
                    pw_cwt.append(None)

            n_beats = len(beats)
            print(f"    P-waves: Template={sum(1 for p in pw_template if p.onset_sample>0)}, "
                  f"Phasor={sum(1 for p in pw_phasor if p and p.onset_sample>=0)}, "
                  f"CWT={sum(1 for p in pw_cwt if p and p.onset_sample>=0)} / {n_beats}")

            rec_out = OUTPUT_BASE / rec_name
            plot_overview(ecg_filt, beats, pw_template, pw_phasor, pw_cwt,
                          rec_name, str(rec_out / "overview.png"))

            max_beats = min(len(beats), MAX_BEATS_PER_RECORD)
            for i in range(max_beats):
                b = beats[i]
                pt = pw_template[i] if i < len(pw_template) else None
                pp = pw_phasor[i] if i < len(pw_phasor) else None
                pc = pw_cwt[i] if i < len(pw_cwt) else None
                plot_beat_detail(ecg_filt, b, pt, pp, pc, rec_name,
                                 str(rec_out / f"beat_{b.beat_id:03d}.png"))

            summary = {
                "record": rec_name,
                "lead": "II",
                "fs": DEFAULT_FS,
                "heart_rate_bpm": round(heart_rate, 1),
                "n_beats": n_beats,
                "n_detected": {
                    "template": sum(1 for p in pw_template if p.onset_sample > 0),
                    "phasor": sum(1 for p in pw_phasor if p and p.onset_sample >= 0),
                    "cwt": sum(1 for p in pw_cwt if p and p.onset_sample >= 0),
                },
                "beats": [],
            }
            for i in range(n_beats):
                b = beats[i]
                pt = pw_template[i] if i < len(pw_template) else None
                pp = pw_phasor[i] if i < len(pw_phasor) else None
                pc = pw_cwt[i] if i < len(pw_cwt) else None
                summary["beats"].append({
                    "beat_id": b.beat_id,
                    "r_peak_sample": b.r_peak,
                    "template": _pw_to_dict(pt),
                    "phasor": _pw_to_dict(pp),
                    "cwt": _pw_to_dict(pc),
                })
            with open(rec_out / "summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)
            all_summaries.append(summary)
            print(f"    Saved → {rec_out}")

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    if all_summaries:
        with open(OUTPUT_BASE / "aggregate_summary.json", "w") as f:
            json.dump({"n_records": len(all_summaries), "records": all_summaries},
                      f, indent=2, default=str)
    print(f"\nDone. Output: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
