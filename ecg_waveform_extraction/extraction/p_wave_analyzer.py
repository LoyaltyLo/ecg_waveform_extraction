"""P-wave morphological and clinical analysis.

Computes duration, amplitude, area, PR interval, P-wave dispersion,
and morphology classification from refined P-wave boundaries.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class PWaveFeatures:
    """Clinical measurements for a single P-wave.

    Attributes
    ----------
    beat_id : int
        Beat index.
    onset_sample : int
        P-wave onset sample.
    offset_sample : int
        P-wave offset sample.
    peak_sample : int
        P-wave peak sample.
    duration_ms : float
        P-wave duration (ms).
    peak_amplitude : float
        P-wave peak amplitude (absolute value).
    area : float
        Integrated absolute amplitude under P-wave.
    morphology_score : float
        Simple morphology metric (area / (duration * peak_amplitude)), proxy for shape.
    pr_interval_ms : float or None
        PR interval (P onset to QRS onset) in ms, if available.
    """
    beat_id: int
    onset_sample: int
    offset_sample: int
    peak_sample: int
    duration_ms: float
    peak_amplitude: float
    area: float
    morphology_score: float
    pr_interval_ms: float | None = None


@dataclass
class PWaveSummary:
    """Aggregate P-wave statistics across a recording.

    Attributes
    ----------
    n_beats : int
        Number of beats with valid P-waves.
    duration_mean_ms : float
        Mean P-wave duration.
    duration_std_ms : float
        Std dev of P-wave duration.
    duration_range_ms : tuple[float, float]
        (min, max) duration.
    dispersion_ms : float
        P-wave dispersion = max_duration - min_duration.
    amplitude_mean : float
        Mean peak amplitude.
    pr_mean_ms : float or None
        Mean PR interval.
    pr_std_ms : float or None
        Std dev of PR interval.
    flagged_beats : list[int]
        Beat IDs with abnormal P-wave morphology.
    """
    n_beats: int = 0
    duration_mean_ms: float = 0.0
    duration_std_ms: float = 0.0
    duration_range_ms: tuple[float, float] = (0.0, 0.0)
    dispersion_ms: float = 0.0
    amplitude_mean: float = 0.0
    pr_mean_ms: float | None = None
    pr_std_ms: float | None = None
    flagged_beats: list[int] = field(default_factory=list)


class PWaveAnalyzer:
    """Compute clinical P-wave measurements from refined boundaries.

    Parameters
    ----------
    fs : float
        Sampling frequency (Hz).
    duration_normal_range_ms : tuple[float, float]
        Normal P-wave duration range for flagging (default 80-120ms).
    pr_normal_range_ms : tuple[float, float]
        Normal PR interval range (default 120-200ms).
    """

    def __init__(self, fs: float = 250.0,
                 duration_normal_range_ms: tuple[float, float] = (80.0, 120.0),
                 pr_normal_range_ms: tuple[float, float] = (120.0, 200.0)):
        self.fs = fs
        self.duration_normal = duration_normal_range_ms
        self.pr_normal = pr_normal_range_ms

        self._ms_per_sample = 1000.0 / fs

    # ------------------------------------------------------------------
    # Per-beat analysis
    # ------------------------------------------------------------------
    def analyze(self, p_wave_results, filtered_ecg: np.ndarray,
                beats=None) -> list[PWaveFeatures]:
        """Compute P-wave features for each beat.

        Parameters
        ----------
        p_wave_results : list[PWaveResult]
            Output from PWaveExtractor.extract().
        filtered_ecg : np.ndarray, shape (T,)
            Preprocessed ECG signal.
        beats : list[BeatBoundary] or None
            Stage 1 beat boundaries, for PR interval computation.

        Returns
        -------
        list[PWaveFeatures]
        """
        features_list = []

        for pw in p_wave_results:
            if pw.onset_sample < 0 or pw.offset_sample < 0:
                continue

            # Duration
            duration_ms = pw.duration_ms
            if duration_ms <= 0:
                duration_ms = (pw.offset_sample - pw.onset_sample + 1) * self._ms_per_sample

            # Peak amplitude
            p_ecg = filtered_ecg[pw.onset_sample:pw.offset_sample + 1]
            if len(p_ecg) == 0:
                continue
            peak_amplitude = float(np.max(np.abs(p_ecg)))

            # Area (integrated absolute amplitude)
            area = float(np.sum(np.abs(p_ecg)) * self._ms_per_sample)

            # Morphology score
            if peak_amplitude > 1e-8 and duration_ms > 0:
                morph_score = area / (duration_ms * peak_amplitude)
            else:
                morph_score = 0.0

            # PR interval
            pr_interval = None
            if beats is not None and pw.beat_id < len(beats):
                beat = beats[pw.beat_id]
                if beat.q_onset > 0 and pw.onset_sample > 0:
                    pr_interval = (beat.q_onset - pw.onset_sample) * self._ms_per_sample

            features_list.append(PWaveFeatures(
                beat_id=pw.beat_id,
                onset_sample=pw.onset_sample,
                offset_sample=pw.offset_sample,
                peak_sample=pw.peak_sample,
                duration_ms=round(duration_ms, 2),
                peak_amplitude=round(peak_amplitude, 4),
                area=round(area, 4),
                morphology_score=round(morph_score, 4),
                pr_interval_ms=round(pr_interval, 2) if pr_interval is not None else None,
            ))

        return features_list

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    def summarize(self, features: list[PWaveFeatures]) -> PWaveSummary:
        """Compute aggregate P-wave statistics.

        Parameters
        ----------
        features : list[PWaveFeatures]

        Returns
        -------
        PWaveSummary
        """
        if not features:
            return PWaveSummary(n_beats=0)

        durations = np.array([f.duration_ms for f in features])
        amplitudes = np.array([f.peak_amplitude for f in features])
        prs = np.array([f.pr_interval_ms for f in features
                        if f.pr_interval_ms is not None])

        # Flag abnormal beats
        d_min, d_max = self.duration_normal
        flagged = [f.beat_id for f in features
                   if f.duration_ms < d_min or f.duration_ms > d_max]

        summary = PWaveSummary(
            n_beats=len(features),
            duration_mean_ms=round(float(np.mean(durations)), 2),
            duration_std_ms=round(float(np.std(durations)), 2),
            duration_range_ms=(round(float(np.min(durations)), 2),
                               round(float(np.max(durations)), 2)),
            dispersion_ms=round(float(np.max(durations) - np.min(durations)), 2),
            amplitude_mean=round(float(np.mean(amplitudes)), 4),
            flagged_beats=flagged,
        )

        if len(prs) > 0:
            summary.pr_mean_ms = round(float(np.mean(prs)), 2)
            summary.pr_std_ms = round(float(np.std(prs)), 2)

        return summary
