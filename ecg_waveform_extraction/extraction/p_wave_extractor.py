"""Stage 2: P-wave extraction from segmented waveforms.

Uses a focused 3-state HSMM (ISO_before, P, PR_after) applied to a narrow window
around the Stage 1 P-wave region to refine P-wave onset and offset boundaries.
"""

from dataclasses import dataclass
import numpy as np

from ..hsmm.hsmm_model import HSMMModel
from ..hsmm.hsmm_decoder import HSMMDecoder
from ..features.extractor import FeatureExtractor
from ..preprocessing.filters import ECGPreprocessor
from ..segmentation.segmenter import SegmentResult


@dataclass
class PWaveResult:
    """Refined P-wave extraction for a single beat.

    Attributes
    ----------
    beat_id : int
        Beat index in the recording.
    onset_sample : int
        P-wave onset sample index in the original signal.
    offset_sample : int
        P-wave offset sample index.
    peak_sample : int
        P-wave peak sample (max absolute amplitude within P region).
    samples : np.ndarray
        ECG signal values within the P-wave [onset:offset+1].
    duration_ms : float
        P-wave duration in milliseconds.
    confidence : float
        Quality score (0-1) from the focused Viterbi.
    """
    beat_id: int
    onset_sample: int
    offset_sample: int
    peak_sample: int
    samples: np.ndarray
    duration_ms: float
    confidence: float = 1.0


class PWaveExtractor:
    """Refines P-wave boundaries using a focused local HSMM.

    For each P-wave detected in Stage 1, a 3-state HSMM (ISO_before → P → PR_after)
    is decoded on a narrow window to get precise onset/offset.

    Parameters
    ----------
    fs : float
        Sampling frequency.
    window_before_ms : float
        Padding before Stage 1 P_onset (ms).
    window_after_ms : float
        Padding after Stage 1 P_offset (ms).
    """

    def __init__(self, fs: float = 250.0,
                 window_before_ms: float = 100.0,
                 window_after_ms: float = 100.0):
        self.fs = fs
        self.window_before_ms = window_before_ms
        self.window_after_ms = window_after_ms

        self._window_before = int(np.round(window_before_ms / 1000.0 * fs))
        self._window_after = int(np.round(window_after_ms / 1000.0 * fs))

        self._preprocessor = ECGPreprocessor(fs=fs)
        self._feature_extractor = FeatureExtractor(fs=fs)
        self._decoder = HSMMDecoder()

    # ------------------------------------------------------------------
    # Build focused 3-state P-wave model
    # ------------------------------------------------------------------
    def _build_p_wave_model(self) -> HSMMModel:
        """Create a 3-state HSMM for fine P-wave boundary detection.

        States: 0=ISO_before, 1=P, 2=PR_after
        """
        model = HSMMModel(
            n_states=3,
            state_labels=["ISO_before", "P", "PR_after"],
            n_features=3,
            n_gmm_components=2,
            fs=self.fs,
        )

        # Left-right topology: 0→1, 1→2
        model.A = np.zeros((3, 3))
        model.A[0, 1] = 1.0  # ISO_before → P
        model.A[1, 2] = 1.0  # P → PR_after
        model.A[2, 2] = 1.0  # PR_after self-loop
        model.pi = np.array([1.0, 0.0, 0.0])

        # Tight physiological duration priors for P-wave
        # P at 250 Hz: ~25 samples (100ms)
        p_samples = int(np.round(100.0 / 1000.0 * self.fs))
        from ..hsmm.distributions import DurationDistribution
        model.dur_dists[0] = DurationDistribution(mu=10, sigma=10, d_min=1)
        model.dur_dists[1] = DurationDistribution(mu=p_samples, sigma=p_samples * 0.2, d_min=8)
        model.dur_dists[2] = DurationDistribution(mu=15, sigma=15, d_min=1)

        model._compute_D_max()
        return model

    # ------------------------------------------------------------------
    # Extract per-beat P-waves
    # ------------------------------------------------------------------
    def extract(self, segment_result: SegmentResult) -> list[PWaveResult]:
        """Refine P-wave boundaries for every beat in the segment result.

        Parameters
        ----------
        segment_result : SegmentResult
            Stage 1 output.

        Returns
        -------
        list[PWaveResult]
            One result per beat with a detected P-wave.
        """
        ecg = segment_result.filtered_ecg
        T = len(ecg)
        p_wave_results = []

        for beat in segment_result.beats:
            if beat.p_onset < 0:
                # No P-wave detected for this beat (e.g., atrial fibrillation)
                continue

            # Define extraction window
            win_start = max(0, beat.p_onset - self._window_before)
            win_end = min(T - 1, beat.p_offset + self._window_after)

            if win_end - win_start < 15:
                # Window too small — use Stage 1 boundaries directly
                p_wave_results.append(self._fallback_p_wave(ecg, beat))
                continue

            # Extract window
            ecg_window = ecg[win_start:win_end + 1]

            # Preprocess & extract features
            clean_window = self._preprocessor.preprocess(ecg_window)
            features = self._feature_extractor.extract(clean_window)

            # Build and fit the focused P-wave model to this window
            model = self._build_p_wave_model()

            # Initialize GMMs from the window data
            self._init_gmms_from_window(model, features)

            # Decode with focused model
            result = self._decoder.decode(model, features)
            labels = result["state_labels"]

            # Find P-state boundaries in window coordinates
            p_indices = np.where(labels == 1)[0]  # state 1 = P
            if len(p_indices) < 3:
                # Focused model didn't find a clear P — use Stage 1
                p_wave_results.append(self._fallback_p_wave(ecg, beat))
                continue

            p_onset_win = p_indices[0]
            p_offset_win = p_indices[-1]

            # Map back to original signal coordinates
            onset_sample = win_start + p_onset_win
            offset_sample = win_start + p_offset_win

            # Find peak within P region
            p_ecg = ecg[onset_sample:offset_sample + 1]
            peak_offset = np.argmax(np.abs(p_ecg))
            peak_sample = onset_sample + peak_offset

            # Compute confidence from Viterbi vs. forward likelihood
            confidence = np.exp(result["log_likelihood"] / max(len(features), 1))
            confidence = float(np.clip(confidence, 0.0, 1.0))

            duration_ms = (offset_sample - onset_sample + 1) / self.fs * 1000.0

            p_wave_results.append(PWaveResult(
                beat_id=beat.beat_id,
                onset_sample=onset_sample,
                offset_sample=offset_sample,
                peak_sample=peak_sample,
                samples=ecg[onset_sample:offset_sample + 1].copy(),
                duration_ms=duration_ms,
                confidence=confidence,
            ))

        return p_wave_results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _init_gmms_from_window(self, model: HSMMModel, features: np.ndarray):
        """Quick-fit GMMs on window data to initialize the focused model."""
        T = features.shape[0]
        third = T // 3
        if third < 3:
            return

        # ISO before (first third)
        seg0 = features[:third]
        if len(seg0) >= model.n_gmm_components:
            try:
                model.obs_dists[0].fit(seg0, max_iter=20)
            except ValueError:
                pass

        # P (middle third)
        seg1 = features[third:2 * third]
        if len(seg1) >= model.n_gmm_components:
            try:
                model.obs_dists[1].fit(seg1, max_iter=20)
            except ValueError:
                pass

        # PR after (last third)
        seg2 = features[2 * third:]
        if len(seg2) >= model.n_gmm_components:
            try:
                model.obs_dists[2].fit(seg2, max_iter=20)
            except ValueError:
                pass

    def _fallback_p_wave(self, ecg: np.ndarray,
                         beat) -> PWaveResult:
        """Use Stage 1 boundaries directly when focused extraction fails."""
        onset = beat.p_onset
        offset = beat.p_offset
        if onset < 0 or offset < 0:
            return PWaveResult(
                beat_id=beat.beat_id,
                onset_sample=-1, offset_sample=-1, peak_sample=-1,
                samples=np.array([]), duration_ms=0.0, confidence=0.0,
            )

        p_ecg = ecg[onset:offset + 1]
        peak = onset + np.argmax(np.abs(p_ecg))
        dur = (offset - onset + 1) / self.fs * 1000.0
        return PWaveResult(
            beat_id=beat.beat_id,
            onset_sample=onset,
            offset_sample=offset,
            peak_sample=peak,
            samples=p_ecg.copy(),
            duration_ms=dur,
            confidence=0.5,
        )
