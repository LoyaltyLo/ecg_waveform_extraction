"""ECG Waveform Segmentation: orchestrates preprocessing, feature extraction,
and HSMM decoding to produce per-beat waveform boundaries.
"""

from dataclasses import dataclass, field
import numpy as np

from ..preprocessing.filters import ECGPreprocessor
from ..features.extractor import FeatureExtractor
from ..hsmm.hsmm_model import HSMMModel, STATE_LABELS
from ..hsmm.hsmm_decoder import HSMMDecoder


@dataclass
class BeatBoundary:
    """Boundary information for a single cardiac beat."""

    beat_id: int
    iso_start: int = -1
    p_onset: int = -1
    p_offset: int = -1
    pr_start: int = -1
    q_onset: int = -1
    r_peak: int = -1
    s_offset: int = -1
    st_start: int = -1
    t_onset: int = -1
    t_offset: int = -1
    tp_start: int = -1

    # Convenience aliases
    @property
    def qrs_onset(self) -> int:
        return self.q_onset

    @property
    def qrs_offset(self) -> int:
        return self.s_offset


@dataclass
class SegmentResult:
    """Result of HSMM waveform segmentation on a single ECG recording.

    Attributes
    ----------
    state_labels : np.ndarray, shape (T,), dtype int
        Per-sample state label (0-8).
    state_names : list[str]
        Per-sample state name.
    beats : list[BeatBoundary]
        Detected beats with waveform boundaries.
    log_likelihood : float
        Viterbi path log-probability.
    filtered_ecg : np.ndarray, shape (T,)
        Preprocessed ECG signal.
    features : np.ndarray, shape (T, 3)
        Feature vectors used for decoding.
    fs : float
        Sampling frequency.
    """
    state_labels: np.ndarray
    state_names: list[str]
    beats: list[BeatBoundary]
    log_likelihood: float
    filtered_ecg: np.ndarray
    features: np.ndarray
    fs: float = 250.0

    def __repr__(self) -> str:
        n_beats = len(self.beats)
        return (f"SegmentResult(n_beats={n_beats}, "
                f"T={len(self.state_labels)}, ll={self.log_likelihood:.1f})")


class ECGSegmenter:
    """High-level orchestrator for Stage 1: ECG waveform segmentation.

    Parameters
    ----------
    preprocessor : ECGPreprocessor or None
        Signal preprocessing pipeline. Created with defaults if None.
    feature_extractor : FeatureExtractor or None
        Feature extraction pipeline. Created with defaults if None.
    model : HSMMModel or None
        HSMM model. Must be trained before segmenting.
    decoder : HSMMDecoder or None
        Viterbi decoder.
    fs : float
        Sampling frequency.
    """

    def __init__(self, preprocessor: ECGPreprocessor | None = None,
                 feature_extractor: FeatureExtractor | None = None,
                 model: HSMMModel | None = None,
                 decoder: HSMMDecoder | None = None,
                 fs: float = 250.0):
        self.preprocessor = preprocessor or ECGPreprocessor(fs=fs)
        self.feature_extractor = feature_extractor or FeatureExtractor(fs=fs)
        self.model = model
        self.decoder = decoder or HSMMDecoder()
        self.fs = fs

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------
    def load_model(self, model_path: str):
        """Load a pre-trained HSMMModel from disk."""
        self.model = HSMMModel.load(model_path)

    def save_model(self, model_path: str):
        """Save the trained model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        self.model.save(model_path)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def segment(self, raw_ecg: np.ndarray) -> SegmentResult:
        """Run full Stage 1 pipeline: preprocess → features → HSMM decode → boundaries.

        Parameters
        ----------
        raw_ecg : np.ndarray, shape (N,)
            Raw ECG signal.

        Returns
        -------
        SegmentResult
        """
        if self.model is None:
            raise RuntimeError(
                "No model loaded. Train or load a model before segmenting."
            )

        # Step 1: Preprocess
        filtered = self.preprocessor.preprocess(raw_ecg)

        # Step 2: Feature extraction
        features = self.feature_extractor.extract(filtered)

        # Step 3: HSMM decode
        result = self.decoder.decode(self.model, features)

        # Step 4: State sequence → beat boundaries
        beats = self._extract_beats(result["state_sequence"], filtered)

        # Step 5: Per-sample state names
        state_names = [self.model.get_state_name(lbl)
                       if lbl >= 0 else "UNKNOWN"
                       for lbl in result["state_labels"]]

        return SegmentResult(
            state_labels=result["state_labels"],
            state_names=state_names,
            beats=beats,
            log_likelihood=result["log_likelihood"],
            filtered_ecg=filtered,
            features=features,
            fs=self.fs,
        )

    # ------------------------------------------------------------------
    # Beat boundary extraction
    # ------------------------------------------------------------------
    def _extract_beats(self,
                       segments: list[tuple[int, int, int]],
                       filtered_ecg: np.ndarray | None = None) -> list[BeatBoundary]:
        """Convert HSMM state segments into per-beat boundary structures.

        Uses R-peak-centered extraction: each detected R segment defines a
        beat. We then look backward for the nearest P and Q segments, and
        forward for the nearest S and T segments. This is robust to missing
        intermediate states (PR, ST, TP) and handles non-canonical sequences
        (e.g., PACs without a preceding P-wave).

        Falls back to the strict left-right parser when fewer than 2 R
        segments are found (degenerate case).

        Parameters
        ----------
        segments : list of (state_idx, start_sample, end_sample)
        filtered_ecg : np.ndarray or None
            Preprocessed signal, used to place the R peak at the max
            absolute amplitude within the R segment (falls back to the
            segment midpoint when not provided).

        Returns
        -------
        list[BeatBoundary]
        """
        if not segments:
            return []

        # ---- Index segments by state type ----
        state_to_label = {}
        if self.model:
            for i in range(self.model.n_states):
                state_to_label[i] = self.model.get_state_name(i)
        else:
            state_to_label = {i: lbl for i, lbl in enumerate(STATE_LABELS)}

        # Group segment indices by label
        by_label: dict[str, list[tuple[int, int, int]]] = {}
        for state_idx, start, end in segments:
            label = state_to_label.get(state_idx, 'UNKNOWN')
            by_label.setdefault(label, []).append((state_idx, start, end))

        r_segs = by_label.get('R', [])
        if len(r_segs) < 1:
            return []

        beats = []

        # For each R segment, find the surrounding wave components
        for beat_id, (r_state, r_start, r_end) in enumerate(r_segs):
            # R peak
            if filtered_ecg is not None and r_end > r_start:
                seg = filtered_ecg[r_start:r_end + 1]
                r_peak = r_start + int(np.argmax(np.abs(seg)))
            else:
                r_peak = (r_start + r_end) // 2

            beat = BeatBoundary(beat_id=beat_id)
            beat.r_peak = r_peak

            # ---- Look backward: find nearest P, Q, ISO segments ----
            p_candidates = [(s, e) for (_, s, e) in by_label.get('P', []) if e < r_peak]
            q_candidates = [(s, e) for (_, s, e) in by_label.get('Q', []) if e < r_peak]
            pr_candidates = [(s, e) for (_, s, e) in by_label.get('PR', []) if e < r_peak]
            iso_candidates = [(s, e) for (_, s, e) in by_label.get('ISO', []) if e < r_peak]

            if p_candidates:
                # Nearest P before R
                beat.p_onset, beat.p_offset = p_candidates[-1]
                beat.pr_start = beat.p_offset + 1
            if q_candidates:
                beat.q_onset, _ = q_candidates[-1]
            if pr_candidates and beat.pr_start < 0:
                beat.pr_start = pr_candidates[-1][0]
            if iso_candidates:
                beat.iso_start = iso_candidates[-1][0]

            # ---- Look forward: find nearest S, T, TP segments ----
            s_candidates = [(s, e) for (_, s, e) in by_label.get('S', []) if s > r_peak]
            t_candidates = [(s, e) for (_, s, e) in by_label.get('T', []) if s > r_peak]
            st_candidates = [(s, e) for (_, s, e) in by_label.get('ST', []) if s > r_peak]
            tp_candidates = [(s, e) for (_, s, e) in by_label.get('TP', []) if s > r_peak]

            if s_candidates:
                _, beat.s_offset = s_candidates[0]
                beat.st_start = beat.s_offset + 1
            if t_candidates:
                beat.t_onset, beat.t_offset = t_candidates[0]
            if st_candidates and beat.st_start < 0:
                beat.st_start = st_candidates[0][0]
            if tp_candidates:
                beat.tp_start = tp_candidates[0][0]

            # ---- Cross-beat boundary: P-offset to Q-onset gap ----
            # If no explicit Q found, use the R-peak minus a window
            if beat.q_onset < 0 and beat.p_offset > 0:
                # Estimate Q as the earliest sample in the R segment
                beat.q_onset = r_start

            beats.append(beat)

        return beats
