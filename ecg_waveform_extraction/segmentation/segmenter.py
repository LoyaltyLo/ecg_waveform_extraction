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
        beats = self._extract_beats(result["state_sequence"])

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
                       segments: list[tuple[int, int, int]]) -> list[BeatBoundary]:
        """Convert HSMM state segments into per-beat boundary structures.

        A beat is defined as the sequence from P through TP.
        The ISO→P transition marks the start of a new beat.

        Parameters
        ----------
        segments : list of (state_idx, start_sample, end_sample)

        Returns
        -------
        list[BeatBoundary]
        """
        beats = []
        current_beat = BeatBoundary(beat_id=-1)
        collecting = False
        beat_id = 0

        for state_idx, start, end in segments:
            label = self.model.get_state_name(state_idx) if self.model else STATE_LABELS[state_idx]

            if label == "ISO" and not collecting:
                # Waiting for next P wave — start new beat collection
                current_beat = BeatBoundary(beat_id=beat_id)
                current_beat.iso_start = start
                collecting = True

            elif label == "P" and collecting:
                current_beat.p_onset = start
                current_beat.p_offset = end
                # PR segment will be after P
                # We approximate PR start = P offset + 1
                current_beat.pr_start = end + 1

            elif label == "PR" and collecting:
                if current_beat.pr_start < 0:
                    current_beat.pr_start = start

            elif label == "Q" and collecting:
                current_beat.q_onset = start

            elif label == "R" and collecting:
                # R peak in the middle of the R segment
                current_beat.r_peak = (start + end) // 2

            elif label == "S" and collecting:
                current_beat.s_offset = end
                current_beat.st_start = end + 1

            elif label == "ST" and collecting:
                if current_beat.st_start < 0:
                    current_beat.st_start = start

            elif label == "T" and collecting:
                current_beat.t_onset = start
                current_beat.t_offset = end

            elif label == "TP" and collecting:
                current_beat.tp_start = start
                # End of beat — finalize
                beats.append(current_beat)
                collecting = False
                beat_id += 1

        return beats
