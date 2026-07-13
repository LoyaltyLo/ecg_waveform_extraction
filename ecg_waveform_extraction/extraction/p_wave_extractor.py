"""Stage 2: P-wave extraction from segmented waveforms.

Optimizations applied (2026-07-13):
  Step 1: Stage 1 boundary-guided GMM init (replaces naive equal-thirds split)
  Step 2: Multi-dimensional confidence (SNR + symmetry + consistency + duration)
  Step 3: Derivative zero-crossing boundary refinement (improves onset/offset precision)
  Step 4: P-wave absence detection (distinguishes AFib from detection failure)
  Step 5: Morphology classification (normal/biphasic/peaked/inverted/absent/low-amp)
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import pearsonr

from ..hsmm.hsmm_model import HSMMModel
from ..hsmm.hsmm_decoder import HSMMDecoder
from ..features.extractor import FeatureExtractor
from ..preprocessing.filters import ECGPreprocessor
from ..segmentation.segmenter import SegmentResult


# ---------------------------------------------------------------------------
# P-wave morphology types
# ---------------------------------------------------------------------------
MORPH_NORMAL         = "normal"
MORPH_BIPHASIC       = "biphasic"
MORPH_PEAKED         = "peaked"
MORPH_INVERTED       = "inverted"
MORPH_ABSENT         = "absent"
MORPH_LOW_AMPLITUDE  = "low_amplitude"
MORPH_UNDETERMINED   = "undetermined"


@dataclass
class PWaveResult:
    """Refined P-wave extraction for a single beat.

    Attributes
    ----------
    beat_id : int
        Beat index in the recording.
    onset_sample : int
        P-wave onset sample index (refined). -1 if absent.
    offset_sample : int
        P-wave offset sample index (refined). -1 if absent.
    peak_sample : int
        P-wave peak sample (max absolute amplitude within P region).
    samples : np.ndarray
        ECG signal values within the P-wave [onset:offset+1].
    duration_ms : float
        P-wave duration in milliseconds.
    confidence : float
        Multi-dimensional quality score (0-1).
    morphology : str
        P-wave morphology classification.
    absence_type : str or None
        If P-wave is absent: 'afib_flat', 'noise', or None.
    snr_db : float
        Signal-to-noise ratio in dB.
    symmetry : float
        Rising/falling slope symmetry (0-1, 1=perfect).
    consistency : float
        Correlation with neighboring P-waves (0-1).
    """
    beat_id: int
    onset_sample: int
    offset_sample: int
    peak_sample: int
    samples: np.ndarray
    duration_ms: float
    confidence: float = 1.0
    morphology: str = MORPH_UNDETERMINED
    absence_type: str | None = None
    snr_db: float = 0.0
    symmetry: float = 0.0
    consistency: float = 0.0


class PWaveExtractor:
    """Extracts and refines P-wave boundaries using focused HSMM + post-processing.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    window_before_ms : float
        Padding before Stage 1 P_onset in ms.
    window_after_ms : float
        Padding after Stage 1 P_offset in ms.
    refine_boundaries : bool
        If True, apply derivative zero-crossing refinement on HSMM boundaries.
    enable_template_fallback : bool
        If True, use template-matching fallback when HSMM fails to find a P-wave.
    """

    def __init__(self, fs: float = 250.0,
                 window_before_ms: float = 100.0,
                 window_after_ms: float = 100.0,
                 refine_boundaries: bool = True,
                 enable_template_fallback: bool = True):
        self.fs = fs
        self.window_before_ms = window_before_ms
        self.window_after_ms = window_after_ms
        self.refine_boundaries = refine_boundaries
        self.enable_template_fallback = enable_template_fallback

        self._window_before = int(np.round(window_before_ms / 1000.0 * fs))
        self._window_after = int(np.round(window_after_ms / 1000.0 * fs))

        self._preprocessor = ECGPreprocessor(fs=fs)
        self._feature_extractor = FeatureExtractor(fs=fs)
        self._decoder = HSMMDecoder()

    # ------------------------------------------------------------------
    # Focused 3-state P-wave HSMM
    # ------------------------------------------------------------------
    def _build_p_wave_model(self, beat=None, hr=None) -> HSMMModel:
        """Build a 3-state HSMM (ISO_before → P → PR_after).

        Parameters
        ----------
        beat : BeatBoundary or None
            Stage 1 beat for HR-adaptive priors.
        hr : float or None
            Heart rate override.
        """
        model = HSMMModel(
            n_states=3,
            state_labels=["ISO_before", "P", "PR_after"],
            n_features=3,
            n_gmm_components=2,
            fs=self.fs,
        )
        # Left-right topology
        model.A = np.zeros((3, 3))
        model.A[0, 1] = 1.0
        model.A[1, 2] = 1.0
        model.A[2, 2] = 1.0
        model.pi = np.array([1.0, 0.0, 0.0])

        # --- HR-adaptive P-wave duration prior ---
        rr_ms = 1000.0  # default (60 BPM)
        if hr is not None and hr > 0:
            rr_ms = 60000.0 / hr
        elif beat is not None:
            # Estimate from adjacent beats if available (rough)
            pass

        # P-wave duration: ~80ms at 150bpm to ~120ms at 50bpm
        p_dur_ms = np.clip(80.0 + (rr_ms - 600.0) * 0.05, 60.0, 140.0)
        p_samples = int(np.round(p_dur_ms / 1000.0 * self.fs))

        # PR duration: ~100ms at 150bpm to ~220ms at 50bpm
        pr_dur_ms = np.clip(120.0 + (rr_ms - 600.0) * 0.08, 90.0, 220.0)
        pr_samples = int(np.round(pr_dur_ms / 1000.0 * self.fs))

        from ..hsmm.distributions import DurationDistribution
        model.dur_dists[0] = DurationDistribution(mu=15, sigma=15, d_min=2)
        model.dur_dists[1] = DurationDistribution(mu=p_samples, sigma=max(p_samples*0.25, 3), d_min=4)
        model.dur_dists[2] = DurationDistribution(mu=pr_samples, sigma=max(pr_samples*0.3, 4), d_min=1)

        model._compute_D_max()
        return model

    # ------------------------------------------------------------------
    # Main extraction loop
    # ------------------------------------------------------------------
    def extract(self, segment_result: SegmentResult,
                heart_rate: float | None = None) -> list[PWaveResult]:
        """Refine P-wave boundaries for every beat in the segmentation result.

        Parameters
        ----------
        segment_result : SegmentResult
            Stage 1 output from ECGSegmenter.
        heart_rate : float or None
            Heart rate in BPM. If None, estimated from beat intervals.

        Returns
        -------
        list[PWaveResult] — one per beat with detected P-wave.
        """
        ecg = segment_result.filtered_ecg
        T = len(ecg)
        beats = segment_result.beats
        n_beats = len(beats)

        # Estimate heart rate if not provided
        if heart_rate is None and n_beats >= 2:
            r_peaks = [b.r_peak for b in beats if b.r_peak > 0]
            if len(r_peaks) >= 2:
                rr_mean = np.mean(np.diff(r_peaks)) / self.fs * 1000.0  # ms
                heart_rate = 60000.0 / rr_mean if rr_mean > 0 else 60
        if heart_rate is None:
            heart_rate = 60.0

        results = []
        # Accumulate good P-wave templates for cross-beat consistency
        template_pool = []

        for beat in beats:
            pw = self._extract_single(ecg, T, beat, heart_rate, template_pool)
            results.append(pw)
            if pw.onset_sample > 0 and pw.confidence >= 0.5:
                template_pool.append(pw.samples)

        # ---- Cross-beat consistency post-processing ----
        self._postprocess_consistency(results)

        return results

    # ------------------------------------------------------------------
    # Extract single P-wave
    # ------------------------------------------------------------------
    def _extract_single(self, ecg, T, beat, heart_rate,
                        template_pool) -> PWaveResult:
        """Extract P-wave for one beat."""
        # ---- P-wave absent check ----
        if beat.p_onset < 0:
            return self._handle_absent(ecg, beat, "No Stage 1 P-onset")

        win_start = max(0, beat.p_onset - self._window_before)
        win_end = min(T - 1, beat.p_offset + self._window_after)

        if win_end - win_start < 10:
            return self._fallback(ecg, beat, "Window too small")

        ecg_window = ecg[win_start:win_end + 1]
        window_len = len(ecg_window)

        # ---- Check if there's actually signal in the P region ----
        p_rel_start = beat.p_onset - win_start
        p_rel_end = beat.p_offset - win_start
        p_rel_start = max(0, min(p_rel_start, window_len - 1))
        p_rel_end = max(p_rel_start + 3, min(p_rel_end, window_len - 1))
        p_region_signal = ecg_window[p_rel_start:p_rel_end + 1]
        iso_region = ecg_window[:max(1, p_rel_start)]

        p_std = np.std(p_region_signal) if len(p_region_signal) > 1 else 0
        iso_std = np.std(iso_region) if len(iso_region) > 1 else 1e-6

        # True absence check: P region is flat compared to ISO
        if p_std < iso_std * 0.5 and len(p_region_signal) > 5:
            # Possible AFib or true P absence
            if self.enable_template_fallback and template_pool:
                pw = self._template_match(ecg_window, win_start, beat, template_pool)
                if pw is not None:
                    return pw
            return self._handle_absent(ecg, beat, "Flat P region (possible AFib)",
                                       absence_type="afib_flat")

        # ---- Build + decode focused HSMM ----
        features = self._feature_extractor.extract(
            self._preprocessor.preprocess(ecg_window))
        model = self._build_p_wave_model(beat=beat, hr=heart_rate)
        # Step 1: boundary-guided GMM init
        self._init_gmms_by_boundary(model, features, beat, win_start, window_len)
        result = self._decoder.decode(model, features)
        labels = result["state_labels"]

        p_indices = np.where(labels == 1)[0]
        if len(p_indices) < 3:
            if self.enable_template_fallback and template_pool:
                pw = self._template_match(ecg_window, win_start, beat, template_pool)
                if pw is not None:
                    return pw
            return self._fallback(ecg, beat, "HSMM found no clear P (<3 samples)")

        p_onset_win = int(p_indices[0])
        p_offset_win = int(p_indices[-1])

        # Step 3: derivative zero-crossing refinement
        if self.refine_boundaries:
            p_onset_win, p_offset_win = self._refine_by_slope(
                ecg_window, p_onset_win, p_offset_win)

        onset_sample = win_start + p_onset_win
        offset_sample = win_start + p_offset_win

        # Ensure valid range
        onset_sample = max(0, min(onset_sample, T - 1))
        offset_sample = max(onset_sample + 1, min(offset_sample, T - 1))

        p_ecg = ecg[onset_sample:offset_sample + 1]
        peak_offset = np.argmax(np.abs(p_ecg))
        peak_sample = onset_sample + peak_offset
        duration_ms = len(p_ecg) / self.fs * 1000.0

        # Step 2: multi-dimensional confidence
        snr_db, symmetry, consistency = self._compute_quality(
            p_ecg, iso_region if len(iso_region) > 0 else ecg_window[:max(1, p_rel_start)],
            template_pool)
        confidence = self._aggregate_confidence(
            snr_db, symmetry, consistency, duration_ms, heart_rate)

        # Step 5: morphology classification
        morphology = self._classify_morphology(p_ecg, duration_ms, heart_rate)

        return PWaveResult(
            beat_id=beat.beat_id,
            onset_sample=onset_sample,
            offset_sample=offset_sample,
            peak_sample=peak_sample,
            samples=p_ecg.copy(),
            duration_ms=round(duration_ms, 2),
            confidence=round(confidence, 3),
            morphology=morphology,
            snr_db=round(snr_db, 1),
            symmetry=round(symmetry, 3),
            consistency=round(consistency, 3),
        )

    # ==================================================================
    # Step 1: Boundary-guided GMM initialization
    # ==================================================================
    def _init_gmms_by_boundary(self, model, features, beat,
                                win_start, window_len):
        """Initialize 3-state GMMs using Stage 1 P-wave boundaries.

        Instead of naive equal-thirds, uses the Stage 1 beat boundaries
        to map the ISO_before / P / PR_after regions correctly.
        """
        T = features.shape[0]
        if T < 9:
            return

        # Map Stage 1 boundaries into window coordinates
        p_on_rel = beat.p_onset - win_start
        p_off_rel = beat.p_offset - win_start

        # Clamp to valid range
        p_on_rel = max(2, min(p_on_rel, T - 6))
        p_off_rel = max(p_on_rel + 3, min(p_off_rel, T - 2))

        # Three regions:
        #   ISO_before: [0, p_on_rel - 2]
        #   P:          [p_on_rel, p_off_rel]
        #   PR_after:   [p_off_rel + 1, T-1]
        regions = [
            (0, max(3, p_on_rel - 2)),          # ISO_before
            (p_on_rel, p_off_rel),              # P
            (p_off_rel + 1, T - 1),             # PR_after
        ]

        for state_idx, (start, end) in enumerate(regions):
            start = max(0, min(start, T - 3))
            end = max(start + 3, min(end, T - 1))
            seg = features[start:end + 1]
            if len(seg) > model.n_gmm_components * 2:
                try:
                    model.obs_dists[state_idx].fit(seg, max_iter=25)
                except (ValueError, np.linalg.LinAlgError):
                    pass

    # ==================================================================
    # Step 2: Multi-dimensional confidence
    # ==================================================================
    def _compute_quality(self, p_ecg, iso_region, template_pool):
        """Compute SNR, symmetry, and consistency for a P-wave segment.

        Returns
        -------
        snr_db : float
            Signal-to-noise ratio in dB.
        symmetry : float
            0-1, 1 = perfectly symmetric.
        consistency : float
            0-1, 1 = highly consistent with neighbors.
        """
        if len(p_ecg) < 4:
            return 0.0, 0.0, 0.0

        # ---- SNR ----
        p_amp = np.max(np.abs(p_ecg - np.mean(p_ecg)))
        iso_sigma = np.std(iso_region) if len(iso_region) > 1 else 1e-6
        snr = p_amp / max(iso_sigma, 1e-6)
        snr_db = 20.0 * np.log10(max(snr, 1e-6))

        # ---- Symmetry ----
        # Split P-wave at its peak, compare rising vs falling slopes
        peak_idx = np.argmax(np.abs(p_ecg - np.mean(p_ecg)))
        if peak_idx > 0 and peak_idx < len(p_ecg) - 1:
            rising = np.abs(np.diff(p_ecg[:peak_idx + 1]))
            falling = np.abs(np.diff(p_ecg[peak_idx:]))
            rising_mean = np.mean(np.abs(rising)) if len(rising) > 0 else 1e-6
            falling_mean = np.mean(np.abs(falling)) if len(falling) > 0 else 1e-6
            symmetry = min(rising_mean, falling_mean) / max(rising_mean, falling_mean, 1e-6)
        else:
            symmetry = 0.5

        # ---- Consistency ----
        consistency = 0.0
        if template_pool:
            # Resample P-wave to fixed length for comparison
            target_len = 40  # fixed interpolation length
            try:
                from scipy.interpolate import interp1d
                x_orig = np.linspace(0, 1, len(p_ecg))
                x_new = np.linspace(0, 1, target_len)
                f_interp = interp1d(x_orig, p_ecg, kind='linear',
                                    bounds_error=False, fill_value=0)
                p_resampled = f_interp(x_new)
            except Exception:
                p_resampled = p_ecg

            correlations = []
            for tpl in template_pool[-5:]:  # last 5 templates
                try:
                    if len(tpl) >= 4:
                        tp_x = np.linspace(0, 1, len(tpl))
                        tp_new = np.interp(x_new, tp_x, tpl) if 'x_new' in dir() else tpl
                        if len(tp_new) == len(p_resampled):
                            r, _ = pearsonr(p_resampled, tp_new)
                            correlations.append(max(0, r))
                except Exception:
                    pass
            if correlations:
                consistency = float(np.clip(np.mean(correlations), 0.0, 1.0))

        return snr_db, symmetry, consistency

    def _aggregate_confidence(self, snr_db, symmetry, consistency,
                               duration_ms, heart_rate):
        """Combine quality metrics into a single confidence score (0-1)."""
        # SNR score: 0dB→0.2, 10dB→0.7, 20dB→1.0
        snr_score = np.clip(snr_db / 25.0, 0.0, 1.0)

        # Duration score: close to HR-expected → 1.0
        rr_ms = 60000.0 / max(heart_rate, 30)
        expected_dur = np.clip(80.0 + (rr_ms - 600.0) * 0.05, 60.0, 140.0)
        dur_error = abs(duration_ms - expected_dur) / max(expected_dur, 1.0)
        dur_score = np.exp(-dur_error * 3.0)  # Gaussian penalty

        # Weighted combination
        weights = {"snr": 0.30, "symmetry": 0.20, "consistency": 0.25, "duration": 0.25}
        confidence = (
            weights["snr"] * snr_score +
            weights["symmetry"] * symmetry +
            weights["consistency"] * consistency +
            weights["duration"] * dur_score
        )
        return float(np.clip(confidence, 0.0, 1.0))

    # ==================================================================
    # Step 3: Derivative zero-crossing boundary refinement
    # ==================================================================
    def _refine_by_slope(self, ecg_window, p_onset_win, p_offset_win):
        """Refine P-wave onset/offset using first-derivative zero-crossings.

        From the HSMM-estimated boundaries, walk outward until the slope
        returns to baseline (mean of first 20% of window ± 2σ).
        """
        T = len(ecg_window)
        d1 = np.gradient(ecg_window)

        # Baseline slope statistics from the quiet early part of the window
        baseline_end = min(T // 4, p_onset_win - 1)
        if baseline_end < 5:
            baseline_end = max(T // 5, 5)
        baseline_d1 = d1[:baseline_end]
        d1_mean = np.mean(baseline_d1)
        d1_std = np.std(baseline_d1)
        threshold = d1_std * 2.5 + 1e-6

        # ---- Refine onset: walk left from p_onset_win ----
        refined_onset = p_onset_win
        for i in range(p_onset_win, max(2, p_onset_win - 20), -1):
            if abs(d1[i]) <= threshold:
                refined_onset = i
            else:
                break
        # Walk a few more samples to the most quiescent point
        for i in range(refined_onset, max(2, refined_onset - 8), -1):
            if abs(d1[i]) <= threshold and abs(d1[i - 1]) <= threshold:
                refined_onset = i
                break

        # ---- Refine offset: walk right from p_offset_win ----
        refined_offset = p_offset_win
        for i in range(p_offset_win, min(T - 2, p_offset_win + 20)):
            if abs(d1[i]) <= threshold:
                refined_offset = i
            else:
                break
        for i in range(refined_offset, min(T - 2, refined_offset + 8)):
            if abs(d1[i]) <= threshold and abs(d1[i + 1]) <= threshold:
                refined_offset = i
                break

        # Don't let refinement collapse the P-wave
        min_p_samples = 4
        if refined_offset - refined_onset < min_p_samples:
            return p_onset_win, p_offset_win

        return max(0, refined_onset), min(T - 1, refined_offset)

    # ==================================================================
    # Step 4: Absence detection + template fallback
    # ==================================================================
    def _handle_absent(self, ecg, beat, reason,
                       absence_type=None) -> PWaveResult:
        """Create PWaveResult for beats where P-wave is genuinely absent."""
        return PWaveResult(
            beat_id=beat.beat_id,
            onset_sample=-1,
            offset_sample=-1,
            peak_sample=-1,
            samples=np.array([]),
            duration_ms=0.0,
            confidence=0.0,
            morphology=MORPH_ABSENT,
            absence_type=absence_type or "undetermined",
        )

    def _template_match(self, ecg_window, win_start, beat,
                        template_pool) -> PWaveResult | None:
        """Fallback: match the average P-wave template against the P region.

        Returns PWaveResult if a plausible match is found, else None.
        """
        if not template_pool:
            return None

        # Build average template
        max_len = max(len(t) for t in template_pool)
        template = np.zeros(max_len)
        counts = np.zeros(max_len)
        for tpl in template_pool:
            n = min(len(tpl), max_len)
            template[:n] += tpl[:n]
            counts[:n] += 1
        counts = np.maximum(counts, 1)
        template /= counts
        # Zero-center
        template = template - np.mean(template)
        template_norm = template / (np.std(template) + 1e-8)

        if len(template) < 4 or len(ecg_window) < len(template):
            return None

        # Cross-correlate template with window
        cc = np.correlate(ecg_window - np.mean(ecg_window),
                          template_norm, mode='valid')
        if len(cc) == 0:
            return None
        cc_norm = cc / (np.std(ecg_window[:len(cc)]) * len(template) + 1e-8)
        peak_cc = np.max(cc_norm)

        if peak_cc < 0.4:  # weak correlation → not a real P-wave
            return None

        best_idx = np.argmax(cc)
        onset = win_start + best_idx
        offset = onset + len(template) - 1

        p_ecg = ecg_window[best_idx:best_idx + len(template)]
        peak_offset = np.argmax(np.abs(p_ecg))
        peak_sample = onset + peak_offset
        duration_ms = len(p_ecg) / self.fs * 1000.0

        return PWaveResult(
            beat_id=beat.beat_id,
            onset_sample=onset,
            offset_sample=offset,
            peak_sample=peak_sample,
            samples=p_ecg.copy(),
            duration_ms=round(duration_ms, 2),
            confidence=round(float(min(peak_cc, 1.0)), 3),
            morphology=MORPH_LOW_AMPLITUDE,
            snr_db=0.0, symmetry=0.0, consistency=float(min(peak_cc, 1.0)),
        )

    # ==================================================================
    # Step 5: P-wave morphology classification
    # ==================================================================
    def _classify_morphology(self, p_ecg, duration_ms, heart_rate):
        """Classify P-wave morphology into one of several types.

        Uses peak count, net area sign, and amplitude to determine type.
        """
        if len(p_ecg) < 4:
            return MORPH_UNDETERMINED

        p_centered = p_ecg - np.mean(p_ecg[:5]) if len(p_ecg) > 5 else p_ecg - np.mean(p_ecg)
        peak_amplitude = np.max(np.abs(p_centered))
        net_area = np.sum(p_centered)

        # ---- Inverted check ----
        if net_area < 0 and abs(net_area) > peak_amplitude * 2:
            return MORPH_INVERTED

        # ---- Low amplitude ----
        if peak_amplitude < 0.03:  # < 0.03 normalized units
            return MORPH_LOW_AMPLITUDE

        # ---- Absent ----
        if peak_amplitude < 0.01:
            return MORPH_ABSENT

        # ---- Peak count for biphasic / peaked ----
        peaks, properties = find_peaks(p_centered, height=peak_amplitude * 0.2,
                                        distance=max(3, int(0.015 * self.fs)))
        troughs, _ = find_peaks(-p_centered, height=peak_amplitude * 0.2,
                                 distance=max(3, int(0.015 * self.fs)))
        n_distinct = len(peaks) + len(troughs)

        # ---- Biphasic (P mitrale) ----
        if n_distinct >= 2 and duration_ms > 120:
            # Check if two peaks are significant and separated
            if len(peaks) >= 2:
                peak_sep = (peaks[1] - peaks[0]) / self.fs * 1000.0
                if peak_sep > 30:
                    return MORPH_BIPHASIC
            if len(peaks) >= 1 and len(troughs) >= 1:
                return MORPH_BIPHASIC

        # ---- Peaked (P pulmonale) ----
        if peak_amplitude > 0.15 and duration_ms <= 120 and n_distinct <= 1:
            return MORPH_PEAKED

        # ---- Normal ----
        if 80 <= duration_ms <= 120 and net_area > 0:
            return MORPH_NORMAL

        return MORPH_NORMAL  # default for borderline cases

    # ==================================================================
    # Cross-beat consistency post-processing
    # ==================================================================
    def _postprocess_consistency(self, results):
        """Smooth P-wave metrics across beats and flag outliers."""
        n = len(results)
        if n < 5:
            return

        # Extract durations from valid beats
        durs = np.array([r.duration_ms for r in results if r.onset_sample > 0])
        if len(durs) < 5:
            return

        # 5-beat sliding median
        from scipy.ndimage import uniform_filter1d
        smoothed = uniform_filter1d(durs, size=min(5, len(durs)))

        # Flag beats deviating > 3σ from the local smoothed value
        idx = 0
        for r in results:
            if r.onset_sample > 0 and idx < len(durs):
                dev = abs(durs[idx] - smoothed[min(idx, len(smoothed) - 1)])
                sigma = np.std(durs) + 1e-6
                if dev > 3 * sigma:
                    r.confidence *= 0.5  # Penalize outliers
                    r.confidence = round(r.confidence, 3)
                idx += 1

    # ==================================================================
    # Fallback / helpers
    # ==================================================================
    def _fallback(self, ecg, beat, reason) -> PWaveResult:
        """Use Stage 1 boundaries directly when focused extraction fails."""
        onset = beat.p_onset
        offset = beat.p_offset
        if onset < 0 or offset < 0 or offset <= onset:
            return PWaveResult(
                beat_id=beat.beat_id,
                onset_sample=-1, offset_sample=-1, peak_sample=-1,
                samples=np.array([]), duration_ms=0.0,
                confidence=0.0, morphology=MORPH_UNDETERMINED,
            )

        p_ecg = ecg[onset:offset + 1]
        peak = onset + np.argmax(np.abs(p_ecg))
        dur = len(p_ecg) / self.fs * 1000.0
        return PWaveResult(
            beat_id=beat.beat_id,
            onset_sample=onset, offset_sample=offset,
            peak_sample=peak,
            samples=p_ecg.copy(),
            duration_ms=round(dur, 2),
            confidence=0.4,  # lower confidence for fallback
            morphology=MORPH_UNDETERMINED,
        )
