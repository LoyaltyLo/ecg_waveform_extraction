"""6-Lead Limb HSMM Processor.

Processes all 6 limb leads (I, II, III, AVR, AVL, AVF) through the HSMM
pipeline, extracting per-beat QRS metrics (v2 polarity) and P-wave features
for each lead. This is the foundation for limb lead reversal detection.

Usage:
    from ecg_waveform_extraction.src.limb_lead_processor import (
        LimbLeadProcessor, LIMB_LEADS,
    )
    from ecg_waveform_extraction.src.utils.aecg_parser import parse_aecg

    processor = LimbLeadProcessor(max_samples=16000)
    aecg = parse_aecg('path/to/file.aECG')
    result = processor.process_record(aecg)
    # result.leads['I']  -> LeadResult for Lead I
    # result.leads['AVR'] -> LeadResult for aVR
"""

import os, json, time, gc
from dataclasses import dataclass, field
from collections import Counter
import numpy as np

from .preprocessing import ECGPreprocessor
from .features import FeatureExtractor
from .hsmm import HSMMModel, smart_initialize_gmms
from .segmentation import ECGSegmenter
from .extraction.qrs_refiner import (
    refine_qrs_boundaries, compute_qrs_polarity_v2,
)

# Optional prominence-based P/T refinement (Emrich et al., EUSIPCO 2024).
# Missing package -> keep raw HSMM boundaries instead of failing.
try:
    from .delineation.prominence_stage import refine_p_t_boundaries
    _HAS_PROMINENCE = True
except ImportError:
    _HAS_PROMINENCE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LIMB_LEADS = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF']

# Lead display order ( Cabrera )
LEAD_PLOT_ORDER = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF']

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LeadResult:
    """HSMM processing result for a single lead.

    Attributes
    ----------
    lead_name : str
    n_beats : int
        Number of beats detected and processed.
    beats : list[dict]
        Per-beat QRS metrics (polarity, rs_ratio, net_area, etc.).
    p_waves : list[dict]
        Per-beat P-wave metrics (net_area, polarity, duration).
    polarity_counts : dict
        QRS polarity histogram, e.g. {'positive': 5, 'negative': 2}.
    p_polarity_counts : dict
        P-wave polarity histogram.
    mean_qrs_dur_ms, mean_rs_ratio, mean_qrs_net, mean_p_net : float
        Aggregate metrics across all beats.
    mean_r_amplitude, mean_s_amplitude : float
        Mean R and S amplitudes.
    """
    lead_name: str
    n_beats: int
    beats: list = field(default_factory=list)
    p_waves: list = field(default_factory=list)
    t_waves: list = field(default_factory=list)
    polarity_counts: dict = field(default_factory=dict)
    p_polarity_counts: dict = field(default_factory=dict)
    t_polarity_counts: dict = field(default_factory=dict)
    mean_qrs_dur_ms: float = 0.0
    mean_rs_ratio: float = 0.0
    mean_qrs_net: float = 0.0
    mean_p_net: float = 0.0
    mean_t_net: float = 0.0
    mean_r_amplitude: float = 0.0
    mean_s_amplitude: float = 0.0


@dataclass
class LimbLeadResult:
    """Full 6-lead processing result for one record.

    Attributes
    ----------
    record : str
        Record identifier.
    fs : float
        Sampling frequency (Hz).
    n_samples : int
        Signal length in samples.
    leads : dict[str, LeadResult or None]
        Lead name → LeadResult (None if lead missing).
    measurements : dict
        Machine measurements from aECG (HR, QRS_dur, P_axis, QRS_axis, etc.).
    interpretation : str
        Clinical interpretation text from aECG.
    n_total_beats : int
        Sum of beats across all 6 leads.
    processing_time_sec : float
        Wall-clock time for this record.
    """
    record: str
    fs: float
    n_samples: int
    leads: dict
    measurements: dict = field(default_factory=dict)
    interpretation: str = ''
    n_total_beats: int = 0
    processing_time_sec: float = 0.0


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class LimbLeadProcessor:
    """Process all 6 limb leads through the HSMM pipeline.

    For each lead, runs:
      1. ECGPreprocessor → bandpass + notch + baseline removal + normalize
      2. FeatureExtractor → 3D features [amplitude, d1, d2]
      3. HSMMModel → 9-state left-right GMM initialization
      4. ECGSegmenter → Viterbi decode → beat boundaries
      5. refine_qrs_boundaries + compute_qrs_polarity_v2 → QRS metrics
      6. P-wave extraction from HSMM P-state boundaries

    Parameters
    ----------
    fs : float
        Sampling frequency override (default 250; actual fs from aECG is used).
    max_samples : int or None
        Truncate signals to this many samples (None = full signal).
        Default 16000 = full 10 s at 1 kHz (4000 was the 250 Hz-era value,
        i.e. only 4 s of a 1 kHz record).
    use_prominence_delineation : bool
        Refine P/T beat boundaries with the prominence delineator
        (Emrich et al., EUSIPCO 2024) after HSMM segmentation. Falls back to
        the raw HSMM boundaries per beat where the delineator finds nothing,
        and entirely if the package is missing.
    """

    def __init__(self, fs: float = 250.0, max_samples: int = 16000,
                 use_prominence_delineation: bool = True):
        self.fs = fs
        self.max_samples = max_samples
        self.use_prominence_delineation = use_prominence_delineation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_record(self, aecg_data: dict,
                       record_name: str | None = None) -> tuple:
        """Run HSMM pipeline on all 6 limb leads for one record.

        Parameters
        ----------
        aecg_data : dict
            Output of parse_aecg(). Must contain 'signals' key with
            at least some of the 6 limb leads.
        record_name : str or None
            Record label. Falls back to aecg_data['filename'].

        Returns
        -------
        (LimbLeadResult, dict)
            LimbLeadResult: aggregated metrics.
            dict: lead_name -> segment data dict for plotting:
                {lead_name: {filtered_ecg, state_labels, state_names, fs, beats}}
        """
        t0 = time.time()

        signals = aecg_data.get('signals', {})
        # `or` (not .get default) so an explicit fs=None from a partial
        # parser result falls back too, instead of propagating None.
        fs_actual = aecg_data.get('fs') or self.fs
        record_name = record_name or aecg_data.get('filename', 'unknown')

        leads: dict[str, LeadResult | None] = {}
        seg_data: dict[str, dict | None] = {}
        total_beats = 0

        for lead_name in LIMB_LEADS:
            sig = signals.get(lead_name)
            if sig is None:
                leads[lead_name] = None
                seg_data[lead_name] = None
                continue

            lead_result, sd = self._process_lead(sig, lead_name, fs_actual)
            leads[lead_name] = lead_result
            seg_data[lead_name] = sd
            if lead_result is not None:
                total_beats += lead_result.n_beats

        dt = time.time() - t0

        return LimbLeadResult(
            record=record_name,
            fs=fs_actual,
            n_samples=aecg_data.get('n_samples', 0),
            leads=leads,
            measurements=aecg_data.get('measurements', {}),
            interpretation=aecg_data.get('interpretation', ''),
            n_total_beats=total_beats,
            processing_time_sec=round(dt, 1),
        ), seg_data

    # ------------------------------------------------------------------
    # Single-lead pipeline
    # ------------------------------------------------------------------
    def _process_lead(self, signal: np.ndarray,
                      lead_name: str, fs_actual: float) -> tuple:
        """Run full HSMM pipeline on one lead.

        Steps:
          1. Truncate + preprocess
          2. Feature extraction
          3. HSMM model init + segment
          4. QRS refinement + polarity v2 per beat
          5. P-wave extraction per beat
          6. Aggregate metrics

        Returns
        -------
        (LeadResult | None, dict | None)
            LeadResult: aggregated metrics, or None if no beats.
            dict: raw segment data for plotting:
                filtered_ecg, state_labels, state_names, fs
        """
        # Truncate
        sig = signal[:self.max_samples].astype(np.float64) if self.max_samples \
              else signal.astype(np.float64)

        # ---- Step 1: Preprocess ----
        prep = ECGPreprocessor(fs=fs_actual)
        clean = prep.preprocess(sig)

        # ---- Step 2: Features ----
        fe = FeatureExtractor(fs=fs_actual)
        features = fe.extract(clean)

        # ---- Step 3: HSMM segment ----
        model = HSMMModel(fs=fs_actual)
        model.initialize_with_priors()
        model.set_left_right_topology()
        smart_initialize_gmms(model, features)

        seg = ECGSegmenter(preprocessor=prep, feature_extractor=fe,
                           model=model, fs=fs_actual)
        seg_result = seg.segment(sig)

        # ---- Raw segment data for plotting ----
        seg_data = {
            'filtered_ecg': clean,
            'state_labels': seg_result.state_labels,
            'state_names': seg_result.state_names,
            'fs': fs_actual,
            'beats': seg_result.beats,  # BeatBoundary objects
        }

        # ---- Step 3.5: Prominence refinement of P/T boundaries ----
        # R-anchored physiological-window delineation (Emrich et al. 2024)
        # overwrites BeatBoundary p_onset/p_offset/t_onset/t_offset where it
        # finds valid waves; HSMM boundaries stay as the per-beat fallback.
        # QRS boundaries are untouched (package R_on/R_off are exploratory).
        prom_refined = 0
        if self.use_prominence_delineation and _HAS_PROMINENCE and seg_result.beats:
            prom_refined = refine_p_t_boundaries(
                seg_result.beats, clean, fs_actual)
        seg_data['prominence_refined_beats'] = prom_refined

        # ---- Steps 4 & 5: Per-beat extraction ----
        beats = []
        p_waves = []
        t_waves = []
        T = len(clean)

        for b in seg_result.beats:
            if b.q_onset <= 0 or b.r_peak <= 0 or b.s_offset <= 0:
                continue

            # QRS refined boundaries
            q_on, r_pk, s_off = refine_qrs_boundaries(
                clean, b.q_onset, b.r_peak, b.s_offset, fs_actual)

            # QRS polarity (C1-only: Dominant Deflection)
            pol = compute_qrs_polarity_v2(
                clean, q_on, r_pk, s_off, fs_actual, lead_name=lead_name,
                criterion_mode='c1')

            # QRS segment metrics
            bl_win = int(0.12 * fs_actual)  # 120 ms pre-QRS baseline
            bl = float(np.mean(clean[max(0, q_on - bl_win):q_on])) if q_on >= bl_win \
                 else float(np.median(clean[q_on:min(q_on + 10, T)]))
            r_amp = float(clean[r_pk] - bl) if 0 <= r_pk < T else 0.0
            dur_ms = (s_off - q_on) / fs_actual * 1000.0

            beats.append({
                'beat_id': b.beat_id,
                'q_onset': int(q_on), 'r_peak': int(r_pk), 's_offset': int(s_off),
                'polarity': pol['polarity'],
                'confidence': pol['confidence'],
                'polarity_score': pol['polarity_score'],
                'energy_ratio': pol['energy_ratio'],
                'peak_count': pol['peak_count'],
                'rs_ratio': pol['rs_ratio'],
                'qrs_net_area': pol['qrs_net_area'],
                'duration_ms': round(dur_ms, 2),
                'r_amplitude': round(r_amp, 4),
            })

            # P-wave from HSMM P-state boundaries
            pw = self._extract_p_wave(clean, b, fs_actual)
            if pw is not None:
                p_waves.append(pw)

            # T-wave from HSMM T-state boundaries
            tw = self._extract_t_wave(clean, b, fs_actual)
            if tw is not None:
                t_waves.append(tw)

        if not beats:
            return None, seg_data

        # ---- Step 6: Aggregate ----
        pol_counts = dict(Counter(b['polarity'] for b in beats))
        p_pol_counts = dict(Counter(p['polarity'] for p in p_waves)) if p_waves else {}
        t_pol_counts = dict(Counter(t['polarity'] for t in t_waves)) if t_waves else {}

        mean_dur = np.mean([b['duration_ms'] for b in beats])
        mean_rs = np.mean([b['rs_ratio'] for b in beats])
        mean_qrs_net = np.mean([b['qrs_net_area'] for b in beats])
        mean_r_amp = np.mean([abs(b['r_amplitude']) for b in beats])
        mean_p_net = np.mean([p['net_area'] for p in p_waves]) if p_waves else 0.0
        mean_t_net = np.mean([t['net_area'] for t in t_waves]) if t_waves else 0.0

        # S amplitude: nadir in the post-R portion of each QRS
        s_vals = []
        for b in beats:
            seg = clean[b['q_onset']:b['s_offset'] + 1]
            r_idx = b['r_peak'] - b['q_onset']
            if 0 <= r_idx < len(seg):
                s_vals.append(float(np.min(seg[r_idx:])))
        mean_s_amp = float(np.mean(np.abs(s_vals))) if s_vals else 0.0

        lr = LeadResult(
            lead_name=lead_name,
            n_beats=len(beats),
            beats=beats,
            p_waves=p_waves,
            t_waves=t_waves,
            polarity_counts=pol_counts,
            p_polarity_counts=p_pol_counts,
            t_polarity_counts=t_pol_counts,
            mean_qrs_dur_ms=round(float(mean_dur), 1),
            mean_rs_ratio=round(float(mean_rs), 4),
            mean_qrs_net=round(float(mean_qrs_net), 4),
            mean_p_net=round(float(mean_p_net), 4),
            mean_t_net=round(float(mean_t_net), 4),
            mean_r_amplitude=round(float(mean_r_amp), 4),
            mean_s_amplitude=round(float(mean_s_amp), 4),
        )
        return lr, seg_data

    # ------------------------------------------------------------------
    # P-wave extraction
    # ------------------------------------------------------------------
    def _extract_p_wave(self, clean: np.ndarray, beat,
                        fs: float) -> dict | None:
        """Extract P-wave metrics from HSMM Stage 1 P-state boundaries.

        Uses the HSMM-segmented P region directly (no focused Stage 2).
        This gives per-beat P-wave net area and polarity at low cost.

        Parameters
        ----------
        clean : np.ndarray
            Preprocessed ECG signal.
        beat : BeatBoundary
            HSMM beat with p_onset / p_offset.
        fs : float
            Sampling frequency.

        Returns
        -------
        dict or None — keys: beat_id, onset, offset, duration_ms,
                      net_area, peak_amplitude, polarity
        """
        if beat.p_onset < 0 or beat.p_offset < 0:
            return None
        if beat.p_offset <= beat.p_onset:
            return None

        T = len(clean)
        p_on = max(0, beat.p_onset)
        p_off = min(T - 1, beat.p_offset)

        if p_off - p_on < 3:
            return None

        p_seg = clean[p_on:p_off + 1]

        # Baseline from pre-P quiet segment
        bl_start = max(0, p_on - int(0.05 * fs))
        if bl_start < p_on and (p_on - bl_start) >= 5:
            bl = float(np.median(clean[bl_start:p_on]))
        else:
            bl = float(np.median(p_seg[:min(5, len(p_seg))]))

        detrend = p_seg - bl
        net_area = float(np.sum(detrend))
        p_peak = float(np.max(np.abs(detrend)))
        duration_ms = len(p_seg) / fs * 1000.0

        # Simple polarity from net area
        noise_floor = 0.01 * max(abs(detrend).max(), 0.001)
        if net_area > noise_floor:
            pol = 'positive'
        elif net_area < -noise_floor:
            pol = 'negative'
        else:
            pol = 'biphasic'

        return {
            'beat_id': beat.beat_id,
            'onset': int(p_on),
            'offset': int(p_off),
            'duration_ms': round(duration_ms, 2),
            'net_area': round(net_area, 4),
            'peak_amplitude': round(p_peak, 4),
            'polarity': pol,
            'source': getattr(beat, 'p_source', 'hsmm'),
        }

    # ------------------------------------------------------------------
    # T-wave extraction
    # ------------------------------------------------------------------
    def _extract_t_wave(self, clean: np.ndarray, beat,
                        fs: float) -> dict | None:
        """Extract T-wave metrics from HSMM Stage 1 T-state boundaries.

        Uses the HSMM-segmented T region directly. The T wave is typically
        150-400ms after the R peak, with a duration of 150-250ms.

        Returns
        -------
        dict or None — keys: beat_id, onset, offset, duration_ms,
                      net_area, peak_amplitude, polarity
        """
        if beat.t_onset < 0 or beat.t_offset < 0:
            return None
        if beat.t_offset <= beat.t_onset:
            return None

        T_len = len(clean)
        t_on = max(0, beat.t_onset)
        t_off = min(T_len - 1, beat.t_offset)

        if t_off - t_on < 3:
            return None

        t_seg = clean[t_on:t_off + 1]

        # Baseline from ST segment (between S offset and T onset)
        st_start = max(0, beat.s_offset + 1) if beat.s_offset > 0 else t_on - int(0.05 * fs)
        st_end = t_on
        if st_end - st_start >= 5:
            bl = float(np.median(clean[st_start:st_end]))
        else:
            bl = float(np.median(t_seg[:min(5, len(t_seg))]))

        detrend = t_seg - bl
        net_area = float(np.sum(detrend))
        t_peak = float(np.max(np.abs(detrend)))
        duration_ms = len(t_seg) / fs * 1000.0

        # Polarity: T wave is normally positive in most leads
        noise_floor = 0.01 * max(abs(detrend).max(), 0.001)
        if net_area > noise_floor:
            pol = 'positive'
        elif net_area < -noise_floor:
            pol = 'negative'
        else:
            pol = 'biphasic'

        return {
            'beat_id': beat.beat_id,
            'onset': int(t_on),
            'offset': int(t_off),
            'duration_ms': round(duration_ms, 2),
            'net_area': round(net_area, 4),
            'peak_amplitude': round(t_peak, 4),
            'polarity': pol,
            'source': getattr(beat, 't_source', 'hsmm'),
        }


# ---------------------------------------------------------------------------
# Convenience: summary helpers
# ---------------------------------------------------------------------------

def result_to_dict(result: LimbLeadResult) -> dict:
    """Serialize a LimbLeadResult to a JSON-safe dict."""
    def _lead_to_dict(lr: LeadResult | None) -> dict | None:
        if lr is None:
            return None
        return {
            'lead_name': lr.lead_name,
            'n_beats': lr.n_beats,
            'polarity_counts': lr.polarity_counts,
            'p_polarity_counts': lr.p_polarity_counts,
            't_polarity_counts': lr.t_polarity_counts,
            'mean_qrs_dur_ms': lr.mean_qrs_dur_ms,
            'mean_rs_ratio': lr.mean_rs_ratio,
            'mean_qrs_net': lr.mean_qrs_net,
            'mean_p_net': lr.mean_p_net,
            'mean_t_net': lr.mean_t_net,
            'mean_r_amplitude': lr.mean_r_amplitude,
            'mean_s_amplitude': lr.mean_s_amplitude,
        }

    return {
        'record': result.record,
        'fs': result.fs,
        'n_samples': result.n_samples,
        'n_total_beats': result.n_total_beats,
        'processing_time_sec': result.processing_time_sec,
        'measurements': result.measurements,
        'interpretation': result.interpretation,
        'leads': {name: _lead_to_dict(lr)
                  for name, lr in result.leads.items()},
    }


def build_summary_table(result: LimbLeadResult) -> str:
    """Build a human-readable summary table for one record.

    Parameters
    ----------
    result : LimbLeadResult

    Returns
    -------
    str — multi-line summary table.
    """
    lines = []
    lines.append(f"{'='*90}")
    lines.append(f"  Record: {result.record}  |  "
                 f"fs={result.fs:.0f} Hz  |  "
                 f"{result.n_samples} samples  |  "
                 f"{result.processing_time_sec:.1f}s")
    lines.append(f"  Measurements: {result.measurements}")
    lines.append(f"{'─'*90}")
    header = (f"  {'Lead':<6} {'Beats':>5}  "
              f"{'QRS Polarity':>28}  {'P Polarity':>20}  "
              f"{'QRSdur':>7}  {'R/S':>7}  {'QRSnet':>8}  {'Pnet':>8}")
    lines.append(header)
    lines.append(f"  {'─'*6} {'─'*5}  "
                 f"{'─'*28}  {'─'*20}  "
                 f"{'─'*7}  {'─'*7}  {'─'*8}  {'─'*8}")

    for lead_name in LIMB_LEADS:
        lr = result.leads.get(lead_name)
        if lr is None:
            lines.append(f"  {lead_name:<6} {'—':>5}  {'—':>28}  {'—':>20}")
            continue

        # QRS polarity string
        qrs_str = ', '.join(f'{k}:{v}' for k, v in
                           sorted(lr.polarity_counts.items()))
        if len(qrs_str) > 27:
            qrs_str = qrs_str[:24] + '...'

        # P polarity string
        p_str = ', '.join(f'{k}:{v}' for k, v in
                         sorted(lr.p_polarity_counts.items()))
        if len(p_str) > 19:
            p_str = p_str[:16] + '...'

        lines.append(
            f"  {lead_name:<6} {lr.n_beats:>5}  "
            f"{qrs_str:>28}  {p_str:>20}  "
            f"{lr.mean_qrs_dur_ms:>6.0f}ms  "
            f"{lr.mean_rs_ratio:>6.2f}  "
            f"{lr.mean_qrs_net:>8.1f}  "
            f"{lr.mean_p_net:>8.3f}"
        )

    lines.append(f"{'='*90}")
    return '\n'.join(lines)


def compare_polarity_across_leads(result: LimbLeadResult) -> dict:
    """Compute cross-lead polarity summary for reversal analysis.

    Returns a dict with:
      - per-lead dominant QRS polarity
      - per-lead dominant P polarity
      - Einthoven consistency check (Lead II ≈ Lead I + Lead III)
    """
    qrs_dominant = {}
    p_dominant = {}

    for lead_name in LIMB_LEADS:
        lr = result.leads.get(lead_name)
        if lr is None or not lr.polarity_counts:
            qrs_dominant[lead_name] = 'N/A'
            p_dominant[lead_name] = 'N/A'
            continue

        # Dominant QRS polarity (excluding 'uncertain')
        pc = {k: v for k, v in lr.polarity_counts.items() if k != 'uncertain'}
        qrs_dominant[lead_name] = max(pc, key=pc.get) if pc else 'uncertain'

        # Dominant P polarity
        pp = {k: v for k, v in lr.p_polarity_counts.items() if k != 'biphasic'}
        p_dominant[lead_name] = max(pp, key=pp.get) if pp else 'biphasic'

    return {
        'record': result.record,
        'qrs_dominant': qrs_dominant,
        'p_dominant': p_dominant,
        'measurements': result.measurements,
    }
