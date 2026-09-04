"""Prominence-based P/T wave delineation stage.

Wraps the ``prominence-delineator`` package (Emrich, Gargano, Koka & Muma,
"Physiology-Informed ECG Delineation Based on Peak Prominence", EUSIPCO 2024;
GPL-3.0, PyPI: prominence-delineator). The delineator classifies P/QRS/T
extrema by peak prominence inside R-anchored physiological search windows;
all limits are defined in seconds and converted to samples at construction,
so the stage works natively at any sampling rate (the real aECG dataset is
1 kHz).

Scope in this repo: refinement of P and T onset/offset on top of the HSMM
segmentation. The package's R_on/R_off are documented by its authors as
exploratory and may not bracket the full QRS complex, so QRS boundaries
remain the job of ``extraction.qrs_refiner.refine_qrs_boundaries``.

Known limitations (see output/traditional_delineation_recommendations):
- The package searches P among local maxima of the signal. This wrapper
  therefore runs a second pass on the negated signal and accepts an
  inverted-P candidate when the upright one is missing, so retrograde /
  aVR-negative P waves are covered.
- False-P control: every P candidate (upright or inverted) must pass a
  local amplitude gate — detrended P amplitude >= 2.5x the MAD-based noise
  sigma of the quiet TP segment just before the P window. A fabricated P on
  a flat TP segment (ectopic beat, AF) fails this gate and the HSMM
  boundary is kept. NOTE: ``audit_spectral_consistency`` audits the HSMM
  state sequence only and cannot see these refined windows (window-level
  audit is future work).
- Monotonicity: a post-pass clamps every T offset below the next beat's
  earliest anchor (p_onset / q_onset / r_peak); degenerate windows are
  dropped (set to -1), so T windows never run into the next beat's P.
- Requires at least 2 R-peaks: the package indexes ``rr[0]``/``rr[-1]`` and
  crashes on fewer, so the wrapper returns an empty result instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    'ProminenceBeat',
    'ProminenceStage',
    'delineate_beats',
    'refine_p_t_boundaries',
]


@dataclass
class ProminenceBeat:
    """Per-beat prominence delineation result.

    All fields are absolute sample indices into the signal passed to
    :meth:`ProminenceStage.delineate`; -1 means the wave was not detected.
    ``r_onset``/``r_offset`` are the package's exploratory R-on/R-off and
    must NOT be used as QRS onset/offset (see module docstring).
    """
    beat_id: int
    r_peak: int
    p_peak: int = -1
    q_peak: int = -1
    s_peak: int = -1
    t_peak: int = -1
    p_onset: int = -1
    p_offset: int = -1
    t_onset: int = -1
    t_offset: int = -1
    r_onset: int = -1
    r_offset: int = -1
    # Inverted-P candidates from the negated-signal pass (upright P missing)
    p_inv_peak: int = -1
    p_inv_onset: int = -1
    p_inv_offset: int = -1


def _idx(value) -> int:
    """Map a package wave entry (int or None) to a sample index / -1."""
    return int(value) if value is not None else -1


class ProminenceStage:
    """Beat-aligned wrapper around ``prominence_delineator.ProminenceDelineator``.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz. Required (no default): the package converts
        every second-based physiological limit to samples in the constructor,
        so an instance is bound to exactly one sampling rate.

    Raises
    ------
    ImportError
        If the ``prominence-delineator`` package is not installed.
    ValueError
        If fs is not a positive sampling rate.
    """

    def __init__(self, fs: float):
        if fs is None or fs <= 0:
            raise ValueError(f"fs must be a positive sampling rate, got {fs}")
        self.fs = float(fs)
        try:
            from prominence_delineator import ProminenceDelineator
        except ImportError as exc:
            raise ImportError(
                "prominence-delineator is required for ProminenceStage "
                "(pip install prominence-delineator)") from exc
        self._delineator = ProminenceDelineator(sampling_frequency=self.fs)

    def delineate(self, ecg_clean: np.ndarray,
                  r_peaks) -> list[ProminenceBeat]:
        """Delineate P/QRS/T waves for one lead.

        Parameters
        ----------
        ecg_clean : np.ndarray
            Filtered single-lead ECG. Amplitude units are irrelevant
            (prominence is scale-based); the repo's preprocessed (z-scored)
            signal is fine. Upright R-peaks are assumed.
        r_peaks : sequence of int
            R-peak sample indices (e.g. ``[b.r_peak for b in beats]``).
            Duplicates are dropped; at least 2 unique peaks are required.

        Returns
        -------
        list[ProminenceBeat]
            One entry per R-peak, beat-aligned (``include_nodetections=True``
            under the hood), sorted by r_peak. Empty if fewer than 2 R-peaks.
        """
        r_peaks = np.asarray(sorted({int(r) for r in r_peaks}), dtype=int)
        if len(r_peaks) < 2:
            return []

        sig = np.asarray(ecg_clean, dtype=np.float64)
        waves = self._delineator.find_waves(
            sig, r_peaks, include_nodetections=True)
        # Second pass on the negated signal: the package's P search scans
        # local maxima only, so inverted P waves are only visible after
        # mirroring. Only the P fields of this pass are meaningful.
        inv_waves = self._delineator.find_waves(
            -sig, r_peaks, include_nodetections=True)

        beats = []
        for i, r in enumerate(r_peaks):
            beats.append(ProminenceBeat(
                beat_id=i,
                r_peak=int(r),
                p_peak=_idx(waves['P'][i]),
                q_peak=_idx(waves['Q'][i]),
                s_peak=_idx(waves['S'][i]),
                t_peak=_idx(waves['T'][i]),
                p_onset=_idx(waves['P_on'][i]),
                p_offset=_idx(waves['P_off'][i]),
                t_onset=_idx(waves['T_on'][i]),
                t_offset=_idx(waves['T_off'][i]),
                r_onset=_idx(waves['R_on'][i]),
                r_offset=_idx(waves['R_off'][i]),
                p_inv_peak=_idx(inv_waves['P'][i]),
                p_inv_onset=_idx(inv_waves['P_on'][i]),
                p_inv_offset=_idx(inv_waves['P_off'][i]),
            ))
        return beats


def delineate_beats(ecg_clean: np.ndarray, r_peaks, fs: float) -> list[ProminenceBeat]:
    """Functional convenience wrapper: ``ProminenceStage(fs).delineate(...)``."""
    return ProminenceStage(fs).delineate(ecg_clean, r_peaks)


def _valid_p(p_on: int, p_off: int, r_peak: int, q_onset: int = -1) -> bool:
    """P boundaries must lie strictly before the QRS onset.

    The package's P_off is the right prominence base of the P peak, which can
    land at/after the QRS onset when the P wave sits close to the Q (observed
    on real data). Falls back to the R peak when q_onset is unknown.
    """
    if not (0 <= p_on < p_off < r_peak):
        return False
    if q_onset > 0 and p_off > q_onset:
        return False
    return True


def _valid_t(pb: ProminenceBeat, r_peak: int, n_samples: int,
             next_anchor: int = -1) -> bool:
    """T boundaries must lie after the R peak and before the next beat.

    Without a forward cap the refined T window may run into the next beat's
    P/QRS (the package only bounds the window by its RR-half split).
    ``next_anchor`` is the next beat's p_onset / q_onset / r_peak (whichever
    is available), -1 if none.
    """
    if not (r_peak < pb.t_onset < pb.t_offset < n_samples):
        return False
    if next_anchor > r_peak and pb.t_offset > next_anchor:
        return False
    return True


def _next_anchor(beats, i: int, r_peak: int) -> int:
    """First available forward anchor of the next beat (p_onset/q_onset/r_peak)."""
    if i + 1 >= len(beats):
        return -1
    nb = beats[i + 1]
    for v in (getattr(nb, 'p_onset', -1), getattr(nb, 'q_onset', -1),
              getattr(nb, 'r_peak', -1)):
        if v > r_peak:
            return v
    return -1


# Local-noise amplitude gate for P candidates. 2.5x the MAD-based sigma of
# the quiet TP segment just before the P window; deliberately permissive
# (an absolute 3-5x full-signal RMS gate was measured to discard 50-77% of
# true P waves).
P_SNR_FACTOR = 2.5


def _p_passes_snr(ecg: np.ndarray, p_on: int, p_off: int, fs: float) -> bool:
    """Detrended P amplitude must exceed the local TP-segment noise.

    The noise reference is the 40-160 ms window ending 20 ms before the P
    onset (20 ms guard so the P upstroke itself is excluded). If the
    reference window is unavailable (signal start) the gate passes — the
    geometry guards still apply.
    """
    n = len(ecg)
    ref_end = p_on - int(round(0.02 * fs))
    ref_start = max(0, p_on - int(round(0.16 * fs)))
    seg = ecg[p_on:p_off + 1]
    if len(seg) == 0:
        return False
    amp = float(np.max(np.abs(seg - np.median(seg))))
    if ref_end - ref_start >= int(round(0.03 * fs)):
        ref = ecg[ref_start:ref_end]
        med = np.median(ref)
        sigma = 1.4826 * float(np.median(np.abs(ref - med)))
        if sigma > 1e-12 and amp < P_SNR_FACTOR * sigma:
            return False
    return True


def refine_p_t_boundaries(beats, ecg_clean: np.ndarray, fs: float) -> int:
    """Refine P/T boundaries of HSMM ``BeatBoundary`` objects in place.

    For every beat whose prominence delineation yields physiologically valid
    boundaries, ``p_onset``/``p_offset``/``t_onset``/``t_offset`` are
    overwritten. Beats where the delineator finds nothing (low-amplitude or
    inverted P, absent T) keep their HSMM boundaries — the HSMM acts as the
    fallback, the delineator as the primary.

    Parameters
    ----------
    beats : list[BeatBoundary]
        HSMM beats (``segmentation.segmenter.BeatBoundary``), modified in place.
    ecg_clean : np.ndarray
        Filtered single-lead ECG, same index space as the beat boundaries.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    int
        Number of beats whose P boundaries were refined.
    """
    r_peaks = [b.r_peak for b in beats if b.r_peak is not None and b.r_peak > 0]
    if len(r_peaks) < 2:
        return 0

    try:
        stage = ProminenceStage(fs)
    except ImportError:
        return 0

    by_rpeak = {pb.r_peak: pb for pb in stage.delineate(ecg_clean, r_peaks)}
    if not by_rpeak:
        return 0

    n_samples = len(ecg_clean)
    n_refined = 0
    for i, b in enumerate(beats):
        pb = by_rpeak.get(b.r_peak)
        if pb is None:
            continue
        # P: upright candidate first, then the inverted-P candidate from the
        # negated-signal pass. Both must pass geometry AND the local SNR
        # gate (fabricated P on flat TP segments fails the amplitude check).
        if (_valid_p(pb.p_onset, pb.p_offset, b.r_peak, q_onset=b.q_onset)
                and _p_passes_snr(ecg_clean, pb.p_onset, pb.p_offset, fs)):
            b.p_onset = pb.p_onset
            b.p_offset = pb.p_offset
            if hasattr(b, 'p_source'):
                b.p_source = 'prominence'
            n_refined += 1
        elif (pb.p_inv_onset >= 0
              and _valid_p(pb.p_inv_onset, pb.p_inv_offset, b.r_peak,
                           q_onset=b.q_onset)
              and _p_passes_snr(ecg_clean, pb.p_inv_onset, pb.p_inv_offset, fs)):
            b.p_onset = pb.p_inv_onset
            b.p_offset = pb.p_inv_offset
            if hasattr(b, 'p_source'):
                b.p_source = 'prominence'
            n_refined += 1
        if _valid_t(pb, b.r_peak, n_samples,
                    next_anchor=_next_anchor(beats, i, b.r_peak)):
            b.t_onset = pb.t_onset
            b.t_offset = pb.t_offset
            if hasattr(b, 't_source'):
                b.t_source = 'prominence'

    # Monotonicity post-pass: the next beat's P may have been refined to an
    # earlier onset after this beat's T was written, so re-clamp every T
    # below the (possibly new) next anchor; drop degenerate windows.
    min_t_samples = int(round(0.08 * fs))
    for i in range(len(beats) - 1):
        anchor = _next_anchor(beats, i, beats[i].r_peak)
        if anchor > 0 and beats[i].t_offset >= anchor:
            new_off = anchor - 1
            if new_off - beats[i].t_onset >= min_t_samples:
                beats[i].t_offset = new_off
            else:
                beats[i].t_onset = -1
                beats[i].t_offset = -1
    return n_refined
