"""Martinez-style wavelet delineation stage (wavedet wrapper).

Wraps the PyPI ``wavedet`` package — a high-fidelity Python port of the
ecg-kit ``wavedet_3D`` wavelet delineator (Martinez et al. 2004 family,
quadratic-spline dyadic wavelet, modmax QRS + zero-crossing P/T; ~99%
mark parity with MATLAB per the port's own tests). NOTE: this is the
faithful lineage; NeuroKit2's ``dwt`` is a loose adaptation and was measured
as the worst traditional delineator in the 2026 independent benchmark —
do not confuse the two.

Role in this repo: SECOND OPINION on QRS boundaries. The production QRS
boundaries stay with ``extraction.qrs_refiner``; this stage's ``qrs_on`` /
``qrs_off`` are exposed for cross-checking (flagging beats where the
derivative-walk refiner and the wavelet disagree, e.g. the diagnosed
early-q_onset bias). Its P/T marks are available but not production refs.

All wavedet config times are seconds converted to samples internally with
the fs passed to :func:`wavedet.delineate`, so any sampling rate works
(real data is 1 kHz). Positions are 0-based sample indices; NaN = not found.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    'WaveletBeat',
    'WaveletStage',
    'crosscheck_qrs_boundaries',
]


@dataclass
class WaveletBeat:
    """Per-beat wavelet delineation (sample indices, -1 = not found)."""
    r_peak: int
    qrs_onset: int = -1
    qrs_offset: int = -1
    p_onset: int = -1
    p_offset: int = -1
    t_onset: int = -1
    t_offset: int = -1


def _idx(value) -> int:
    """Map a wavedet entry (float, possibly NaN) to a sample index / -1."""
    v = float(value)
    return int(v) if np.isfinite(v) else -1


class WaveletStage:
    """Wrapper around ``wavedet.delineate`` for one lead.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz. Required: wavedet converts all its
        second-based tunables to samples with this value.

    Raises
    ------
    ImportError
        If the ``wavedet`` package is not installed.
    ValueError
        If fs is not a positive sampling rate.
    """

    def __init__(self, fs: float):
        if fs is None or fs <= 0:
            raise ValueError(f"fs must be a positive sampling rate, got {fs}")
        self.fs = float(fs)
        try:
            import wavedet
        except ImportError as exc:
            raise ImportError(
                "wavedet is required for WaveletStage "
                "(pip install wavedet)") from exc
        self._wavedet = wavedet

    def delineate(self, ecg_clean: np.ndarray) -> list[WaveletBeat]:
        """Delineate one lead; beats are keyed by wavedet's own R peaks.

        wavedet detects its own beats (no external R-peaks API), so the
        returned beats may not align 1:1 with HSMM beats — use
        :func:`crosscheck_qrs_boundaries` for R-anchored matching.
        """
        sig = np.asarray(ecg_clean, dtype=np.float64)
        try:
            d = self._wavedet.delineate(sig, int(self.fs))
        except Exception:
            # upstream raises on degenerate signals; mirror the per-lead
            # try/catch of wavedet_interface.m by returning no beats
            return []

        beats = []
        for i in range(len(d.qrs_on)):
            r = _idx(d.r[i])
            beats.append(WaveletBeat(
                r_peak=r,
                qrs_onset=_idx(d.qrs_on[i]),
                qrs_offset=_idx(d.qrs_off[i]),
                p_onset=_idx(d.p_on[i]),
                p_offset=_idx(d.p_off[i]),
                t_onset=_idx(d.t_on[i]),
                t_offset=_idx(d.t_off[i]),
            ))
        return beats


def crosscheck_qrs_boundaries(beats, ecg_clean: np.ndarray, fs: float,
                              tol_ms: float = 20.0) -> dict:
    """Compare production QRS boundaries against the wavelet second opinion.

    Matches wavedet beats to ``beats`` by R-peak proximity (within 50 ms)
    and measures the onset/offset shift. Read-only: boundaries are NOT
    modified — the wavelet marks are a reference, not a replacement.

    Parameters
    ----------
    beats : list[BeatBoundary]
        Production beats (already QRS-refined).
    ecg_clean : np.ndarray
        Filtered single-lead ECG, same index space as the beat boundaries.
    fs : float
        Sampling frequency in Hz.
    tol_ms : float
        Disagreement threshold: beats whose onset or offset shift exceeds
        this are counted in ``n_disagree``.

    Returns
    -------
    dict with keys:
        n_production, n_wavelet, n_matched, n_disagree,
        median_on_shift_ms, median_off_shift_ms,
        matches: list[dict(r_peak, wd_qrs_onset, wd_qrs_offset,
                           on_shift_ms, off_shift_ms, disagree)]
    Empty dict (all zero) if wavedet is missing or finds nothing.
    """
    empty = {'n_production': 0, 'n_wavelet': 0, 'n_matched': 0,
             'n_disagree': 0, 'median_on_shift_ms': 0.0,
             'median_off_shift_ms': 0.0, 'matches': []}
    try:
        stage = WaveletStage(fs)
    except ImportError:
        return empty

    wd = stage.delineate(ecg_clean)
    wd = [w for w in wd if w.r_peak > 0 and w.qrs_onset > 0 and w.qrs_offset > 0]
    prod = [b for b in beats if b.r_peak is not None and b.r_peak > 0
            and b.q_onset > 0 and b.s_offset > 0]
    empty['n_production'] = len(prod)
    empty['n_wavelet'] = len(wd)
    if not prod or not wd:
        return empty

    tol = int(round(0.05 * fs))  # 50 ms R-match window
    matches = []
    on_shifts, off_shifts = [], []
    for b in prod:
        best = min(wd, key=lambda w: abs(w.r_peak - b.r_peak), default=None)
        if best is None or abs(best.r_peak - b.r_peak) > tol:
            continue
        on_ms = (best.qrs_onset - b.q_onset) / fs * 1000.0
        off_ms = (best.qrs_offset - b.s_offset) / fs * 1000.0
        on_shifts.append(on_ms)
        off_shifts.append(off_ms)
        matches.append({
            'r_peak': int(b.r_peak),
            'wd_qrs_onset': int(best.qrs_onset),
            'wd_qrs_offset': int(best.qrs_offset),
            'on_shift_ms': round(on_ms, 1),
            'off_shift_ms': round(off_ms, 1),
            'disagree': abs(on_ms) > tol_ms or abs(off_ms) > tol_ms,
        })

    if not matches:
        return empty
    return {
        'n_production': len(prod),
        'n_wavelet': len(wd),
        'n_matched': len(matches),
        'n_disagree': sum(1 for m in matches if m['disagree']),
        'median_on_shift_ms': float(np.median(on_shifts)),
        'median_off_shift_ms': float(np.median(off_shifts)),
        'matches': matches,
    }
