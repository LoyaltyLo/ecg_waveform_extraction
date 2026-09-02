"""Spectrogram-based RA-LA (left/right arm) lead reversal detection.

Why cross-lead phase and not magnitude spectra: RA-LA reversal inverts
lead I (I' = -I) and swaps II and III, but a pure polarity inversion
leaves a magnitude spectrum unchanged (|FFT(-x)| = |FFT(x)|). The
reversal IS visible in the *cross-lead* time-frequency phase:

    normal axis:  lead I and lead II QRS are in phase      (phi ~ 0)
    RA-LA swap :  I' = -I is anti-phase with II' (= III)   (phi ~ pi)

Method: complex Morlet CWT of leads I and II on a shared frequency grid;
cross-wavelet spectrum X = W_I * conj(W_II); weighted circular statistics
of X's phase over the QRS band rows inside windows anchored on the R peaks
of lead I (anchors from tf_segmentation.segment_tf). The concentration
(mean resultant length R-bar) of the phase samples is the confidence
measure. A time-domain companion feature — the median Pearson correlation
between the I and II QRS windows (broadband, threshold-free) — encodes the
same physics and gates the verdict.

Decision rule (calibrated on the RA-LA_Reversal dataset, see below): the
anti-phase boundary is |phi| >= 90 deg, NOT the a-priori 120/60 deg pair.
On this dataset many morphologically normal records sit at phi = +60..+90
(a slow negative wave after the QRS dominates the lead-I weight), so a
record is called reversed only when BOTH features agree on anti-phase,
and normal only when the correlation is clearly positive.

Validation: `synthetic_swap` builds a reversed copy of any record
(I' = -I, II' = III, III' = II) and `flip_lead_i` flips only lead I, so
every record yields a labeled (normal, reversed) pair. NOTE: the
cardiologist-confirmed xlsx (C:/LoyaltyLo/datasets/反接/左右手反接/,
26 records, 16-digit IDs) has ZERO ID overlap with the 666-record
RA-LA_Reversal working set (8-char IDs), so there is no true-label
check on this data — the main package's aECG_v10_triple.csv verdicts
(59% swapped) serve as the comparison baseline instead.
"""

from dataclasses import dataclass

import numpy as np

from .spectrogram import compute_cwt_complex
from .tf_segmentation import segment_tf

# ---------------------------------------------------------------------------
# Parameters (fs = 1000 Hz)
# ---------------------------------------------------------------------------
REV_FREQ_RANGE = (2.0, 30.0)  # Hz, shared CWT grid for both leads
REV_N_VOICES = 32

QRS_BAND_REV = (10.0, 25.0)   # Hz — cross-phase evaluated here
P_BAND_REV = (3.0, 8.0)       # Hz — secondary (informational) vote
QRS_WIN_MS = (-80.0, 80.0)    # QRS window around each R peak (phase + corr)
P_WIN_MS = (-220.0, -60.0)    # around each R peak (expected P timing)

ANTI_PHASE_MIN_DEG = 90.0  # |phi| >= this -> QRS vote is anti-phase
CORR_REVERSED_MAX = 0.0    # corr(I, II) QRS <= this -> corr vote anti-phase
CORR_NORMAL_MIN = 0.10     # corr(I, II) QRS >= this -> corr vote in-phase
REVERSED_MIN_RBAR = 0.30
P_USE_RBAR = 0.20          # P vote only when its phase is concentrated
QRS_VOTE_WEIGHT = 0.7
P_VOTE_WEIGHT = 0.3


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class TFReversalResult:
    """Outcome of the spectrogram-based RA-LA reversal check.

    verdict : 'normal' | 'reversed' | 'uncertain'
    confidence : 0-1, from the weighted circular concentration
    phase_qrs_deg / rbar_qrs : circular mean cross-phase (deg) and
        concentration in the QRS band (the primary vote)
    corr_qrs : median Pearson correlation of the I/II QRS windows
        (secondary gate; None when too few clean beats)
    phase_p_deg / rbar_p : cross-phase in the P band (informational;
        rbar_p ~ 0 when P is weak, which is common in this dataset)
    n_beats : number of R-peak anchors used
    qrs_power_ratio : mean QRS-band power lead I / lead II (informational)
    """

    verdict: str
    confidence: float
    phase_qrs_deg: float | None
    rbar_qrs: float | None
    corr_qrs: float | None
    phase_p_deg: float | None
    rbar_p: float | None
    n_beats: int
    qrs_power_ratio: float | None

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        for k in ('phase_qrs_deg', 'phase_p_deg', 'qrs_power_ratio'):
            if d[k] is not None:
                d[k] = round(float(d[k]), 1 if 'phase' in k else 3)
        for k in ('confidence', 'rbar_qrs', 'corr_qrs', 'rbar_p'):
            if d[k] is not None:
                d[k] = round(float(d[k]), 4)
        return d


# ---------------------------------------------------------------------------
# Circular statistics helpers
# ---------------------------------------------------------------------------
def _circular_stats(phase: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
    """Weighted circular mean (rad) and mean resultant length (0-1)."""
    wsum = float(np.sum(weight))
    if wsum <= 0:
        return np.nan, 0.0
    z = np.sum(weight * np.exp(1j * phase)) / wsum
    return float(np.angle(z)), float(np.abs(z))


def _wrap_deg(phase_deg: float) -> float:
    """Wrap to (-180, 180]."""
    return (phase_deg + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------
def detect_tf_reversal(clean_i: np.ndarray, clean_ii: np.ndarray, fs: float,
                       clean_iii: np.ndarray | None = None,
                       return_extras: bool = False):
    """Detect RA-LA reversal from preprocessed leads I and II.

    Parameters
    ----------
    clean_i, clean_ii : preprocessed (bandpassed, z-scored) lead I / II.
    clean_iii : optional preprocessed lead III (only used for the
        informational QRS power ratio when II/III are swapped).
    fs : sampling rate in Hz.
    return_extras : when True, return (result, extras) where extras holds
        the intermediate arrays for plotting: cross-phase (rad), weight,
        frequency axis, R-peak anchors and per-beat correlations.
    """
    clean_i = np.asarray(clean_i, dtype=np.float64)
    clean_ii = np.asarray(clean_ii, dtype=np.float64)
    n = min(len(clean_i), len(clean_ii))
    clean_i, clean_ii = clean_i[:n], clean_ii[:n]

    uncertain = TFReversalResult(
        verdict='uncertain', confidence=0.0, phase_qrs_deg=None, rbar_qrs=None,
        corr_qrs=None, phase_p_deg=None, rbar_p=None, n_beats=0,
        qrs_power_ratio=None,
    )

    # QRS anchors from lead I's TF segmentation
    seg = segment_tf(clean_i, fs)
    r_peaks = [b.r_peak for b in seg.beats]
    extras: dict = {'cross_phase': None, 'weight': None, 'freqs': None,
                    'r_peaks': r_peaks, 'beat_corr': []}
    if not r_peaks:
        return (uncertain, extras) if return_extras else uncertain

    # ---- Time-domain companion feature: I/II QRS-window correlation ----
    half = int(abs(QRS_WIN_MS[0]) * fs / 1000.0)
    corrs = []
    for r in r_peaks:
        if r - half < 0 or r + half > n:
            continue
        wi, w2 = clean_i[r - half:r + half], clean_ii[r - half:r + half]
        if wi.std() > 1e-6 and w2.std() > 1e-6:
            corrs.append(float(np.corrcoef(wi, w2)[0, 1]))
    corr_qrs = float(np.median(corrs)) if corrs else None
    extras['beat_corr'] = corrs

    # Shared CWT grid for both leads
    cwt_i = compute_cwt_complex(clean_i, fs=fs, wavelet='morlet',
                                freq_range=REV_FREQ_RANGE, n_voices=REV_N_VOICES)
    cwt_ii = compute_cwt_complex(clean_ii, fs=fs, wavelet='morlet',
                                 freq_range=REV_FREQ_RANGE, n_voices=REV_N_VOICES)
    w_i = np.asarray(cwt_i.data)
    w_ii = np.asarray(cwt_ii.data)
    freqs = cwt_i.freqs
    if w_i.shape != w_ii.shape:
        return (uncertain, extras) if return_extras else uncertain

    cross = w_i * np.conj(w_ii)
    phase = np.angle(cross)
    weight = np.abs(w_i) * np.abs(w_ii)
    extras['cross_phase'] = phase
    extras['weight'] = weight
    extras['freqs'] = freqs

    def band_vote(band: tuple[float, float], win_ms: tuple[float, float]) -> tuple[float, float]:
        fmask = (freqs >= band[0]) & (freqs <= band[1])
        if not fmask.any():
            return np.nan, 0.0
        ph_rows = phase[fmask, :]
        wt_rows = weight[fmask, :]
        ph_all, wt_all = [], []
        for r in r_peaks:
            lo = max(0, r + int(win_ms[0] * fs / 1000.0))
            hi = min(n, r + int(win_ms[1] * fs / 1000.0))
            if hi - lo < 10:
                continue
            ph_all.append(ph_rows[:, lo:hi].ravel())
            wt_all.append(wt_rows[:, lo:hi].ravel())
        if not ph_all:
            return np.nan, 0.0
        return _circular_stats(np.concatenate(ph_all), np.concatenate(wt_all))

    phi_qrs, rbar_qrs = band_vote(QRS_BAND_REV, QRS_WIN_MS)
    phi_p, rbar_p = band_vote(P_BAND_REV, P_WIN_MS)

    phase_qrs_deg = None if np.isnan(phi_qrs) else _wrap_deg(np.degrees(phi_qrs))
    phase_p_deg = None if np.isnan(phi_p) else _wrap_deg(np.degrees(phi_p))

    # Informational QRS-band power ratio (I / II)
    ratio = None
    if clean_iii is not None:
        iii = np.asarray(clean_iii, dtype=np.float64)[:n]
        fmask = (freqs >= QRS_BAND_REV[0]) & (freqs <= QRS_BAND_REV[1])
        if fmask.any():
            p_i = float(np.mean(np.abs(w_i[fmask, :]) ** 2))
            p_ii = float(np.mean(np.abs(w_ii[fmask, :]) ** 2))
            ratio = p_i / p_ii if p_ii > 0 else None

    if phase_qrs_deg is None or rbar_qrs is None:
        return (uncertain, extras) if return_extras else uncertain

    # ---- Decision: phase vote gated by the correlation vote ----
    # Anti-phase means the QRS-band cross-lead relationship is inverted,
    # i.e. |phi| beyond the perpendicular (>= 90 deg). The correlation
    # feature must agree (same physics, broadband view): negative
    # correlation = anti-phase, clearly positive = in-phase. Records
    # whose lead I is dominated by a slow post-QRS wave sit at phi =
    # +60..+90 with weak positive correlation -> 'uncertain', correctly
    # refusing to vote on ambiguous morphology.
    anti_deg = abs(phase_qrs_deg)  # 0 = in-phase, 180 = anti-phase
    phase_anti = anti_deg >= ANTI_PHASE_MIN_DEG and rbar_qrs >= REVERSED_MIN_RBAR
    corr_anti = corr_qrs is not None and corr_qrs <= CORR_REVERSED_MAX
    corr_in = corr_qrs is not None and corr_qrs >= CORR_NORMAL_MIN

    if phase_anti and corr_anti:
        verdict = 'reversed'
    elif not phase_anti and corr_in:
        verdict = 'normal'
    else:
        verdict = 'uncertain'

    if verdict == 'reversed':
        conf_qrs = rbar_qrs * (anti_deg / 180.0)
        confidence = QRS_VOTE_WEIGHT * conf_qrs \
            + P_VOTE_WEIGHT * (-corr_qrs if corr_qrs is not None else 0.0)
    elif verdict == 'normal':
        conf_qrs = rbar_qrs * (1.0 - anti_deg / 180.0)
        confidence = QRS_VOTE_WEIGHT * conf_qrs \
            + P_VOTE_WEIGHT * (corr_qrs if corr_qrs is not None else 0.0)
    else:
        # Uncertain: confidence reflects how far from the boundary we are
        sep = abs(anti_deg - ANTI_PHASE_MIN_DEG) / 90.0
        c = corr_qrs if corr_qrs is not None else 0.0
        confidence = QRS_VOTE_WEIGHT * min(sep, abs(c) * 2.0)
    if phase_p_deg is not None and rbar_p is not None and rbar_p >= P_USE_RBAR:
        anti_p = abs(phase_p_deg)
        agree = (anti_p >= 90.0) == (verdict == 'reversed') if verdict != 'uncertain' \
            else False
        if agree:
            confidence += P_VOTE_WEIGHT * 0.5 * rbar_p
    confidence = float(min(confidence, 0.99))

    result = TFReversalResult(
        verdict=verdict, confidence=confidence,
        phase_qrs_deg=phase_qrs_deg, rbar_qrs=rbar_qrs,
        corr_qrs=corr_qrs,
        phase_p_deg=phase_p_deg, rbar_p=rbar_p,
        n_beats=len(r_peaks), qrs_power_ratio=ratio,
    )
    return (result, extras) if return_extras else result


# ---------------------------------------------------------------------------
# Convenience wrapper: raw signals in, verdict out
# ---------------------------------------------------------------------------
def detect_record_reversal(signals: dict, fs: float) -> TFReversalResult:
    """Preprocess raw lead signals and run the reversal check.

    Parameters
    ----------
    signals : dict lead name -> raw 1-D array (as returned by parse_aecg).
    fs : sampling rate in Hz.
    """
    from ecg_waveform_extraction.src.preprocessing.filters import ECGPreprocessor

    prep = ECGPreprocessor(fs=fs)
    clean = {name: prep.preprocess(np.asarray(sig, dtype=np.float64))
             for name, sig in signals.items() if sig is not None}
    return detect_tf_reversal(
        clean.get('I'), clean.get('II'), fs, clean_iii=clean.get('III'),
    )


# ---------------------------------------------------------------------------
# Synthetic reversal for validation
# ---------------------------------------------------------------------------
def synthetic_swap(signals: dict) -> dict:
    """Return a copy of the record with RA-LA swapped.

    I' = -I, II' = III, III' = II; all other leads unchanged. This is the
    full anatomical reversal: applying the detection to this copy must
    yield 'reversed' if the method works. Note the II<->III half is what
    makes some records hard — the detector sees II' = III, whose morphology
    differs from II, so the phase relationship is no longer a pure pi shift.
    """
    out = dict(signals)
    if 'I' in signals:
        out['I'] = -np.asarray(signals['I'])
    if 'II' in signals and 'III' in signals:
        out['II'] = np.asarray(signals['III'])
        out['III'] = np.asarray(signals['II'])
    return out


def flip_lead_i(signals: dict) -> dict:
    """Return a copy with only lead I inverted (I' = -I, II untouched).

    The minimal signature of RA-LA reversal as seen by this detector
    (I-vs-II anti-phase), without the II<->III morphology confound.
    Used alongside `synthetic_swap` in --validate-synthetic to separate
    "phase machinery works" from "anatomy cooperates".
    """
    out = dict(signals)
    if 'I' in signals:
        out['I'] = -np.asarray(signals['I'])
    return out
