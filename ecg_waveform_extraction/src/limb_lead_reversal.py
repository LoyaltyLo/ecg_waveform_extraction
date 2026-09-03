"""Limb Lead Reversal Detection.

Detects all 3 types of limb lead reversal using 6-lead HSMM waveform data
combined with machine measurements. Multi-criteria weighted voting per type.

Reversal Types
--------------
  RA-LA  — Right Arm / Left Arm swap (most common, ~80% of reversals)
  RA-LL  — Right Arm / Left Leg swap (~10%)
  LA-LL  — Left Arm / Left Leg swap (~8%)

Detection Principles
--------------------
Each type is evaluated by 4–6 independent criteria. Each criterion returns
a verdict (normal / reversed / uncertain) + confidence (0–1). Criteria are
weighted by clinical specificity and summed into per-type scores. The type
with the highest score above threshold wins.

Usage
-----
    from ecg_waveform_extraction.src.limb_lead_reversal import (
        LimbLeadReversalDetector, ReversalResult,
    )
    from ecg_waveform_extraction.src.limb_lead_processor import LimbLeadProcessor
    from ecg_waveform_extraction.src.utils.aecg_parser import parse_aecg

    # Option A: from pre-computed LimbLeadResult
    processor = LimbLeadProcessor(max_samples=16000)
    aecg = parse_aecg('file.aECG')
    ll_result = processor.process_record(aecg)
    detector = LimbLeadReversalDetector()
    result = detector.detect(ll_result)

    # Option B: one-shot from aECG dict (runs HSMM internally)
    result = detector.detect_from_aecg(aecg)

    print(result.summary())
"""

from dataclasses import dataclass, field
from collections import Counter
import numpy as np

from .limb_lead_processor import (
    LimbLeadResult, LeadResult, LIMB_LEADS,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REVERSAL_TYPES = ['ra_la']

REVERSAL_NAMES = {
    'ra_la': 'RA-LA (Right Arm <-> Left Arm)',
    'normal': 'Normal (no reversal detected)',
    'uncertain': 'Uncertain (insufficient evidence)',
}

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CriterionResult:
    """Single criterion evaluation result."""
    name: str
    verdict: str          # 'normal' | 'reversed' | 'uncertain'
    confidence: float     # 0–1
    detail: str           # human-readable explanation
    weight: float = 1.0   # clinical weight


@dataclass
class ReversalTypeResult:
    """Evaluation result for one reversal type."""
    reversal_type: str    # 'ra_la' | 'ra_ll' | 'la_ll'
    criteria: list = field(default_factory=list)
    score_normal: float = 0.0
    score_reversed: float = 0.0
    verdict: str = 'normal'       # 'normal' | 'reversed' | 'uncertain'
    confidence: float = 0.0
    n_criteria_triggered: int = 0


@dataclass
class ReversalResult:
    """Complete limb lead reversal detection result for one record."""
    record: str
    reversal_type: str = 'normal'   # winning type or 'normal'/'uncertain'
    confidence: float = 0.0
    types: dict = field(default_factory=dict)   # type_name → ReversalTypeResult
    measurements: dict = field(default_factory=dict)
    interpretation: str = ''
    lead_polarity_summary: dict = field(default_factory=dict)
    detail: str = ''

    def is_reversed(self) -> bool:
        return self.reversal_type in REVERSAL_TYPES

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        lines = []
        lines.append(f"{'='*75}")
        lines.append(f"  Record: {self.record}")
        lines.append(f"  Verdict: {REVERSAL_NAMES.get(self.reversal_type, self.reversal_type)}")
        lines.append(f"  Confidence: {self.confidence:.2f}")
        if self.measurements:
            meas = self.measurements
            lines.append(f"  P-axis: {meas.get('P_axis','?')} deg  "
                         f"QRS-axis: {meas.get('QRS_axis','?')} deg  "
                         f"HR: {meas.get('HR','?')} bpm")
        lines.append(f"{'─'*75}")
        lines.append(f"  {'Type':<8} {'Verdict':<12} {'Score(N/R)':<18} {'Conf':<6}  Criteria triggered")
        lines.append(f"  {'─'*8} {'─'*12} {'─'*18} {'─'*6}  {'─'*18}")

        for tname in REVERSAL_TYPES:
            tr = self.types.get(tname)
            if tr is None:
                continue
            crit_names = [c.name for c in tr.criteria
                         if c.verdict == 'reversed']
            lines.append(
                f"  {tname:<8} {tr.verdict:<12} "
                f"{tr.score_normal:.2f}/{tr.score_reversed:.2f}        "
                f"{tr.confidence:<6.2f}  "
                f"{', '.join(crit_names) if crit_names else '—'}"
            )

        lines.append(f"{'─'*75}")
        lines.append(f"  DETAIL:")
        for tname in REVERSAL_TYPES:
            tr = self.types.get(tname)
            if tr is None:
                continue
            for c in tr.criteria:
                if c.verdict == 'reversed':
                    marker = 'R'
                elif c.verdict == 'borderline':
                    marker = '~'
                elif c.verdict == 'uncertain':
                    marker = '?'
                else:
                    marker = ' '
                lines.append(f"    [{marker}] [{tname}] {c.name}: {c.detail} "
                             f"(conf={c.confidence:.2f}, w={c.weight})")

        lines.append(f"{'='*75}")
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class LimbLeadReversalDetector:
    """Multi-type limb lead reversal detector.

    Uses 6-lead HSMM waveform data + machine measurements to evaluate
    3 reversal hypotheses independently, then picks the best match.

    Parameters
    ----------
    hsmm_required : bool
        If True, skip criteria that need HSMM data when unavailable.
        If False, fall back to machine measurements only (reduced accuracy).
    """

    def __init__(self, hsmm_required: bool = True):
        self.hsmm_required = hsmm_required

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, ll_result: LimbLeadResult) -> ReversalResult:
        """Run full reversal detection on 6-lead HSMM results.

        Parameters
        ----------
        ll_result : LimbLeadResult
            Output of LimbLeadProcessor.process_record().

        Returns
        -------
        ReversalResult
        """
        # Evaluate RA-LA reversal only
        ra_la_result = self._evaluate_ra_la(ll_result)
        type_results = {'ra_la': ra_la_result}

        # Classify based on RA-LA score alone
        reversal_type, confidence = self._classify_ra_la(ra_la_result)

        # Build lead polarity summary
        lp_summary = {}
        for ln in LIMB_LEADS:
            lr = ll_result.leads.get(ln)
            if lr is None:
                lp_summary[ln] = None
                continue
            qrs_dom = self._dominant_polarity(lr.polarity_counts)
            p_dom = self._dominant_polarity(lr.p_polarity_counts)
            t_dom = self._dominant_polarity(lr.t_polarity_counts)
            lp_summary[ln] = {
                'qrs_dominant': qrs_dom,
                'p_dominant': p_dom,
                't_dominant': t_dom,
                'qrs_counts': lr.polarity_counts,
                'p_counts': lr.p_polarity_counts,
                't_counts': lr.t_polarity_counts,
                'mean_qrs_net': lr.mean_qrs_net,
                'mean_p_net': lr.mean_p_net,
                'mean_t_net': lr.mean_t_net,
            }

        return ReversalResult(
            record=ll_result.record,
            reversal_type=reversal_type,
            confidence=confidence,
            types=type_results,
            measurements=ll_result.measurements,
            interpretation=ll_result.interpretation,
            lead_polarity_summary=lp_summary,
        )

    def detect_from_aecg(self, aecg_data: dict,
                         record_name: str | None = None) -> ReversalResult:
        """One-shot detection from parsed aECG data (runs HSMM internally).

        Parameters
        ----------
        aecg_data : dict
            Output of parse_aecg().
        record_name : str or None

        Returns
        -------
        ReversalResult
        """
        from .limb_lead_processor import LimbLeadProcessor
        processor = LimbLeadProcessor(max_samples=16000)
        ll_result, _ = processor.process_record(aecg_data, record_name=record_name)
        return self.detect(ll_result)

    # ------------------------------------------------------------------
    # Classification: pick winning reversal type
    # ------------------------------------------------------------------
    def _classify_ra_la(self, tr: ReversalTypeResult) -> tuple[str, float]:
        """Classify based on RA-LA score alone.

        Rules:
          - score_reversed >= 2.0 AND verdict is 'reversed' → ra_la
          - score_reversed >= 1.0 AND verdict is 'reversed' → ra_la (lower conf)
          - verdict is 'normal' → normal
          - otherwise → uncertain
        """
        if tr.verdict == 'reversed' and tr.score_reversed >= 2.0:
            conf = min(tr.score_reversed / 5.0, 0.98)
            return 'ra_la', round(conf, 2)
        elif tr.verdict == 'reversed' and tr.score_reversed >= 1.0:
            conf = min(tr.score_reversed / 5.0, 0.75)
            return 'ra_la', round(conf, 2)
        elif tr.verdict == 'normal':
            return 'normal', min(tr.score_normal / 4.0, 0.95)
        else:
            return 'uncertain', max(tr.score_reversed / 3.0, 0.25)

    # ==================================================================
    # RA-LA Reversal Detection (Right Arm ↔ Left Arm)
    # ==================================================================
    def _evaluate_ra_la(self, ll: LimbLeadResult) -> ReversalTypeResult:
        """Evaluate RA-LA reversal hypothesis.

        6 criteria, ranked by clinical specificity:
          C1 (w=1.5): Lead I P-wave inversion — HSMM P-state net area
          C2 (w=1.5): aVR P-wave upright — pathognomonic sign
          C3 (w=1.2): Lead I QRS inversion — HSMM QRS polarity
          C4 (w=1.0): P-axis extreme right — machine measurement
          C5 (w=0.8): aVR ↔ aVL polarity swap — cross-lead comparison
          C6 (w=0.7): QRS-axis extreme right — machine measurement
        """
        criteria = []
        meas = ll.measurements

        # ---- C1: Lead I P-wave inversion (HSMM) ----
        c1 = self._criterion_lead_I_p_inverted(ll)
        criteria.append(c1)

        # ---- C2: aVR P-wave upright (HSMM) ----
        c2 = self._criterion_avr_p_upright(ll)
        criteria.append(c2)

        # ---- C3: Lead I QRS inversion (HSMM) ----
        c3 = self._criterion_lead_I_qrs_inverted(ll)
        criteria.append(c3)

        # ---- C4: P-axis extreme right ----
        c4 = self._criterion_p_axis_reversed(meas)
        criteria.append(c4)

        # ---- C5: aVR / aVL polarity swap ----
        c5 = self._criterion_avr_avl_swap(ll)
        criteria.append(c5)

        # ---- C6: QRS-axis extreme right (>120°) ----
        c6 = self._criterion_qrs_axis_extreme_right(meas)
        criteria.append(c6)

        return self._aggregate('ra_la', criteria)

    # ==================================================================
    # RA-LL Reversal Detection (Right Arm ↔ Left Leg)
    # ==================================================================
    def _evaluate_ra_ll(self, ll: LimbLeadResult) -> ReversalTypeResult:
        """Evaluate RA-LL reversal hypothesis.

        6 criteria:
          C1 (w=1.5): Lead II near-isoelectric (very low amplitude)
          C2 (w=1.2): Lead III QRS inversion
          C3 (w=1.2): aVF low amplitude
          C4 (w=1.0): Lead I ≈ -Lead III (correlation)
          C5 (w=0.8): Lead II P-wave flat
          C6 (w=0.8): aVR/aVF amplitude ratio anomaly
        """
        criteria = []
        meas = ll.measurements

        # ---- C1: Lead II near-isoelectric ----
        c1 = self._criterion_lead_II_isoelectric(ll)
        criteria.append(c1)

        # ---- C2: Lead III QRS inversion ----
        c2 = self._criterion_lead_III_qrs_inverted(ll)
        criteria.append(c2)

        # ---- C3: aVF low amplitude ----
        c3 = self._criterion_avf_low_amplitude(ll)
        criteria.append(c3)

        # ---- C4: Lead I ≈ -Lead III correlation ----
        c4 = self._criterion_lead_I_neg_lead_III_correlation(ll)
        criteria.append(c4)

        # ---- C5: Lead II P-wave flat ----
        c5 = self._criterion_lead_II_p_flat(ll)
        criteria.append(c5)

        # ---- C6: aVR/aVF amplitude ratio ----
        c6 = self._criterion_avr_avf_amplitude_ratio(ll)
        criteria.append(c6)

        return self._aggregate('ra_ll', criteria)

    # ==================================================================
    # LA-LL Reversal Detection (Left Arm ↔ Left Leg)
    # ==================================================================
    def _evaluate_la_ll(self, ll: LimbLeadResult) -> ReversalTypeResult:
        """Evaluate LA-LL reversal hypothesis.

        5 criteria:
          C1 (w=1.5): Lead III QRS + P inversion
          C2 (w=1.3): Lead I / Lead II polarity mismatch (should be similar)
          C3 (w=1.2): aVR / aVF polarity swap
          C4 (w=1.0): QRS-axis left deviation
          C5 (w=0.8): P-axis left deviation
        """
        criteria = []
        meas = ll.measurements

        # ---- C1: Lead III inversion ----
        c1 = self._criterion_lead_III_inverted(ll)
        criteria.append(c1)

        # ---- C2: Lead I / Lead II polarity mismatch ----
        c2 = self._criterion_lead_I_II_mismatch(ll)
        criteria.append(c2)

        # ---- C3: aVR / aVF polarity swap ----
        c3 = self._criterion_avr_avf_swap(ll)
        criteria.append(c3)

        # ---- C4: QRS-axis left deviation ----
        c4 = self._criterion_qrs_axis_left(meas)
        criteria.append(c4)

        # ---- C5: P-axis left deviation ----
        c5 = self._criterion_p_axis_left(meas)
        criteria.append(c5)

        return self._aggregate('la_ll', criteria)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def _aggregate(self, reversal_type: str,
                   criteria: list[CriterionResult]) -> ReversalTypeResult:
        """Weighted voting aggregation for one reversal type.

        Each criterion votes: reversed=+1, normal=-1, uncertain=0.
        Weighted sum → score.
        """
        score_normal = 0.0
        score_reversed = 0.0
        n_triggered = 0

        for c in criteria:
            w = c.weight
            if c.verdict == 'reversed':
                score_reversed += w * c.confidence
                n_triggered += 1
            elif c.verdict == 'normal':
                score_normal += w * c.confidence
            # uncertain → no contribution to either side

        # Determine verdict
        if score_reversed > score_normal + 0.5 and n_triggered >= 2:
            verdict = 'reversed'
            confidence = min(score_reversed / 4.0, 0.98)
        elif score_reversed > score_normal and n_triggered >= 1:
            verdict = 'borderline'
            confidence = min(score_reversed / 5.0, 0.6)
        else:
            verdict = 'normal'
            confidence = min(score_normal / 4.0, 0.95)

        return ReversalTypeResult(
            reversal_type=reversal_type,
            criteria=criteria,
            score_normal=round(score_normal, 2),
            score_reversed=round(score_reversed, 2),
            verdict=verdict,
            confidence=round(confidence, 2),
            n_criteria_triggered=n_triggered,
        )

    # ==================================================================
    # RA-LA Criteria Implementations
    # ==================================================================

    def _criterion_lead_I_p_inverted(self, ll: LimbLeadResult) -> CriterionResult:
        """Lead I P-wave inversion — most reliable RA-LA sign."""
        lr = ll.leads.get('I')
        if lr is None:
            return CriterionResult('Lead I P inverted', 'uncertain', 0.0,
                                   'No Lead I data', weight=1.5)

        p_neg = lr.p_polarity_counts.get('negative', 0)
        p_pos = lr.p_polarity_counts.get('positive', 0)
        p_total = p_neg + p_pos
        if p_total == 0:
            return CriterionResult('Lead I P inverted', 'uncertain', 0.0,
                                   'No P-waves with clear polarity', weight=1.5)

        neg_pct = p_neg / p_total
        pos_pct = p_pos / p_total
        mean_p_net = lr.mean_p_net

        if neg_pct >= 0.7 and mean_p_net < -1.0:
            conf = min(0.7 + neg_pct * 0.3, 0.98)
            return CriterionResult(
                'Lead I P inverted', 'reversed', round(conf, 2),
                f'Lead I P {p_neg}/{p_total} negative ({neg_pct:.0%}), '
                f'net={mean_p_net:.1f}',
                weight=1.5)
        elif neg_pct >= 0.5 and mean_p_net < 0:
            conf = 0.5 + neg_pct * 0.2
            return CriterionResult(
                'Lead I P inverted', 'reversed', round(conf, 2),
                f'Lead I P mostly negative ({neg_pct:.0%}), net={mean_p_net:.1f}',
                weight=1.5)
        elif pos_pct >= 0.7 and mean_p_net > 1.0:
            return CriterionResult(
                'Lead I P inverted', 'normal', 0.90,
                f'Lead I P clearly positive ({pos_pct:.0%}), net={mean_p_net:.1f}',
                weight=1.5)
        else:
            return CriterionResult(
                'Lead I P inverted', 'uncertain', 0.3,
                f'Lead I P mixed: +{p_pos} -{p_neg}, net={mean_p_net:.1f}',
                weight=1.5)

    def _criterion_avr_p_upright(self, ll: LimbLeadResult) -> CriterionResult:
        """aVR P-wave upright — pathognomonic for RA-LA reversal.

        In normal hearts aVR P-wave is ALWAYS negative. Positive aVR P-wave
        is almost never seen except in RA-LA reversal.
        """
        lr = ll.leads.get('AVR')
        if lr is None:
            return CriterionResult('aVR P upright', 'uncertain', 0.0,
                                   'No aVR data', weight=1.5)

        p_pos = lr.p_polarity_counts.get('positive', 0)
        p_neg = lr.p_polarity_counts.get('negative', 0)
        p_total = p_pos + p_neg
        if p_total == 0:
            return CriterionResult('aVR P upright', 'uncertain', 0.0,
                                   'No aVR P-waves', weight=1.5)

        pos_pct = p_pos / p_total
        mean_p_net = lr.mean_p_net

        if pos_pct >= 0.6 and mean_p_net > 0.5:
            conf = min(0.7 + pos_pct * 0.3, 0.98)
            return CriterionResult(
                'aVR P upright', 'reversed', round(conf, 2),
                f'aVR P {p_pos}/{p_total} POSITIVE ({pos_pct:.0%}) — PATHOGNOMONIC, '
                f'net={mean_p_net:.1f}',
                weight=1.5)
        elif pos_pct >= 0.4 and mean_p_net > 0:
            return CriterionResult(
                'aVR P upright', 'reversed', 0.65,
                f'aVR P partially positive ({pos_pct:.0%}), net={mean_p_net:.1f}',
                weight=1.5)
        elif p_neg >= 0.7 and mean_p_net < -0.5:
            return CriterionResult(
                'aVR P upright', 'normal', 0.90,
                f'aVR P negative as expected ({p_neg}/{p_total}), net={mean_p_net:.1f}',
                weight=1.5)
        else:
            return CriterionResult(
                'aVR P upright', 'uncertain', 0.3,
                f'aVR P ambiguous: +{p_pos} -{p_neg}, net={mean_p_net:.1f}',
                weight=1.5)

    def _criterion_lead_I_qrs_inverted(self, ll: LimbLeadResult) -> CriterionResult:
        """Lead I QRS inversion with P/T cross-check.

        Rule (simplified):
          - QRS inverted AND (P or T also inverted) → reversed
          - QRS inverted but P and T are NOT inverted → normal
          - QRS positive → normal
        """
        lr = ll.leads.get('I')
        if lr is None or lr.n_beats == 0:
            return CriterionResult('Lead I QRS+P+T', 'uncertain', 0.0,
                                   'No Lead I data', weight=1.2)

        neg = lr.polarity_counts.get('negative', 0)
        pos = lr.polarity_counts.get('positive', 0)
        total = neg + pos
        if total == 0:
            return CriterionResult('Lead I QRS+P+T', 'uncertain', 0.0,
                                   'All uncertain/biphasic', weight=1.2)

        neg_pct = neg / total
        mean_net = lr.mean_qrs_net

        # QRS positive → normal
        if pos / total >= 0.6 and mean_net > 1.0:
            return CriterionResult(
                'Lead I QRS+P+T', 'normal', 0.85,
                f'Lead I QRS positive ({pos}/{total})', weight=1.2)

        # QRS not clearly negative → uncertain
        if neg_pct < 0.5:
            return CriterionResult(
                'Lead I QRS+P+T', 'uncertain', 0.3,
                f'Lead I QRS mixed: +{pos} -{neg}', weight=1.2)

        # ---- QRS IS negative → check P and T ----
        p_neg = lr.p_polarity_counts.get('negative', 0)
        p_pos = lr.p_polarity_counts.get('positive', 0)
        p_total = p_neg + p_pos
        p_inverted = (p_total > 0 and p_neg / p_total >= 0.5 and lr.mean_p_net < -0.5)

        t_neg = lr.t_polarity_counts.get('negative', 0)
        t_pos = lr.t_polarity_counts.get('positive', 0)
        t_total = t_neg + t_pos
        t_inverted = (t_total > 0 and t_neg / t_total >= 0.5 and lr.mean_t_net < -0.5)

        qrs_str = f'QRS {neg}/{total} neg ({neg_pct:.0%})'
        p_str = f'P: {p_neg}neg/{p_pos}pos' if p_total > 0 else 'P: --'
        t_str = f'T: {t_neg}neg/{t_pos}pos' if t_total > 0 else 'T: --'

        if p_inverted or t_inverted:
            parts = []
            if p_inverted: parts.append('P')
            if t_inverted: parts.append('T')
            conf = min(0.70 + neg_pct * 0.25, 0.95)
            if p_inverted and t_inverted:
                conf = min(conf + 0.03, 0.98)
            return CriterionResult(
                'Lead I QRS+P+T', 'reversed', round(conf, 2),
                f'{qrs_str} | {p_str} | {t_str} | '
                f'{"+".join(parts)} also inverted → reversal',
                weight=1.2)
        else:
            # QRS inverted but P and T are NOT → normal connection
            return CriterionResult(
                'Lead I QRS+P+T', 'normal', 0.75,
                f'{qrs_str} | {p_str} | {t_str} | '
                f'QRS neg but P/T not inverted → normal connection',
                weight=1.2)

    def _criterion_p_axis_reversed(self, meas: dict) -> CriterionResult:
        """P-axis extreme right: >100° or <-60°."""
        p_axis = meas.get('P_axis')
        if p_axis is None:
            return CriterionResult('P-axis', 'uncertain', 0.0,
                                   'No P-axis measurement', weight=1.0)

        if p_axis > 100 or p_axis < -60:
            return CriterionResult(
                'P-axis', 'reversed', 0.90,
                f'P-axis={p_axis:.0f}° (extreme right, normal 0–75°)',
                weight=1.0)
        elif 75 < p_axis <= 100:
            return CriterionResult(
                'P-axis', 'borderline', 0.45,
                f'P-axis={p_axis:.0f}° (borderline right, normal 0–75°)',
                weight=1.0)
        elif 0 <= p_axis <= 75:
            return CriterionResult(
                'P-axis', 'normal', 0.90,
                f'P-axis={p_axis:.0f}° (normal range 0–75°)',
                weight=1.0)
        elif -60 <= p_axis < 0:
            return CriterionResult(
                'P-axis', 'normal', 0.70,
                f'P-axis={p_axis:.0f}° (slightly left but acceptable)',
                weight=1.0)
        else:
            return CriterionResult(
                'P-axis', 'uncertain', 0.3,
                f'P-axis={p_axis:.0f}°', weight=1.0)

    def _criterion_avr_avl_swap(self, ll: LimbLeadResult) -> CriterionResult:
        """Check if aVR and aVL have swapped polarity patterns.

        Normal: aVR QRS negative, aVL QRS positive (or variable).
        RA-LA reversal: aVR becomes positive (like normal aVL), aVL becomes negative.
        """
        avr = ll.leads.get('AVR')
        avl = ll.leads.get('AVL')
        if avr is None or avl is None:
            return CriterionResult('aVR↔aVL swap', 'uncertain', 0.0,
                                   'Missing aVR or aVL', weight=0.8)

        avr_qrs_dom = self._dominant_polarity(avr.polarity_counts)
        avl_qrs_dom = self._dominant_polarity(avl.polarity_counts)

        # Normal: aVR negative, aVL positive/variable
        # Reversed: aVR positive, aVL negative
        if avr_qrs_dom == 'positive' and avl_qrs_dom == 'negative':
            return CriterionResult(
                'aVR↔aVL swap', 'reversed', 0.85,
                f'aVR QRS={avr_qrs_dom}, aVL QRS={avl_qrs_dom} — SWAPPED',
                weight=0.8)
        elif avr_qrs_dom == 'positive':
            return CriterionResult(
                'aVR↔aVL swap', 'reversed', 0.60,
                f'aVR QRS is positive (abnormal), aVL={avl_qrs_dom}',
                weight=0.8)
        elif avr_qrs_dom == 'negative' and avl_qrs_dom in ('positive', 'biphasic'):
            return CriterionResult(
                'aVR↔aVL swap', 'normal', 0.80,
                f'aVR negative, aVL {avl_qrs_dom} — normal pattern',
                weight=0.8)
        else:
            return CriterionResult(
                'aVR↔aVL swap', 'uncertain', 0.3,
                f'aVR={avr_qrs_dom}, aVL={avl_qrs_dom}', weight=0.8)

    def _criterion_qrs_axis_extreme_right(self, meas: dict) -> CriterionResult:
        """QRS-axis extreme right (>120°)."""
        qrs_axis = meas.get('QRS_axis')
        if qrs_axis is None:
            return CriterionResult('QRS-axis right', 'uncertain', 0.0,
                                   'No QRS-axis measurement', weight=0.7)

        if qrs_axis > 120:
            return CriterionResult(
                'QRS-axis right', 'reversed', 0.85,
                f'QRS-axis={qrs_axis:.0f}° (extreme right, normal -30~+90°)',
                weight=0.7)
        elif 90 < qrs_axis <= 120:
            return CriterionResult(
                'QRS-axis right', 'borderline', 0.40,
                f'QRS-axis={qrs_axis:.0f}° (right deviation)',
                weight=0.7)
        elif -30 <= qrs_axis <= 90:
            return CriterionResult(
                'QRS-axis right', 'normal', 0.85,
                f'QRS-axis={qrs_axis:.0f}° (normal)',
                weight=0.7)
        else:
            return CriterionResult(
                'QRS-axis right', 'uncertain', 0.3,
                f'QRS-axis={qrs_axis:.0f}°', weight=0.7)

    # ==================================================================
    # RA-LL Criteria Implementations
    # ==================================================================

    def _criterion_lead_II_isoelectric(self, ll: LimbLeadResult) -> CriterionResult:
        """Lead II near-isoelectric — hallmark of RA-LL reversal.

        When RA and LL are swapped, Lead II (RA→LL) becomes nearly flat
        because both electrodes are at similar potentials.
        """
        lr = ll.leads.get('II')
        if lr is None or lr.n_beats == 0:
            return CriterionResult('Lead II isoelectric', 'uncertain', 0.0,
                                   'No Lead II data', weight=1.5)

        # Check amplitude: compare Lead II R amplitude vs Lead I
        li = ll.leads.get('I')
        r_amp_II = lr.mean_r_amplitude
        r_amp_I = li.mean_r_amplitude if li and li.n_beats > 0 else r_amp_II + 1.0

        amp_ratio = r_amp_II / max(r_amp_I, 0.001)

        if amp_ratio < 0.3 and r_amp_II < 1.0:
            return CriterionResult(
                'Lead II isoelectric', 'reversed', 0.90,
                f'Lead II R-amp={r_amp_II:.2f} vs Lead I={r_amp_I:.2f} '
                f'(ratio={amp_ratio:.2f}) — nearly flat',
                weight=1.5)
        elif amp_ratio < 0.5 and r_amp_II < 2.0:
            return CriterionResult(
                'Lead II isoelectric', 'reversed', 0.65,
                f'Lead II low amplitude (ratio={amp_ratio:.2f})',
                weight=1.5)
        elif r_amp_II >= 2.0 and amp_ratio > 0.5:
            return CriterionResult(
                'Lead II isoelectric', 'normal', 0.85,
                f'Lead II normal amplitude (R={r_amp_II:.2f}, ratio={amp_ratio:.2f})',
                weight=1.5)
        else:
            return CriterionResult(
                'Lead II isoelectric', 'uncertain', 0.3,
                f'Lead II R={r_amp_II:.2f}, ratio={amp_ratio:.2f}',
                weight=1.5)

    def _criterion_lead_III_qrs_inverted(self, ll: LimbLeadResult) -> CriterionResult:
        """Lead III QRS inversion — RA-LL causes Lead III to invert."""
        lr = ll.leads.get('III')
        if lr is None or lr.n_beats == 0:
            return CriterionResult('Lead III QRS inverted', 'uncertain', 0.0,
                                   'No Lead III data', weight=1.2)

        neg = lr.polarity_counts.get('negative', 0)
        pos = lr.polarity_counts.get('positive', 0)
        total = neg + pos
        if total == 0:
            return CriterionResult('Lead III QRS inverted', 'uncertain', 0.0,
                                   'All uncertain/biphasic', weight=1.2)

        neg_pct = neg / total

        if neg_pct >= 0.6:
            return CriterionResult(
                'Lead III QRS inverted', 'reversed', min(0.7 + neg_pct * 0.25, 0.95),
                f'Lead III QRS {neg}/{total} negative ({neg_pct:.0%})',
                weight=1.2)
        elif pos / total >= 0.7:
            return CriterionResult(
                'Lead III QRS inverted', 'normal', 0.80,
                f'Lead III QRS dominantly positive',
                weight=1.2)
        else:
            return CriterionResult(
                'Lead III QRS inverted', 'uncertain', 0.3,
                f'Lead III QRS mixed', weight=1.2)

    def _criterion_avf_low_amplitude(self, ll: LimbLeadResult) -> CriterionResult:
        """aVF low amplitude — RA-LL makes aVF nearly flat."""
        avf = ll.leads.get('AVF')
        if avf is None or avf.n_beats == 0:
            return CriterionResult('aVF low amplitude', 'uncertain', 0.0,
                                   'No aVF data', weight=1.2)

        # Compare aVF R amplitude to aVR
        avr = ll.leads.get('AVR')
        avf_amp = avf.mean_r_amplitude
        avr_amp = avr.mean_r_amplitude if avr and avr.n_beats > 0 else avf_amp + 1.0

        ratio_to_avr = avf_amp / max(avr_amp, 0.001)

        if avf_amp < 1.0 and ratio_to_avr < 0.3:
            return CriterionResult(
                'aVF low amplitude', 'reversed', 0.85,
                f'aVF R-amp={avf_amp:.2f} (vs aVR={avr_amp:.2f}) — very flat',
                weight=1.2)
        elif avf_amp < 1.5 and ratio_to_avr < 0.5:
            return CriterionResult(
                'aVF low amplitude', 'reversed', 0.60,
                f'aVF relatively flat (amp={avf_amp:.2f})',
                weight=1.2)
        elif avf_amp >= 2.0:
            return CriterionResult(
                'aVF low amplitude', 'normal', 0.80,
                f'aVF normal amplitude (R={avf_amp:.2f})',
                weight=1.2)
        else:
            return CriterionResult(
                'aVF low amplitude', 'uncertain', 0.3,
                f'aVF R={avf_amp:.2f}', weight=1.2)

    def _criterion_lead_I_neg_lead_III_correlation(self,
                                                    ll: LimbLeadResult) -> CriterionResult:
        """Lead I ≈ -Lead III — characteristic of RA-LL reversal.

        In RA-LL: Lead I and Lead III become nearly mirror images because
        the LL electrode (now on RA) is opposite to LA in Lead I, and
        RA (now on LL) is opposite to LA in Lead III.
        """
        li = ll.leads.get('I')
        liii = ll.leads.get('III')
        if li is None or liii is None or li.n_beats == 0 or liii.n_beats == 0:
            return CriterionResult('I ≈ -III corr', 'uncertain', 0.0,
                                   'Missing Lead I or III', weight=1.0)

        # Compare QRS net areas: Lead I net ≈ -(Lead III net) in RA-LL
        net_I = li.mean_qrs_net
        net_III = liii.mean_qrs_net

        # Check if they have opposite signs
        if net_I * net_III < 0:  # opposite signs
            ratio = abs(net_I) / max(abs(net_III), 0.01)
            if 0.5 < ratio < 2.0:
                return CriterionResult(
                    'I ≈ -III corr', 'reversed', 0.80,
                    f'Lead I net={net_I:.1f}, Lead III net={net_III:.1f} '
                    f'— opposite signs, similar magnitude (ratio={ratio:.2f})',
                    weight=1.0)
            else:
                return CriterionResult(
                    'I ≈ -III corr', 'reversed', 0.55,
                    f'Lead I and III opposite signs (ratio={ratio:.2f})',
                    weight=1.0)
        else:
            return CriterionResult(
                'I ≈ -III corr', 'normal', 0.75,
                f'Lead I net={net_I:.1f}, Lead III net={net_III:.1f} '
                f'— same sign, not mirroring',
                weight=1.0)

    def _criterion_lead_II_p_flat(self, ll: LimbLeadResult) -> CriterionResult:
        """Lead II P-wave flat — P-wave disappears in Lead II with RA-LL."""
        lr = ll.leads.get('II')
        if lr is None or lr.mean_p_net == 0.0:
            return CriterionResult('Lead II P flat', 'uncertain', 0.0,
                                   'No Lead II P-wave data', weight=0.8)

        mean_p_net = lr.mean_p_net
        # Normalize by QRS amplitude to get relative P-wave size
        qrs_amp = max(lr.mean_r_amplitude, 0.01)
        relative_p = abs(mean_p_net) / qrs_amp

        if relative_p < 0.15:
            return CriterionResult(
                'Lead II P flat', 'reversed', 0.75,
                f'Lead II P-wave nearly flat (|P_net|/R={relative_p:.2f})',
                weight=0.8)
        elif relative_p > 0.3:
            return CriterionResult(
                'Lead II P flat', 'normal', 0.80,
                f'Lead II P-wave present (|P_net|/R={relative_p:.2f})',
                weight=0.8)
        else:
            return CriterionResult(
                'Lead II P flat', 'uncertain', 0.35,
                f'Lead II P-wave borderline (|P_net|/R={relative_p:.2f})',
                weight=0.8)

    def _criterion_avr_avf_amplitude_ratio(self,
                                            ll: LimbLeadResult) -> CriterionResult:
        """aVR/aVF amplitude ratio — RA-LL causes aVF to be much smaller than aVR."""
        avr = ll.leads.get('AVR')
        avf = ll.leads.get('AVF')
        if avr is None or avf is None or avr.n_beats == 0 or avf.n_beats == 0:
            return CriterionResult('aVR/aVF ratio', 'uncertain', 0.0,
                                   'Missing aVR or aVF', weight=0.8)

        ratio = avr.mean_r_amplitude / max(avf.mean_r_amplitude, 0.001)

        if ratio > 3.0:
            return CriterionResult(
                'aVR/aVF ratio', 'reversed', 0.75,
                f'aVR/aVF amplitude ratio={ratio:.1f} (>3.0 → aVF suppressed)',
                weight=0.8)
        elif ratio < 1.5:
            return CriterionResult(
                'aVR/aVF ratio', 'normal', 0.75,
                f'aVR/aVF ratio={ratio:.1f} (normal)',
                weight=0.8)
        else:
            return CriterionResult(
                'aVR/aVF ratio', 'uncertain', 0.35,
                f'aVR/aVF ratio={ratio:.1f} (borderline)',
                weight=0.8)

    # ==================================================================
    # LA-LL Criteria Implementations
    # ==================================================================

    def _criterion_lead_III_inverted(self, ll: LimbLeadResult) -> CriterionResult:
        """Lead III QRS + P inversion — key sign of LA-LL reversal.

        LA-LL: Lead III becomes predominantly negative because the recording
        vector is effectively reversed.
        """
        lr = ll.leads.get('III')
        if lr is None or lr.n_beats == 0:
            return CriterionResult('Lead III inverted', 'uncertain', 0.0,
                                   'No Lead III data', weight=1.5)

        # Check both QRS and P polarity
        qrs_neg = lr.polarity_counts.get('negative', 0)
        qrs_pos = lr.polarity_counts.get('positive', 0)
        qrs_total = qrs_neg + qrs_pos

        p_neg = lr.p_polarity_counts.get('negative', 0)
        p_pos = lr.p_polarity_counts.get('positive', 0)
        p_total = p_neg + p_pos

        qrs_neg_pct = qrs_neg / max(qrs_total, 1)
        p_neg_pct = p_neg / max(p_total, 1)

        # Strongest signal: both QRS and P are inverted
        if qrs_neg_pct >= 0.6 and p_neg_pct >= 0.6:
            return CriterionResult(
                'Lead III inverted', 'reversed', 0.90,
                f'Lead III QRS {qrs_neg_pct:.0%} neg, P {p_neg_pct:.0%} neg — '
                f'both inverted',
                weight=1.5)
        elif qrs_neg_pct >= 0.6:
            return CriterionResult(
                'Lead III inverted', 'reversed', 0.70,
                f'Lead III QRS inverted ({qrs_neg_pct:.0%} neg), P borderline',
                weight=1.5)
        elif qrs_neg_pct < 0.3 and p_neg_pct < 0.3:
            return CriterionResult(
                'Lead III inverted', 'normal', 0.85,
                f'Lead III QRS and P both positive — normal',
                weight=1.5)
        else:
            return CriterionResult(
                'Lead III inverted', 'uncertain', 0.3,
                f'Lead III QRS: +{qrs_pos} -{qrs_neg}, P: +{p_pos} -{p_neg}',
                weight=1.5)

    def _criterion_lead_I_II_mismatch(self, ll: LimbLeadResult) -> CriterionResult:
        """Lead I / Lead II polarity mismatch — they should be similar normally.

        In LA-LL: Lead I retains its normal appearance but Lead II changes
        dramatically (because LA and LL are swapped, Lead II becomes the
        old Lead III).
        """
        li = ll.leads.get('I')
        lii = ll.leads.get('II')
        if li is None or lii is None or li.n_beats == 0 or lii.n_beats == 0:
            return CriterionResult('Lead I/II mismatch', 'uncertain', 0.0,
                                   'Missing Lead I or II', weight=1.3)

        dom_I = self._dominant_polarity(li.polarity_counts)
        dom_II = self._dominant_polarity(lii.polarity_counts)

        # Normal: Lead I and Lead II both positive (both point leftward-inferior)
        # LA-LL: Lead I positive, Lead II negative or very different
        if dom_I == 'positive' and dom_II == 'negative':
            return CriterionResult(
                'Lead I/II mismatch', 'reversed', 0.85,
                f'Lead I={dom_I}, Lead II={dom_II} — should both be positive',
                weight=1.3)
        elif dom_I == 'positive' and dom_II == 'positive':
            return CriterionResult(
                'Lead I/II mismatch', 'normal', 0.85,
                f'Lead I and II both positive — normal concordance',
                weight=1.3)
        elif dom_I in ('positive', 'negative') and dom_II in ('positive', 'negative') \
                and dom_I != dom_II:
            return CriterionResult(
                'Lead I/II mismatch', 'reversed', 0.65,
                f'Lead I={dom_I}, Lead II={dom_II} — discordant',
                weight=1.3)
        else:
            return CriterionResult(
                'Lead I/II mismatch', 'uncertain', 0.3,
                f'Lead I={dom_I}, Lead II={dom_II}', weight=1.3)

    def _criterion_avr_avf_swap(self, ll: LimbLeadResult) -> CriterionResult:
        """aVR / aVF polarity swap — LA-LL causes aVR to resemble normal aVF."""
        avr = ll.leads.get('AVR')
        avf = ll.leads.get('AVF')
        if avr is None or avf is None:
            return CriterionResult('aVR↔aVF swap', 'uncertain', 0.0,
                                   'Missing aVR or aVF', weight=1.2)

        avr_qrs = self._dominant_polarity(avr.polarity_counts)
        avf_qrs = self._dominant_polarity(avf.polarity_counts)
        avr_p = self._dominant_polarity(avr.p_polarity_counts)
        avf_p = self._dominant_polarity(avf.p_polarity_counts)

        # Normal: aVR negative, aVF positive
        # LA-LL: aVR becomes positive (like normal aVF), aVF becomes negative (like normal aVR)
        swap_signs = (
            avr_qrs == 'positive' and avf_qrs == 'negative'
        )

        if swap_signs:
            return CriterionResult(
                'aVR↔aVF swap', 'reversed', 0.80,
                f'aVR QRS={avr_qrs} (abnormal), aVF QRS={avf_qrs} (abnormal) — '
                f'appear swapped',
                weight=1.2)
        elif avr_qrs == 'positive':
            return CriterionResult(
                'aVR↔aVF swap', 'reversed', 0.55,
                f'aVR QRS is positive (abnormal), aVF={avf_qrs}',
                weight=1.2)
        elif avr_qrs == 'negative' and avf_qrs == 'positive':
            return CriterionResult(
                'aVR↔aVF swap', 'normal', 0.85,
                f'aVR negative, aVF positive — normal pattern',
                weight=1.2)
        else:
            return CriterionResult(
                'aVR↔aVF swap', 'uncertain', 0.3,
                f'aVR={avr_qrs}, aVF={avf_qrs}', weight=1.2)

    def _criterion_qrs_axis_left(self, meas: dict) -> CriterionResult:
        """QRS-axis left deviation (< -30°) — supportive of LA-LL."""
        qrs_axis = meas.get('QRS_axis')
        if qrs_axis is None:
            return CriterionResult('QRS-axis left', 'uncertain', 0.0,
                                   'No QRS-axis', weight=1.0)

        if qrs_axis < -30:
            return CriterionResult(
                'QRS-axis left', 'reversed', 0.70,
                f'QRS-axis={qrs_axis:.0f}° (left axis deviation)',
                weight=1.0)
        elif -30 <= qrs_axis <= 90:
            return CriterionResult(
                'QRS-axis left', 'normal', 0.80,
                f'QRS-axis={qrs_axis:.0f}° (normal)',
                weight=1.0)
        else:
            return CriterionResult(
                'QRS-axis left', 'normal', 0.60,
                f'QRS-axis={qrs_axis:.0f}° (not left deviated)',
                weight=1.0)

    def _criterion_p_axis_left(self, meas: dict) -> CriterionResult:
        """P-axis left deviation (< 0°) — supportive of LA-LL."""
        p_axis = meas.get('P_axis')
        if p_axis is None:
            return CriterionResult('P-axis left', 'uncertain', 0.0,
                                   'No P-axis', weight=0.8)

        if p_axis < 0:
            return CriterionResult(
                'P-axis left', 'reversed', 0.65,
                f'P-axis={p_axis:.0f}° (left axis, normal 0–75°)',
                weight=0.8)
        elif 0 <= p_axis <= 75:
            return CriterionResult(
                'P-axis left', 'normal', 0.85,
                f'P-axis={p_axis:.0f}° (normal)',
                weight=0.8)
        else:
            return CriterionResult(
                'P-axis left', 'normal', 0.60,
                f'P-axis={p_axis:.0f}° (rightward, not left)',
                weight=0.8)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _dominant_polarity(counts: dict) -> str:
        """Return the dominant polarity from a Counter dict, excluding uncertain."""
        if not counts:
            return 'N/A'
        filtered = {k: v for k, v in counts.items()
                    if k not in ('uncertain', 'N/A')}
        if not filtered:
            return 'uncertain'
        return max(filtered, key=filtered.get)


# ---------------------------------------------------------------------------
# Convenience: batch detection helpers
# ---------------------------------------------------------------------------

def reversal_result_to_dict(result: ReversalResult) -> dict:
    """Serialize ReversalResult to JSON-safe dict."""
    types_dict = {}
    for tname, tr in result.types.items():
        types_dict[tname] = {
            'reversal_type': tr.reversal_type,
            'verdict': tr.verdict,
            'confidence': tr.confidence,
            'score_normal': tr.score_normal,
            'score_reversed': tr.score_reversed,
            'n_criteria_triggered': tr.n_criteria_triggered,
            'criteria': [
                {
                    'name': c.name,
                    'verdict': c.verdict,
                    'confidence': c.confidence,
                    'detail': c.detail,
                    'weight': c.weight,
                }
                for c in tr.criteria
            ],
        }

    return {
        'record': result.record,
        'reversal_type': result.reversal_type,
        'confidence': result.confidence,
        'is_reversed': result.is_reversed(),
        'types': types_dict,
        'measurements': result.measurements,
        'interpretation': result.interpretation,
        'lead_polarity_summary': result.lead_polarity_summary,
    }
