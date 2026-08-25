# P-Wave Detection Optimization Plan

## Current State (2026-07-13)

P-wave extraction uses a 3-state focused HSMM (ISO_before → P → PR_after) on a
window around the Stage 1 P-wave region. While effective (96.5%+ sensitivity on
MIT-BIH), there are several concrete areas for improvement.

## Priority 1: Boundary Refinement (High Impact)

### 1.1 Fix `_init_gmms_from_window` to use Stage 1 boundaries
**Problem**: Current init splits window into equal thirds, assuming P is in the
middle third. When P is shifted (due to tachycardia, ectopy, or PR segment
variation), the GMM initializes on wrong regions.
**Fix**: Use Stage 1 beat.p_onset/p_offset relative positions to guide the
three-way split, keeping ISO_before GMM in the pre-P region, P GMM inside
the Stage 1 P boundaries, and PR_after GMM post-P.

### 1.2 Slope zero-crossing boundary refinement
**Problem**: HSMM state boundaries are coarse (limited by duration grid
resolution). The true electrophysiological P-wave onset is where the slope
first deviates from baseline.
**Approach**: Starting from HSMM-detected boundaries, walk outward until the
first derivative (d1) crosses the ISO baseline range (mean ± 2σ), or until
curvature sign changes. This gains ~5-10ms precision.

### 1.3 Smaller D_max grid for Stage 2
**Problem**: The focused HSMM uses a duration grid with step=1 sample, which
limits precision to ~4ms at 250Hz.
**Approach**: Already at step=1 — no change needed. But D_max can be narrowed
to μ±3σ instead of μ±4σ for faster decoding without precision loss.

## Priority 2: Confidence Metric Rework (Medium Impact)

### 2.1 Replace Viterbi-LL confidence with multi-dimensional score
**Problem**: Current `confidence = exp(ll/T)` always ≈1.0, providing no
discrimination between good and poor P-wave detections.
**Replace with**:
- **SNR**: P-wave peak amplitude / ISO baseline std-dev
- **Symmetry**: rising/falling slope ratio (0-1, 1=perfect symmetry)
- **Consistency**: correlation with adjacent beats' P-wave morphology
- **Duration validity**: how close P-width is to HR-expected value

### 2.2 Add P-wave quality flag per beat
- `good`: confidence > 0.7
- `fair`: confidence 0.4–0.7
- `poor`: confidence < 0.4

## Priority 3: Morphology Classification (New Feature)

### 3.1 P-wave type classification
| Class | Criteria | Clinical Meaning |
|-------|----------|------------------|
| Normal | mono-phasic, 80-120ms, positive area | Normal atrial depolarization |
| Biphasic (P mitrale) | two peaks, interval > 40ms, width > 120ms | Left atrial enlargement |
| Peaked (P pulmonale) | single peak, amplitude > 2.5× normal, normal width | Right atrial enlargement |
| Inverted | net area negative in Lead II | Ectopic atrial focus / RA-LA reversal |
| Absent | no detectable P-wave | Atrial fibrillation / flutter |
| Low-amplitude | present but peak < 0.05 mV | Pericardial effusion / obesity |

### 3.2 PR segment analysis
- PR segment duration (P-offset to QRS-onset)
- PR segment depression/elevation (deviation from ISO baseline)
- PR segment slope (normally flat; sloping suggests atrial repolarization
  abnormality or pericarditis)

## Priority 4: P-Wave Absence Handling (Medium Impact)

### 4.1 Distinguish true absence from detection failure
**True absence** (AFib): flat baseline in P-region, no organized signal.
→ Report `P_absent=True, absence_type='afib_flat'`
**Detection failure**: signal present but HSMM missed it.
→ Fall back to template matching against neighboring P-waves.

### 4.2 Template-matching fallback
When HSMM fails but signal is present:
1. Compute average P-wave template from successfully detected beats
2. Cross-correlate template with the P-region window
3. If correlation peak > threshold, extract P-wave at peak location

## Priority 5: Multi-lead Fusion (aECG Data)

### 5.1 Compute P-wave in all 6 limb leads
For aECG data with 12 leads, run HSMM on each lead independently.

### 5.2 PCA-enhanced P-wave
Stack P-region signals from all leads → PCA → keep first PC as enhanced P-wave.

### 5.3 Lead selection
For single-lead recordings, auto-select the best lead for P-wave analysis:
- Choose lead with highest P-wave SNR across all beats
- Default priority for 12-lead: II > V1 > III > aVF

## Priority 6: Adaptive Priors (Low Impact, Quick Win)

### 6.1 Heart-rate-adaptive duration priors
```python
rr_ms = 60000 / heart_rate  # RR interval in ms
p_dur_prior = 80 + (rr_ms - 600) * 0.05  # ms, capped at [60, 140]
pr_dur_prior = 120 + (rr_ms - 600) * 0.08  # ms, capped at [100, 220]
```

### 6.2 Cross-beat consistency smoothing
Apply 5-beat sliding median filter to P-wave duration across consecutive beats.
Flag beats where P-duration deviates > 3σ from the median as suspicious.

## Implementation Order

| Step | Module | Expected Gain |
|------|--------|---------------|
| 1. GMM init fix | p_wave_extractor.py | +30% boundary accuracy |
| 2. Confidence rewrite | p_wave_extractor.py | meaningful quality scores |
| 3. Slope zero-crossing | p_wave_extractor.py (new method) | MAE ↓ 5–10ms |
| 4. Absence detection | p_wave_extractor.py | AFib handling |
| 5. Morphology classify | p_wave_analyzer.py | new clinical feature |
| 6. Multi-lead fusion | extractors/p_wave_multi_lead.py | aECG accuracy ↑ |
| 7. Adaptive priors | p_wave_extractor.py | +HR robustness |
| 8. Consistency post-process | p_wave_extractor.py (postprocess method) | outlier removal |
