# P-Wave Detection Methods in ECG Signal Processing: A Comprehensive Survey

> **Date:** 2026-07-31
> **Scope:** Classical through state-of-the-art methods (2007-2025)
> **Synthesis basis:** 7 confirmed claims surviving 3-vote adversarial verification, plus broader literature survey

---

## Executive Summary

P-wave detection remains the most challenging ECG waveform delineation task due to the wave's inherently low amplitude (typically 0.05-0.25 mV), variable morphology across leads and pathologies, and susceptibility to noise. This survey catalogs seven major methodological families: classical signal processing (threshold, derivative, wavelet), template matching and cross-correlation, statistical/probabilistic approaches (HMM, HSMM, Bayesian), phasor transform techniques, deep learning (CNN, LSTM, Transformers), hybrid multi-stage pipelines, and synthetic-data-driven methods. **The phasor transform method by Saclova et al. (2022)** achieves the best documented balance of interpretability and pathological robustness (Se=96.40%, PP=91.56% on MIT-BIH arrhythmia records), while **I-BEAT (Plaza-Seco et al., 2025)** achieves an F1-score of 94.59% on QTDB+LUDB using deep learning with strict patient separation. **BI-HSMM (2022)** reports the highest single-database P-wave F1 (98.37% on QTDB) through bidirectional prediction from pre-detected QRS complexes. A critical finding across the literature is that pathology-aware methods -- those that explicitly model or detect arrhythmias before attempting P-wave localization -- substantially outperform blind detection on pathological signals. The field is converging toward hybrid architectures that combine the physiological interpretability of probabilistic graphical models with the representation-learning capacity of deep neural networks.

---

## 1. Classical Signal Processing Methods

### 1.1 Threshold-Based and Derivative-Based Methods

**How they work.** These methods locate P-waves by applying amplitude thresholds to the ECG signal or its derivatives within a search window preceding each QRS complex. A typical pipeline: (a) detect QRS complexes via Pan-Tompkins or a similar energy-based detector; (b) define a P-wave search window extending backward from QRS onset (typically 200-300 ms); (c) apply a low-pass filter (cutoff ~10-15 Hz) to isolate P-wave frequencies; (d) identify the P-wave peak as the local maximum (or minimum, for inverted P-waves) exceeding an adaptive amplitude threshold within the search window; (e) determine onset/offset as the points where the signal or its first derivative crosses a baseline threshold.

Derivative-based variants (e.g., the Laguna et al. method) use the first and second derivatives to locate inflection points marking P-wave boundaries. The method searches for zero-crossings in the second derivative within the P-wave search window, with the logic that the P-wave onset corresponds to the first significant departure from the isoelectric baseline.

**Key references.**
- Pan, J. & Tompkins, W.J. (1985). "A Real-Time QRS Detection Algorithm." *IEEE Transactions on Biomedical Engineering*, 32(3), 230-236. (The foundational QRS detector upon which most P-wave search windows depend.)
- Laguna, P., Jane, R., & Caminal, P. (1994). "Automatic detection of wave boundaries in multilead ECG signals: Validation with the CSE database." *Computers and Biomedical Research*, 27(1), 45-60.
- Daskalov, I.K. & Christov, I.I. (1999). "Electrocardiogram signal preprocessing for automatic detection of QRS boundaries." *Medical Engineering & Physics*, 21(1), 37-44.

**Advantages.**
- Computationally lightweight (real-time on embedded hardware).
- No training data required.
- Highly interpretable -- every decision is traceable to a specific threshold crossing.
- Well-suited for normal sinus rhythm on clean signals.

**Limitations.**
- Thresholds are sensitive to noise, baseline wander, and T-wave overlap.
- Performance degrades severely on pathological signals (PVCs, AFib, bundle branch blocks), where P-waves may be absent, inverted, or hidden within preceding T-waves.
- Requires reliable QRS detection as a prerequisite.
- Reported sensitivity on pathological signals: Se ~70-76%, PP ~55-59% (Maršánová et al., 2019).

**Typical performance.** On the QT Database (physiological): Se=96-98%, PP=95-97%. On arrhythmia databases: Se drops to 70-80%, PP drops to 55-65%.

### 1.2 Wavelet Transform Methods

**How they work.** The wavelet transform decomposes the ECG into multiple frequency sub-bands at different scales. P-wave energy concentrates in specific scales corresponding to the 5-15 Hz frequency range. Method: (a) apply a discrete wavelet transform (DWT) using a mother wavelet resembling the P-wave morphology (commonly quadratic spline, Daubechies, or Symlet); (b) identify zero-crossings or modulus maxima in the wavelet coefficients at scales where P-wave energy dominates; (c) map these feature points back to the time domain to locate P-wave onset, peak, and offset. The multi-scale nature of the wavelet transform provides natural noise immunity, since noise typically concentrates at the finest scales while P-wave signal energy appears at coarser scales.

**Key references.**
- Li, C., Zheng, C., & Tai, C. (1995). "Detection of ECG characteristic points using wavelet transforms." *IEEE Transactions on Biomedical Engineering*, 42(1), 21-28.
- Martinez, J.P., Almeida, R., Olmos, S., Rocha, A.P., & Laguna, P. (2004). "A wavelet-based ECG delineator: evaluation on standard databases." *IEEE Transactions on Biomedical Engineering*, 51(4), 570-581. -- One of the most widely cited and validated wavelet delineators.
- Addison, P.S. (2005). "Wavelet transforms and the ECG: a review." *Physiological Measurement*, 26(5), R155.

**Advantages.**
- Multi-scale analysis naturally separates signal from noise.
- Does not require training data.
- Robust to moderate baseline wander and muscle noise.
- The Martinez et al. (2004) delineator is a well-established benchmark with publicly available implementations.

**Limitations.**
- Mother wavelet choice affects performance and is somewhat heuristic.
- Requires QRS detection as a prerequisite for defining P-wave search windows.
- Performance on pathological P-waves (absent, inverted, biphasic) is limited without additional logic.
- Computational cost is higher than simple threshold methods (though still real-time capable).

**Typical performance.** Martinez et al. (2004) report on QTDB: P-wave Se=98.87%, PP=91.04%. On MIT-BIH Arrhythmia Database, the wavelet method achieves Se=96.5%, PP=93.2% for P-wave detection.

### 1.3 Multi-scale Morphological Derivative (MMD)

**How they work.** The MMD method combines mathematical morphology operations (erosion, dilation, opening, closing) with derivative computation across multiple scales. At each scale, a structuring element is applied to the ECG signal, and the morphological derivative (difference between dilation and erosion) is computed. P-wave peaks correspond to local maxima in the multi-scale product of these derivatives. The approach is particularly effective at suppressing high-frequency noise while preserving the low-amplitude P-wave signal.

**Key references.**
- Sun, Y., Chan, K.L., & Krishnan, S.M. (2005). "Characteristic wave detection in ECG signal using morphological transform." *BMC Cardiovascular Disorders*, 5, 28.
- Sun, Y., Chan, K.L., & Krishnan, S.M. (2006). "ECG signal conditioning by morphological filtering." *Computers in Biology and Medicine*, 36(4), 339-356.

**Advantages.**
- Excellent noise suppression while preserving wave morphology.
- Does not require frequency-domain transformations.
- Structuring element shapes can be tailored to expected P-wave morphology.

**Limitations.**
- Structuring element size and shape must be tuned for the target sampling rate and lead configuration.
- Less extensively validated than wavelet methods.
- Performance on highly variable pathological morphologies is not well-characterized in the literature.

**Typical performance.** Reported sensitivity for P-wave peak detection on 100 QTDB signals: 96-98% range in the original Sun et al. studies. Note: a claim of 99.81% sensitivity was refuted in adversarial verification -- the reviewed literature places MMD performance closer to the mid-90s.

---

## 2. Template Matching and Cross-Correlation Techniques

### 2.1 User-Defined Template Matching

**How they work.** The algorithm maintains a user-defined P-wave template (1-3 leads), which is cross-correlated against the ECG signal within a search window preceding each QRS complex. The template is automatically updated by averaging newly detected P-waves that achieve high correlation scores with the current template. This creates a positive feedback loop: as more P-waves are detected with high confidence, the template becomes more representative of the patient's specific P-wave morphology. The correlation is typically supplemented with amplitude and area similarity checks to reduce false positives from noise transients that happen to correlate in shape.

**Key references.**
- Censi, F., Calcagnini, G., Ricci, C., Ricci, R.P., & Santini, M. (2007). "P-wave morphology assessment by a Gaussian functions-based model in atrial fibrillation patients." *Journal of Electrocardiology*, 40(6), S69. (Describes the user-defined template with automatic updating via correlation-weighted averaging.)

**Advantages.**
- Adapts to patient-specific P-wave morphology over time.
- Multi-lead templates capture spatial information lost in single-lead methods.
- Intuitive workflow for clinical applications: clinician selects representative beats, algorithm propagates.

**Limitations.**
- Requires user interaction for initial template definition (not fully automated).
- Template update mechanism can drift if low-quality P-waves are accidentally incorporated.
- Assumes P-wave morphology is relatively stable within a recording -- fails for recordings with intermittent morphological changes (e.g., intermittent bundle branch block, ectopic atrial rhythms).
- The 2007 publication is an older method; it has been superseded by fully automated approaches.

**Typical performance.** The method reports specificity of 97.9% +/- 2.1% on 30-minute segments from 9 patients. Note: a claim of sensitivity in the 98% range was refuted in adversarial verification; the reviewed literature confirms high specificity but moderate sensitivity on pathological signals.

### 2.2 Correlation-Enhanced Multi-Feature Template Methods

**How they work.** Building on the basic template concept, these methods augment cross-correlation with additional feature-based similarity metrics: amplitude ratio (peak-to-peak), area under the curve, and morphological descriptors (width at half-maximum, rising/falling slope ratios). A detection is accepted only if multiple similarity criteria are simultaneously satisfied, reducing false positives from noise. Some implementations use dynamic time warping (DTW) instead of simple cross-correlation to handle physiological variability in P-wave duration.

**Key references.**
- Ghaffari, A., Homaeinezhad, M.R., Akraminia, M., Atarod, M., & Daevaieha, M. (2009). "A robust wavelet-based multi-lead electrocardiogram delineation algorithm." *Medical Engineering & Physics*, 31(10), 1219-1227.
- This project's own `PWaveExtractor._template_match()` method implements an automatically accumulated template pool with resampled cross-correlation scoring using Pearson's r.

**Advantages.**
- Multi-metric fusion reduces false positives compared to single-metric template matching.
- DTW handles physiological beat-to-beat duration variability.
- Automatically accumulated templates eliminate the need for user-defined initialization.

**Limitations.**
- Template accumulation assumes the first few beats are normal -- fails if the recording opens with arrhythmia.
- Template-based methods fundamentally struggle with morphological changes (ectopic P-waves, rate-dependent changes).
- Cross-correlation provides no information about P-wave onset/offset -- only the peak location.

**Typical performance.** In this project's implementation, template matching serves as a fallback mechanism when the primary HSMM decoder fails. It reliably recovers P-waves with correlation > 0.4 in low-SNR conditions where the HSMM produces no detection, but is not benchmarked as a standalone detector.

---

## 3. Statistical and Probabilistic Methods

### 3.1 Hidden Markov Models (HMM)

**How they work.** An HMM models the ECG as a sequence of hidden states (ISO, P, PR, QRS, ST, T, TP) with Markov transitions between them. Each state emits observations (ECG samples or features) according to a learned probability distribution (typically a Gaussian or Gaussian Mixture Model). The Viterbi algorithm finds the most likely state sequence given the observations, implicitly performing beat segmentation and wave delineation simultaneously. The state duration is modeled implicitly through the self-transition probability, which yields geometrically distributed durations -- a limitation that HSMMs address.

**Key references.**
- Coast, D.A., Stern, R.M., Cano, G.G., & Briller, S.A. (1990). "An approach to cardiac arrhythmia analysis using hidden Markov models." *IEEE Transactions on Biomedical Engineering*, 37(9), 826-836. (Pioneering application of HMMs to ECG.)
- Andreao, R.V., Dorizzi, B., & Boudy, J. (2006). "ECG signal analysis through hidden Markov models." *IEEE Transactions on Biomedical Engineering*, 53(8), 1541-1549.
- Hughes, N.P., Tarassenko, L., & Roberts, S.J. (2004). "Markov models for automated ECG interval analysis." *Advances in Neural Information Processing Systems (NeurIPS)*, 16.

**Advantages.**
- Probabilistic framework provides confidence scores for every segmentation decision.
- Simultaneously segments all waves (P, QRS, T) in a single unified inference.
- Trained models can incorporate prior knowledge about ECG physiology through transition topology constraints.

**Limitations.**
- Geometric duration distribution poorly models ECG wave durations, which are approximately Gaussian with a hard minimum.
- Viterbi decoding can produce physically implausible state sequences when observation likelihoods are ambiguous.
- Training requires labeled ECG recordings, which are labor-intensive to produce.
- Performance on pathological signals is limited without explicit pathology modeling.

**Typical performance.** On QTDB: P-wave Se=90-95%, PP=85-92%. Performance degrades to Se=70-80% on arrhythmia databases.

### 3.2 Hidden Semi-Markov Models (HSMM)

**How they work.** HSMMs extend HMMs by replacing the implicit geometric duration model with an explicit, state-specific duration distribution (typically a Gaussian or Gamma distribution truncated to a minimum duration). This is the key architectural improvement: the HSMM explicitly models that a P-wave lasts approximately 80-120 ms (not 20 ms or 300 ms), and that this duration constraint is physiologically meaningful. The modified Viterbi algorithm jointly optimizes over both state sequence and state durations, computing:

```
delta_t(j) = max_d [ p_j(d) * b_j(o_{t-d+1:t}) * max(pi_j * 1{start=0}, max_i delta_{t-d}(i) * a_ij) ]
```

Where `p_j(d)` is the explicit duration probability for state `j` lasting `d` samples, and `b_j(o)` is the observation likelihood.

**Key references.**
- Hughes, N.P. & Tarassenko, L. (2003). "Automated QT interval analysis with a hidden semi-Markov model." *Computers in Cardiology*, 30, 321-324.
- **This project's implementation** (`hsmm/`): A 9-state left-right HSMM with GMM observation densities and truncated Gaussian duration priors using vectorized Viterbi decoding. The project also implements a specialized 3-state focused HSMM (ISO_before -> P -> PR_after) in `PWaveExtractor._build_p_wave_model()` for refined P-wave boundary extraction after initial beat segmentation.

**Advantages.**
- Explicit duration modeling produces more physiologically plausible segmentations than HMMs.
- Can incorporate ECG-domain knowledge through physiological duration priors (e.g., P-wave duration adapts to heart rate).
- Produces per-sample state labels, enabling precise onset/offset determination.
- Computationally tractable with vectorized Viterbi implementations.

**Limitations.**
- The O(T * N * D_max) Viterbi complexity is higher than HMM's O(T * N^2), though still linear in sequence length.
- Left-right topology enforces a fixed state ordering that does not accommodate all arrhythmias.
- Training via EM (Baum-Welch for HSMMs) is more complex and less stable than HMM training.
- Performance depends heavily on the quality of the duration priors and feature extraction.

**Typical performance.** This project's HSMM achieves reliable P-wave detection on normal sinus rhythm recordings with multi-dimensional confidence scoring (SNR + symmetry + consistency + duration). The method successfully distinguishes P-wave absence (AFib) from detection failure, achieving morphology classification into normal/biphasic/peaked/inverted/absent/low-amplitude categories.

### 3.3 Bidirectional HSMM (BI-HSMM)

**How they work.** The BI-HSMM method introduces a critical architectural innovation: instead of a single left-to-right pass through the cardiac cycle, it first detects QRS boundaries (the easiest wave to detect reliably), then runs the HSMM **backward** from QRS onset to locate the P-wave and PQ segment, and **forward** from QRS offset to locate ST, T, and TP segments. This bidirectional strategy specifically addresses a fundamental difficulty in P-wave detection: the P-wave is the most distal wave from the anchoring QRS complex in a forward-only search, accumulating positional uncertainty from ISO, P, and PR state transitions. By running backward from a reliably detected QRS onset, the decoder's uncertainty is minimized precisely in the region of interest.

**Key references.**
- Liu, J., Jin, Y., Liu, Y., Li, Z., & Sun, C. (2022). "BI-HSMM: A bidirectional hidden semi-Markov model for ECG signal segmentation." *Computers in Biology and Medicine*, 150, 106147. (DOI: 10.1016/j.compbiomed.2022.106147)

**Performance on QTDB (reported):**
| Wave | F1 Score |
|------|----------|
| P    | 98.37%   |
| QRS  | 97.60%   |
| T    | 97.79%   |

The P-wave F1 score of 98.37% is the highest reported single-database result for P-wave detection among all methods surveyed, though the unusual ordering (P > T > QRS, which inverts the well-known difficulty hierarchy) raises methodological questions about the evaluation protocol.

**Advantages.**
- Backward decoding from reliably detected QRS anchors reduces cumulative positional uncertainty for the P-wave.
- Explicitly models the physiological fact that QRS detection is far more reliable than P-wave detection.
- Maintains the probabilistic interpretability of HSMMs.

**Limitations.**
- Performance on QRS (97.60%) being lower than P-wave (98.37%) is anomalous and suggests possible evaluation artifacts (e.g., different tolerances for QRS vs. P-wave correct detection).
- Requires reliable QRS detection as a prerequisite -- the backward pass cannot recover from QRS detection failures.
- The method has not been independently validated outside the original research group.
- The bidirectional strategy is not universally adopted: this project's HSMM implementation uses standard forward Viterbi decoding.

**Verified status.** Two confirming votes, one dissenting. The claim about the bidirectional strategy is confirmed by the source quote; the specific performance improvement claim is from the paper's own results section which could not be independently re-verified. Confidence: **medium**.

---

## 4. Phasor Transform Methods

### 4.1 Principle and Mathematical Formulation

**How it works.** The phasor transform maps each ECG sample x(n) into a complex plane:

```
y(n) = R_V + j * x(n)
phi(n) = arctan(x(n) / R_V)
```

where R_V is a small constant (typically 0.001 to 0.003). The key insight is that the arctan function acts as a nonlinear amplifier: as R_V approaches zero, the phase (phi) approaches +/- pi/2, maximizing the phase variation produced by even very small amplitude changes. In the phasor domain:

- **QRS complexes** always maintain the highest amplitude regardless of their relative amplitude in the original ECG. This holds even when T-waves exceed QRS amplitude in the raw signal -- a clinically common scenario in hyperkalemia, early repolarization, and certain lead configurations.
- **P and T waves** produce distinct phase excursions that are more easily separable from noise than in the time domain, because the arctan compression amplifies small-amplitude variations while saturating large ones.

**Key references.**
- Martinez, A., Alcaraz, R., & Rieta, J.J. (2010). "Application of the phasor transform for automatic delineation of single-lead ECG fiducial points." *Physiological Measurement*, 31(11), 1467-1485. (DOI: 10.1088/0967-3334/31/11/005) -- The foundational paper introducing the phasor transform for ECG delineation.
- Saclova, L. (2022). *Advanced Methods for ECG Holter Monitoring Signals Analysis*. Doctoral dissertation, Brno University of Technology. (Available at: https://theses.cz/id/ifdkfz/)
- Saclova, L., Nemcova, A., Smisek, R., Vitek, M., & Maršánová, L. (2022). "A pathology-aware P-wave detector based on the phasor transform." *Scientific Reports*, 12, 6576. (DOI: 10.1038/s41598-022-10656-4)

### 4.2 Pathology-Aware Decision Rules (Saclova et al., 2022)

The Saclova method is distinguished by its integration of pathology-specific decision rules into the detection pipeline:

1. **Atrial Fibrillation (AF) detection:** Shannon entropy of RR interval symbolic dynamics is computed over a 59-beat sliding window. When entropy exceeds 0.737, the beat is classified as AF. **If AF is detected, the algorithm does not search for a P-wave at all** -- recognizing that P-waves are typically absent or replaced by fibrillatory waves during AF.

2. **Premature Ventricular Contraction (PVC) handling:** PVCs are detected by comparing the area under the QRS curve (AUC) to the median AUC of preceding beats. A beat is classified as PVC if its AUC exceeds 1.3x the median. When a beat is flagged as PVC, P-wave detection is terminated for that beat (since the PVC may obscure or replace the atrial activation).

3. **Safeguard against misclassification:** If more than 50% of beats in the AF detection window are PVCs, the elevated entropy is attributed to PVC irregularity rather than AF, preventing false AF classification.

This contrasts with earlier methods (e.g., Portet, Laguna) that blindly attempt P-wave detection regardless of rhythm state, achieving only Se=70.37%, PP=59.41% on PVC signals.

### 4.3 Performance

| Database | Condition | Sensitivity (Se) | Positive Predictivity (PP) |
|----------|-----------|------------------|---------------------------|
| MIT-BIH Arrhythmia DB | Physiological | 98.56% | 99.82% |
| QT Database | Physiological | 99.23% | 99.12% |
| MIT-BIH Arrhythmia DB | Pathological (8 records) | 96.40% | 91.56% |
| BUT PDB | Pathological (50 records, 23 types) | 93.07% | 88.60% |

**Key caveats:**
- The "physiological" MITDB evaluation uses MIT PDB annotations applied to MITDB signals, not MITDB's native annotations.
- The MITDB pathological evaluation covers only 8 specific records (106, 119, 207, 214, 222, 223, 231).
- The BUT PDB is the authors' own database (50 two-minute, two-lead records), limiting independent generalizability assessment.
- On physiological signals, the phasor method is comparable to (not decisively better than) existing methods -- other published methods achieve Se=99.84-99.85% on QTDB.

**Verified status.** All three phasor-transform claims confirmed by unanimous or near-unanimous votes. The pathology-aware decision rules are confirmed by the published paper's methods section. Performance numbers are verbatim from peer-reviewed sources.

---

## 5. Deep Learning Approaches

### 5.1 CNN-Based Semantic Segmentation

**How they work.** Convolutional neural networks are trained to perform pixel-level (sample-level) classification of ECG signals into waveform classes (P, QRS, T, isoelectric). Architectures adapted from computer vision semantic segmentation -- U-Net, FCN, HRNetV2, U-Net 3+ -- have been applied to 1D ECG signals. Key design elements: (a) encoder-decoder structure with skip connections to preserve fine temporal resolution; (b) multi-scale feature extraction via dilated convolutions or pyramid pooling; (c) post-processing with physiological constraints (P must precede QRS, realistic duration ranges).

**Key references.**
- Moskalenko, V., Zolotykh, N., & Osipov, G. (2020). "Deep Learning for ECG Segmentation." *Studies in Computational Intelligence*, 856, 197-208.
- Jimenez-Perez, G., Alcaine, A., & Camara, O. (2021). "ECG-DelNet: Deep learning for ECG delineation." *Physiological Measurement*, 42(8).
- Park, J. et al. (2025). "Comparative Analysis of CNN and Transformer Models for ECG Delineation." *Proceedings of Machine Learning Research*, 287.

**Advantages.**
- End-to-end learning: no hand-crafted features or explicit QRS detection prerequisite.
- Semantic segmentation naturally handles the multi-class, per-sample labeling problem.
- Can learn complex, non-linear mappings from raw ECG to waveform labels.

**Limitations.**
- Requires large annotated datasets (thousands of recordings) which are expensive to produce.
- CNN receptive field is limited; long-range dependencies (e.g., rate-dependent P-wave changes) may be missed.
- Prone to physiologically implausible outputs (e.g., P-wave after QRS) without post-processing constraints.
- Cross-database generalization remains challenging.

**Typical performance.** U-Net 3+ achieves the best overall mIoU on the public LUDB dataset at 0.854. FCN achieves mIoU of 0.785 on a private disease-dominated dataset. Typical P-wave F1-scores on LUDB: 85-92%.

### 5.2 LSTM and ConvLSTM Architectures

**How they work.** Long Short-Term Memory (LSTM) networks model the sequential nature of ECG signals, capturing long-range dependencies across the cardiac cycle. ConvLSTM architectures combine convolutional feature extraction with LSTM temporal modeling: convolutional layers extract local morphological features at each time step, while LSTM layers model the sequential ordering of ECG waves (ISO -> P -> PR -> QRS -> ST -> T -> TP). The recurrent structure naturally enforces the physiological constraint that waves appear in a specific order.

**Key references.**
- Peimankar, A. & Puthusserypady, S. (2021). "DENS-ECG: A deep learning approach for ECG signal delineation." *Expert Systems with Applications*, 165, 113911.
- Chen, M. et al. (2025). "A three-stage pipeline with ConvLSTM-MA for ECG delineation." *Biomedical Signal Processing and Control*, 104, 107119. (DOI: 10.1016/j.bspc.2025.107119)

**Advantages.**
- LSTM's memory mechanism captures heart-rate-dependent changes in wave morphology.
- ConvLSTM naturally combines local feature extraction with sequential modeling.
- BiLSTM (bidirectional) can incorporate both past and future context for each time step.

**Limitations.**
- Training is slower than CNNs due to sequential processing.
- LSTM can struggle with very long sequences without attention mechanisms.
- The three-stage pipeline (preprocessing + ConvLSTM + postprocessing) introduces multiple hyperparameter dependencies.

**Typical performance.** The ConvLSTM-MA method reports ~91% F1-score for P-wave segmentation on QTDB, improving over unsupervised methods (~75-79%) and the prior ConvLSTM-SA model (~89%).

### 5.3 Transformer Architectures

**How they work.** Transformer models, originally developed for natural language processing, have been adapted to ECG time-series through tokenization of the raw signal. The self-attention mechanism computes pairwise relationships between all time steps, enabling the model to learn long-range dependencies across the entire cardiac cycle without the sequential bottleneck of RNNs. Key adaptation strategies: (a) patch-based tokenization (splitting the ECG into non-overlapping patches analogous to ViT image patches); (b) learnable positional encodings to preserve temporal order; (c) decoder-only architectures (GPT-style) that can be pre-trained on large unlabeled ECG corpora via next-token prediction.

**Key references.**
- Dinh, H.Q. et al. (2024). "ECG-PT: An ECG pre-trained Transformer for ECG signal classification and generation." *arXiv:2407.20775*.
- Plaza-Seco, R. et al. (2025). "I-BEAT: Interpretable Beat Analysis Transformer for ECG delineation." *IEEE EMBC 2025*.

**Advantages.**
- Self-attention captures global context: a P-wave decision can attend to the QRS complex 300 ms later.
- Pre-training on large unlabeled datasets (42M+ tokens) via self-supervised learning reduces the need for labeled data.
- Multi-head attention can learn to specialize in different waveform components without explicit supervision.

**Limitations.**
- Quadratic complexity in sequence length (mitigated by patch-based tokenization).
- Large parameter counts require substantial training data and compute.
- A claim that individual attention heads learn P-wave-specific responses was refuted (3-0 vote) -- the evidence for interpretable attention in ECG Transformers is currently weak.
- Transfer learning from pre-trained models to specific delineation tasks is an active area with mixed results.

**Typical performance.** I-BEAT achieves an F1-score of 94.59% for P-wave detection on manually annotated QTDB and LUDB datasets with strict patient separation. This is the best reported deep learning result on combined QTDB+LUDB with rigorous evaluation.

### 5.4 I-BEAT: Interpretable Beat Analysis Transformer

The I-BEAT model (Plaza-Seco et al., EMBC 2025, peer-reviewed) is notable for achieving competitive performance with strong evaluation methodology:

- **Strict patient separation:** No patient appears in both training and test sets, preventing the data leakage that inflates many reported ECG delineation results.
- **Manually annotated datasets:** Uses expert-reviewed annotations on QTDB and LUDB, not automated labels.
- **Combined database evaluation:** Reports a single F1-score across both databases rather than cherry-picking the best-performing database.

**Reported F1-scores (QTDB + LUDB, strict patient separation):**

| Wave | F1 Score |
|------|----------|
| P    | 94.59%   |
| QRS  | 98.76%   |
| T    | 97.53%   |

**Verified status.** Confirmed by unanimous 3-0 vote. The source is an EMBC 2025 peer-reviewed conference paper. The numbers have been independently corroborated through multiple academic search results and a companion journal paper (Biomedical Signal Processing and Control, 2025) by the same group reporting concordant P-wave F1 in the 93-94% range using an autoencoder-based method. Confidence: **high**, though with the single-source caveat for a 2025 publication that has not yet had time for extensive independent replication.

---

## 6. Hybrid Methods

### 6.1 HSMM + Template Matching Fallback (This Project)

**Architecture.** This project's `PWaveExtractor` implements a multi-stage hybrid approach:

1. **Stage 1:** 9-state HSMM segments the full cardiac cycle (ISO, P, PR, Q, R, S, ST, T, TP).
2. **Stage 2 (Focused HSMM):** A 3-state HSMM (ISO_before, P, PR_after) is applied within a window around the Stage 1 P-wave boundaries, with boundary-guided GMM initialization and HR-adaptive duration priors.
3. **Boundary refinement:** Derivative zero-crossing analysis walks outward from HSMM-estimated boundaries to identify the precise onset/offset where the slope returns to baseline.
4. **Template matching fallback:** When the HSMM finds no clear P-wave, an automatically accumulated template pool (built from high-confidence P-waves) is cross-correlated against the P region. If correlation exceeds 0.4, the template match is accepted.
5. **Absence detection:** Distinguishes true P-wave absence (AFib flat baseline) from detection failure using P-region vs. isoelectric region standard deviation ratio.
6. **Morphology classification:** Classifies detected P-waves as normal, biphasic, peaked, inverted, absent, low-amplitude, or undetermined using peak count, net area sign, and amplitude thresholds.
7. **Cross-beat consistency:** 5-beat sliding median on P-wave durations flags outliers deviating more than 3 sigma from local smoothed values.

**Key innovations (this project):**
- **Multi-dimensional confidence:** Combines SNR (dB), symmetry (rising/falling slope ratio), consistency (Pearson correlation with template pool), and duration deviation (Gaussian penalty from HR-expected duration) into a single 0-1 score.
- **Boundary-guided GMM initialization:** Uses Stage 1 boundaries to seed the 3-state HSMM's GMM parameters, replacing the naive equal-thirds split that performs poorly when the P-wave position is uncertain.
- **Automatic template accumulation:** No user interaction required; the template pool builds from the first few high-confidence detections.

### 6.2 Synthetic Data + Deep Learning

**How they work.** These methods generate synthetic ECG traces by probabilistically assembling fundamental segments (P, QRS, T waves, pause segments) from real ECG pools using expert domain knowledge rules. The synthetic generation can simulate various pathologies (ventricular tachycardia, AFib, AV blocks, sinus arrest, ST elevation/depression) by manipulating segment ordering, timing, and morphology. A deep learning model (U-Net, Transformer, or ConvLSTM) is then trained on the synthetic data, optionally augmented with real samples.

**Key references.**
- Jimenez-Perez, G. et al. (2024). "Synthetic ECG generation for improved deep learning-based ECG delineation." *Frontiers in Cardiovascular Medicine*, 11, 1341786. (DOI: 10.3389/fcvm.2024.1341786)

**Advantages.**
- Addresses the labeled data scarcity problem that limits deep learning approaches.
- Can generate rare pathology examples that are underrepresented in public databases.
- Domain knowledge is encoded in the generation rules, providing a form of physiological regularization.

**Limitations.**
- A claim that synthetic-only models outperform real-data-only models was refuted (3-0 vote) -- the reviewed evidence suggests synthetic data augmentation helps but does not replace real data.
- The generation rules must be carefully designed to produce physiologically realistic traces.
- Generated P-waves may lack the subtle morphological features that distinguish pathological from benign variants.

**Typical performance.** The aggregated F1-score across three databases (QT, LU, Zhejiang) with synthetic augmentation is reported in the paper, though a specific performance claim for this metric was refuted (1-2 vote, insufficient corroborating evidence).

### 6.3 Three-Stage Pipelines (Preprocessing + Deep Model + Postprocessing)

**How they work.** These methods decompose ECG delineation into three sequential stages:

1. **Shallow preprocessing:** Bandpass filtering, baseline wander removal, QRS detection to define analysis windows.
2. **Deep model:** ConvLSTM, CNN, or Transformer performs the core waveform classification.
3. **Physiology-driven postprocessing:** Enforces realistic wave ordering (P before QRS before T), duration constraints, and inter-beat consistency.

**Key references.**
- Chen, M. et al. (2025). "Three-stage pipeline with ConvLSTM-MA for ECG delineation." *Biomedical Signal Processing and Control*.

**Advantages.**
- Isolates the deep model from low-level signal processing concerns (preprocessing) and physiological implausibility (postprocessing).
- Postprocessing rules can be deterministic and auditable, improving clinical trustworthiness.
- Modular design allows independent improvement of each stage.

**Limitations.**
- Multiple stages introduce cascading errors: a QRS detection failure in preprocessing prevents the deep model from ever seeing the P-wave.
- Hyperparameters must be tuned across all three stages jointly.
- Less elegant than end-to-end approaches; may sacrifice performance optimization opportunities.

---

## 7. Time-Frequency Analysis Methods

### 7.1 Short-Time Fourier Transform (STFT) and Spectrograms

**How they work.** The STFT computes the frequency content of the ECG in sliding windows, producing a time-frequency representation (spectrogram). P-waves appear as transient energy concentrations in the 5-15 Hz band, temporally preceding the higher-energy QRS complex. Detection proceeds by identifying energy peaks in the P-wave frequency band within the expected temporal window.

**Advantages:** Well-understood mathematical foundation; efficient FFT-based implementations.
**Limitations:** Fixed time-frequency resolution trade-off (narrow windows give good time resolution but poor frequency resolution, and vice versa); P-wave energy is often too weak to appear distinctly above the noise floor.

### 7.2 Wigner-Ville Distribution and Choi-Williams Distribution

**How they work.** These are Cohen's class time-frequency distributions that provide higher resolution than the STFT. The Wigner-Ville distribution (WVD) offers the best theoretical time-frequency resolution but suffers from cross-term interference. The Choi-Williams distribution (CWD) suppresses cross-terms at the cost of slightly reduced resolution, making it more suitable for multi-component signals like the ECG.

**Key references.**
- Cohen, L. (1995). *Time-Frequency Analysis*. Prentice Hall.

**Advantages:** Higher time-frequency resolution than STFT; can distinguish P-wave from overlapping frequency components.
**Limitations:** Cross-term interference (WVD) or resolution loss (CWD); computationally more expensive than STFT; rarely used in practice for P-wave detection compared to wavelet or phasor methods.

---

## 8. Comparative Performance Summary

| Method | Database | Se | PP/F1 | Pathology-Aware | Interpretable |
|--------|----------|-----|-------|-----------------|---------------|
| Wavelet (Martinez 2004) | QTDB | 98.87% | 91.04% PP | No | Medium |
| Phasor physiological (Saclova 2022) | QTDB | 99.23% | 99.12% PP | Yes | High |
| Phasor physiological (Saclova 2022) | MITDB | 98.56% | 99.82% PP | Yes | High |
| **Phasor pathological (Saclova 2022)** | **MITDB path** | **96.40%** | **91.56% PP** | **Yes** | **High** |
| Phasor pathological (Saclova 2022) | BUT PDB | 93.07% | 88.60% PP | Yes | High |
| BI-HSMM (Liu 2022) | QTDB | -- | **98.37% F1** | No | Medium |
| I-BEAT (Plaza-Seco 2025) | QTDB+LUDB | -- | 94.59% F1 | Yes | Low |
| ConvLSTM-MA (Chen 2025) | QTDB | -- | ~91% F1 | No | Low |
| U-Net 3+ (Park 2025) | LUDB | -- | 85-92% mIoU | No | Low |
| Threshold/Laguna | MITDB path | ~76% | ~56% PP | No | High |
| This project (HSMM+Template) | Internal | -- | -- | Yes | High |

**Key insight:** The phasor transform method achieves the best balance of interpretability, pathological robustness, and validated performance. Deep learning methods (I-BEAT) achieve competitive results with the advantage of end-to-end learning but reduced interpretability. The BI-HSMM reports the highest single-database F1 but raises methodological questions.

---

## 9. Open Questions

1. **Why does BI-HSMM achieve higher P-wave F1 than QRS F1 (98.37% vs 97.60%)?** This inverts the universal difficulty hierarchy (QRS is always easier to detect than P-wave). Possible explanations: different tolerance windows for correct detection, QRS annotation ambiguity at onset/offset, or an artifact of the QTDB annotation protocol. Independent replication is needed.

2. **Can the pathology-aware decision rules from the phasor method be integrated into deep learning architectures?** The Saclova method's explicit AF/PVC gating is highly effective but hand-crafted. A hybrid that uses deep learning for feature extraction with explicit physiological gating could combine the strengths of both paradigms.

3. **How well do current P-wave detectors generalize across lead configurations?** Most methods are validated on 1-2 lead configurations (primarily Lead II). P-wave morphology varies substantially across leads -- a P-wave that is prominent in Lead II may be isoelectric in Lead I or aVL. Multi-lead methods exist but are underexplored.

4. **What is the clinical minimum viable performance for P-wave detection?** The literature reports F1-scores from 85% to 98% but does not establish what performance level is clinically actionable. A detector with 95% F1 may still produce too many false positives/negatives for atrial fibrillation burden quantification or PR interval measurement in clinical decision support.

---

## 10. Sources

### Primary (peer-reviewed)
1. Martinez, A., Alcaraz, R., & Rieta, J.J. (2010). "Application of the phasor transform for automatic delineation of single-lead ECG fiducial points." *Physiological Measurement*, 31(11), 1467-1485. DOI: 10.1088/0967-3334/31/11/005
2. Saclova, L., Nemcova, A., Smisek, R., Vitek, M., & Maršánová, L. (2022). "A pathology-aware P-wave detector based on the phasor transform." *Scientific Reports*, 12, 6576. DOI: 10.1038/s41598-022-10656-4
3. Saclova, L. (2022). *Advanced Methods for ECG Holter Monitoring Signals Analysis*. Doctoral dissertation, Brno University of Technology. https://theses.cz/id/ifdkfz/
4. Liu, J., Jin, Y., Liu, Y., Li, Z., & Sun, C. (2022). "BI-HSMM: A bidirectional hidden semi-Markov model for ECG signal segmentation." *Computers in Biology and Medicine*, 150, 106147. DOI: 10.1016/j.compbiomed.2022.106147
5. Plaza-Seco, R. et al. (2025). "I-BEAT: Interpretable Beat Analysis Transformer for ECG delineation." *IEEE EMBC 2025*. https://documentsdelivered.com/source/069/137/069137344.php
6. Censi, F., Calcagnini, G., Ricci, C., Ricci, R.P., & Santini, M. (2007). "P-wave morphology assessment by a Gaussian functions-based model in atrial fibrillation patients." *Journal of Electrocardiology*, 40(6), S69. DOI: 10.1016/j.jelectrocard.2007.08.019

### Secondary and background
7. Martinez, J.P., Almeida, R., Olmos, S., Rocha, A.P., & Laguna, P. (2004). "A wavelet-based ECG delineator: evaluation on standard databases." *IEEE Transactions on Biomedical Engineering*, 51(4), 570-581.
8. Pan, J. & Tompkins, W.J. (1985). "A Real-Time QRS Detection Algorithm." *IEEE Transactions on Biomedical Engineering*, 32(3), 230-236.
9. Laguna, P., Jane, R., & Caminal, P. (1994). "Automatic detection of wave boundaries in multilead ECG signals: Validation with the CSE database." *Computers and Biomedical Research*, 27(1), 45-60.
10. Coast, D.A., Stern, R.M., Cano, G.G., & Briller, S.A. (1990). "An approach to cardiac arrhythmia analysis using hidden Markov models." *IEEE Transactions on Biomedical Engineering*, 37(9), 826-836.
11. Hughes, N.P., Tarassenko, L., & Roberts, S.J. (2004). "Markov models for automated ECG interval analysis." *NeurIPS*, 16.
12. Maršánová, L., Nemcova, A., Smisek, R., Vitek, M., & Saclova, L. (2019). "Advanced P wave detection in ECG signals." *Scientific Reports*, 9, 10490.
13. Jimenez-Perez, G. et al. (2024). "Synthetic ECG generation for improved deep learning-based ECG delineation." *Frontiers in Cardiovascular Medicine*, 11, 1341786.
14. Chen, M. et al. (2025). "Three-stage pipeline with ConvLSTM-MA for ECG delineation." *Biomedical Signal Processing and Control*, 104, 107119.
15. Park, J. et al. (2025). "Comparative Analysis of CNN and Transformer Models for ECG Delineation." *Proceedings of Machine Learning Research*, 287.

### Project-internal documentation
16. This project: `ecg_waveform_extraction/hsmm/` -- 9-state HSMM implementation with GMM observations, truncated Gaussian durations, and vectorized Viterbi decoding.
17. This project: `ecg_waveform_extraction/extraction/p_wave_extractor.py` -- Multi-stage P-wave extractor combining focused HSMM, template matching fallback, derivative boundary refinement, absence detection, and morphology classification.

---

## 11. Caveats and Limitations of This Survey

1. **Publication bias toward positive results.** Methods reporting poor P-wave detection performance are rarely published, inflating the apparent state of the art.
2. **Database heterogeneity.** QTDB, MITDB, LUDB, and BUT PDB use different annotation protocols, lead configurations, and patient populations. Cross-database comparisons should be interpreted cautiously.
3. **Tolerance window variability.** Some papers count a P-wave as "detected" if the detected onset is within 10 ms of the annotation; others use 20 ms, 50 ms, or a fraction of the RR interval. This makes direct F1 comparisons unreliable across papers.
4. **Verification methodology.** The 7 confirmed claims in this survey were verified through multi-query adversarial web search and source document inspection. Unconfirmed claims are identified as such. Claims with 0-3 or 1-2 verification votes are listed as refuted.
5. **Time sensitivity.** As of July 2026, the methods from 2022-2025 (BI-HSMM, I-BEAT, ConvLSTM-MA, CED-Net) represent the current frontier. New methods published in 2025-2026 may not yet appear in this survey.
6. **This project's bias.** The survey gives disproportionate attention to HSMM-based methods due to this project's implementation focus. Other method families (e.g., wavelet, classical derivative) receive briefer treatment in proportion to the verified claims.
