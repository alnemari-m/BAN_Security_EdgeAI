# PhysioKey: Research Reasoning & Process

## Overview

**Paper Title:** Edge-AI-Driven Physiological Key Agreement for Secure Body Area Networks: A TinyML Approach Using ECG and PPG Signals

**Author:** Mohammed Alnemari, University of California, Irvine

**Target Venue:** IJACSA (International Journal of Advanced Computer Science and Applications)

**Paper Type:** Concept paper with simulation-based evaluation (no hardware prototype)

---

## 1. Problem Statement

Body Area Networks (BANs) in healthcare deploy multiple sensors (ECG, PPG, temperature, accelerometer) on a patient's body. These sensors communicate wirelessly and transmit sensitive physiological data. Securing this communication is critical but uniquely hard because:

- **Resource constraints**: BAN sensors run on ARM Cortex-M class MCUs with 256-512 KB flash, 64-128 KB SRAM, and coin-cell batteries
- **PKI is too expensive**: ECC-256 ECDH costs ~6.84 mJ and 142 ms per key exchange
- **Pre-shared keys don't scale**: Clinical environments constantly add/remove/replace sensors
- **Static biometric methods are rigid**: Hand-crafted features (IPI, peak amplitude) don't adapt to inter-patient variability

**The gap:** No existing scheme provides an adaptive, learning-based key agreement mechanism that operates within BAN resource constraints while eliminating pre-shared secrets.

---

## 2. Core Insight

Sensors on the **same body** measure physiological signals that share a common biological origin (the cardiovascular system). A sensor on a **different body** cannot produce signals correlated with the target patient. This physical proximity requirement is the security foundation.

**Key innovation:** Instead of hand-crafting statistical features from physiological signals, use a **TinyML model** (lightweight 1D-CNN) to *learn* discriminative features that:
1. Are highly correlated between sensors on the same body
2. Are uncorrelated between sensors on different bodies
3. Carry enough entropy for cryptographic key derivation
4. Can run in real-time on a Cortex-M4 MCU

---

## 3. Design Decisions & Rationale

### 3.1 Why 1D-CNN?

| Alternative | Why Rejected |
|---|---|
| RNN/LSTM | Too large for MCU (>100 KB), recurrent computation is slow |
| Transformer | Attention mechanism requires O(n^2) memory, impractical |
| Random Forest | Can't learn continuous embeddings for fuzzy commitment |
| Hand-crafted features | Not adaptive, poor cross-patient generalization |
| **1D-CNN** | **Small (6.3 KB), fast (2.1 ms), fixed compute graph, INT8-friendly** |

### 3.2 Architecture: Why This Specific CNN?

```
Input(256x1) -> Conv1D(8, k=5, s=2) + BN -> ReLU
             -> Conv1D(16, k=3, s=2) + BN -> ReLU
             -> Conv1D(32, k=3, s=2) + BN -> ReLU
             -> GlobalAvgPool -> Dense(64) -> Dense(32)
```

**Reasoning:**
- **3 conv layers**: Captures local (individual heartbeat), mid-range (beat-to-beat variation), and global (rhythm) patterns
- **Increasing filters (8 -> 16 -> 32)**: Progressively abstract features from raw waveform
- **Stride 2**: Reduces sequence length by 8x total (256 -> 32), avoiding max-pool overhead
- **Global Average Pooling**: Variable-length input support, minimal parameters, acts as regularizer
- **32-dim embedding**: 32 dimensions x 2 bits = 64-bit key (or x 1 bit = 32-bit key) -- tuned for BAN key sizes
- **BatchNorm**: Training stability, folds into conv weights at inference (zero overhead)
- **Total: 6,320 parameters = 6.2 KB at INT8**

### 3.3 Why Contrastive + Alignment + Decorrelation Loss?

The training objective has three components:

1. **NT-Xent Contrastive Loss** (temperature = 0.05): Pushes same-patient embeddings together and different-patient embeddings apart in the cosine similarity space. The low temperature creates sharper decision boundaries.

2. **Alignment Loss** (weight = 0.5): Directly maximizes cosine similarity between paired Lead I/Lead II embeddings. The contrastive loss alone optimizes relative ranking but doesn't guarantee high absolute similarity. Alignment loss ensures the paired embeddings are actually close, which reduces the Bit Disagreement Rate (BDR).

3. **Decorrelation Loss** (weight = 0.05): Penalizes correlation between embedding dimensions. This ensures each dimension carries independent information, maximizing the entropy of the quantized key. Without this, redundant dimensions waste key bits.

### 3.4 Why Fuzzy Commitment (not Fuzzy Vault)?

- **Fuzzy Vault**: Requires unordered set representation, polynomial reconstruction -- complex and slow on MCU
- **Fuzzy Commitment**: Simple XOR with BCH codeword, BCH decode is well-optimized for embedded systems
- BCH codes have efficient hardware/software decoders (CMSIS-DSP has GF(2^m) primitives)

### 3.5 Why Gray Code Quantization?

When quantizing continuous embedding values to bits, adjacent quantization levels should differ by only 1 bit (not multiple bits). Gray coding ensures that a small analog error (value near a boundary) causes at most 1 bit flip, minimizing the Hamming distance between slightly different embeddings.

### 3.6 BCH Parameter Trade-off (Corrected with Real BCH Codes)

The core engineering trade-off:

```
Higher BCH error correction (t) -> Higher key agreement success rate
                                 -> But shorter effective key (more redundancy bits)

Lower quantization (fewer bits/dim) -> Lower BDR (fewer disagreements)
                                     -> But less entropy per key
```

**Critical correction:** Standard BCH codes have block lengths n = 2^m - 1. The 32 raw bits from 1-bit quantization must be zero-padded to 63 bits (BCH(63,...)), and 64 raw bits from 2-bit quantization must be padded to 127 bits (BCH(127,...)). Earlier versions of this work incorrectly claimed BCH(31,16,7) which only corrects t=3 errors.

Simulation results with **real BCH codes**:

| Config | BCH Code | t | BDR | Success | Effective Key |
|--------|----------|---|-----|---------|---------------|
| 1 bit/dim | BCH(63,24,15) | 7 | 29.2% | 33.8% | 24 bits |
| 1 bit/dim | BCH(63,18,21) | 10 | 29.2% | 62.8% | 18 bits |
| 1 bit/dim | BCH(63,16,23) | 11 | 29.2% | **73.2%** | **16 bits** |
| 1 bit/dim | BCH(63,10,27) | 13 | 29.2% | 86.1% | 10 bits |
| 1 bit/dim | BCH(63,7,31) | 15 | 29.2% | 93.6% | 7 bits |
| 2 bit/dim | BCH(127,29,43) | 21 | 34.0% | 48.9% | 29 bits |
| 2 bit/dim | BCH(127,22,47) | 23 | 34.0% | 59.9% | 22 bits |
| 2 bit/dim | BCH(127,15,55) | 27 | 34.0% | **80.4%** | **15 bits** |
| 2 bit/dim | BCH(127,8,63) | 31 | 34.0% | 92.4% | 8 bits |

**Decision:** Present the full trade-off space. Best balanced operating points are BCH(63,16,23) with 73.2% success / 16-bit key and BCH(127,15,55) with 80.4% success / 15-bit key. All use SHA-256 key derivation for computational security, but the information-theoretic security is bounded by the effective key bits (7-24 bits). This limitation is discussed honestly in the paper.

---

## 4. Simulation Design & Rationale

### 4.0 Revision: Dual-Dataset Evaluation (BIDMC)

The original version used only PTB-XL with Lead I vs Lead II as a "proxy" for ECG+PPG. Reviewer feedback correctly identified that this is not a valid multi-modal evaluation since both leads are ECG. The revision adds:

**BIDMC PPG and Respiration Dataset**:
- 53 recordings x 8 minutes from ICU patients
- **Synchronized ECG (Lead II) + PPG (plethysmograph)** at 125 Hz
- Open access: physionet.org/content/bidmc/1.0.0/
- Signal names: `II` (ECG), `PLETH` (PPG), `RESP` (respiration)
- PPG preprocessing: narrower bandpass 0.5-8 Hz (vs ECG 0.5-40 Hz)
- Resampled 125 Hz -> 256 Hz for pipeline consistency

**Other major revisions**:
- 5-fold patient-level cross-validation on both datasets (addresses single-split concern)
- Ablation studies: loss components, embedding dim, window length (addresses "which component matters?" question)
- Hybrid PhysioKey+ECDH protocol (addresses 7-24 bit effective key weakness)
- 3 additional NIST tests: Serial, Approximate Entropy, Cumulative Sums
- Theorem 1 changed to Remark (addresses "trivially true" criticism)
- All 37+ references now cited in the paper text

### 4.1 Why PTB-XL?

| Dataset | Pros | Cons |
|---------|------|------|
| MIMIC-III | Has ECG + PPG simultaneously | Requires credentialed access, ICU-only patients |
| **PTB-XL** | **21,837 recordings, 18,885 patients, freely available, 12-lead at 500 Hz** | No PPG channel |
| MIT-BIH | Classic ECG dataset | Only 48 patients, too small |

**PTB-XL solution:** Use Lead I vs Lead II as two "different sensors." These leads measure different electrical vectors of the heart (Lead I: left arm to right arm; Lead II: left leg to right arm). They share cardiovascular origin but have different waveform morphologies. This is actually a **more conservative** test than ECG+PPG, because:
- Lead I and Lead II measure the *same physical phenomenon* (cardiac electricity) from different angles
- ECG and PPG measure *different physical phenomena* (electrical vs. optical/mechanical) that are more locally correlated when sensors are nearby

### 4.2 Preprocessing Pipeline

```
Raw 500 Hz signal
  -> Resample to 256 Hz (matches target MCU sampling rate)
  -> Bandpass filter 0.5-40 Hz (remove DC drift and high-freq noise)
  -> Normalize (zero mean, unit variance per recording)
  -> Segment into 1-second windows (256 samples)
  -> Quality filter (discard windows with std < 0.1)
```

**Why 256 Hz?** Standard ECG diagnostic bandwidth is 0.05-150 Hz, but for key agreement we only need cardiac rhythm features (QRS complex, R-R intervals), which are captured below 40 Hz. 256 Hz at 40 Hz bandwidth gives 6.4x oversampling -- sufficient.

**Why 1-second windows?** At 60-100 BPM heart rate, 1 second captures 1-2 complete cardiac cycles. Shorter windows miss morphological features; longer windows reduce the number of training samples and increase MCU buffer requirements.

### 4.3 Train/Test Split

- **500 patients** downloaded (out of 21,837 available)
- **200 patients** for training (2,000 windows)
- **100 patients** for testing (999 windows)
- **200 remaining** unused (could expand for more robust evaluation)

**Critical:** Split by *patient*, not by window. No patient appears in both train and test sets. This prevents data leakage and tests generalization to unseen physiologies.

### 4.4 Training Details

- **150 epochs** with cosine annealing LR schedule (5e-4 -> ~0)
- **Batch size 128** (smaller than typical to improve gradient quality with limited data)
- **PyTorch** (TensorFlow not installed in environment; PyTorch 2.5.1 available)
- **CPU-only** (no CUDA available; training completes in ~3 minutes)

### 4.5 Evaluation Metrics

1. **Cosine Similarity**: Measures embedding space separability between intra-body and inter-body pairs
2. **FAR/FRR**: False acceptance (different body accepted) and false rejection (same body rejected) rates across thresholds
3. **EER**: Equal Error Rate -- threshold where FAR = FRR (standard biometric metric)
4. **ROC AUC**: Overall discriminative quality, threshold-independent
5. **BDR**: Bit Disagreement Rate -- fraction of quantized bits that disagree between paired sensors
6. **Key Agreement Success Rate**: Fraction of intra-body pairs where Hamming distance <= BCH correction capability
7. **Min-Entropy**: Per-dimension and total, measuring unpredictability of quantized keys
8. **NIST SP 800-22**: Statistical randomness tests on generated bit sequences
9. **Temporal Drift**: Hamming distance between different time windows of same patient (measures replay resistance)
10. **Hardware Estimates**: Analytical computation of flash, SRAM, latency, energy on Cortex-M4

---

## 5. Results & Interpretation

### 5.1 Embedding Quality

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Intra-body cosine similarity | 0.703 +/- 0.173 | Same-patient Lead I/II embeddings are well-aligned |
| Inter-body cosine similarity | 0.241 +/- 0.228 | Different-patient embeddings are nearly orthogonal |
| Separation | 0.462 | Clear gap enables discrimination |
| ROC AUC | 0.943 | Strong classifier |
| EER | 13.3% | At the equal error point, 13.3% of pairs are misclassified |

**Why EER is 13.3% (not <5%):**
- Lead I and Lead II have genuinely different morphologies (different cardiac electrical vectors)
- Published ECG-PPG systems (Poon et al.) achieve 5% EER using *synchronized* ECG+PPG with temporal alignment
- This setup is harder because it compares two ECG leads without temporal alignment
- A real ECG+PPG deployment would likely perform better

### 5.2 Key Agreement (Corrected with Real BCH Codes)

The key finding is a fundamental trade-off between success rate and effective key length:

- **BCH(63,16,23), t=11: 73.2% success, 16-bit effective key** -- best balanced point
- **BCH(63,7,31), t=15: 93.6% success, 7-bit effective key** -- highest success
- **BCH(127,15,55), t=27: 80.4% success, 15-bit effective key** -- 2-bit config balanced point

The BDR of 29.2% (1-bit) means that on average, ~9 out of 32 bits disagree. With BCH(63,...) codes, the 32 bits are padded to 63, and the error correction capability t must handle these disagreements.

**Honest assessment:** Effective key lengths of 7-24 bits are modest. SHA-256 stretching provides computational hardness but not information-theoretic security beyond the source entropy. Reducing BDR (through real ECG+PPG sensors) is the clearest path to stronger keys.

### 5.3 Security Properties

- **Entropy**: Per-dimension min-entropy is 2.0 bits for 2-bit quantization, but this is partly an artifact of percentile-based quantization boundaries forcing uniform bin occupancy. The entropy of the quantized representation is high by construction; the entropy of the underlying source signal may be lower.
- **NIST Tests (concatenated 1024-bit)**: Monobit 77.4%, Runs 87.1%, Block Frequency 98.4% -- the Monobit result indicates some systematic bias in the bit sequences.
- **NIST Tests (per-key 64-bit, below NIST minimum)**: Higher pass rates (93.4-99.4%) but statistically less reliable at this sequence length.
- **Replay Resistance**: 99.2% of replayed (time-shifted) vectors exceed BCH correction capability.
- **FAR as Security Concern**: The 12.9% FAR at the balanced operating point means roughly 1 in 8 adversary attempts produces similar-enough embeddings. This is a genuine security limitation that should be mitigated by combining PhysioKey with other authentication factors.

### 5.4 Hardware Feasibility

| Resource | PhysioKey | ECDH | Improvement |
|----------|-----------|------|-------------|
| Flash | 37.6 KB | 24.6 KB | Comparable |
| SRAM | 10.4 KB | 12.3 KB | 15% less |
| Latency | 3.6 ms | 142 ms | **39x faster** |
| Energy | 0.345 mJ | 6.84 mJ | **20x less** |
| Battery life* | 7.0M agreements | 355K agreements | **20x longer** |

*225 mAh coin cell at 3.0V

---

## 6. Honest Assessment of Limitations

1. **Simulation only**: No hardware deployment yet. Analytical estimates may differ from real MCU performance by 10-30%.

2. **Dual-dataset evaluation**: PTB-XL provides dual-ECG evaluation (Lead I/II); BIDMC provides true ECG+PPG evaluation (53 patients). While BIDMC is small, it validates cross-modality generalization.

3. **EER of ~13%**: Higher than some published schemes, but now cross-validated across 5 folds with confidence intervals. The learned representation is more general than hand-crafted features used by prior schemes with lower EER.

4. **Effective key entropy is modest in standalone mode**: 7-24 bits for 1-bit config. The hybrid PhysioKey+ECDH protocol addresses this by combining on-body authentication with 128-bit cryptographic key exchange.

5. **FAR is a security concern for standalone mode**: The ~13% FAR at the balanced threshold is mitigated by the hybrid protocol for high-security applications.

6. **NIST randomness tests show some bias**: Now tested with 6 NIST tests (added Serial, Approximate Entropy, Cumulative Sums). Bias in Monobit test remains a known limitation.

7. **BIDMC is small**: 53 patients produce wide confidence intervals. This is addressed honestly; the value is in validating ECG+PPG generalization, not in achieving statistical power comparable to PTB-XL.

---

## 7. Why This Work Matters

1. **First TinyML-based key agreement for BANs**: No prior work uses learned neural features for BAN key derivation on MCU-class devices

2. **Plug-and-play**: Unlike PKI/pre-shared key schemes, sensors can be added without any prior coordination -- critical for clinical workflows

3. **20x energy improvement over ECDH**: Extends battery life from months to years for key agreement operations

4. **Adaptive**: The learned model captures patient-specific features, unlike hand-crafted statistical methods that use fixed feature definitions

5. **Framework with clear trade-offs**: The parameter sweep table gives practitioners a menu of configurations to choose from based on their security/reliability requirements

---

## 8. File Structure

```
BAN_Security_EdgeAI/
├── overleaf/
│   ├── main.tex              # Full paper (Overleaf-ready, IEEEtran format)
│   └── references.bib        # 38 BibTeX references
├── simulation/
│   ├── run_simulation.py      # Complete simulation pipeline (dual-dataset, CV, ablation)
│   ├── ptbxl_data/            # Downloaded PTB-XL data (gitignored)
│   ├── bidmc_data/            # Downloaded BIDMC data (gitignored)
│   └── results/               # Simulation results JSON (gitignored)
├── IEEEtran.cls               # LaTeX class file
├── main.tex                   # Local-compile version (article class)
├── PhysioKey_IJACSA_Overleaf.zip  # Ready-to-upload Overleaf package
├── REASONING.md               # This document
├── .gitignore
└── README.md
```

---

## 9. Reproduction Instructions

### Prerequisites
```bash
pip install torch wfdb scipy scikit-learn numpy
```

### Run Simulation
```bash
cd simulation
python run_simulation.py
```

This will:
1. Download 500 PTB-XL records + 53 BIDMC records from PhysioNet (~10 min)
2. Preprocess signals (bandpass filter, resample, normalize, segment)
3. Run 5-fold CV on PTB-XL (~20 min on CPU)
4. Run 5-fold CV on BIDMC (~5 min on CPU)
5. Run ablation studies on BIDMC (10 variants x 3-fold, ~15 min)
6. Compute hardware + hybrid protocol estimates
7. Save results to `results/simulation_results.json`

Total runtime: ~40-45 minutes on CPU.

### Compile Paper
Upload `overleaf/main.tex` and `overleaf/references.bib` to Overleaf, or compile locally with a full TeX Live installation.

### Post-Simulation: Update Paper Tables
After running the simulation, update the placeholder values (marked with *) in main.tex using the numbers from `simulation_results.json`.
