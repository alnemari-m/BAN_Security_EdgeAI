# PhysioKey: Edge-AI-Driven Physiological Key Agreement for Secure Body Area Networks

A TinyML approach using ECG and PPG signals for lightweight, plug-and-play cryptographic key agreement in Body Area Networks (BANs).

## Overview

Body Area Networks in healthcare deploy multiple wireless sensors on a patient's body to monitor vital signs. Securing communication between these sensors is critical but challenging due to extreme resource constraints (ARM Cortex-M class MCUs, coin-cell batteries). Traditional approaches like PKI (ECC-256 ECDH) are too expensive, and pre-shared keys don't scale in clinical environments.

**PhysioKey** leverages the insight that sensors on the same body measure physiologically correlated signals. A lightweight 1D-CNN (6,320 parameters, 6.2 KB at INT8) learns discriminative embeddings from ECG/PPG signals, which are then used for cryptographic key derivation via a fuzzy commitment scheme with BCH error correction.

## Key Results

| Metric | Value |
|--------|-------|
| Model Size | 6.2 KB (INT8) |
| Inference Latency | 3.6 ms (Cortex-M4 @ 64 MHz) |
| Energy per Key Agreement | 0.345 mJ (20x less than ECDH) |
| ROC AUC | 0.945 |
| EER | 12.7% |
| Key Agreement Success | 94.5% (1-bit/dim, BCH t=15) |
| NIST Randomness Tests | 94.5-99.0% pass rate |
| Replay Attack Failure | 99.2% |

## Repository Structure

```
BAN_Security_EdgeAI/
├── overleaf/
│   └── main.tex              # Full paper (Overleaf-ready, IEEEtran format)
├── simulation/
│   ├── run_simulation.py      # Complete simulation pipeline
│   ├── ptbxl_data/            # Downloaded PTB-XL data (gitignored)
│   └── results/               # Simulation results JSON (gitignored)
├── references.bib             # 35 BibTeX references
├── IEEEtran.cls               # IEEE LaTeX class file
├── main.tex                   # Local-compile version (article class)
├── PhysioKey_IJACSA_Overleaf.zip  # Ready-to-upload Overleaf package
├── REASONING.md               # Detailed research reasoning & process
├── .gitignore
└── README.md
```

## Running the Simulation

### Prerequisites

```bash
pip install torch wfdb scipy scikit-learn numpy
```

### Execute

```bash
cd simulation
python run_simulation.py
```

This will:
1. Download 500 PTB-XL records from PhysioNet (~5 min)
2. Preprocess signals (bandpass filter, resample, normalize, segment)
3. Train the 1D-CNN with contrastive + alignment + decorrelation loss (150 epochs, ~3 min on CPU)
4. Compute all evaluation metrics (cosine similarity, FAR/FRR, EER, entropy, NIST tests, key agreement sweep)
5. Save results to `results/simulation_results.json`

### Dataset

Uses the [PTB-XL dataset](https://physionet.org/content/ptb-xl/) (Lead I vs Lead II as ECG sensor proxy). This is a conservative evaluation -- real ECG+PPG deployment would likely yield better results due to stronger hemodynamic correlation between co-located sensors.

## Paper

**Title:** Edge-AI-Driven Physiological Key Agreement for Secure Body Area Networks: A TinyML Approach Using ECG and PPG Signals

**Author:** Mohammed Alnemari, University of California, Irvine

**Target Venue:** IJACSA (International Journal of Advanced Computer Science and Applications)

The paper is available in `overleaf/main.tex` (IEEEtran format) or as `PhysioKey_IJACSA_Overleaf.zip` for direct Overleaf upload.

## Research Process

See [REASONING.md](REASONING.md) for a detailed account of the research reasoning, design decisions, simulation methodology, and honest assessment of limitations.

## License

This project is for academic research purposes.
