# Overcoming Topographic Inversion in Monocular Lunar Imagery

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Metric](https://img.shields.io/badge/Metric-Balanced%20Accuracy-green.svg)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)
[![Paper](https://img.shields.io/badge/Preprint-arXiv%20%2F%20IEEE-red.svg)](paper.tex)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An end-to-end, physics-grounded machine learning framework designed to resolve the crater-mound shape-from-shading ambiguity (*topographic pareidolia*) in monocular lunar orbital imagery. The pipeline classifies ambiguous lunar landforms into **Class 0 (Depression / Crater)** versus **Class 1 (Elevation / Mound)** using solar azimuth normalization, 3-channel slope/curvature tensors, and multi-scale ensemble spatial gating.

---

## Academic Manuscript & Citation

The methodology, mathematical derivations, and ablation findings are documented in the companion research paper:

* **Title:** *Overcoming Topographic Inversion in Monocular Lunar Imagery via Solar-Aligned 3-Channel Physics Tensors and Gated Multi-Scale Ensembling*
* **Author:** Aayush Chavanke
* **LaTeX Source:** [`paper.tex`](paper.tex)
* **Bibliography:** [`references.bib`](references.bib)
* **Markdown Preprint:** [`PAPER.md`](PAPER.md)

---

## Executive Summary & Computational Scale

| Metric / Dimension | Value |
| :--- | :--- |
| **Total Processed Dataset** | 9,854 orbital lunar crops (7,854 labeled ground-truth + 2,000 unlabelled evaluation scenes) |
| **Domain-Adapted Training Set** | 8,182 samples (7,854 ground-truth + 328 high-confidence pseudo-labels) |
| **Cumulative Training Passes** | 1,387,405 forward / backward passes across 16 neural models |
| **Stratified 5-Fold Balanced Accuracy** | **71.88% ± 0.78%** (Peak single fold: 73.05%) |
| **Inference Density** | 80,000 total forward passes (40 passes per sample across 5 folds) |
| **Calibrated Decision Threshold** | $\tau^* = 0.4950$ (Optimized for Balanced Accuracy) |
| **Hardware Platform** | NVIDIA GeForce RTX GPU (CUDA accelerated) |
| **Compute Time** | 2.66 hours dedicated neural training |

---

## Methodology & Architectural Overview

```
Raw Lunar Scene (256x256) + Solar Azimuth (θ)
                     │
                     ▼
[Reflection-Padded Affine Rotation: -θ]  --> Enforces Canonical Sun Angle at North (y = 0)
                     │
                     ▼
[3-Channel Topographic Physics Tensor]    --> T = [Albedo I, Slope ∇y I, Curvature ∇²I]
                     │
                     ▼
[5-Fold Stratified ConvNeXt Ensemble]     --> 5 Independent Deep Feature Extractors
                     │
                     ▼
[40-Pass Deep Test-Time Augmentation]     --> 8 Symmetry-Preserving Views × 5 Folds
                     │
                     ▼
[Gated Multi-Scale Spatial Profiler]      --> Fuses Full (256), Mid (192), and Core (160)
                     │
                     ▼
[Calibrated Threshold (τ* = 0.4950)]      --> Class 0 (Crater) / Class 1 (Mound)
```

### 1. Illumination Normalization & Reflection Padding
Under monocular orbital observation, reversing the direction of illumination inverts the perceived relief of a surface. By rotating each image by $-\theta_{\text{azimuth}}$ using reflection padding ($N_{\text{pad}} = 128$ pixels), we align the subsolar illumination vector strictly along the negative vertical axis ($y = -\infty$ / North).

```
Canonical Solar-Aligned Geometry (Sun Positioned at Top):

Depression (Crater)                     Elevation (Mound)
┌─────────────────────────┐             ┌─────────────────────────┐
│ ████ SHADOW REGION ████ │ (Sunward)   │ ░░░░ LIGHT REGION ░░░░  │ (Sunward)
│                         │             │                         │
│ ░░░░ LIGHT REGION ░░░░  │ (Rim)       │ ████ SHADOW REGION ████ │ (Cast)
└─────────────────────────┘             └─────────────────────────┘
```

### 2. Differential 3-Channel Physics Tensors
Rather than triplicating grayscale intensities, each sample is converted into a 3-channel differential tensor $\mathbf{T}(x, y) \in \mathbb{R}^{3 \times H \times W}$:
1. **Channel 0 ($I$):** Normalized surface radiance $I(x, y) \in [0, 1]$.
2. **Channel 1 ($\nabla_y I = \frac{\partial I}{\partial y}$):** Vertical directional derivative, encoding local surface slope normal ($\hat{n} \cdot \hat{s}$) relative to the subsolar vector.
3. **Channel 2 ($\nabla^2 I$):** Spatial Laplacian operator, isolating circular rim crests and break-of-slope terrain discontinuities.

---

## Experimental Results & Ablation Analysis

### 5-Fold Cross-Validation Performance

| Cross-Validation Fold | Validation Loss | Accuracy | Balanced Accuracy | ROC AUC |
| :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 0.5510 | 74.28% | 70.90% | 0.782 |
| **Fold 1** | 0.5342 | 75.81% | 73.05% | 0.801 |
| **Fold 2** | 0.5401 | 75.14% | 72.37% | 0.794 |
| **Fold 3** | 0.5582 | 74.60% | 71.20% | 0.779 |
| **Fold 4** | 0.5489 | 75.09% | 71.90% | 0.789 |
| **Mean ± Std** | **0.5465 ± 0.008** | **74.98% ± 0.52%** | **71.88% ± 0.78%** | **0.789 ± 0.008** |

### Progressive Architectural Ablation

| Experimental Stage | Architectural Configuration | OOF Balanced Accuracy | Stability Rate |
| :--- | :--- | :---: | :---: |
| Stage 1 | Single ConvNeXt Baseline ($[I,I,I]$) | 68.42% | 82.1% |
| Stage 2 | + Solar Azimuth Alignment ($R(-\theta)$) | 70.15% | 88.6% |
| Stage 3 | + 5-Fold Stratified Ensemble | 71.12% | 92.4% |
| Stage 4 | + 3-Channel Physics Tensors ($I, \nabla_y I, \nabla^2 I$) | 71.88% | 95.7% |
| **Stage 5** | **+ Domain Adaptation + Gated 40-Pass TTA** | **72.90%** | **96.8%** |

---

## Repository Structure

```
.
├── train.py                    # Official 5-fold training entry point
├── inference.py                # Official 40-pass TTA inference engine
├── dataset.py                  # 3-Channel differential physics dataset loader
├── model.py                    # ConvNeXt / ResNet architecture definitions
├── train_grandmaster.py        # Grandmaster domain-adapted training script
├── inference_grandmaster.py    # Deep TTA + Gated multi-scale inference engine
├── requirements.txt            # Python environment dependencies
├── paper.tex                   # Academic research paper (LaTeX source)
├── references.bib              # Complete BibTeX citations
├── PAPER.md                    # Markdown version of research paper
├── submission.csv              # Final verified competition submission (2,000 samples)
├── train_metadata.csv          # Training ground-truth metadata
├── test_metadata.csv           # Evaluation set metadata
├── run_training.bat            # One-click execution script for training
└── run_inference.bat           # One-click execution script for inference
```

---

## Quickstart & Reproduction

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Train 5-Fold Physics Ensemble
```bash
python train.py
# or run: run_training.bat
```

### 3. Generate Submission with 40-Pass Deep TTA
```bash
python inference.py
# or run: run_inference.bat
```
The output file [`submission.csv`](submission.csv) will be generated with exactly 2,000 evaluated samples formatted for evaluation.

---

## License
This project is open-source under the [MIT License](LICENSE).
