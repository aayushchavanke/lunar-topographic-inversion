# The Pareidolia Paradox: Solving Lunar Topographic Inversion

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Metric](https://img.shields.io/badge/Metric-Balanced%20Accuracy-green.svg)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)
[![Inference](https://img.shields.io/badge/Ensemble-40%20Passes%20%2B%203Ch%20Physics-purple.svg)](https://github.com/aayushchavanke/lunar-topographic-inversion)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An end-to-end, physics-grounded machine learning system designed to resolve the crater-mound shape-from-shading ambiguity (*topographic pareidolia*) in monocular lunar orbital imagery. The pipeline classifies ambiguous lunar landforms into **Class 0 (Depression / Crater)** versus **Class 1 (Elevation / Mound)** evaluated on **Balanced Accuracy** ($\frac{\text{Recall}_0 + \text{Recall}_1}{2}$).

---

## Executive Summary & Computational Benchmarks

| Metric / Parameter | Value / Measured Benchmark |
| :--- | :--- |
| **Total Processed Dataset** | 9,854 orbital lunar crops (7,854 ground-truth + 2,000 blind evaluation scenes) |
| **Domain-Adapted Training Set** | 8,182 samples (7,854 ground-truth + 328 high-confidence pseudo-labels) |
| **Cumulative Neural Training Passes** | 1,387,405 forward / backward passes across 16 neural models |
| **Stratified 5-Fold Balanced Accuracy** | **71.88% ± 0.78%** (Peak single-fold: 73.05%) |
| **Inference Density** | 80,000 total forward passes (40 passes per sample across 5 folds) |
| **Calibrated Decision Threshold** | $\tau^* = 0.4950$ (Optimized for Balanced Accuracy) |
| **Hardware Platform** | NVIDIA GeForce RTX GPU (CUDA accelerated) |
| **Dedicated GPU Compute Time** | 2.66 hours |

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

### 1. Optical Illumination Normalization
Under monocular observation without stereoscopy, reversing illumination inverts perceived 3D surface relief. Rotating each crop by $-\theta_{\text{azimuth}}$ with reflection padding ($N_{\text{pad}} = 128$ px) locks the subsolar vector strictly to North ($y = 0$).

![Optical Illumination Geometry](figures/fig1_optical_geometry.png)

### 2. Differential 3-Channel Physics Tensors
Rather than triplicating grayscale intensities ($[I, I, I]$), each sample is converted into a 3-channel differential tensor $\mathbf{T}(x, y) \in \mathbb{R}^{3 \times H \times W}$:
1. **Channel 0 ($I$):** Normalized surface radiance $I(x, y) \in [0, 1]$.
2. **Channel 1 ($\nabla_y I = \frac{\partial I}{\partial y}$):** Vertical directional derivative, encoding surface slope normal ($\hat{n} \cdot \hat{s}$) relative to the subsolar vector.
3. **Channel 2 ($\nabla^2 I$):** Spatial Laplacian operator isolating circular rim crests and break-of-slope terrain discontinuities.

![3-Channel Topographic Physics Tensors](figures/fig2_physics_tensors.png)

---

## Experimental Results & Validation Metrics

### 5-Fold Stratified Cross-Validation

| Cross-Validation Fold | Validation Loss | Accuracy | Balanced Accuracy | Crater Recall | ROC AUC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 0.5510 | 74.28% | 70.90% | 64.12% | 0.782 |
| **Fold 1** | 0.5342 | 75.81% | 73.05% | 67.54% | 0.801 |
| **Fold 2** | 0.5401 | 75.14% | 72.37% | 66.82% | 0.794 |
| **Fold 3** | 0.5582 | 74.60% | 71.20% | 64.78% | 0.779 |
| **Fold 4** | 0.5489 | 75.09% | 71.90% | 65.91% | 0.789 |
| **Mean ± Std** | **0.5465 ± 0.008** | **74.98% ± 0.52%** | **71.88% ± 0.78%** | **65.83% ± 1.22%** | **0.789 ± 0.008** |

### Decision Threshold Calibration ($\tau^* = 0.4950$)
Because the evaluation metric is Balanced Accuracy, the decision threshold $\tau^*$ is optimized across Out-of-Fold validation predictions:

![ROC Curves and Threshold Optimization](figures/fig3_roc_threshold_calibration.png)

---

## Repository Structure

```
.
├── train.py                    # 5-fold training entry point
├── inference.py                # 40-pass deep TTA inference engine
├── dataset.py                  # 3-Channel differential physics dataset loader
├── model.py                    # ConvNeXt / ResNet architecture definitions
├── train_grandmaster.py        # Grandmaster domain-adapted training script
├── inference_grandmaster.py    # Deep TTA + Gated multi-scale inference engine
├── requirements.txt            # Python environment dependencies
├── submission.csv              # Official competition submission (2,000 samples)
├── train_metadata.csv          # Training ground-truth metadata
├── test_metadata.csv           # Evaluation set metadata
├── figures/                    # Scientific diagrams & visual figures
├── run_training.bat            # Batch execution script for training
└── run_inference.bat           # Batch execution script for inference
```

---

## Quickstart & Reproduction

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Train 5-Fold Models
```bash
python train.py
# or: run_training.bat
```

### 3. Generate Verified Submission
```bash
python inference.py
# or: run_inference.bat
```
Generates [`submission.csv`](submission.csv) containing exactly 2,000 predictions (408 Craters / 1,592 Mounds) with 0 nulls.

---

## License
This project is open-source under the [MIT License](LICENSE).
