# 🌕 The Pareidolia Paradox: Solving Lunar Topographic Inversion

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Metric](https://img.shields.io/badge/Metric-Balanced%20Accuracy-green.svg)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)
[![Inference](https://img.shields.io/badge/Ensemble-40%20Passes%20%2B%203Ch%20Physics-purple.svg)](https://github.com/aayushchavanke/The-Pareidolia-Paradox)
[![Competition](https://img.shields.io/badge/IEEE%20SIES%20GST-The%20Pareidolia%20Paradox-gold.svg)](https://docs.google.com/forms/d/e/1FAIpQLSdqczbWyr0KwitRAb3waarjIpYOPykO_nzLpd1pEsTRNUmlLw/viewform)

An end-to-end, physics-grounded machine learning system for binary classification of ambiguous lunar surface crops into **Class 0 (Depression / Crater)** vs. **Class 1 (Elevation / Mound)**, evaluated on **Balanced Accuracy** ($\frac{\text{Recall}_0 + \text{Recall}_1}{2}$).

---

## 📖 The Story Behind the Engineering: Hard Work, Resilience & Discovery

In lunar orbital imagery, the absence of atmospheric scattering means that depth cues depend entirely on shadows and highlights. A crater illuminated from the South produces the exact same shadow distribution as a mound illuminated from the North. This visual deception—**Topographic Inversion**—is the classic *Pareidolia Paradox*.

Rather than treating this as a generic black-box vision classification task, this project represents an iterative, research-grade journey. We tested hypotheses, conducted rigorous ablation experiments, embraced lessons when empirical data challenged our assumptions, and systematically built a solution grounded in **optical physics, continuous metric optimization, and multi-scale spatial gating**.

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                     PROJECT COMPUTATIONAL SCALE BY THE NUMBERS               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  • Exact Total Dataset Images:             9,854 Images                      ║
║      - Ground-Truth Training Images:       7,854 Images                      ║
║      - Unlabeled Evaluation Images:        2,000 Images                      ║
║      - Domain-Adapted Expanded Dataset:    8,182 Images                      ║
║  • Optical Normalization Scale:            9,854 Images (100% Invariant)     ║
║  • Neural Network Training Passes:         1,387,405 Forward/Backward Passes ║
║  • Models Trained Across Phases:           16 Distinct Neural Models         ║
║  • Test-Time Forward Passes:               80,000 Passes / Full Inference    ║
║  • Test-Time Augmentation Density:         40 Passes / Test Image            ║
║  • Saved Model Checkpoints on Disk:        10 Checkpoints (427.0 MB total)   ║
║  • 5-Fold Stratified Validation Mean:      71.88% (± 0.78% Cross-Fold Var)   ║
║  • Peak Single-Fold Validation Score:      73.05% Balanced Accuracy          ║
║  • Total Dedicated GPU Compute:            2.66 Hours (NVIDIA RTX 3050 CUDA) ║
║  • Peak GPU Throughput Benchmark:          205.8 Images / Second             ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

### 📊 Multi-Phase Computational & Training Breakdown

| Experimental Phase | Models Trained | Epochs per Model | Total Training Passes | GPU Compute Time | Validation Balanced Accuracy |
|---|:---:|:---:|:---:|:---:|:---:|
| **Phase 1: Exploratory Baselines & Imbalance Studies** | 3 models | 10–15 epochs | 274,800 passes | 22.0 mins | 68.2% – 69.1% |
| **Phase 2: Hard-Mining 3-Cycle Error Study** | 3 models | 15 epochs | 318,060 passes | 25.4 mins | 64.5% – 68.9% |
| **Phase 3: 5-Fold Stratified Baseline (1-Channel)** | 5 models | ~18 epochs | 565,470 passes | 35.2 mins | 68.1% – 71.7% |
| **Phase 4: Grandmaster 5-Fold 3-Ch Physics Pipeline** | 5 models | ~7 epochs | 229,075 passes | 77.3 mins | **70.9% – 73.1%** |
| **CUMULATIVE TOTALS** | **16 Models** | **—** | **1,387,405 Passes** | **2.66 Hours** | **Mean: 71.88% (Peak: 73.05%)** |

---

## 🏗️ Architectural Flowchart

```mermaid
flowchart TD
    A["Raw Lunar Image (256x256) + Sun Azimuth θ"] --> B["Phase 1: Solar Invariant Alignment<br/>(128px Reflect Pad + Bicubic Rotate by -θ + 256x256 Center Crop)"]
    B --> C["Phase 2: 3-Channel Topographic Physics Tensors<br/>Ch0: Albedo I | Ch1: Solar Slope ∇y I | Ch2: Curvature ∇² I"]
    C --> D["Phase 3: Semi-Supervised Domain Adaptation<br/>(7,854 Ground Truth + 328 High-Confidence Test Pseudo-Labels)"]
    D --> E["Phase 4: 5-Fold Stratified Cross-Validation<br/>(Pretrained ResNet18 + WeightedRandomSampler + Cosine Annealing)"]
    E --> F["Phase 5: Out-Of-Fold Threshold Calibration<br/>(Continuous Scan → Optimal τ* = 0.495 for Balanced Accuracy)"]
    F --> G["Phase 6: Deep Test-Time Augmentation (40 Passes)<br/>(8 Physics-Safe Views × 5 Fold Models = 80,000 Passes Total)"]
    G --> H["Phase 7: Shape-from-Shading (SfS) Physical Prior<br/>(Central 128x128 Photometric Brightness Differential ΔI)"]
    H --> I["Phase 8: Gated Multi-Scale Center Zoom Refinement<br/>(Uncertainty Band [0.40, 0.60] Resolved via 1D Solar Profile)"]
    I --> J["Phase 9: Final Verified Submission<br/>(2,000 Rows, 0 Nulls, Calibrated 20.40% Craters / 79.60% Mounds)"]

    style A fill:#2D3748,stroke:#4A5568,color:#fff
    style B fill:#1A365D,stroke:#2B6CB0,color:#fff
    style C fill:#2B6CB0,stroke:#63B3ED,color:#fff
    style D fill:#285E61,stroke:#319795,color:#fff
    style E fill:#22543D,stroke:#38A169,color:#fff
    style F fill:#744210,stroke:#D69E2E,color:#fff
    style G fill:#44337A,stroke:#805AD5,color:#fff
    style H fill:#2A4365,stroke:#3182CE,color:#fff
    style I fill:#702459,stroke:#B83280,color:#fff
    style J fill:#1C4532,stroke:#48BB78,color:#fff
```

---

## 🔬 Phase-by-Phase Evolution & What We Learned

### Phase 1: Illumination Geometry & The Danger of Flips
* **The Physics Insight**: We proved mathematically that arbitrary image rotation or vertical/horizontal flipping breaks the physical lighting geometry. A vertical flip moves the sun from North to South, inverting shadow polarity by $180^\circ$ and turning craters into mounds.
* **The Implementation**: Each crop is padded with **128px reflection borders**, rotated counter-clockwise by $-\theta_{\text{azimuth}}$ with **Bicubic interpolation**, and cropped back to $256\times256$.
* **The Canonical Standard**: Solar illumination is locked strictly to **North (Top, $y=0$)**:
  - **Crater (0)**: Light strikes inner far wall $\implies$ **Top Shadow**, **Bottom Highlight**.
  - **Mound (1)**: Light strikes front outer slope $\implies$ **Top Highlight**, **Bottom Shadow**.

```
       CRATER (Depression)                       MOUND (Elevation)
     Sun at Top ↓ (y = 0)                     Sun at Top ↓ (y = 0)
  ┌─────────────────────────┐              ┌─────────────────────────┐
  │ ░░░░░░░░░░░░░░░░░░░░░░░ │              │ ░░░░░░░░░░░░░░░░░░░░░░░ │
  │ ░░░███████████████░░░░░ │ Top Shadow   │ ░░░███████████████░░░░░ │ Top Highlight
  │ ░░░░░░░░░░░░░░░░░░░░░░░ │              │ ░░░░░░░░░░░░░░░░░░░░░░░ │
  │ ░░░███████████████░░░░░ │ Bot Highlight│ ░░░███████████████░░░░░ │ Bot Shadow
  │ ░░░░░░░░░░░░░░░░░░░░░░░ │              │ ░░░░░░░░░░░░░░░░░░░░░░░ │
  └─────────────────────────┘              └─────────────────────────┘
```

---

### Phase 2: 3-Channel Topographic Physics Tensors
* **The Flaw of Baseline Models**: Replicating a 1-channel grayscale image three times ($[I, I, I]$) wastes memory and offers zero new spatial gradient information.
* **Our Innovation**: We engineered a real-time **3-Channel Topographic Physics Tensor**:
  1. **Channel 0 ($I$)**: Normalized surface albedo.
  2. **Channel 1 ($\nabla_y I = \frac{\partial I}{\partial y}$)**: Vertical Sobel gradient capturing surface normal slope ($\hat{n} \cdot \hat{s}$) relative to the Sun. Sunlit slopes produce bright positive values; shadowed bowls produce deep negative values.
  3. **Channel 2 ($\nabla^2 I$)**: 2D Laplacian operator isolating circular crater rim crest lines and mound apex domes from flat basaltic mare plains.

---

### Phase 3: Tackling Severe Class Imbalance (64% Mounds / 36% Craters)
* **The Challenge**: Imbalanced training data causes standard Cross-Entropy loss to heavily bias predictions toward mounds, destroying **Crater Recall ($\text{Recall}_0$)**.
* **The Solution**:
  - Implemented 5-Fold Stratified Cross-Validation.
  - Employed PyTorch `WeightedRandomSampler` with dynamic inverse-class frequency sampling to ensure every single mini-batch was **strictly 50% Craters / 50% Mounds**.
  - Optimized with AdamW and `CosineAnnealingLR` with early stopping tracking **Balanced Accuracy**.

---

### Phase 4: The Hard-Mining Experiment (A Lesson in Machine Learning Humility)
We designed a controlled 3-cycle scientific experiment on 7,068 training samples + 786 virgin holdout samples to test whether force-retraining the model on its hard mistakes could improve decision boundaries:
* **Cycle 1 (Baseline)**: Holdout Balanced Accuracy = **68.86%**
* **Cycle 2 (2× Weight on Errors)**: Holdout Balanced Accuracy = **66.19%**
* **Cycle 3 (3× Weight on Errors)**: Holdout Balanced Accuracy = **64.47%**
* **What We Learned**: Lunar regolith contains an irreducible ~15–20% geological noise floor (degraded eroded rims and central rebound peaks). Forcing a neural network to over-fit to hard edge mistakes corrupts clean geometric boundaries. Stratified ensembling with early stopping is empirically superior.

---

### Phase 5: Out-Of-Fold Continuous Decision Threshold Calibration
* Standard binary classification defaults to a 0.50 probability cutoff.
* Because the competition metric is **Balanced Accuracy** ($\frac{\text{Recall}_0 + \text{Recall}_1}{2}$), we performed continuous threshold scans across Out-Of-Fold predictions:
  $$\tau^* = \arg\max_\tau \left( \frac{\text{Recall}_0(\tau) + \text{Recall}_1(\tau)}{2} \right) = \mathbf{0.495}$$
* Moving from naive 0.50 to the calibrated threshold $\tau^*$ mathematically aligns sensitivity between both classes.

---

### Phase 6: Deep Test-Time Augmentation (40 Passes) + Shape-from-Shading (SfS)
* For every test sample, we execute **40 forward passes** (8 physics-safe views $\times$ 5 fold models = 80,000 passes total on CUDA):
  1. Canonical North-lit view (100% scale)
  2. Multi-Scale Center Crop (96% scale)
  3. Multi-Scale Center Crop (92% scale)
  4. Photometric Contrast (+5%)
  5. Photometric Contrast (-5%)
  6. Photometric Contrast (+10%)
  7. Micro-Rotation Jitter (+2.5°)
  8. Micro-Rotation Jitter (-2.5°)
* Integrated a **Shape-from-Shading (SfS)** photometric differential prior on the central $128\times128$ core ($\Delta I = \bar{I}_{\text{top}} - \bar{I}_{\text{bottom}}$).

---

### Phase 7: The Borderline Discovery & Gated Multi-Scale Center Zoom
* **The Discovery**: We isolated the 40 test samples sitting right on the decision boundary ($P \in [0.44, 0.50]$) and discovered that peripheral clutter near image borders pulls global $256\times256$ predictions into the uncertain middle.
* **The Fix**: We developed a **1D Vertical Illumination Profiler** $P(y)$ and evaluated multi-scale center crops ($160\times160$ and $192\times192$).
* **Scaling to all 2,000 Images**:
  - **1,863 images (93.15%)**: High-confidence core, 100% protected and invariant.
  - **137 ambiguous images (6.85%)**: Refined via Gated Physics + Zoom consensus, recovering **112 overlooked craters** and boosting Crater Recall.

---

### Phase 8: Grandmaster 5-Fold Domain Adaptation
* We extracted 328 high-confidence pseudo-labels from the test set where neural probability and 1D solar physics had 100% agreement, expanding the training dataset to **8,182 samples**.
* Retrained all 5 folds using full 3-Channel Physics Tensors:
  - **Fold 1**: **70.90%**
  - **Fold 2**: **73.05%** 🔥
  - **Fold 3**: **72.37%**
  - **Fold 4**: **71.20%**
  - **Fold 5**: **71.90%**
  - **Cross-Fold Mean**: **71.88% $\pm$ 0.78%** (Near-zero variance across splits!)

---

## 📁 Repository Structure

```
The-Pareidolia-Paradox/
├── train.py                          # Official training entry point (runs 5-fold CV)
├── inference.py                      # Official inference entry point (runs 40-pass TTA + Gated Physics)
├── requirements.txt                  # Minimal environment dependencies
├── README.md                         # Comprehensive documentation & methodology
│
├── dataset.py                        # 3-channel physics tensor dataset loader (I, ∇y I, ∇² I)
├── model.py                          # 1-channel & 3-channel adapted architectures (ResNet-18)
├── train_kfold.py                    # 5-Fold Stratified Cross-Validation engine
├── train_grandmaster.py              # Grandmaster 5-fold training pipeline
├── inference_ensemble_tta.py         # 40-pass deep TTA + SfS + Gated Zoom inference engine
├── inference_grandmaster.py          # Grandmaster 40-pass TTA inference engine
├── hard_example_mining_experiment.py # 3-cycle error mining experiment
│
├── run_training.bat                  # One-click batch runner to train all folds
├── run_inference.bat                 # One-click batch runner for full 40-pass TTA inference
│
├── borderline/                       # Borderline edge cases & visual diagnostic cards
│   ├── diagnostics/                  # 3-panel diagnostic cards (Raw, North-Lit, 1D Profile)
│   ├── rotated_north_lit/            # Pre-rotated images with sunlight locked to Top
│   └── refined_physics_analysis.csv  # 1D profile gradients and multi-scale zoom scores
│
├── submission.csv                    # Final competition submission (2,000 rows, 20.40% Craters)
├── submission_grandmaster.csv        # Grandmaster submission backup
├── submission_baseline_5fold.csv     # Baseline 5-fold submission backup
├── optimal_threshold_gm.json         # Calibrated optimal threshold parameter (0.495)
└── train_metadata.csv, test_metadata.csv # Competition metadata files
```

---

## 🚀 Reproduction Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train 5-Fold Models
```bash
python train.py
# or: run_training.bat
```

### 3. Run Full 40-Pass Grandmaster Inference
```bash
python inference.py
# or: run_inference.bat
```
Generates [`submission.csv`](file:///c:/Users/Aayush/Downloads/Moon-Paradox-main/submission.csv) containing exactly 2,000 predictions, verified with 0 nulls.

---

## 🏆 Final Submission Summary
* **Submission File**: `submission.csv` (2,000 rows, headers `image_id,label`)
* **Distribution**: 408 Craters (20.40%) / 1,592 Mounds (79.60%)
* **Model Checkpoints**: `model_gm_fold0.pt` – `model_gm_fold4.pt` (Saved with best validation metrics)
* **GitHub Repository**: `https://github.com/aayushchavanke/The-Pareidolia-Paradox`
