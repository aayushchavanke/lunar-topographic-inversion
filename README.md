# 🌑 The Pareidolia Paradox: Lunar Surface Topography Classification
> **Binary Classification of Lunar Surface Crops (Depth vs. Rise) Evaluated on Balanced Accuracy**

A production-grade, physics-aligned deep learning pipeline developed for "The Pareidolia Paradox" competition. Distinguishes lunar depressions (**Class 0: Craters, holes**) from elevations (**Class 1: Mounds, boulders**) using standardized solar illumination geometry and a **5-Fold Stratified ResNet-18 Ensemble with Deep Test-Time Augmentation (TTA)**.

---

## 🧭 End-to-End Pipeline Architecture

```mermaid
graph TD
    A["Raw 256x256 Lunar Image & Sun Azimuth (θ)"] --> B["1. Planetary Illumination Normalization (-θ Rotation)"]
    B --> C["2. Symmetric Reflect Padding (128px) & Bicubic Resampling"]
    C --> D["3. Canonical North-Lit Image (Sun locked at Top/North)"]
    
    subgraph Training Phase
        D --> E["5-Fold Stratified Split (80% Train / 20% Holdout)"]
        E --> F["Inverse-Frequency Weighted Sampler (50/50 Batches)"]
        F --> G["ResNet-18 (1-Channel Adapted Pretrained Backbone)"]
        G --> H["Save Checkpoints: model_fold0.pt ... model_fold4.pt"]
        H --> I["Compile 7,854 Out-Of-Fold Predictions (oof_predictions.csv)"]
        I --> J["Scan Optimal Decision Cutoff: τ* = 0.47 (69.75% Bal Acc)"]
    end

    subgraph Inference Phase (2,000 Test Images)
        D --> K["Generate 8 Physics-Safe TTA Views per Image"]
        K --> L["Evaluate across all 5 Trained Fold Models (40 Passes / Sample)"]
        L --> M["Validation-Weighted Soft Probability Averaging"]
        M --> N["Apply Calibrated Threshold (τ* = 0.47)"]
        N --> O["Final Verified submission.csv (2,000 Rows)"]
    end
```

---

## 🚀 Key Architectural & Physical Innovations

### 1. Planetary Illumination Alignment ($-\theta_{\text{azimuth}}$ Rotation)
* Lunar shape-from-shading depends strictly on the Sun's azimuth angle ($\theta_{\text{azimuth}}$).
* All crops are normalized by counter-clockwise rotation by $-\theta_{\text{azimuth}}$ with **128px symmetric reflection padding** (`mode='reflect'`) and **Bicubic interpolation**, followed by center-cropping to 256×256.
* **Result:** Sunlight is mathematically locked to come strictly from the **North (Top of image)** across the entire dataset with zero black-border artifacts.

```
                     Lunar North (0° / Moon's Spin Axis)
                                    ▲
                                    │
                                    │   θ (sun_azimuth_angle)
                                    │  ↗
                                    │ / 
                                    │/  ☀ (Apparent Sun Position)
                                    ┼──────────────► East (90°)
                                   /│
                                  / │
                           Local Lunar Surface Patch
```

### 2. Strict Purge of Shadow-Inverting Flips
* In a North-lit image, applying a vertical flip moves the Sun to the South ($180^\circ$ inversion), turning craters into mounds and corrupting topography cues.
* All horizontal and vertical flips were completely eliminated from training and inference, replaced with orientation-preserving transforms (`RandomResizedCrop(256, scale=(0.92, 1.0))` and subtle `ColorJitter`).

### 3. Pretrained 1-Channel Grayscale ResNet-18
* ImageNet-pretrained ResNet-18 adapted for 1-channel grayscale by averaging RGB first-layer convolutional weights:
  $$W_{\text{gray}} = \frac{W_R + W_G + W_B}{3}$$
* Retains 100% of ImageNet visual edge/shading representations without introducing parameter bloat or overfitting on ~7,800 images.

### 4. Physical Batch Balancing (50/50 Mini-Batches)
* The dataset has a natural ~64% Mound vs ~36% Crater imbalance.
* An inverse-frequency `WeightedRandomSampler` is applied during training to force exact 50/50 class distributions in every mini-batch, preventing majority-class shortcut collapse.

---

## 🔍 How Inference Works: 8 Views × 5 Models = 40 Passes per Image

To ensure maximum prediction stability and eliminate sensor noise on borderline terrain, every test image is evaluated **40 separate times**:

```
                       [ Single Test Image: eval_00001.png ]
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        │ 8 Different "Camera Lenses / Angles" (Physics-Safe TTA Views)   │
        │   1. Standard 100% scale (Sun locked to North)                  │
        │   2. 96% Center-Crop (Zoomed in slightly)                       │
        │   3. 92% Center-Crop (Zoomed in a bit more)                     │
        │   4. +5% Brightness/Contrast                                    │
        │   5. -5% Brightness/Contrast                                    │
        │   6. +10% High Contrast                                         │
        │   7. +2.5° Micro-tilt (Reflect-padded)                          │
        │   8. -2.5° Micro-tilt (Reflect-padded)                          │
        └────────────────────────────────┬────────────────────────────────┘
                                         │
                 Feed all 8 Views into all 5 Trained Models:
                                         │
       ┌───────────┬───────────┬─────────┴─┬───────────┬───────────┐
       ▼           ▼           ▼           ▼           ▼           ▼
   [Model 0]   [Model 1]   [Model 2]   [Model 3]   [Model 4]
   (Weight:    (Weight:    (Weight:    (Weight:    (Weight:
    18.30%)     20.15%)     24.53%⭐)   16.97%)     20.06%)
       │           │           │           │           │
   8 passes    8 passes    8 passes    8 passes    8 passes
       └───────────┴───────────┼───────────┴───────────┘
                               ▼
            TOTAL = 8 x 5 = 40 FORWARD PASSES
                               │
                               ▼
            Weighted Continuous Average Probability
                               │
                               ▼
            Optimal Decision Cutoff Comparison (τ = 0.47)
                               │
                               ▼
                Final Prediction: Class 0 or Class 1
```

* Across all 2,000 test images: $2,000 \times 40 = \mathbf{80,000 \text{ total forward passes}}$.

---

## 📊 Cross-Validation Performance Summary

| Fold | Checkpoint | Val Balanced Accuracy | Model Voting Weight |
| :---: | :--- | :---: | :---: |
| **Fold 0** | `model_fold0.pt` | **68.77%** | **18.30%** |
| **Fold 1** | `model_fold1.pt` | **69.69%** | **20.15%** |
| **Fold 2** | `model_fold2.pt` | **71.73%** ⭐ | **24.53%** *(Highest Trust)* |
| **Fold 3** | `model_fold3.pt` | **68.08%** | **16.97%** |
| **Fold 4** | `model_fold4.pt` | **69.65%** | **20.06%** |
| **Mean $\pm$ Std** | — | **69.58% ($\pm 1.23\%$)** | — |
| **Global OOF Optimal** | `optimal_threshold.json` | **$\tau^* = 0.47 \rightarrow \mathbf{69.75\%}$** | — |

---

## 🤝 Model Consensus on the 2,000 Test Images

| Consensus Level | Agreement Description | Test Images Count | Percentage |
| :--- | :--- | :---: | :---: |
| **5 / 5** | **Unanimous Agreement** *(All 5 models agreed 100%)* | **1,333** | **66.65%** |
| **4 / 5** | **Strong Majority** *(4 models agreed, 1 disagreed)* | **444** | **22.20%** |
| **3 / 5** | **Split Decision** *(Ambiguous terrain resolved by soft weighting)* | **223** | **11.15%** |
| **Total High Confidence** | **(4/5 and 5/5 Consensus)** | **1,777** | **88.85%** |

---

## 🛠 Directory Structure

```text
├── train_images/                  # 7,854 raw training lunar PNGs (ignored by git)
├── test_images/                   # 2,000 test lunar PNGs (ignored by git)
├── train_metadata.csv             # Training metadata: image_id, sun_azimuth_angle, label
├── test_metadata.csv              # Test metadata: image_id, sun_azimuth_angle
│
├── dataset.py                     # Reflection-padded azimuth rotation & dataset loaders
├── model.py                       # 1-Channel adapted pretrained ResNet-18 architecture
├── train_kfold.py                 # 5-Fold stratified training & OOF threshold scanner
├── inference_ensemble_tta.py      # Deep 8-view TTA + weighted 5-fold ensemble inference
├── tune_threshold.py              # Standalone Out-Of-Fold decision threshold analyzer
├── hard_example_mining_experiment.py # Standalone 3-cycle error-mining validation script
│
├── optimal_threshold.json         # Calibrated decision threshold metadata
├── oof_predictions.csv            # Complete 7,854 Out-Of-Fold validation predictions
├── submission.csv                 # Final competition submission (2,000 test predictions)
├── submission_final.csv           # Backup verified submission file
│
├── run_training.bat               # Automated 1-click training launcher
└── run_inference.bat              # Automated 1-click TTA inference launcher
```

---

## 🏃 Quick Start & Reproduction

### 1. Environment Setup
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run 5-Fold Cross-Validation Training
```bash
# Windows 1-Click:
.\run_training.bat

# Or direct Python:
python train_kfold.py --epochs 20 --batch_size 64 --patience 6
```
*Saves `model_fold0.pt` through `model_fold4.pt`, `oof_predictions.csv`, and `optimal_threshold.json`.*

### 3. Generate Final Submission (Deep TTA + Weighted Ensemble)
```bash
# Windows 1-Click:
.\run_inference.bat

# Or direct Python:
python inference_ensemble_tta.py
```
*Evaluates 8 views × 5 fold models (40 passes/sample) and writes out verified `submission.csv`.*

---

## 🔬 Experimental Hard-Example Mining (Reinforcement / Boosting Study)

We conducted an isolated 3-cycle error-mining experiment (`hard_example_mining_experiment.py`) with a 10% virgin holdout vault (786 images never exposed to training):
* **Experiment Result:** Repeatedly forcing the neural network to retrain on hard misclassified samples caused training pool errors to drop to 0.2%, but holdout test accuracy declined from **68.86% down to 64.47%** due to overfitting on ambiguous regolith noise.
* **Conclusion:** Validated that **Early Stopping (Epoch 2–3) with 5-Fold Ensembling** is the optimal, generalization-maximizing configuration.
* **Preservation Notice:** The production `submission.csv` was **not** modified by this experiment and remains strictly generated by the clean 5-fold ensemble.

---

## 👥 Authors & Collaborators
* Param Patil ([@ParamPatil-03](https://github.com/ParamPatil-03))
* Aayush Chavanke ([@aayushchavanke](https://github.com/aayushchavanke))
