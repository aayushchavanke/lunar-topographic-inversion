# 🌑 The Pareidolia Paradox: Lunar Surface Topography Classification
> **Binary Classification of Lunar Surface Crops (Depth vs. Rise) Evaluated on Balanced Accuracy**

A production-grade, physics-aligned deep learning pipeline developed for "The Pareidolia Paradox" competition. Distinguishes lunar depressions (Class 0: Craters, holes) from elevations (Class 1: Mounds, boulders) using standardized solar illumination geometry and a 5-Fold Stratified ResNet-18 Ensemble with Deep Test-Time Augmentation (TTA).

---

## 🚀 Key Architectural & Physical Innovations

### 1. Planetary Illumination Alignment ($-\theta_{\text{azimuth}}$ Rotation)
* Lunar shape-from-shading depends strictly on the Sun's azimuth angle ($\theta_{\text{azimuth}}$).
* All crops are normalized by counter-clockwise rotation by $-\theta_{\text{azimuth}}$ with **128px symmetric reflection padding** (`mode='reflect'`) and **Bicubic interpolation**, followed by center-cropping to 256×256.
* **Result:** Sunlight is mathematically locked to come strictly from the **North (Top of image)** across the entire dataset with zero black-border artifacts.

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

### 5. 5-Fold Stratified Cross-Validation & Out-Of-Fold (OOF) Calibration
* Data is partitioned into 5 stratified folds. Each model is trained on 80% of the data and validated on a 20% holdout.
* Full Out-Of-Fold predictions are compiled in `oof_predictions.csv` to find the global optimal decision threshold:
  $$\tau^* = 0.47 \quad \longrightarrow \quad \mathbf{69.75\% \text{ Global OOF Balanced Accuracy}}$$

### 6. Deep 8-View Physics-Safe TTA & Validation-Weighted Ensemble
* Evaluates 8 illumination-safe views per image across all 5 models (**40 forward passes per test image** / 80,000 total evaluations).
* Folds are weighted by their cross-validation performance ($w_i \propto (\text{Score}_i - 0.50)^2$), giving highest voting weight to **Fold 2 (24.53% weight, 71.73% Val Acc)**.

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
* **Finding:** Repeatedly forcing the neural network to retrain on hard misclassified samples caused training pool errors to drop to 0.2%, but holdout test accuracy declined from **68.86% down to 64.47%** due to overfitting on ambiguous regolith noise.
* **Conclusion:** Validated that **Early Stopping (Epoch 2–3) with 5-Fold Ensembling** is the optimal, generalization-maximizing configuration for planetary vision datasets.

---

## 👥 Authors & Collaborators
* Param Patil ([@ParamPatil-03](https://github.com/ParamPatil-03))
* Aayush Chavanke ([@aayushchavanke](https://github.com/aayushchavanke))
