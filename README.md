# The Pareidolia Paradox: Lunar Binary Image Classification

A machine learning pipeline for detecting subtle lunar pareidolia formations under variable sun illumination angles, evaluated on **Balanced Accuracy**.

---

## 🚀 Key Pipeline Features

1. **Azimuth Normalization with Reflect Padding:**
   - Lunar shadows depend strictly on the sun azimuth angle ($\theta_{\text{azimuth}}$).
   - Images are normalized by rotating counter-clockwise by $-\theta_{\text{azimuth}}$ with a 128px reflection border and center-cropping, eliminating artificial black border artifacts.
2. **First-Layer Adaptation:**
   - Pretrained ImageNet ResNet-18 modified for single-channel grayscale input by averaging RGB weights $\frac{W_R + W_G + W_B}{3}$.
3. **Physical Batch Rebalancing:**
   - Uses `WeightedRandomSampler` to enforce 50/50 minority/majority class mini-batches, preventing shortcut class collapse.
4. **5-Fold Stratified Cross-Validation:**
   - 5 independent fold models preserving stratified class balance.
5. **Test-Time Augmentation (TTA) & Ensembling:**
   - Evaluates 3 spatial views (Original, Horizontal Flip, Vertical Flip) across all 5 models (15 predictions per sample) and averages probabilities.

---

## 🛠 Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <REPO_URL>
   cd <REPO_NAME>
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux / macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: For GPU acceleration on Windows/Linux, ensure you install the CUDA-enabled PyTorch build for your system).*

4. **Directory Structure:**
   Place the competition dataset folders in the root directory:
   ```text
   ├── train_images/          # 7,854 grayscale lunar PNGs
   ├── eval_images/           # 2,000 test lunar PNGs
   ├── train_metadata.csv     # image_id, sun_azimuth_angle, label
   ├── test_metadata.csv      # image_id, sun_azimuth_angle
   ├── dataset.py             # Dataset loader & azimuth rotation logic
   ├── model.py               # 1-channel adapted ResNet18 architecture
   ├── train_kfold.py         # 5-fold cross-validation training script
   ├── tune_threshold.py      # Threshold scanning script
   ├── inference_ensemble_tta.py # 5-fold TTA ensemble inference script
   └── submission_final.csv   # Final submission file
   ```

---

## 🏃 Reproducing Training & Inference

### 1. Run 5-Fold Cross-Validation
Trains 5 fold models and saves `model_fold0.pt` through `model_fold4.pt`:
```bash
python train_kfold.py --epochs 3 --batch_size 64
```

### 2. Decision Threshold Tuning (Validation Only)
Scans thresholds on held-out validation data and plots trade-off curves:
```bash
python tune_threshold.py
```

### 3. Generate Final TTA + Ensemble Submission
Runs 15-view TTA ensemble across all 5 models to output `submission_final.csv`:
```bash
python inference_ensemble_tta.py --threshold 0.50
```

---

## ⚠️ Notes for Collaborators

- **Large Files:** Model checkpoints (`*.pt`, `*.pth`) and raw image folders (`train_images/`, `eval_images/`) are ignored by `.gitignore` to prevent exceeding GitHub file limits (>100MB).
- **Checkpoints:** If sharing pre-trained weights with teammates, use Google Drive / OneDrive or GitHub Releases.
