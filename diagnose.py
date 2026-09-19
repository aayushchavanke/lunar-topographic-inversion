import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PareidoliaDataset
from model import build_model

def run_diagnostics(checkpoint_path="best_model.pt", 
                    val_csv="val_split.csv", 
                    train_csv="train_split.csv", 
                    images_dir="train_images", 
                    task_log_path=r"C:\Users\PARAM\.gemini\antigravity-ide\brain\75654c39-7f35-483e-9e5a-67cbad8a0ae5\.system_generated\tasks\task-171.log"):
    
    print("=" * 70)
    print("MODEL COLLAPSE & PERFORMANCE DIAGNOSTIC SUITE")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Active device: {device.type.upper()}")

    # -------------------------------------------------------------------------
    # (3) CHECK CLASS WEIGHTS (Intended vs Computed vs Saved)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("(3) CLASS WEIGHTS VERIFICATION")
    print("=" * 70)
    if os.path.exists(train_csv):
        train_df = pd.read_csv(train_csv)
        y_train = train_df["label"].values
        n_total = len(y_train)
        n_0 = (y_train == 0).sum()
        n_1 = (y_train == 1).sum()

        expected_w0 = n_total / (2.0 * n_0)
        expected_w1 = n_total / (2.0 * n_1)

        sk_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train)

        print(f"Training set sample counts: Class 0 = {n_0:,} ({n_0/n_total*100:.2f}%), Class 1 = {n_1:,} ({n_1/n_total*100:.2f}%)")
        print(f"Formula weights [N / (2 * N_c)]: Class 0 = {expected_w0:.4f}, Class 1 = {expected_w1:.4f}")
        print(f"sklearn.compute_class_weight:    Class 0 = {sk_weights[0]:.4f}, Class 1 = {sk_weights[1]:.4f}")
    else:
        print(f"Warning: '{train_csv}' not found.")

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_weights = ckpt.get("class_weights", None)
        saved_epoch = ckpt.get("epoch", None)
        saved_val_acc = ckpt.get("best_val_balanced_acc", None)
        print(f"Checkpoint '{checkpoint_path}' saved epoch: {saved_epoch}")
        print(f"Checkpoint saved best val balanced acc: {saved_val_acc}")
        print(f"Checkpoint stored class weights:        {saved_weights}")
    else:
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found.")

    # -------------------------------------------------------------------------
    # (4) CHECK TRAINING LOG / HISTORY ACROSS ALL EPOCHS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("(4) TRAINING HISTORY & VALIDATION BALANCED ACCURACY CHECK")
    print("=" * 70)
    if os.path.exists(task_log_path):
        with open(task_log_path, "r", encoding="utf-8", errors="replace") as f:
            log_content = f.read()
        print("Raw Epoch Lines from Training Log:")
        epoch_lines = [line for line in log_content.splitlines() if line.startswith("Epoch [")]
        for el in epoch_lines:
            print("  " + el)
    else:
        print(f"Note: Training task log '{task_log_path}' not found.")

    # -------------------------------------------------------------------------
    # (1) & (2) RUN EVALUATION ON HELD-OUT VALIDATION SET
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("(1) VALIDATION EVALUATION: CONFUSION MATRIX & PER-CLASS METRICS")
    print("=" * 70)

    val_df = pd.read_csv(val_csv)
    print(f"[+] Loaded validation split '{val_csv}' with {len(val_df):,} samples.")

    # Instantiate model & load checkpoint
    model = build_model(num_classes=2, pretrained=False, in_channels=1)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    val_dataset = PareidoliaDataset(
        metadata=val_df,
        images_dir=images_dir,
        is_train=False,
        is_labeled=True
    )
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    all_targets = []
    all_preds = []
    all_probs_c1 = []
    all_probs_c0 = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            preds = torch.argmax(probs, dim=1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs_c0.extend(probs[:, 0].cpu().numpy().tolist())
            all_probs_c1.extend(probs[:, 1].cpu().numpy().tolist())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs_c1 = np.array(all_probs_c1)
    all_probs_c0 = np.array(all_probs_c0)

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    # cm[0,0] = True Neg (0 predicted as 0)
    # cm[0,1] = False Pos (0 predicted as 1)
    # cm[1,0] = False Neg (1 predicted as 0)
    # cm[1,1] = True Pos (1 predicted as 1)
    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix (Ground Truth Rows x Predicted Columns):")
    print(f"               Predicted 0    Predicted 1    Total True")
    print(f"True Class 0:  {cm[0, 0]:>11}    {cm[0, 1]:>11}    {cm[0].sum():>10}")
    print(f"True Class 1:  {cm[1, 0]:>11}    {cm[1, 1]:>11}    {cm[1].sum():>10}")
    print(f"Total Pred:    {cm[:, 0].sum():>11}    {cm[:, 1].sum():>11}    {len(all_targets):>10}")

    print("\nDetailed Per-Class Classification Report:")
    report = classification_report(all_targets, all_preds, target_names=["Class 0", "Class 1"], digits=4)
    print(report)

    overall_acc = accuracy_score(all_targets, all_preds)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    sens_c0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Recall for Class 0
    sens_c1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall for Class 1

    print(f"Class 0 Recall (Specificity): {sens_c0:.4f} ({tn}/{tn+fp})")
    print(f"Class 1 Recall (Sensitivity): {sens_c1:.4f} ({tp}/{tp+fn})")
    print(f"Balanced Accuracy:            {bal_acc:.4f} (Average of recalls: ({sens_c0:.4f} + {sens_c1:.4f})/2)")
    print(f"Standard Accuracy:            {overall_acc:.4f}")

    # -------------------------------------------------------------------------
    # (2) RAW SOFTMAX PROBABILITY DISTRIBUTION FOR CLASS 1
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("(2) RAW SOFTMAX PROBABILITY DISTRIBUTION (CLASS 1)")
    print("=" * 70)
    print(f"Summary Statistics for P(Class 1):")
    print(f"  - Min:    {np.min(all_probs_c1):.6f}")
    print(f"  - 25th %: {np.percentile(all_probs_c1, 25):.6f}")
    print(f"  - Median: {np.median(all_probs_c1):.6f}")
    print(f"  - Mean:   {np.mean(all_probs_c1):.6f}")
    print(f"  - 75th %: {np.percentile(all_probs_c1, 75):.6f}")
    print(f"  - 90th %: {np.percentile(all_probs_c1, 90):.6f}")
    print(f"  - 95th %: {np.percentile(all_probs_c1, 95):.6f}")
    print(f"  - Max:    {np.max(all_probs_c1):.6f}")
    print(f"  - Std:    {np.std(all_probs_c1):.6f}")

    print(f"\nP(Class 1) conditioned on true label:")
    p1_when_true_0 = all_probs_c1[all_targets == 0]
    p1_when_true_1 = all_probs_c1[all_targets == 1]
    print(f"  When Ground Truth = 0: mean P(Class 1) = {np.mean(p1_when_true_0):.4f}, median = {np.median(p1_when_true_0):.4f}")
    print(f"  When Ground Truth = 1: mean P(Class 1) = {np.mean(p1_when_true_1):.4f}, median = {np.median(p1_when_true_1):.4f}")

    threshold_above_05 = (all_probs_c1 >= 0.5).sum()
    print(f"\nSamples where P(Class 1) >= 0.5 (standard threshold): {threshold_above_05}/{len(all_probs_c1)} ({threshold_above_05/len(all_probs_c1)*100:.2f}%)")

    # Threshold scan for balanced accuracy
    print("\nOptimal Threshold Search for P(Class 1):")
    thresholds = np.linspace(0.05, 0.95, 19)
    best_thresh = 0.5
    best_thresh_bal_acc = 0.0
    for t in thresholds:
        t_preds = (all_probs_c1 >= t).astype(int)
        t_bal = balanced_accuracy_score(all_targets, t_preds)
        if t_bal > best_thresh_bal_acc:
            best_thresh_bal_acc = t_bal
            best_thresh = t
        print(f"  Threshold {t:.2f} -> Balanced Acc: {t_bal:.4f} | Pred Class 1: {(t_preds==1).sum():>4}/{len(t_preds)}")
    print(f"\nOptimal Decision Threshold: {best_thresh:.2f} (Yields Balanced Acc: {best_thresh_bal_acc:.4f})")
    print("=" * 70)

if __name__ == "__main__":
    run_diagnostics()
