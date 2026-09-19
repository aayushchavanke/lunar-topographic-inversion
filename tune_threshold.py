import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

import torch
from torch.utils.data import DataLoader

from dataset import PareidoliaDataset
from model import build_model

def tune_threshold(checkpoint_path="best_model.pt", 
                   val_csv="val_split.csv", 
                   images_dir="train_images", 
                   plot_out="threshold_tuning_plot.png",
                   brain_artifact_dir=r"C:\Users\PARAM\.gemini\antigravity-ide\brain\75654c39-7f35-483e-9e5a-67cbad8a0ae5"):
    
    print("=" * 75)
    print("STEP 10: DECISION THRESHOLD TUNING (ON VALIDATION SET ONLY)")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device: {device.type.upper()}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found.")
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"Validation CSV '{val_csv}' not found.")

    val_df = pd.read_csv(val_csv)
    print(f"[+] Loaded validation split: '{val_csv}' ({len(val_df):,} samples)")

    # 1. Load Model
    print(f"[+] Loading model from '{checkpoint_path}'...")
    model = build_model(num_classes=2, pretrained=False, in_channels=1)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
    else:
        model = ckpt
        
    model.to(device)
    model.eval()

    # 2. Validation DataLoader
    val_dataset = PareidoliaDataset(
        metadata=val_df,
        images_dir=images_dir,
        is_train=False,
        is_labeled=True
    )
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    # 3. Compute raw Softmax probabilities for all validation samples
    print(f"[+] Extracting validation softmax probabilities...")
    all_targets = []
    all_probs_c1 = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs_c1.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_prob = np.array(all_probs_c1)

    print(f"    Validation samples evaluated: {len(y_true):,}")
    print(f"    True Class 0 count: {(y_true == 0).sum():,} ({(y_true == 0).mean()*100:.2f}%)")
    print(f"    True Class 1 count: {(y_true == 1).sum():,} ({(y_true == 1).mean()*100:.2f}%)")
    print(f"    P(Class 1) Range: [{y_prob.min():.4f}, {y_prob.max():.4f}], Mean: {y_prob.mean():.4f}, Median: {np.median(y_prob):.4f}")

    # 4. Sweep decision thresholds from 0.10 to 0.90
    print("\n" + "-" * 75)
    print("THRESHOLD SWEEP TABLE (0.10 to 0.90)")
    print("-" * 75)
    print(f"{'Threshold':<10} | {'Bal Acc':<9} | {'C0 Recall':<10} | {'C1 Recall':<10} | {'Accuracy':<9} | {'Pred C0 / C1'}")
    print("-" * 75)

    thresholds = np.linspace(0.10, 0.90, 81)  # Fine step of 0.01 for search
    display_thresholds = np.linspace(0.10, 0.90, 17)  # Step of 0.05 for clean display table

    results = []
    best_threshold = 0.50
    best_bal_acc = 0.0
    best_metrics = None

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        c0_rec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        c1_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        bal_acc = balanced_accuracy_score(y_true, preds)
        acc = accuracy_score(y_true, preds)

        entry = {
            "threshold": t,
            "bal_acc": bal_acc,
            "c0_recall": c0_rec,
            "c1_recall": c1_rec,
            "accuracy": acc,
            "pred_c0": tn + fn,
            "pred_c1": tp + fp,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn
        }
        results.append(entry)

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_threshold = t
            best_metrics = entry

    # Print summary table at step of 0.05
    for r in results:
        # Match display thresholds within float tolerance
        if any(np.isclose(r["threshold"], dt, atol=1e-4) for dt in display_thresholds):
            is_best = np.isclose(r["threshold"], best_threshold, atol=1e-4)
            best_mark = " [* BEST]" if is_best else ""
            print(f"{r['threshold']:<10.2f} | {r['bal_acc']:<9.4f} | {r['c0_recall']:<10.4f} | {r['c1_recall']:<10.4f} | {r['accuracy']:<9.4f} | {r['pred_c0']:>4} / {r['pred_c1']:<4}{best_mark}")

    print("-" * 75)

    # 5. Plot Threshold Curve
    res_df = pd.DataFrame(results)

    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(res_df["threshold"], res_df["bal_acc"], label="Balanced Accuracy (Metric)", color="#1f77b4", linewidth=2.5)
    plt.plot(res_df["threshold"], res_df["c0_recall"], label="Class 0 Recall (Specificity)", color="#2ca02c", linestyle="--", linewidth=1.8)
    plt.plot(res_df["threshold"], res_df["c1_recall"], label="Class 1 Recall (Sensitivity)", color="#d62728", linestyle=":", linewidth=1.8)
    plt.plot(res_df["threshold"], res_df["accuracy"], label="Standard Accuracy", color="#7f7f7f", linestyle="-.", linewidth=1.2, alpha=0.7)

    # Highlight default 0.50 threshold
    r_50 = res_df.iloc[(res_df["threshold"] - 0.50).abs().argmin()]
    plt.axvline(x=0.50, color="gray", linestyle="--", alpha=0.6, label=f"Default 0.50 (Bal Acc={r_50['bal_acc']:.4f})")

    # Highlight optimal threshold
    plt.axvline(x=best_threshold, color="#ff7f0e", linestyle="-", linewidth=2, label=f"Optimal Threshold {best_threshold:.2f} (Bal Acc={best_bal_acc:.4f})")
    plt.scatter([best_threshold], [best_bal_acc], color="#ff7f0e", s=120, zorder=5, edgecolor="black")

    plt.title("Decision Threshold Tuning for Balanced Accuracy (Validation Set Only)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Decision Threshold $\\tau$ for P(Class 1)", fontsize=11)
    plt.ylabel("Validation Metric Score", fontsize=11)
    plt.xlim(0.10, 0.90)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="lower center", framealpha=0.95, fontsize=10)
    plt.tight_layout()

    plt.savefig(plot_out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\n[+] Saved threshold tuning plot to '{plot_out}'.")

    if os.path.exists(brain_artifact_dir):
        dest_brain = os.path.join(brain_artifact_dir, plot_out)
        shutil.copy2(plot_out, dest_brain)
        print(f"[+] Synced plot to artifacts directory at '{dest_brain}'.")

    # 6. Print Final Verdict
    print("\n" + "=" * 75)
    print("OPTIMAL THRESHOLD REPORT")
    print("=" * 75)
    print(f"  * Best Decision Threshold:      {best_threshold:.2f}")
    print(f"  * Peak Validation Balanced Acc: {best_bal_acc:.4f} ({best_bal_acc*100:.2f}%)")
    print(f"  * Class 0 Recall @ Best:        {best_metrics['c0_recall']:.4f} ({best_metrics['tn']}/{best_metrics['tn']+best_metrics['fp']})")
    print(f"  * Class 1 Recall @ Best:        {best_metrics['c1_recall']:.4f} ({best_metrics['tp']}/{best_metrics['tp']+best_metrics['fn']})")
    print(f"  * Standard Accuracy @ Best:     {best_metrics['accuracy']:.4f}")
    print(f"  * Default 0.50 Threshold Score: {r_50['bal_acc']:.4f}")
    print(f"  * Improvement over 0.50:        +{(best_bal_acc - r_50['bal_acc'])*100:.2f}%")
    print("=" * 75)

    return best_threshold, best_bal_acc

if __name__ == "__main__":
    tune_threshold()
