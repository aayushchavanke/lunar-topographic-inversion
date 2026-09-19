import os
import sys
import time
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import PareidoliaDataset
from model import build_model

def create_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Creates a WeightedRandomSampler that balances class representation per batch (50/50)."""
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float)
    return WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )

def evaluate_fold(model, val_loader, criterion, device):
    """Evaluates validation loss, recall per class, and balanced accuracy."""
    model.eval()
    val_loss_total = 0.0
    val_samples = 0
    all_targets = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            bs = images.size(0)
            val_loss_total += loss.item() * bs
            val_samples += bs

            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    val_loss = val_loss_total / max(1, val_samples)
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    c0_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    c1_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    acc = accuracy_score(all_targets, all_preds)

    return val_loss, bal_acc, c0_recall, c1_recall, acc, (tn, fp, fn, tp), all_targets, all_probs

def train_one_fold(fold_idx, train_sub_df, val_sub_df, args, device):
    print(f"\n" + "-" * 75)
    print(f"FOLD {fold_idx}: Train={len(train_sub_df):,} images | Val={len(val_sub_df):,} images (Target: {args.epochs} Epochs)")
    print(f"  Train Class Dist: C0={(train_sub_df['label']==0).sum()} ({(train_sub_df['label']==0).mean()*100:.1f}%), C1={(train_sub_df['label']==1).sum()} ({(train_sub_df['label']==1).mean()*100:.1f}%)")
    print(f"  Val Class Dist:   C0={(val_sub_df['label']==0).sum()} ({(val_sub_df['label']==0).mean()*100:.1f}%), C1={(val_sub_df['label']==1).sum()} ({(val_sub_df['label']==1).mean()*100:.1f}%)")
    print("-" * 75)

    train_dataset = PareidoliaDataset(train_sub_df, args.images_dir, is_train=True, is_labeled=True)
    val_dataset = PareidoliaDataset(val_sub_df, args.images_dir, is_train=False, is_labeled=True)

    sampler = create_weighted_sampler(train_sub_df["label"].values)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    # Initialize fresh ResNet-18 adapted for 1-channel grayscale
    model = build_model(num_classes=2, pretrained=True, in_channels=1)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    backbone_params = [p for name, p in model.named_parameters() if not name.startswith("fc")]
    head_params = [p for name, p in model.named_parameters() if name.startswith("fc")]

    # Freeze backbone for epoch 1 warm-up
    for p in backbone_params:
        p.requires_grad = False
    optimizer = torch.optim.AdamW(head_params, lr=args.head_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_bal_acc = 0.0
    best_epoch = 0
    best_val_probs = None
    no_improve_epochs = 0
    fold_checkpoint_path = f"model_fold{fold_idx}.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Unfreeze backbone after epoch 1
        if epoch == 2:
            for p in backbone_params:
                p.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": backbone_params, "lr": args.lr},
                {"params": head_params, "lr": args.head_lr}
            ], weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - 1, eta_min=1e-6)

        model.train()
        train_loss_total = 0.0
        train_samples = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            bs = images.size(0)
            train_loss_total += loss.item() * bs
            train_samples += bs

        train_loss = train_loss_total / max(1, train_samples)
        scheduler.step()

        val_loss, bal_acc, c0_rec, c1_rec, acc, (tn, fp, fn, tp), targets_arr, probs_arr = evaluate_fold(
            model, val_loader, criterion, device
        )
        elapsed = time.time() - t0

        saved_msg = ""
        if bal_acc > best_val_bal_acc:
            best_val_bal_acc = bal_acc
            best_epoch = epoch
            best_val_probs = probs_arr
            no_improve_epochs = 0
            
            torch.save({
                "fold": fold_idx,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_balanced_acc": bal_acc,
                "c0_recall": c0_rec,
                "c1_recall": c1_rec,
            }, fold_checkpoint_path)
            saved_msg = f" -> [* Saved {fold_checkpoint_path}]"
        else:
            no_improve_epochs += 1

        print(f"  Epoch [{epoch:>2}/{args.epochs:>2}] ({elapsed:>4.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"C0 Rec: {c0_rec:.4f} ({tn:>3}/{tn+fp}) | "
              f"C1 Rec: {c1_rec:.4f} ({tp:>3}/{tp+fn}) | "
              f"Bal Acc: {bal_acc:.4f}{saved_msg}", flush=True)

        if args.patience > 0 and no_improve_epochs >= args.patience and epoch >= 10:
            print(f"  [!] Early stopping: No validation improvement for {args.patience} epochs.")
            break

    print(f"[+] Fold {fold_idx} Best Val Balanced Accuracy: {best_val_bal_acc:.4f} (Epoch {best_epoch})")
    return best_val_bal_acc, best_val_probs

def main():
    parser = argparse.ArgumentParser(description="5-Fold Stratified ResNet-18 Cross-Validation & OOF Calibration")
    parser.add_argument("--data_csv", type=str, default="train_metadata.csv", help="Full training metadata CSV")
    parser.add_argument("--images_dir", type=str, default="train_images", help="Training images folder")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per fold (default: 20)")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Backbone learning rate")
    parser.add_argument("--head_lr", type=float, default=1e-3, help="FC head learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="Label smoothing epsilon")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers (0 for Windows)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print("=" * 75)
    print(f"5-FOLD STRATIFIED TRAINING PIPELINE: RESNET-18 (EPOCHS={args.epochs})")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Active Device: {device.type.upper()}")
    if device.type == "cuda":
        print(f"    GPU Model: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(args.data_csv):
        raise FileNotFoundError(f"Data file '{args.data_csv}' not found.")

    df = pd.read_csv(args.data_csv)
    print(f"[+] Total Dataset: {len(df):,} samples from '{args.data_csv}'")
    print(f"    Class 0 (Depth): {(df['label'] == 0).sum():,} ({(df['label'] == 0).mean()*100:.2f}%)")
    print(f"    Class 1 (Rise):  {(df['label'] == 1).sum():,} ({(df['label'] == 1).mean()*100:.2f}%)")

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    fold_scores = []
    oof_df = df.copy()
    oof_df["oof_prob_class1"] = np.nan
    oof_df["fold"] = -1

    start_time = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
        train_sub = df.iloc[train_idx].reset_index(drop=True)
        val_sub = df.iloc[val_idx].reset_index(drop=True)

        torch.manual_seed(args.seed + fold_idx)
        np.random.seed(args.seed + fold_idx)

        score, best_probs = train_one_fold(fold_idx, train_sub, val_sub, args, device)
        fold_scores.append(score)

        oof_df.loc[val_idx, "oof_prob_class1"] = best_probs
        oof_df.loc[val_idx, "fold"] = fold_idx

    total_time = time.time() - start_time
    fold_scores = np.array(fold_scores)

    # Save Out-Of-Fold predictions
    oof_csv_path = "oof_predictions.csv"
    oof_df.to_csv(oof_csv_path, index=False)
    print(f"\n[+] Saved full Out-Of-Fold predictions to '{oof_csv_path}'")

    # Global Out-Of-Fold Balanced Accuracy Calculation
    y_true_all = oof_df["label"].values
    y_prob_all = oof_df["oof_prob_class1"].values

    thresholds = np.linspace(0.10, 0.90, 81)
    best_oof_threshold = 0.50
    best_oof_bal_acc = 0.0

    for t in thresholds:
        preds = (y_prob_all >= t).astype(int)
        score = balanced_accuracy_score(y_true_all, preds)
        if score > best_oof_bal_acc:
            best_oof_bal_acc = score
            best_oof_threshold = t

    oof_50_score = balanced_accuracy_score(y_true_all, (y_prob_all >= 0.50).astype(int))

    # Save optimal threshold metadata to JSON
    threshold_info = {
        "model_name": "resnet18",
        "optimal_threshold": float(best_oof_threshold),
        "oof_balanced_accuracy_at_optimal": float(best_oof_bal_acc),
        "oof_balanced_accuracy_at_050": float(oof_50_score),
        "mean_fold_score": float(fold_scores.mean()),
        "std_fold_score": float(fold_scores.std())
    }
    with open("optimal_threshold.json", "w") as f:
        json.dump(threshold_info, f, indent=4)
    print(f"[+] Saved optimal threshold metadata to 'optimal_threshold.json'")

    print("\n" + "=" * 75)
    print("CROSS-VALIDATION RESULTS SUMMARY (5 FOLDS)")
    print("=" * 75)
    print(f"{'Fold':<10} | {'Checkpoint':<18} | {'Val Balanced Accuracy'}")
    print("-" * 75)
    for i, score in enumerate(fold_scores):
        print(f"Fold {i:<5} | model_fold{i}.pt     | {score:.4f} ({score*100:.2f}%)")
    print("-" * 75)
    print(f"Mean Fold Balanced Accuracy:     {fold_scores.mean():.4f} ({fold_scores.mean()*100:.2f}%)")
    print(f"Standard Deviation:             {fold_scores.std():.4f} ({fold_scores.std()*100:.2f}%)")
    print(f"Global OOF Balanced Acc @ 0.50: {oof_50_score:.4f} ({oof_50_score*100:.2f}%)")
    print(f"Global OOF Optimal Threshold:   {best_oof_threshold:.2f} -> Bal Acc: {best_oof_bal_acc:.4f} ({best_oof_bal_acc*100:.2f}%)")
    print(f"Total Elapsed Time:             {total_time/60:.2f} minutes")
    print("=" * 75)

if __name__ == "__main__":
    main()

