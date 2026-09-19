import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PareidoliaDataset
from model import build_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet18 for The Pareidolia Paradox")
    parser.add_argument("--train_csv", type=str, default="train_split.csv", help="Path to training metadata CSV")
    parser.add_argument("--val_csv", type=str, default="val_split.csv", help="Path to validation metadata CSV")
    parser.add_argument("--images_dir", type=str, default="train_images", help="Path to directory containing images")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for AdamW optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for AdamW optimizer")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum LR for CosineAnnealingLR")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader worker processes")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth", help="Filepath to save best checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max_train_batches", type=int, default=None, help="Optional limit on train batches per epoch")
    return parser.parse_args()

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 65)
    print("TRAINING RESNET18: The Pareidolia Paradox")
    print("=" * 65)

    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device: {device.type.upper()}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"    CPU Threads: {torch.get_num_threads()} (Total logical cores: {os.cpu_count()})")

    # 2. Check datasets
    if not os.path.exists(args.train_csv) or not os.path.exists(args.val_csv):
        raise FileNotFoundError(f"Missing '{args.train_csv}' or '{args.val_csv}'. Run split_data.py first.")

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    print(f"[+] Loaded Train split: {len(train_df):,} samples | Val split: {len(val_df):,} samples")

    # 3. Compute class weights for CrossEntropyLoss
    labels_train = train_df["label"].values
    classes = np.sort(np.unique(labels_train))
    raw_weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels_train)
    class_weights = torch.tensor(raw_weights, dtype=torch.float32).to(device)

    print("\n" + "-" * 50)
    print("COMPUTED CLASS WEIGHTS (from train split)")
    print("-" * 50)
    for cls, w in zip(classes, class_weights.cpu().numpy()):
        count = (labels_train == cls).sum()
        pct = (count / len(labels_train)) * 100
        print(f"  Class {cls}: {count:,} samples ({pct:.2f}%) -> Loss Weight: {w:.4f}")

    # 4. DataLoaders
    train_dataset = PareidoliaDataset(
        metadata=train_df,
        images_dir=args.images_dir,
        is_train=True,
        is_labeled=True
    )
    val_dataset = PareidoliaDataset(
        metadata=val_df,
        images_dir=args.images_dir,
        is_train=False,
        is_labeled=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    # 5. Build Model
    print(f"\n[+] Building adapted ResNet18 model (single-channel grayscale -> 2 classes)...")
    model = build_model(num_classes=2, pretrained=True, in_channels=1)
    model.to(device)

    # 6. Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    print(f"[+] Hyperparameters: Epochs={args.epochs}, Batch Size={args.batch_size}, Initial LR={args.lr}, Weight Decay={args.weight_decay}")
    print(f"[+] Checkpoint file: '{args.checkpoint}'")

    # 7. Training Loop
    print("\n" + "=" * 65)
    print("STARTING TRAINING")
    print("=" * 65)

    best_val_bal_acc = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Training Phase
        model.train()
        train_loss_total = 0.0
        train_samples = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size_cur = images.size(0)
            train_loss_total += loss.item() * batch_size_cur
            train_samples += batch_size_cur

        epoch_train_loss = train_loss_total / max(1, train_samples)

        # Validation Phase
        model.eval()
        val_loss_total = 0.0
        val_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, targets)

                batch_size_cur = images.size(0)
                val_loss_total += loss.item() * batch_size_cur
                val_samples += batch_size_cur

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        epoch_val_loss = val_loss_total / max(1, val_samples)
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)

        val_bal_acc = balanced_accuracy_score(all_targets, all_preds)
        val_std_acc = accuracy_score(all_targets, all_preds)

        # Step LR scheduler
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Checkpoint saving
        improved = val_bal_acc > best_val_bal_acc
        save_msg = ""
        if improved:
            best_val_bal_acc = val_bal_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_balanced_acc": best_val_bal_acc,
                "val_loss": epoch_val_loss,
                "class_weights": class_weights.cpu(),
                "args": vars(args)
            }, args.checkpoint)
            save_msg = f" -> [*] Saved checkpoint (Best Val Bal Acc: {best_val_bal_acc:.4f})"

        epoch_elapsed = time.time() - epoch_start
        print(f"Epoch [{epoch}/{args.epochs}] ({epoch_elapsed:.1f}s) | "
              f"LR: {current_lr:.2e} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Bal Acc: {val_bal_acc:.4f} | "
              f"Val Acc: {val_std_acc:.4f}{save_msg}", flush=True)

    total_time = time.time() - start_time
    print("=" * 65)
    print(f"[+] Training completed in {total_time/60:.2f} minutes.")
    print(f"[+] Best Validation Balanced Accuracy: {best_val_bal_acc:.4f} ({best_val_bal_acc*100:.2f}%)")
    print(f"[+] Checkpoint saved at '{args.checkpoint}'")
    print("=" * 65)

if __name__ == "__main__":
    main()
