import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PareidoliaDataset
from model import build_model

def parse_args():
    parser = argparse.ArgumentParser(description="Improved Training Pipeline for The Pareidolia Paradox")
    parser.add_argument("--train_csv", type=str, default="train_split.csv", help="Path to training metadata CSV")
    parser.add_argument("--val_csv", type=str, default="val_split.csv", help="Path to validation metadata CSV")
    parser.add_argument("--images_dir", type=str, default="train_images", help="Path to images directory")
    parser.add_argument("--epochs", type=int, default=8, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for fine-tuning")
    parser.add_argument("--head_lr", type=float, default=1e-3, help="Learning rate for classification head")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--freeze_epochs", type=int, default=1, help="Number of initial epochs to freeze backbone")
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="Label smoothing to prevent extreme probability collapse")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader num_workers")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt", help="Path to save best checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def find_best_threshold(y_true, y_probs):
    """Scans decision thresholds to find the one maximizing balanced accuracy."""
    thresholds = np.linspace(0.10, 0.90, 81)
    best_thresh = 0.50
    best_score = 0.0
    
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        score = balanced_accuracy_score(y_true, preds)
        if score > best_score:
            best_score = score
            best_thresh = t
            
    return best_thresh, best_score

def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 70)
    print("IMPROVED RESNET18 TRAINING: The Pareidolia Paradox")
    print("=" * 70)

    # 1. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Active Device: {device.type.upper()}")
    if device.type == "cuda":
        print(f"    GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print(f"    CPU Threads: {torch.get_num_threads()} (Logical cores: {os.cpu_count()})")

    # 2. Data
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    print(f"[+] Loaded Train: {len(train_df):,} samples | Val: {len(val_df):,} samples")

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

    # 3. Model
    print("\n[+] Instantiating pretrained ResNet18 (channel-averaged first layer)...")
    model = build_model(num_classes=2, pretrained=True, in_channels=1)
    model.to(device)

    # Loss function with label smoothing (unweighted to prevent shortcut collapse)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Differential learning rate parameter groups
    backbone_params = [p for name, p in model.named_parameters() if not name.startswith("fc")]
    head_params = [p for name, p in model.named_parameters() if name.startswith("fc")]

    # Freeze backbone if requested for initial warmup
    if args.freeze_epochs > 0:
        for p in backbone_params:
            p.requires_grad = False
        print(f"[*] Frozen backbone feature extractor for initial {args.freeze_epochs} epoch(s).")
        optimizer = torch.optim.AdamW(head_params, lr=args.head_lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": args.lr},
            {"params": head_params, "lr": args.head_lr}
        ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_bal_acc = 0.0
    best_threshold = 0.50
    total_start_time = time.time()

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Check if we should unfreeze backbone
        if epoch == args.freeze_epochs + 1:
            print("\n[*] Unfreezing full backbone for end-to-end fine-tuning...")
            for p in backbone_params:
                p.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": backbone_params, "lr": args.lr},
                {"params": head_params, "lr": args.head_lr}
            ], weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.freeze_epochs, eta_min=1e-6
            )

        # Training Phase
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

        epoch_train_loss = train_loss_total / max(1, train_samples)

        # Validation Phase
        model.eval()
        val_loss_total = 0.0
        val_samples = 0
        all_targets = []
        all_probs_c1 = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, targets)
                probs = torch.softmax(outputs, dim=1)

                bs = images.size(0)
                val_loss_total += loss.item() * bs
                val_samples += bs

                all_targets.extend(targets.cpu().numpy().tolist())
                all_probs_c1.extend(probs[:, 1].cpu().numpy().tolist())

        epoch_val_loss = val_loss_total / max(1, val_samples)
        all_targets = np.array(all_targets)
        all_probs_c1 = np.array(all_probs_c1)

        # Standard 0.50 threshold predictions
        standard_preds = (all_probs_c1 >= 0.50).astype(int)
        std_bal_acc = balanced_accuracy_score(all_targets, standard_preds)
        std_acc = accuracy_score(all_targets, standard_preds)

        # Calibrated threshold search
        opt_thresh, opt_bal_acc = find_best_threshold(all_targets, all_probs_c1)

        scheduler.step()
        epoch_time = time.time() - epoch_start

        # Checkpoint saving: evaluate by calibrated balanced accuracy
        save_tag = ""
        if opt_bal_acc > best_val_bal_acc:
            best_val_bal_acc = opt_bal_acc
            best_threshold = opt_thresh
            
            # Save checkpoint
            save_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_balanced_acc": best_val_bal_acc,
                "best_threshold": best_threshold,
                "val_loss": epoch_val_loss,
                "args": vars(args)
            }
            torch.save(save_payload, args.checkpoint)
            # Also sync to .pth for backwards compatibility
            alt_ckpt = args.checkpoint.replace(".pt", ".pth") if args.checkpoint.endswith(".pt") else args.checkpoint + ".pth"
            torch.save(save_payload, alt_ckpt)
            save_tag = f" -> [*] Saved Checkpoint (Best Bal Acc: {best_val_bal_acc:.4f} @ threshold {best_threshold:.2f})"

        print(f"Epoch [{epoch:>2}/{args.epochs:>2}] ({epoch_time:>5.1f}s) | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Bal Acc (0.50): {std_bal_acc:.4f} | "
              f"Calibrated Bal Acc: {opt_bal_acc:.4f} (th={opt_thresh:.2f}){save_tag}", flush=True)

    total_time = time.time() - total_start_time
    print("=" * 70)
    print(f"[+] Training completed in {total_time/60:.2f} minutes.")
    print(f"[+] Best Validation Balanced Accuracy: {best_val_bal_acc:.4f} ({best_val_bal_acc*100:.2f}%)")
    print(f"[+] Optimal Decision Threshold:        {best_threshold:.2f}")
    print(f"[+] Checkpoint saved to:               '{args.checkpoint}'")
    print("=" * 70)

if __name__ == "__main__":
    main()
