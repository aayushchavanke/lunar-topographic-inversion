import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import PareidoliaDataset
from model import build_model

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss: FL(pt) = - alpha_t * (1 - pt)^gamma * log(pt)
    Down-weights easy well-classified examples and focuses gradients on hard minority examples.
    """
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: (B, C), targets: (B)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # probability of true class
        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

def create_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Creates a WeightedRandomSampler that physically balances batches (50/50 per class)."""
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float)
    return WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )

def evaluate_model(model, val_loader, criterion, device):
    """Evaluates validation loss, per-class recall (sensitivity/specificity), and balanced accuracy."""
    model.eval()
    val_loss_total = 0.0
    val_samples = 0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            bs = images.size(0)
            val_loss_total += loss.item() * bs
            val_samples += bs

            preds = torch.argmax(outputs, dim=1)
            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    val_loss = val_loss_total / max(1, val_samples)
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    c0_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    c1_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    acc = accuracy_score(all_targets, all_preds)

    return val_loss, c0_recall, c1_recall, bal_acc, acc, (tn, fp, fn, tp)

def run_experiment(exp_name: str, 
                   use_sampler: bool, 
                   loss_type: str, 
                   epochs: int = 3, 
                   batch_size: int = 64, 
                   lr: float = 2e-4, 
                   head_lr: float = 1e-3,
                   device = None):
    
    print("\n" + "=" * 70)
    print(f"RUNNING EXPERIMENT: {exp_name}")
    print(f"  Configuration: Sampler={'WeightedRandomSampler' if use_sampler else 'StandardShuffle'} | Loss={loss_type}")
    print("=" * 70)

    train_df = pd.read_csv("train_split.csv")
    val_df = pd.read_csv("val_split.csv")

    train_dataset = PareidoliaDataset(train_df, "train_images", is_train=True, is_labeled=True)
    val_dataset = PareidoliaDataset(val_df, "train_images", is_train=False, is_labeled=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )

    if use_sampler:
        sampler = create_weighted_sampler(train_df["label"].values)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
            drop_last=True
        )

    # Initialize fresh model
    model = build_model(num_classes=2, pretrained=True, in_channels=1)
    model.to(device)

    # Configure Loss
    if loss_type == "FocalLoss":
        criterion = FocalLoss(gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Optimizer & Scheduler (warmup head 1 epoch, then unfreeze)
    backbone_params = [p for name, p in model.named_parameters() if not name.startswith("fc")]
    head_params = [p for name, p in model.named_parameters() if name.startswith("fc")]

    # Freeze backbone for epoch 1
    for p in backbone_params:
        p.requires_grad = False

    optimizer = torch.optim.AdamW(head_params, lr=head_lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    epoch_logs = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        if epoch == 2:
            for p in backbone_params:
                p.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": backbone_params, "lr": lr},
                {"params": head_params, "lr": head_lr}
            ], weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - 1, eta_min=1e-6)

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

        val_loss, c0_rec, c1_rec, bal_acc, acc, (tn, fp, fn, tp) = evaluate_model(
            model, val_loader, criterion, device
        )
        elapsed = time.time() - t0

        log_entry = {
            "epoch": epoch,
            "time_s": elapsed,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "c0_recall": c0_rec,
            "c1_recall": c1_rec,
            "bal_acc": bal_acc,
            "acc": acc,
            "tp": tp, "fn": fn, "tn": tn, "fp": fp
        }
        epoch_logs.append(log_entry)

        print(f"  Epoch [{epoch}/{epochs}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"C0 Recall: {c0_rec:.4f} ({tn}/{tn+fp}) | "
              f"C1 Recall: {c1_rec:.4f} ({tp}/{tp+fn}) | "
              f"Bal Acc: {bal_acc:.4f} | Acc: {acc:.4f}", flush=True)

    return model, epoch_logs

def main():
    print("=" * 70)
    print("STEP 9: IMBALANCE HANDLING EXPERIMENTATION & BENCHMARK")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Active Device: {device.type.upper()}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(42)
    np.random.seed(42)

    # Run 1: WeightedRandomSampler + CrossEntropy
    model_sampler, logs_sampler = run_experiment(
        exp_name="Run 1: WeightedRandomSampler (Physical 50/50 Batch Oversampling)",
        use_sampler=True,
        loss_type="CrossEntropy",
        epochs=3,
        device=device
    )

    # Run 2: Focal Loss (gamma=2.0)
    torch.manual_seed(42)
    np.random.seed(42)
    model_focal, logs_focal = run_experiment(
        exp_name="Run 2: Focal Loss (Dynamic Easy-Sample Down-Weighting, gamma=2.0)",
        use_sampler=False,
        loss_type="FocalLoss",
        epochs=3,
        device=device
    )

    # Summary Comparison Table
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD COMPARISON: PER-EPOCH CLASS RECALL & BALANCED ACCURACY")
    print("=" * 70)
    print(f"{'Method':<25} | {'Epoch':<6} | {'C0 Recall':<10} | {'C1 Recall':<10} | {'Bal Acc':<10} | {'Val Loss'}")
    print("-" * 70)
    for log in logs_sampler:
        print(f"{'WeightedSampler + CE':<25} | {log['epoch']:<6} | {log['c0_recall']:<10.4f} | {log['c1_recall']:<10.4f} | {log['bal_acc']:<10.4f} | {log['val_loss']:.4f}")
    print("-" * 70)
    for log in logs_focal:
        print(f"{'Focal Loss (gamma=2.0)':<25} | {log['epoch']:<6} | {log['c0_recall']:<10.4f} | {log['c1_recall']:<10.4f} | {log['bal_acc']:<10.4f} | {log['val_loss']:.4f}")
    print("=" * 70)

    # Save the better model checkpoint to best_model.pt
    best_sampler_bal = max(l["bal_acc"] for l in logs_sampler)
    best_focal_bal = max(l["bal_acc"] for l in logs_focal)

    if best_sampler_bal >= best_focal_bal:
        winner = "WeightedRandomSampler"
        best_model = model_sampler
        best_score = best_sampler_bal
    else:
        winner = "Focal Loss"
        best_model = model_focal
        best_score = best_focal_bal

    print(f"\n[+] Winning Approach: {winner} (Peak Validation Balanced Acc: {best_score:.4f})")
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "method": winner,
        "best_val_balanced_acc": best_score
    }, "best_model.pt")
    print(f"[+] Saved {winner} model to 'best_model.pt'")

if __name__ == "__main__":
    main()
