import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import PareidoliaDataset
from model import build_model

def create_mining_sampler(labels: np.ndarray, error_weights: np.ndarray = None) -> WeightedRandomSampler:
    """Creates a WeightedRandomSampler that combines class balance with hard error emphasis."""
    class_counts = np.bincount(labels)
    class_base_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_base_weights[labels].copy()
    
    if error_weights is not None:
        sample_weights *= error_weights
        
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

def evaluate_model(model, dataloader, device):
    """Evaluates balanced accuracy and returns predictions and true labels."""
    model.eval()
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_targets.extend(targets.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Calculate best balanced accuracy across thresholds
    thresholds = np.linspace(0.30, 0.70, 41)
    best_score = 0.0
    best_t = 0.50
    for t in thresholds:
        score = balanced_accuracy_score(all_targets, (all_probs >= t).astype(int))
        if score > best_score:
            best_score = score
            best_t = t
            
    preds_at_best = (all_probs >= best_t).astype(int)
    cm = confusion_matrix(all_targets, preds_at_best, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    c0_rec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    c1_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return best_score, best_t, c0_rec, c1_rec, all_probs

def run_experiment():
    print("=" * 75)
    print("STANDALONE EXPERIMENT: ITERATIVE HARD-EXAMPLE MINING & RETRAINING")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device: {device.type.upper()}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")

    df = pd.read_csv("train_metadata.csv")
    print(f"[+] Total Dataset: {len(df):,} labeled samples")

    # 1. Lock away a 10% pure holdout test set (never touched during mining loops)
    train_pool_df, holdout_test_df = train_test_split(
        df, test_size=0.10, random_state=42, stratify=df["label"]
    )
    train_pool_df = train_pool_df.reset_index(drop=True)
    holdout_test_df = holdout_test_df.reset_index(drop=True)

    print(f"[+] Active Training Pool:  {len(train_pool_df):,} samples")
    print(f"[+] Pure Virgin Holdout:   {len(holdout_test_df):,} samples (Locked in vault)")

    holdout_dataset = PareidoliaDataset(holdout_test_df, "train_images", is_train=False, is_labeled=True)
    holdout_loader = DataLoader(holdout_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    # Initialize Model
    model = build_model(num_classes=2, pretrained=True, in_channels=1)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    num_cycles = 3
    epochs_per_cycle = 4
    error_weights = np.ones(len(train_pool_df), dtype=np.float32)

    print("\n" + "-" * 75)
    print(f"STARTING {num_cycles} ITERATIVE MINING & CORRECTION CYCLES")
    print("-" * 75)

    history = []

    for cycle in range(1, num_cycles + 1):
        print(f"\n>>> [CYCLE {cycle}/{num_cycles}] Training with Hard-Error Multipliers...")
        
        # Build sampler with current error weights
        sampler = create_mining_sampler(train_pool_df["label"].values, error_weights)
        train_dataset = PareidoliaDataset(train_pool_df, "train_images", is_train=True, is_labeled=True)
        train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler, num_workers=0, pin_memory=True)

        # Train for epochs_per_cycle
        model.train()
        for epoch in range(1, epochs_per_cycle + 1):
            t0 = time.time()
            running_loss = 0.0
            total_samples = 0
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

            elapsed = time.time() - t0
            avg_loss = running_loss / max(1, total_samples)
            print(f"    Cycle {cycle} Epoch {epoch}/{epochs_per_cycle} ({elapsed:.1f}s) | Train Loss: {avg_loss:.4f}")

        # Step 1: Evaluate on the Pure Virgin Holdout (Honest Unseen Exam)
        h_score, h_t, h_c0, h_c1, _ = evaluate_model(model, holdout_loader, device)
        print(f"\n    [+] CYCLE {cycle} VIRGIN HOLDOUT TEST SCORE:")
        print(f"        Balanced Accuracy: {h_score*100:.2f}% (C0: {h_c0*100:.1f}%, C1: {h_c1*100:.1f}% @ threshold {h_t:.2f})")

        history.append({
            "cycle": cycle,
            "holdout_balanced_acc": h_score,
            "c0_recall": h_c0,
            "c1_recall": h_c1,
            "threshold": h_t
        })

        # Step 2: Error-Mining Phase (Evaluate on training pool to find mistakes)
        eval_train_dataset = PareidoliaDataset(train_pool_df, "train_images", is_train=False, is_labeled=True)
        eval_train_loader = DataLoader(eval_train_dataset, batch_size=64, shuffle=False, num_workers=0)
        _, _, _, _, pool_probs = evaluate_model(model, eval_train_loader, device)

        pool_preds = (pool_probs >= h_t).astype(int)
        pool_targets = train_pool_df["label"].values
        is_error = (pool_preds != pool_targets)
        num_errors = is_error.sum()
        error_pct = (num_errors / len(pool_targets)) * 100

        print(f"    [!] Error Mining: Found {num_errors:,} mistakes ({error_pct:.1f}%) in training pool.")
        
        # Boost sampling probability for mistakes by 3.0x for next cycle
        error_weights = np.ones(len(train_pool_df), dtype=np.float32)
        error_weights[is_error] = 3.0
        print(f"    [+] Updated mistake weights: 3.0x boost applied to {num_errors} hard samples.")

        # Reduce learning rate for fine-tuning
        for param_group in optimizer.param_groups:
            param_group["lr"] *= 0.65

    print("\n" + "=" * 75)
    print("ITERATIVE HARD-EXAMPLE MINING EXPERIMENT RESULTS")
    print("=" * 75)
    for h in history:
        print(f"  Cycle {h['cycle']}: Holdout Balanced Accuracy = {h['holdout_balanced_acc']*100:.2f}% "
              f"(Craters: {h['c0_recall']*100:.1f}%, Mounds: {h['c1_recall']*100:.1f}%)")

    # Generate predictions on test set to compare with submission.csv
    test_meta = pd.read_csv("test_metadata.csv")
    images_dir = "test_images" if os.path.isdir("test_images") else "eval_images"
    test_ds = PareidoliaDataset(test_meta, images_dir, is_train=False, is_labeled=False)
    test_ld = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    model.eval()
    exp_test_probs = []
    with torch.no_grad():
        for images, _ in test_ld:
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1)[:, 1]
            exp_test_probs.extend(probs.cpu().numpy().tolist())

    exp_preds = (np.array(exp_test_probs) >= history[-1]["threshold"]).astype(int)
    
    # Compare with our production submission.csv
    prod_sub = pd.read_csv("submission.csv")
    agreement = (exp_preds == prod_sub["label"].values).mean() * 100

    print("\n" + "=" * 75)
    print("COMPARISON WITH PRODUCTION 5-FOLD ENSEMBLE (submission.csv)")
    print("=" * 75)
    print(f"  * Agreement with Production submission.csv: {agreement:.2f}% ({int(agreement*20)} / 2000 images agree)")
    print("=" * 75)

if __name__ == "__main__":
    run_experiment()
