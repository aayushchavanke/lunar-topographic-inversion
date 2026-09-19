import os
import sys
import argparse
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, recall_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import PareidoliaDataset
from model import build_model

def parse_args():
    parser = argparse.ArgumentParser(description="Grandmaster 5-Fold 3-Channel Training Pipeline")
    parser.add_argument("--train_csv", type=str, default="train_pseudo_labeled.csv", help="Path to training CSV (with pseudo labels)")
    parser.add_argument("--epochs", type=int, default=18, help="Number of training epochs per fold")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for AdamW")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for Stratified K-Fold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--arch", type=str, default="resnet18", help="Backbone architecture")
    return parser.parse_args()

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_optimal_threshold(oof_df):
    y_true = oof_df["label"].values
    y_prob = oof_df["prob_class_1"].values
    
    thresholds = np.linspace(0.20, 0.80, 121)
    best_thresh = 0.50
    best_score = 0.0
    
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        score = balanced_accuracy_score(y_true, preds)
        if score > best_score:
            best_score = score
            best_thresh = t
            
    return float(best_thresh), float(best_score)

def train_grandmaster():
    args = parse_args()
    set_seed(args.seed)
    
    print("=" * 75)
    print("GRANDMASTER 5-FOLD 3-CHANNEL PHYSICS PIPELINE")
    print(f"Arch: {args.arch} | Epochs: {args.epochs} | LR: {args.lr} | Batch: {args.batch_size}")
    print("=" * 75)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Active Device: {device}")
    
    df = pd.read_csv(args.train_csv)
    print(f"[+] Loaded dataset: {len(df):,} samples (Ground Truth + Pseudo-Labels)")
    
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof_predictions = np.zeros(len(df))
    fold_scores = []
    
    start_total_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
        print("\n" + "#" * 60)
        print(f"--- STARTING FOLD {fold + 1}/{args.n_splits} ---")
        print("#" * 60)
        
        train_sub_df = df.iloc[train_idx].reset_index(drop=True)
        val_sub_df = df.iloc[val_idx].reset_index(drop=True)
        
        # 1. Dataset with 3-Channel Physics Tensors
        train_dataset = PareidoliaDataset(metadata=train_sub_df, is_train=True, use_physics_3ch=True)
        val_dataset = PareidoliaDataset(metadata=val_sub_df, is_train=False, use_physics_3ch=True)
        
        # 2. Balanced WeightedRandomSampler for 50/50 batch class balance
        class_counts = train_sub_df["label"].value_counts().to_dict()
        sample_weights = train_sub_df["label"].map(lambda l: 1.0 / class_counts[l]).values
        sampler = WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.double), num_samples=len(sample_weights), replacement=True)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=(device.type=="cuda"))
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))
        
        # 3. Build Model (Full 3-Channel Pretrained Backbone)
        model = build_model(arch=args.arch, num_classes=2, pretrained=True, in_channels=3)
        model.to(device)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        
        best_val_bal_acc = 0.0
        best_c0_recall = 0.0
        best_c1_recall = 0.0
        best_val_probs = None
        patience_counter = 0
        best_model_path = f"model_gm_fold{fold}.pt"
        
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(labels)
                
                if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                    print(f"    [Fold {fold+1} Epoch {epoch:02d}] Batch {batch_idx+1:>3}/{len(train_loader)} | Running Loss: {loss.item():.4f}", flush=True)
                
            scheduler.step()
            train_loss /= len(train_dataset)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            val_probs_epoch = []
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * len(labels)
                    
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    preds = torch.argmax(outputs, dim=1)
                    
                    val_probs_epoch.extend(probs.cpu().numpy().tolist())
                    val_preds.extend(preds.cpu().numpy().tolist())
                    val_targets.extend(labels.cpu().numpy().tolist())
                    
            val_loss /= len(val_dataset)
            val_bal_acc = balanced_accuracy_score(val_targets, val_preds)
            c0_rec = recall_score(val_targets, val_preds, pos_label=0, zero_division=0)
            c1_rec = recall_score(val_targets, val_preds, pos_label=1, zero_division=0)
            
            saved_indicator = ""
            if val_bal_acc > best_val_bal_acc:
                best_val_bal_acc = val_bal_acc
                best_c0_recall = c0_rec
                best_c1_recall = c1_rec
                best_val_probs = np.array(val_probs_epoch)
                patience_counter = 0
                saved_indicator = " [* BEST SAVED]"
                
                torch.save({
                    "fold": fold,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_balanced_acc": best_val_bal_acc,
                    "c0_recall": best_c0_recall,
                    "c1_recall": best_c1_recall
                }, best_model_path)
            else:
                patience_counter += 1
                
            print(f"  --> Fold {fold+1} Epoch [{epoch:02d}/{args.epochs:02d}] TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | Val Bal Acc: {val_bal_acc*100:.2f}% (R0: {c0_rec*100:.1f}%, R1: {c1_rec*100:.1f}%){saved_indicator}", flush=True)
            
            if patience_counter >= args.patience:
                print(f"  [!] Early stopping triggered at epoch {epoch} (Patience: {args.patience})")
                break
                
        print(f"[+] Fold {fold + 1} Best Validation Balanced Accuracy: {best_val_bal_acc * 100:.2f}%")
        fold_scores.append(best_val_bal_acc)
        oof_predictions[val_idx] = best_val_probs
        
    total_elapsed = time.time() - start_total_time
    print("\n" + "=" * 75)
    print("GRANDMASTER 5-FOLD TRAINING COMPLETE!")
    print("=" * 75)
    print(f"Total Training Time: {total_elapsed / 60:.1f} minutes")
    for f_idx, score in enumerate(fold_scores):
        print(f"  Fold {f_idx + 1}: Best Val Balanced Accuracy = {score * 100:.2f}%")
    print(f"Mean 5-Fold Balanced Accuracy: {np.mean(fold_scores) * 100:.2f}% (+/- {np.std(fold_scores) * 100:.2f}%)")
    
    # Save OOF Predictions
    df["prob_class_1"] = oof_predictions
    oof_csv_path = "oof_gm_predictions.csv"
    df.to_csv(oof_csv_path, index=False)
    print(f"[+] Saved OOF predictions to '{oof_csv_path}'")
    
    # Threshold Tuning on OOF
    best_thresh, best_oof_score = compute_optimal_threshold(df)
    print(f"[+] Calibrated Optimal Threshold: tau* = {best_thresh:.4f}")
    print(f"[+] Global OOF Balanced Accuracy at tau* = {best_thresh:.4f}: {best_oof_score * 100:.2f}%")
    
    thresh_data = {
        "optimal_threshold": best_thresh,
        "best_oof_balanced_acc": best_oof_score,
        "mean_fold_score": float(np.mean(fold_scores)),
        "fold_scores": fold_scores
    }
    with open("optimal_threshold_gm.json", "w") as f:
        json.dump(thresh_data, f, indent=4)
    print("[+] Saved threshold parameters to 'optimal_threshold_gm.json'")
    print("=" * 75)

if __name__ == "__main__":
    train_grandmaster()
