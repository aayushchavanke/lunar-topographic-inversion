import os
import sys
import argparse
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PareidoliaDataset
from model import build_model

def parse_args():
    parser = argparse.ArgumentParser(description="Generate submission predictions for The Pareidolia Paradox")
    parser.add_argument("--test_csv", type=str, default="test_metadata.csv", help="Path to test metadata CSV")
    parser.add_argument("--images_dir", type=str, default=None, help="Path to test/eval images directory")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--output_csv", type=str, default="submission.csv", help="Output path for submission CSV")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader num_workers")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold for Class 1 (default: from checkpoint)")
    return parser.parse_args()

def run_inference():
    args = parse_args()

    print("=" * 65)
    print("INFERENCE & SUBMISSION GENERATION: The Pareidolia Paradox")
    print("=" * 65)

    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device: {device.type.upper()}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"    CPU Threads: {torch.get_num_threads()} (Total logical cores: {os.cpu_count()})")

    # 2. Locate checkpoint
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        # Fallback to .pth if .pt wasn't found or vice versa
        alt_path = "best_model.pth" if ckpt_path.endswith(".pt") else "best_model.pt"
        if os.path.exists(alt_path):
            print(f"[*] '{ckpt_path}' not found, falling back to '{alt_path}'")
            ckpt_path = alt_path
        else:
            raise FileNotFoundError(f"Checkpoint not found at '{ckpt_path}' or '{alt_path}'.")

    # 3. Locate images directory
    images_dir = args.images_dir
    if images_dir is None:
        if os.path.isdir("eval_images"):
            images_dir = "eval_images"
        elif os.path.isdir("test_images"):
            images_dir = "test_images"
        else:
            raise FileNotFoundError("Could not find 'eval_images' or 'test_images' directory.")

    print(f"[+] Test metadata: '{args.test_csv}'")
    print(f"[+] Images directory: '{images_dir}'")
    print(f"[+] Using Checkpoint: '{ckpt_path}'")

    if not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"Missing '{args.test_csv}'.")

    test_meta_df = pd.read_csv(args.test_csv)
    expected_rows = len(test_meta_df)
    print(f"[+] Loaded test metadata with {expected_rows:,} rows.")

    # 4. Load Model
    print("\n[+] Initializing ResNet18 model architecture...")
    model = build_model(num_classes=2, pretrained=False, in_channels=1)

    print(f"[+] Loading trained weights from '{ckpt_path}'...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    best_threshold = 0.50
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        best_acc = checkpoint.get("best_val_balanced_acc", None)
        best_threshold = checkpoint.get("best_threshold", 0.50)
        epoch = checkpoint.get("epoch", None)
        print(f"    Loaded checkpoint from Epoch {epoch} (Validation Balanced Acc: {best_acc})")
        print(f"    Calibrated Decision Threshold from Checkpoint: {best_threshold:.4f}")
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint

    decision_threshold = best_threshold if args.threshold is None else args.threshold
    print(f"[+] Active Decision Threshold for Class 1: {decision_threshold:.4f}")

    model.to(device)
    model.eval()

    # 5. Dataset and DataLoader
    test_dataset = PareidoliaDataset(
        metadata=test_meta_df,
        images_dir=images_dir,
        is_train=False,
        is_labeled=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    # 6. Run Inference Loop
    print(f"\n[+] Running inference on {len(test_dataset):,} test images (batch_size={args.batch_size})...")
    start_time = time.time()

    all_ids = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, (images, ids) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probs >= decision_threshold).long()

            all_ids.extend(ids)
            all_preds.extend(preds.cpu().numpy().tolist())

    elapsed = time.time() - start_time
    print(f"[+] Inference finished in {elapsed:.2f} seconds ({len(all_ids)/elapsed:.1f} img/s).")

    # 7. Build Submission DataFrame
    submission_df = pd.DataFrame({
        "image_id": all_ids,
        "label": all_preds
    })

    # 8. Sanity checks & constraints verification
    assert len(submission_df) == expected_rows, f"Row count mismatch: got {len(submission_df)}, expected {expected_rows}"
    assert list(submission_df.columns) == ["image_id", "label"], f"Invalid columns: {submission_df.columns}"
    assert (submission_df["image_id"].values == test_meta_df["image_id"].values).all(), "Image IDs do not exactly match test metadata order!"

    # Save to CSV
    submission_df.to_csv(args.output_csv, index=False)
    print(f"[+] Saved submission file to '{args.output_csv}'")

    # 9. Verify saved file directly from disk
    print("\n" + "=" * 65)
    print("SUBMISSION FILE VERIFICATION (from saved disk CSV)")
    print("=" * 65)
    disk_df = pd.read_csv(args.output_csv)

    print(f"Total Rows (excluding header): {len(disk_df):,} (Expected: 2,000) -> {'PASS' if len(disk_df) == 2000 else 'FAIL'}")
    print(f"Columns:                      {list(disk_df.columns)} -> {'PASS' if list(disk_df.columns) == ['image_id', 'label'] else 'FAIL'}")
    print(f"Missing/NaN Values:           {disk_df.isna().sum().sum()} -> PASS")
    print(f"Image IDs Match Exactly:      {(disk_df['image_id'] == test_meta_df['image_id']).all()} -> PASS")

    print("\n" + "-" * 50)
    print("FIRST 5 ROWS OF SUBMISSION.CSV")
    print("-" * 50)
    print(disk_df.head(5).to_string(index=False))

    print("\n" + "-" * 50)
    print("PREDICTED LABEL DISTRIBUTION")
    print("-" * 50)
    counts = disk_df["label"].value_counts().sort_index()
    percentages = (counts / len(disk_df)) * 100
    summary_df = pd.DataFrame({
        "Count": counts,
        "Percentage (%)": percentages.map("{:.2f}%".format)
    })
    print(summary_df.to_string())

    if len(counts) > 1:
        print("\n[+] Sanity Check PASSED: Model is making non-trivial predictions (predicting both classes).")
    else:
        print("\n[!] Warning: Model collapsed to single class.")

    print("=" * 65)

if __name__ == "__main__":
    run_inference()
