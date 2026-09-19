import os
import sys
import argparse
import time
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from dataset import PareidoliaDataset
from model import build_model

def parse_args():
    parser = argparse.ArgumentParser(description="Test-Time Augmentation (TTA) + 5-Fold Ensemble Inference")
    parser.add_argument("--test_csv", type=str, default="test_metadata.csv", help="Path to test metadata CSV")
    parser.add_argument("--images_dir", type=str, default="eval_images", help="Path to test images directory")
    parser.add_argument("--models_dir", type=str, default=".", help="Directory containing model_fold0.pt to model_fold4.pt")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of fold models to ensemble")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold for Class 1 (default: auto-loaded from optimal_threshold.json or 0.47)")
    parser.add_argument("--sfs_alpha", type=float, default=0.10, help="Shape-from-Shading physical prior blend weight (default: 0.10)")
    parser.add_argument("--output_csv", type=str, default="submission.csv", help="Output submission CSV path")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers (0 for Windows)")
    return parser.parse_args()

def load_fold_models(models_dir, n_folds, device):
    models = []
    val_scores = []
    for fold in range(n_folds):
        ckpt_name = f"model_fold{fold}.pt"
        ckpt_path = os.path.join(models_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint '{ckpt_path}' not found! Ensure Step 11 completed training all folds.")
        
        print(f"[+] Loading fold model {fold} from '{ckpt_path}'...")
        model = build_model(num_classes=2, pretrained=False, in_channels=1)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        val_score = 0.6958
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            val_score = ckpt.get("best_val_balanced_acc", 0.6958)
            print(f"    Loaded fold {fold} (Saved Bal Acc: {val_score:.4f})")
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        else:
            model = ckpt
        
        model.to(device)
        model.eval()
        models.append(model)
        val_scores.append(val_score)

    # Compute validation-quality weights: w_i proportional to (val_score - 0.50)^2
    excess_scores = np.maximum(np.array(val_scores) - 0.50, 0.01) ** 2
    fold_weights = excess_scores / excess_scores.sum()
    print("\n[+] Validation-Score-Weighted Ensembling:")
    for f_idx, (score, w) in enumerate(zip(val_scores, fold_weights)):
        print(f"    Fold {f_idx}: Bal Acc = {score:.4f} | Ensemble Weight = {w*100:.2f}%")
    
    return models, torch.tensor(fold_weights, dtype=torch.float32, device=device)

def run_tta_ensemble_inference():
    args = parse_args()

    print("=" * 75)
    print("STEP 12: DEEP PHYSICS-SAFE TTA (8 VIEWS) + WEIGHTED 5-FOLD ENSEMBLE")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device: {device.type.upper()}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")

    # Resolve images directory automatically
    images_dir = args.images_dir
    if not os.path.isdir(images_dir):
        if os.path.isdir("test_images"):
            images_dir = "test_images"
        elif os.path.isdir("eval_images"):
            images_dir = "eval_images"
        else:
            raise FileNotFoundError(f"Test images directory '{args.images_dir}', 'test_images', or 'eval_images' not found.")

    test_df = pd.read_csv(args.test_csv)
    print(f"[+] Loaded test metadata: '{args.test_csv}' ({len(test_df):,} rows)")
    print(f"[+] Using test images directory: '{images_dir}'")
    assert "image_id" in test_df.columns and "sun_azimuth_angle" in test_df.columns, \
        "test_metadata.csv must contain 'image_id' and 'sun_azimuth_angle'"

    # 1. Load all 5 fold models with validation weights
    fold_models, fold_weights = load_fold_models(args.models_dir, args.n_folds, device)
    print(f"[+] Successfully loaded {len(fold_models)} fold models into {device.type.upper()} memory.")

    # 2. Test Dataset & DataLoader (azimuth normalization is applied inside PareidoliaDataset)
    test_dataset = PareidoliaDataset(
        metadata=test_df,
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

    print(f"\n[+] Executing Deep Orientation-Preserving TTA on {len(test_dataset):,} test images...")
    print("    Physics-Safe TTA Views per sample (Sun strictly locked to North/Top):")
    print("      1) Canonical Azimuth-Rotated View (100% scale)")
    print("      2) Multi-Scale Center Crop (96% scale)")
    print("      3) Multi-Scale Center Crop (92% scale)")
    print("      4) Photometric Perturbation (+5% Contrast)")
    print("      5) Photometric Perturbation (-5% Contrast)")
    print("      6) Photometric Perturbation (+10% Contrast)")
    print("      7) Micro-Rotation (+2.5° Jitter with Reflect-Crop)")
    print("      8) Micro-Rotation (-2.5° Jitter with Reflect-Crop)")
    print(f"    Total Predictions = 8 views x 5 models = 40 passes / image\n")

    all_image_ids = []
    all_ensemble_probs = []
    start_time = time.time()

    # Predefine crops
    h, w = 256, 256
    c96_h, c96_w = int(h * 0.96), int(w * 0.96)
    top_96, left_96 = (h - c96_h) // 2, (w - c96_w) // 2

    c92_h, c92_w = int(h * 0.92), int(w * 0.92)
    top_92, left_92 = (h - c92_h) // 2, (w - c92_w) // 2

    with torch.no_grad():
        for batch_idx, (images, image_ids) in enumerate(test_loader):
            # images shape: (B, 1, 256, 256) - already azimuth-rotated (Sun at North/Top)
            images = images.to(device, non_blocking=True)
            
            # View 1: Original
            v1_orig = images
            
            # View 2: 96% Scale Crop
            v2_crop = images[:, :, top_96:top_96 + c96_h, left_96:left_96 + c96_w]
            v2_scaled = torch.nn.functional.interpolate(v2_crop, size=(256, 256), mode="bicubic", align_corners=False)

            # View 3: 92% Scale Crop
            v3_crop = images[:, :, top_92:top_92 + c92_h, left_92:left_92 + c92_w]
            v3_scaled = torch.nn.functional.interpolate(v3_crop, size=(256, 256), mode="bicubic", align_corners=False)
            
            # Views 4, 5, 6: Contrast variants
            v4_c1 = images * 1.05
            v5_c2 = images * 0.95
            v6_c3 = images * 1.10

            # Views 7 & 8: Micro-rotations (+2.5° and -2.5°) with bicubic grid sample
            # Affine grid rotation without black borders
            theta_pos = 2.5 * (np.pi / 180.0)
            cos_p, sin_p = np.cos(theta_pos), np.sin(theta_pos)
            rot_mat_pos = torch.tensor([[cos_p, -sin_p, 0], [sin_p, cos_p, 0]], dtype=torch.float32, device=device).unsqueeze(0).repeat(images.size(0), 1, 1)
            grid_pos = torch.nn.functional.affine_grid(rot_mat_pos, images.size(), align_corners=False)
            v7_rot_pos = torch.nn.functional.grid_sample(images, grid_pos, mode='bicubic', padding_mode='reflection', align_corners=False)

            theta_neg = -2.5 * (np.pi / 180.0)
            cos_n, sin_n = np.cos(theta_neg), np.sin(theta_neg)
            rot_mat_neg = torch.tensor([[cos_n, -sin_n, 0], [sin_n, cos_n, 0]], dtype=torch.float32, device=device).unsqueeze(0).repeat(images.size(0), 1, 1)
            grid_neg = torch.nn.functional.affine_grid(rot_mat_neg, images.size(), align_corners=False)
            v8_rot_neg = torch.nn.functional.grid_sample(images, grid_neg, mode='bicubic', padding_mode='reflection', align_corners=False)
            
            views = [v1_orig, v2_scaled, v3_scaled, v4_c1, v5_c2, v6_c3, v7_rot_pos, v8_rot_neg]

            model_batch_probs = []
            for model in fold_models:
                view_probs = []
                for v in views:
                    p = torch.softmax(model(v), dim=1)[:, 1]
                    view_probs.append(p)
                p_model_avg = torch.stack(view_probs, dim=0).mean(dim=0)
                model_batch_probs.append(p_model_avg)

            # Weighted average across all 5 fold models according to validation scores
            stacked_probs = torch.stack(model_batch_probs, dim=0) # (5, B)
            neural_probs = (stacked_probs * fold_weights.view(-1, 1)).sum(dim=0) # (B,)

            # Physical Shape-from-Shading (SfS) Prior on central 128x128 crop (Sun locked to North)
            # Top half vs Bottom half photometric luminance gradient
            center_crop = images[:, 0, 64:192, 64:192]
            top_half = center_crop[:, :64, :]
            bot_half = center_crop[:, 64:, :]
            delta_i = top_half.mean(dim=(1, 2)) - bot_half.mean(dim=(1, 2))
            sfs_probs = torch.sigmoid(15.0 * delta_i)

            # Blend neural ensemble with physical SfS prior
            alpha = args.sfs_alpha
            blended_probs = (1.0 - alpha) * neural_probs + alpha * sfs_probs

            all_ensemble_probs.extend(blended_probs.cpu().numpy().tolist())
            all_image_ids.extend(list(image_ids))

            processed = len(all_image_ids)
            if (batch_idx + 1) % 5 == 0 or processed >= len(test_dataset):
                pct = (processed / len(test_dataset)) * 100
                print(f"  [>] Processed {processed:>4}/{len(test_dataset)} images ({pct:>5.1f}%) ...", flush=True)

    elapsed_time = time.time() - start_time
    print(f"[+] Finished inference in {elapsed_time:.1f}s ({elapsed_time/len(test_dataset)*1000:.1f}ms / image).")

    import json
    # 3. Apply Decision Threshold
    if args.threshold is not None:
        chosen_threshold = args.threshold
        print(f"[+] Using user-specified threshold: {chosen_threshold:.4f}")
    elif os.path.exists("optimal_threshold.json"):
        with open("optimal_threshold.json", "r") as f:
            thresh_data = json.load(f)
        chosen_threshold = float(thresh_data.get("optimal_threshold", 0.47))
        print(f"[+] Loaded calibrated optimal threshold from 'optimal_threshold.json': {chosen_threshold:.4f}")
    else:
        chosen_threshold = 0.47
        print(f"[!] 'optimal_threshold.json' not found, defaulting to: {chosen_threshold:.4f}")

    all_ensemble_probs = np.array(all_ensemble_probs)
    preds = (all_ensemble_probs >= chosen_threshold).astype(int)

    submission_df = pd.DataFrame({
        "image_id": all_image_ids,
        "label": preds
    })

    # Save to submission.csv and also submission_final.csv backup
    submission_df.to_csv(args.output_csv, index=False)
    if args.output_csv != "submission_final.csv":
        submission_df.to_csv("submission_final.csv", index=False)
    print(f"\n[+] Saved final submission file to '{args.output_csv}' (and backup 'submission_final.csv')")

    # 4. Strict Validation of Submission Requirements
    print("\n" + "=" * 75)
    print("SUBMISSION VERIFICATION & QUALITY CHECKS")
    print("=" * 75)
    print(f"  * File Path:             {args.output_csv}")
    print(f"  * Exact Row Count:       {len(submission_df):,} rows (Expected: 2,000)")
    print(f"  * Columns:               {list(submission_df.columns)} (Expected: ['image_id', 'label'])")
    print(f"  * Image IDs match test:  {list(submission_df['image_id']) == list(test_df['image_id'])}")
    print(f"  * Decision Threshold:    {chosen_threshold:.2f}")

    c0_count = (preds == 0).sum()
    c1_count = (preds == 1).sum()
    print("\n[+] Final Class Prediction Distribution:")
    print(f"    Class 0: {c0_count:>4} ({c0_count / len(preds) * 100:.2f}%)")
    print(f"    Class 1: {c1_count:>4} ({c1_count / len(preds) * 100:.2f}%)")
    print(f"    Ensemble P(Class 1) - Min: {all_ensemble_probs.min():.4f}, Mean: {all_ensemble_probs.mean():.4f}, Max: {all_ensemble_probs.max():.4f}")

    print("\n[+] First 5 rows of generated submission file:")
    print(submission_df.head(5).to_string(index=False))
    print("=" * 75)

if __name__ == "__main__":
    run_tta_ensemble_inference()
