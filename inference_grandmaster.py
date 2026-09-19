import os
import sys
import argparse
import time
import json
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader

from dataset import PareidoliaDataset, compute_physics_channels
from model import build_model

def parse_args():
    parser = argparse.ArgumentParser(description="Grandmaster 40-Pass TTA Inference Pipeline")
    parser.add_argument("--test_csv", type=str, default="test_metadata.csv", help="Path to test metadata CSV")
    parser.add_argument("--images_dir", type=str, default="test_images", help="Path to test images directory")
    parser.add_argument("--models_dir", type=str, default=".", help="Directory containing model_gm_fold0.pt to model_gm_fold4.pt")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of fold models to ensemble")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold")
    parser.add_argument("--sfs_alpha", type=float, default=0.08, help="Shape-from-Shading physical prior weight")
    parser.add_argument("--output_csv", type=str, default="submission.csv", help="Output submission CSV path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    return parser.parse_args()

def load_grandmaster_models(models_dir, n_folds, device):
    models = []
    val_scores = []
    for fold in range(n_folds):
        ckpt_path = os.path.join(models_dir, f"model_gm_fold{fold}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint '{ckpt_path}' not found! Run train_grandmaster.py first.")
        
        m = build_model(arch="resnet18", num_classes=2, pretrained=False, in_channels=3)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        val_score = 0.70
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            m.load_state_dict(ckpt["model_state_dict"])
            val_score = ckpt.get("best_val_balanced_acc", 0.70)
        elif isinstance(ckpt, dict):
            m.load_state_dict(ckpt)
        else:
            m = ckpt
            
        m.to(device)
        m.eval()
        models.append(m)
        val_scores.append(val_score)
        
    excess = np.maximum(np.array(val_scores) - 0.50, 0.01) ** 2
    fold_weights = torch.tensor(excess / excess.sum(), dtype=torch.float32, device=device)
    print(f"[+] Loaded {len(models)} Grandmaster models. Mean validation score: {np.mean(val_scores)*100:.2f}%")
    return models, fold_weights

def run_grandmaster_inference():
    args = parse_args()
    print("=" * 75)
    print("GRANDMASTER 40-PASS TTA + 3-CHANNEL PHYSICS INFERENCE PIPELINE")
    print("=" * 75)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device: {device}")
    
    images_dir = args.images_dir
    if not os.path.isdir(images_dir):
        if os.path.isdir("eval_images"):
            images_dir = "eval_images"
        elif os.path.isdir("test_images"):
            images_dir = "test_images"
            
    test_df = pd.read_csv(args.test_csv)
    print(f"[+] Loaded test metadata: '{args.test_csv}' ({len(test_df):,} rows)")
    
    # 1. Load 5 Grandmaster fold models
    fold_models, fold_weights = load_grandmaster_models(args.models_dir, args.n_folds, device)
    
    # 2. Test Dataset (3-Channel Physics Enabled)
    test_dataset = PareidoliaDataset(metadata=test_df, images_dir=images_dir, is_train=False, use_physics_3ch=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))
    
    all_ensemble_probs = []
    all_image_ids = []
    start_time = time.time()
    
    print("\n[+] Executing 40-Pass Physics TTA (8 Views x 5 Fold Models)...")
    
    # Crop constants
    c96_h = int(256 * 0.96)
    c96_w = int(256 * 0.96)
    top_96 = (256 - c96_h) // 2
    left_96 = (256 - c96_w) // 2
    
    c92_h = int(256 * 0.92)
    c92_w = int(256 * 0.92)
    top_92 = (256 - c92_h) // 2
    left_92 = (256 - c92_w) // 2
    
    with torch.no_grad():
        for batch_idx, (images, image_ids) in enumerate(test_loader):
            images = images.to(device) # (B, 3, 256, 256)
            
            # View 1: Canonical
            v1_orig = images
            
            # View 2 & 3: Multi-Scale Center Crops (96% and 92%)
            v2_crop = images[:, :, top_96:top_96 + c96_h, left_96:left_96 + c96_w]
            v2_scaled = torch.nn.functional.interpolate(v2_crop, size=(256, 256), mode="bicubic", align_corners=False)
            
            v3_crop = images[:, :, top_92:top_92 + c92_h, left_92:left_92 + c92_w]
            v3_scaled = torch.nn.functional.interpolate(v3_crop, size=(256, 256), mode="bicubic", align_corners=False)
            
            # Views 4, 5, 6: Contrast variants on intensity channel
            v4 = images.clone(); v4[:, 0:1] = images[:, 0:1] * 1.05
            v5 = images.clone(); v5[:, 0:1] = images[:, 0:1] * 0.95
            v6 = images.clone(); v6[:, 0:1] = images[:, 0:1] * 1.10
            
            # Views 7 & 8: Micro-rotations (+2.5° and -2.5°)
            theta_pos = 2.5 * (np.pi / 180.0)
            rot_mat_pos = torch.tensor([[np.cos(theta_pos), -np.sin(theta_pos), 0],
                                        [np.sin(theta_pos),  np.cos(theta_pos), 0]], dtype=torch.float32, device=device).unsqueeze(0).repeat(images.size(0), 1, 1)
            grid_pos = torch.nn.functional.affine_grid(rot_mat_pos, images.size(), align_corners=False)
            v7 = torch.nn.functional.grid_sample(images, grid_pos, mode='bicubic', padding_mode='reflection', align_corners=False)
            
            theta_neg = -2.5 * (np.pi / 180.0)
            rot_mat_neg = torch.tensor([[np.cos(theta_neg), -np.sin(theta_neg), 0],
                                        [np.sin(theta_neg),  np.cos(theta_neg), 0]], dtype=torch.float32, device=device).unsqueeze(0).repeat(images.size(0), 1, 1)
            grid_neg = torch.nn.functional.affine_grid(rot_mat_neg, images.size(), align_corners=False)
            v8 = torch.nn.functional.grid_sample(images, grid_neg, mode='bicubic', padding_mode='reflection', align_corners=False)
            
            views = [v1_orig, v2_scaled, v3_scaled, v4, v5, v6, v7, v8]
            
            model_batch_probs = []
            for model in fold_models:
                view_probs = [torch.softmax(model(v), dim=1)[:, 1] for v in views]
                p_model_avg = torch.stack(view_probs, dim=0).mean(dim=0)
                model_batch_probs.append(p_model_avg)
                
            model_stack = torch.stack(model_batch_probs, dim=1) # (B, 5)
            neural_probs = torch.sum(model_stack * fold_weights.unsqueeze(0), dim=1) # (B,)
            
            # Central Shape-from-Shading Photometric Prior
            ch0 = images[:, 0:1, 64:192, 64:192]
            top_half = ch0[:, :, :64, :].mean(dim=(1, 2, 3))
            bot_half = ch0[:, :, 64:, :].mean(dim=(1, 2, 3))
            delta_i = top_half - bot_half
            sfs_probs = torch.sigmoid(15.0 * delta_i)
            
            alpha = args.sfs_alpha
            blended_probs = (1.0 - alpha) * neural_probs + alpha * sfs_probs
            
            all_ensemble_probs.extend(blended_probs.cpu().numpy().tolist())
            all_image_ids.extend(list(image_ids))
            
            if (batch_idx + 1) % 15 == 0 or len(all_image_ids) >= len(test_dataset):
                print(f"  [>] Processed {len(all_image_ids):>4}/{len(test_dataset)} images ({(len(all_image_ids)/len(test_dataset))*100:.1f}%) ...", flush=True)
                
    elapsed = time.time() - start_time
    print(f"[+] 40-Pass Inference finished in {elapsed:.1f}s ({elapsed/len(test_dataset)*1000:.1f}ms / image).")
    
    # 3. Apply Decision Threshold
    if args.threshold is not None:
        chosen_threshold = args.threshold
    elif os.path.exists("optimal_threshold_gm.json"):
        with open("optimal_threshold_gm.json", "r") as f:
            t_data = json.load(f)
        chosen_threshold = float(t_data.get("optimal_threshold", 0.47))
        print(f"[+] Loaded Grandmaster calibrated optimal threshold: {chosen_threshold:.4f}")
    else:
        chosen_threshold = 0.47
        
    all_ensemble_probs = np.array(all_ensemble_probs)
    raw_preds = (all_ensemble_probs >= chosen_threshold).astype(int)
    
    # 4. Gated Multi-Scale Center Zoom Refinement on Uncertainty Zone [0.40, 0.60]
    print("\n[+] Applying Gated Multi-Scale Center Zoom & Photometric Profile Refinement...")
    refined_preds = []
    flipped_count = 0
    test_meta_indexed = test_df.set_index("image_id")
    pad_size = 128
    
    for i, img_id in enumerate(all_image_ids):
        raw_p = all_ensemble_probs[i]
        curr_label = raw_preds[i]
        
        if 0.40 <= raw_p <= 0.60:
            azimuth = float(test_meta_indexed.loc[img_id, "sun_azimuth_angle"])
            img_path = os.path.join(images_dir, img_id)
            img_pil = Image.open(img_path).convert("L")
            
            img_pad = Image.fromarray(np.pad(np.array(img_pil), pad_size, mode='reflect'))
            img_rot = img_pad.rotate(-azimuth, resample=Image.Resampling.BICUBIC)
            w, h = img_rot.size
            img_north = img_rot.crop((w//2 - 128, h//2 - 128, w//2 + 128, h//2 + 128))
            
            img_np = np.array(img_north, dtype=np.float32) / 255.0
            
            # 1D Solar Profile
            H_c, W_c = img_np.shape
            m_y, m_x = int(H_c * 0.225), int(W_c * 0.225)
            center_crop = img_np[m_y:H_c - m_y, m_x:W_c - m_x]
            prof_y = np.mean(center_crop, axis=1)
            half = len(prof_y) // 2
            delta_i = np.mean(prof_y[:half]) - np.mean(prof_y[half:])
            phys_pred = 1 if delta_i > 0 else 0
            
            # Zoom crops with 3-channel physics
            c160 = img_north.crop((128 - 80, 128 - 80, 128 + 80, 128 + 80)).resize((256, 256), Image.Resampling.BICUBIC)
            c192 = img_north.crop((128 - 96, 128 - 96, 128 + 96, 128 + 96)).resize((256, 256), Image.Resampling.BICUBIC)
            
            t160 = compute_physics_channels(torch.from_numpy(((np.array(c160, dtype=np.float32)/255.0)-0.485)/0.229).unsqueeze(0)).unsqueeze(0).to(device)
            t192 = compute_physics_channels(torch.from_numpy(((np.array(c192, dtype=np.float32)/255.0)-0.485)/0.229).unsqueeze(0)).unsqueeze(0).to(device)
            zoom_batch = torch.cat([t160, t192], dim=0)
            
            with torch.no_grad():
                z_probs = [torch.softmax(m(zoom_batch), dim=1)[:, 1] for m in fold_models]
                center_zoom_p = torch.stack(z_probs, dim=0).mean().item()
                
            zoom_model_pred = 1 if center_zoom_p >= chosen_threshold else 0
            
            if phys_pred == zoom_model_pred:
                final_l = phys_pred
            else:
                if abs(delta_i) > 0.04:
                    final_l = phys_pred
                else:
                    final_l = zoom_model_pred
                    
            if final_l != curr_label:
                flipped_count += 1
            refined_preds.append(final_l)
        else:
            refined_preds.append(curr_label)
            
    print(f"    Total Refined / Flipped in Uncertainty Band: {flipped_count} / {len(all_image_ids)} ({flipped_count/len(all_image_ids)*100:.2f}%)")
    
    sub_df = pd.DataFrame({"image_id": all_image_ids, "label": refined_preds})
    sub_df.to_csv("submission_grandmaster.csv", index=False)
    sub_df.to_csv(args.output_csv, index=False)
    print(f"\n[+] Saved Grandmaster submission file to '{args.output_csv}' and 'submission_grandmaster.csv'")
    
    c0 = (np.array(refined_preds) == 0).sum()
    c1 = (np.array(refined_preds) == 1).sum()
    print("\n[+] Final Grandmaster Prediction Distribution:")
    print(f"    Class 0 (Craters): {c0:>4} ({c0 / len(refined_preds) * 100:.2f}%)")
    print(f"    Class 1 (Mounds):  {c1:>4} ({c1 / len(refined_preds) * 100:.2f}%)")
    print("=" * 75)

if __name__ == "__main__":
    run_grandmaster_inference()
