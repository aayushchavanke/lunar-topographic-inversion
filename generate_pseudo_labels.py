import os
import pandas as pd
import numpy as np

def generate_pseudo_labels(
    scan_csv="full_2000_physics_scan.csv",
    train_csv="train_metadata.csv",
    output_csv="train_pseudo_labeled.csv",
    confidence_upper=0.80,
    confidence_lower=0.20
):
    print("=" * 70)
    print("GRANDMASTER STEP 1: HIGH-CONFIDENCE PSEUDO-LABEL EXTRACTION")
    print("=" * 70)
    
    scan_df = pd.read_csv(scan_csv)
    train_df = pd.read_csv(train_csv)
    
    print(f"[+] Loaded ground truth training set: {len(train_df):,} samples")
    print(f"[+] Loaded test scan predictions: {len(scan_df):,} samples")
    
    # Filter for high-confidence predictions
    # Confidence upper: global_prob >= 0.80 -> label = 1
    # Confidence lower: global_prob <= 0.20 -> label = 0
    # Also enforce agreement with physics prediction for rock-solid pseudo-labels
    high_conf_mounds = scan_df[(scan_df["global_prob"] >= confidence_upper) & (scan_df["physics_pred"] == 1)].copy()
    high_conf_mounds["label"] = 1
    
    high_conf_craters = scan_df[(scan_df["global_prob"] <= confidence_lower) & (scan_df["physics_pred"] == 0)].copy()
    high_conf_craters["label"] = 0
    
    pseudo_df = pd.concat([high_conf_mounds, high_conf_craters], axis=0)
    
    print(f"\n[+] Extracted Pseudo-Labels:")
    print(f"    - High-Confidence Mounds (P >= {confidence_upper}): {len(high_conf_mounds):,}")
    print(f"    - High-Confidence Craters (P <= {confidence_lower}): {len(high_conf_craters):,}")
    print(f"    - Total Pseudo-Labeled Test Samples: {len(pseudo_df):,} ({len(pseudo_df)/len(scan_df)*100:.1f}% of test set)")
    
    # Standardize columns to match train_metadata.csv
    # train_metadata.csv: image_id, sun_azimuth_angle, label, [image_path]
    train_df["is_pseudo"] = 0
    train_df["image_dir"] = "train_images"
    
    pseudo_df["is_pseudo"] = 1
    pseudo_df["image_dir"] = "test_images"
    
    keep_cols = ["image_id", "sun_azimuth_angle", "label", "is_pseudo", "image_dir"]
    combined_df = pd.concat([train_df[keep_cols], pseudo_df[keep_cols]], ignore_index=True)
    
    combined_df.to_csv(output_csv, index=False)
    print(f"\n[+] Successfully saved combined training set to '{output_csv}'")
    print(f"    - Total Training Samples: {len(combined_df):,} (Ground Truth: {len(train_df):,} + Pseudo: {len(pseudo_df):,})")
    print(f"    - Class Balance: Class 0 = {(combined_df['label']==0).sum():,} ({(combined_df['label']==0).mean()*100:.1f}%), Class 1 = {(combined_df['label']==1).sum():,} ({(combined_df['label']==1).mean()*100:.1f}%)")
    print("=" * 70)

if __name__ == "__main__":
    generate_pseudo_labels()
