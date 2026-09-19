import os
import sys
import shutil
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

def rotate_with_reflect_crop(img_gray, azimuth_angle, pad_width=128):
    """
    Rotates a grayscale PIL image counter-clockwise by -sun_azimuth_angle.
    Applies reflect padding before rotation, then center-crops back to 256x256.
    """
    arr = np.array(img_gray)
    h, w = arr.shape
    
    # 1. Reflect pad
    padded_arr = np.pad(arr, pad_width=pad_width, mode='reflect')
    padded_img = Image.fromarray(padded_arr)
    
    # 2. Rotate counter-clockwise by -sun_azimuth_angle
    # PIL.Image.rotate rotates counter-clockwise by default
    ccw_angle = -float(azimuth_angle)
    rotated_padded = padded_img.rotate(ccw_angle, resample=Image.BICUBIC)
    
    # 3. Center crop back to original 256x256
    left = pad_width
    top = pad_width
    right = pad_width + w
    bottom = pad_width + h
    cropped_img = rotated_padded.crop((left, top, right, bottom))
    return cropped_img

def main():
    metadata_path = "train_metadata.csv"
    images_dir = "train_images"
    output_png = "azimuth_rotation_comparison.png"
    brain_artifact_dir = r"C:\Users\PARAM\.gemini\antigravity-ide\brain\75654c39-7f35-483e-9e5a-67cbad8a0ae5"

    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found.")
        return
    if not os.path.isdir(images_dir):
        print(f"Error: {images_dir} directory not found.")
        return

    df = pd.read_csv(metadata_path)
    
    # Pick 6 random images (fixed seed for deterministic run, but representative across quadrants)
    sample_df = df.sample(n=6, random_state=42).reset_index(drop=True)

    print("=" * 65)
    print("AZIMUTH ROTATION VERIFICATION (6 Random Samples)")
    print("=" * 65)
    print(f"{'Index':<6} | {'Image ID':<18} | {'Azimuth (deg)':<15} | {'PIL Rotation (CCW)':<20} | {'Label'}")
    print("-" * 65)

    pairs = []
    for idx, row in sample_df.iterrows():
        img_id = row["image_id"]
        azimuth = float(row["sun_azimuth_angle"])
        label = int(row["label"])
        rot_angle = -azimuth

        print(f"{idx+1:<6} | {img_id:<18} | {azimuth:>13.2f}° | {rot_angle:>18.2f}° | {label}")

        img_path = os.path.join(images_dir, img_id)
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found.")
            continue

        # Load as grayscale
        orig_img = Image.open(img_path).convert("L")
        
        # Apply reflect-pad rotate and center-crop
        rotated_img = rotate_with_reflect_crop(orig_img, azimuth, pad_width=128)

        pairs.append({
            "image_id": img_id,
            "azimuth": azimuth,
            "rot_angle": rot_angle,
            "label": label,
            "orig": orig_img,
            "rotated": rotated_img
        })

    print("-" * 65)

    # Plot 6 pairs in a 3x4 grid: 3 rows, 2 pairs per row
    fig, axes = plt.subplots(3, 4, figsize=(16, 13))
    fig.suptitle("Shadow Direction Normalization: Counter-Clockwise Rotation by -Sun Azimuth Angle\n(Bicubic Resampling + Reflect-Pad & Center-Crop)", 
                 fontsize=15, fontweight='bold')

    for i, item in enumerate(pairs):
        row = i // 2
        col_orig = (i % 2) * 2
        col_rot = col_orig + 1

        # Original
        ax_orig = axes[row, col_orig]
        ax_orig.imshow(item["orig"], cmap="gray", vmin=0, vmax=255)
        ax_orig.set_title(f"Original: {item['image_id']}\nAzimuth: {item['azimuth']:.2f}° | Label: {item['label']}", 
                          fontsize=10, pad=6)
        ax_orig.axis("off")

        # Rotated
        ax_rot = axes[row, col_rot]
        ax_rot.imshow(item["rotated"], cmap="gray", vmin=0, vmax=255)
        ax_rot.set_title(f"Rotated CCW: {item['rot_angle']:.2f}°\n(Reflect-pad -> Center 256x256)", 
                         fontsize=10, color="navy", fontweight="semibold", pad=6)
        ax_rot.axis("off")

    plt.subplots_adjust(top=0.90, bottom=0.03, left=0.03, right=0.97, hspace=0.32, wspace=0.12)
    plt.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\n[+] Saved comparison grid to '{output_png}'.")

    # Also copy to brain artifacts directory if it exists for rendering in UI
    if os.path.exists(brain_artifact_dir):
        dest_brain = os.path.join(brain_artifact_dir, output_png)
        shutil.copy2(output_png, dest_brain)
        print(f"[+] Synced to artifacts directory at '{dest_brain}'.")

if __name__ == "__main__":
    main()
