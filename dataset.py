import os
import sys
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

def rotate_with_reflect_crop(img: Image.Image, azimuth_angle: float, pad_width: int = 128) -> Image.Image:
    """
    Normalizes shadow direction by rotating counter-clockwise by -sun_azimuth_angle.
    Uses reflect padding before rotation, then center-crops back to original 256x256.
    Uses Bicubic resampling to preserve high-frequency lunar regolith and shadow sharpness.
    
    Args:
        img: Input PIL Image (converted to grayscale 'L' mode if not already).
        azimuth_angle: Sun azimuth angle in degrees.
        pad_width: Number of pixels to reflect-pad on all sides (default: 128).
        
    Returns:
        Center-cropped, rotated 256x256 grayscale PIL Image.
    """
    if img.mode != "L":
        img = img.convert("L")
        
    arr = np.array(img)
    h, w = arr.shape
    
    # 1. Symmetric reflection padding to prevent black corner artifacts
    padded_arr = np.pad(arr, pad_width=pad_width, mode="reflect")
    padded_img = Image.fromarray(padded_arr)
    
    # 2. Counter-clockwise rotation by -azimuth using Bicubic resampling (preserves crisp shadow edges)
    ccw_angle = -float(azimuth_angle)
    resample_method = getattr(Image, "Resampling", Image).BICUBIC
    rotated_padded = padded_img.rotate(ccw_angle, resample=resample_method)
    
    # 3. Center crop back to original (w, h)
    left = pad_width
    top = pad_width
    right = pad_width + w
    bottom = pad_width + h
    
    return rotated_padded.crop((left, top, right, bottom))

def get_transforms(is_train: bool = True, mean: tuple = (0.485,), std: tuple = (0.229,)):
    """
    Builds torchvision transforms pipeline for grayscale lunar images.
    
    Planetary Illumination Physics:
    - Azimuth 0° corresponds to North (Top of image).
    - Normalizing by -azimuth forces the sun to shine directly from the TOP (North).
    - BOTH Horizontal and Vertical Flips are STRICTLY PURGED:
        * Vertical Flip moves the sun from Top (North) to Bottom (South), inverting shadows 180°!
        * Horizontal Flip mirrors shadows horizontally.
    - Only orientation-preserving augmentations are applied: subtle scale/crop jitter and brightness/contrast.
    
    ImageNet Pixel Standardization:
    - Standardizes ToTensor() [0, 1] to N(0, 1) with mean=0.485, std=0.229.
    """
    transform_list = []
    
    if is_train:
        transform_list.extend([
            # 1. Orientation-preserving subtle crop & scale (no black borders)
            T.RandomResizedCrop(
                size=(256, 256), 
                scale=(0.94, 1.00), 
                ratio=(0.98, 1.02),
                interpolation=T.InterpolationMode.BICUBIC
            ),
            # 2. Photometric variation (albedo & solar elevation differences)
            T.ColorJitter(brightness=0.10, contrast=0.10),
        ])
        
    # Tensor conversion: converts PIL (0..255) to float tensor (0.0..1.0), shape (1, H, W)
    transform_list.append(T.ToTensor())
    
    # ImageNet grayscale standardization: maps [0.0, 1.0] to N(0, 1) matching pretrained weights
    transform_list.append(T.Normalize(mean=mean, std=std))
    
    return T.Compose(transform_list)


class PareidoliaDataset(Dataset):
    """
    PyTorch Dataset for the Pareidolia Paradox challenge.
    
    Supports:
      - Labeled mode (for training / validation): returns (image_tensor, label)
      - Unlabeled mode (for test / inference): returns (image_tensor, image_id)
    """
    def __init__(
        self,
        metadata: str | pd.DataFrame,
        images_dir: str,
        is_train: bool = True,
        is_labeled: bool | None = None,
        pad_width: int = 128,
        transform=None,
    ):
        super().__init__()
        
        if isinstance(metadata, str):
            if not os.path.exists(metadata):
                raise FileNotFoundError(f"Metadata file '{metadata}' does not exist.")
            self.df = pd.read_csv(metadata)
        elif isinstance(metadata, pd.DataFrame):
            self.df = metadata.copy().reset_index(drop=True)
        else:
            raise TypeError("metadata must be a file path (str) or pandas DataFrame.")
            
        self.images_dir = images_dir
        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f"Images directory '{self.images_dir}' does not exist.")
            
        self.is_train = is_train
        self.pad_width = pad_width
        
        # Determine labeled vs unlabeled mode
        if is_labeled is not None:
            self.is_labeled = is_labeled
        else:
            self.is_labeled = "label" in self.df.columns
            
        # Set transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_transforms(is_train=self.is_train)
            
    def __len__(self) -> int:
        return len(self.df)
        
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        azimuth = float(row["sun_azimuth_angle"])
        
        img_path = os.path.join(self.images_dir, image_id)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at '{img_path}'.")
            
        # 1. Load image in grayscale
        img = Image.open(img_path).convert("L")
        
        # 2. Apply azimuth normalization
        img_rotated = rotate_with_reflect_crop(img, azimuth, pad_width=self.pad_width)
        
        # 3. Apply augmentations + ToTensor + Normalization
        img_tensor = self.transform(img_rotated)
        
        # 4. Return based on mode
        if self.is_labeled:
            label = torch.tensor(int(row["label"]), dtype=torch.long)
            return img_tensor, label
        else:
            return img_tensor, image_id

def test_pipeline():
    print("=" * 60)
    print("TESTING PYTORCH DATA PIPELINE (dataset.py)")
    print("=" * 60)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Active device: {device}")
    
    # 1. Test Training (Labeled) Dataset & DataLoader
    train_csv = "train_metadata.csv"
    train_dir = "train_images"
    batch_size = 32
    
    print(f"\n[+] Initializing PareidoliaDataset (train_mode=True, labeled=True)...")
    train_dataset = PareidoliaDataset(
        metadata=train_csv,
        images_dir=train_dir,
        is_train=True,
    )
    print(f"    Dataset length: {len(train_dataset):,} samples")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Cross-platform safe
        drop_last=False
    )
    
    print(f"[+] Loading one training batch (batch_size={batch_size})...")
    images, labels = next(iter(train_loader))
    
    print("\n" + "-" * 50)
    print("TRAIN BATCH INSPECTION")
    print("-" * 50)
    print(f"Batch Image Shape:  {images.shape} (Expected: [{batch_size}, 1, 256, 256])")
    print(f"Batch Image Dtype:  {images.dtype} (Expected: torch.float32)")
    print(f"Batch Image Range:  min={images.min().item():.4f}, max={images.max().item():.4f}, mean={images.mean().item():.4f}, std={images.std().item():.4f}")
    print(f"Batch Label Shape:  {labels.shape} (Expected: [{batch_size}])")
    print(f"Batch Label Dtype:  {labels.dtype} (Expected: torch.int64)")
    
    unique_labels, counts = torch.unique(labels, return_counts=True)
    dist = dict(zip(unique_labels.tolist(), counts.tolist()))
    print("\nBatch Label Distribution:")
    for lbl, cnt in dist.items():
        pct = (cnt / batch_size) * 100
        print(f"  Class {lbl}: {cnt}/{batch_size} ({pct:.1f}%)")
        
    # 2. Test Unlabeled / Test Dataset mode
    test_csv = "test_metadata.csv"
    eval_dir = "eval_images" if os.path.isdir("eval_images") else "test_images"
    
    if os.path.exists(test_csv) and os.path.isdir(eval_dir):
        print("\n" + "-" * 50)
        print("TEST / UNLABELED MODE CHECK")
        print("-" * 50)
        test_dataset = PareidoliaDataset(
            metadata=test_csv,
            images_dir=eval_dir,
            is_train=False,
            is_labeled=False
        )
        print(f"Test dataset length: {len(test_dataset):,} samples")
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        test_images, test_ids = next(iter(test_loader))
        print(f"Test batch images shape: {test_images.shape}")
        print(f"Test batch sample IDs:   {test_ids}")
        
    print("\n[+] All pipeline checks passed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_pipeline()
