import os
import sys
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

def rotate_with_reflect_crop(img: Image.Image, azimuth_angle: float, pad_width: int = 128) -> Image.Image:
    """
    Normalizes shadow direction by rotating counter-clockwise by -sun_azimuth_angle.
    Uses reflect padding before rotation, then center-crops back to original 256x256.
    Uses Bicubic resampling to preserve crisp lunar regolith and shadow sharpness.
    """
    if img.mode != "L":
        img = img.convert("L")
        
    arr = np.array(img)
    h, w = arr.shape
    
    # 1. Symmetric reflection padding to prevent black corner artifacts
    padded_arr = np.pad(arr, pad_width=pad_width, mode="reflect")
    padded_img = Image.fromarray(padded_arr)
    
    # 2. Counter-clockwise rotation by -azimuth using Bicubic resampling
    ccw_angle = -float(azimuth_angle)
    resample_method = getattr(Image, "Resampling", Image).BICUBIC
    rotated_padded = padded_img.rotate(ccw_angle, resample=resample_method)
    
    # 3. Center crop back to original (w, h)
    left = pad_width
    top = pad_width
    right = pad_width + w
    bottom = pad_width + h
    
    return rotated_padded.crop((left, top, right, bottom))

def compute_physics_channels(img_1ch_tensor: torch.Tensor) -> torch.Tensor:
    """
    Given a (1, H, W) normalized grayscale tensor with sunlight locked to North (Top, y=0),
    computes a (3, H, W) Physics-Engineered Tensor:
      - Channel 0: Normalized Intensity I
      - Channel 1: Vertical Solar Gradient (∇y I = ∂I/∂y) - slope facing toward/away from sun
      - Channel 2: Topographic Laplacian Curvature (∇² I) - ridge vs basin boundary
    """
    # Sobel vertical kernel: detects illumination slope along Sun vector (y-axis)
    sobel_y = torch.tensor([[-1., -2., -1.],
                            [ 0.,  0.,  0.],
                            [ 1.,  2.,  1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0

    # Laplacian kernel: detects 2D topographic curvature / crater rim boundaries
    laplacian = torch.tensor([[0.,  1., 0.],
                             [1., -4., 1.],
                             [0.,  1., 0.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 4.0

    pad_tensor = F.pad(img_1ch_tensor.unsqueeze(0), (1, 1, 1, 1), mode='reflect')
    
    grad_y = F.conv2d(pad_tensor, sobel_y).squeeze(0)
    curv = F.conv2d(pad_tensor, laplacian).squeeze(0)
    
    # Standardize channels to zero-mean and unit scale
    grad_y = (grad_y - grad_y.mean()) / (grad_y.std() + 1e-6)
    curv = (curv - curv.mean()) / (curv.std() + 1e-6)
    
    ch0 = img_1ch_tensor # normalized (1, H, W)
    ch1 = grad_y         # normalized (1, H, W)
    ch2 = curv           # normalized (1, H, W)
    
    return torch.cat([ch0, ch1, ch2], dim=0) # (3, H, W)

def get_transforms(is_train: bool = True):
    transform_list = []
    if is_train:
        transform_list.extend([
            T.RandomResizedCrop(
                size=(256, 256), 
                scale=(0.94, 1.00), 
                ratio=(0.98, 1.02),
                interpolation=T.InterpolationMode.BICUBIC
            ),
            T.ColorJitter(brightness=0.10, contrast=0.10),
        ])
    transform_list.append(T.ToTensor())
    transform_list.append(T.Normalize(mean=(0.485,), std=(0.229,)))
    return T.Compose(transform_list)

class PareidoliaDataset(Dataset):
    def __init__(
        self,
        metadata: str | pd.DataFrame,
        images_dir: str = "train_images",
        is_train: bool = True,
        is_labeled: bool | None = None,
        use_physics_3ch: bool = True
    ):
        if isinstance(metadata, str):
            if not os.path.exists(metadata):
                raise FileNotFoundError(f"Metadata file '{metadata}' not found.")
            self.df = pd.read_csv(metadata)
        elif isinstance(metadata, pd.DataFrame):
            self.df = metadata.copy().reset_index(drop=True)
        else:
            raise TypeError(f"metadata must be a file path string or pandas DataFrame, got {type(metadata)}")
            
        self.default_images_dir = images_dir
        self.is_train = is_train
        self.use_physics_3ch = use_physics_3ch
        
        if is_labeled is None:
            self.is_labeled = "label" in self.df.columns
        else:
            self.is_labeled = is_labeled
            
        self.transforms = get_transforms(is_train=is_train)
        
    def __len__(self) -> int:
        return len(self.df)
        
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])
        azimuth = float(row["sun_azimuth_angle"])
        
        # Support dynamic image directory if specified in row (for pseudo-labels)
        if "image_dir" in row and pd.notna(row["image_dir"]):
            img_dir = str(row["image_dir"])
        else:
            img_dir = self.default_images_dir
            
        img_path = os.path.join(img_dir, image_id)
        if not os.path.exists(img_path):
            alt_dirs = ["train_images", "eval_images", "test_images", "."]
            for alt in alt_dirs:
                candidate = os.path.join(alt, image_id)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
                    
        with Image.open(img_path) as raw_img:
            # 1. Normalize illumination by rotating by -azimuth with reflect padding
            north_lit_img = rotate_with_reflect_crop(raw_img, azimuth_angle=azimuth)
            
            # 2. Apply orientation-safe transforms (ToTensor + Normalize)
            t_img = self.transforms(north_lit_img) # (1, 256, 256)
            
            # 3. If enabled, compute 3-Channel Physics Tensor (I, ∇y I, ∇² I)
            if self.use_physics_3ch:
                t_img = compute_physics_channels(t_img) # (3, 256, 256)
                
        if self.is_labeled:
            label = int(row["label"])
            return t_img, label
        else:
            return t_img, image_id

# Alias for backward compatibility
LunarDataset = PareidoliaDataset
