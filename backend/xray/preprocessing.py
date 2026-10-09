"""
backend/xray/preprocessing.py
Image Preprocessing and Normalization Pipeline for Chest X-ray Classification.

Clinical rationale:
- Chest radiographs are monochrome luminance maps where photon attenuation reflects tissue density.
- Normalization rescales raw pixel intensities to [0, 1] and standardizes mean/std.
- Avoids aggressive spatial distortions (such as horizontal flips that destroy dextrocardia/organ situs markers).
"""

import math
from typing import List, Tuple, Dict, Any

IMAGE_SIZE = (224, 224)
TARGET_WIDTH = 224
TARGET_HEIGHT = 224

# Radiographic intensity standardization constants
CHEST_MEAN = 0.485
CHEST_STD = 0.229

def normalize_intensity(raw_pixel_val: float) -> float:
    """
    Normalizes 8-bit grayscale intensity [0, 255] -> standardized z-score.
    """
    unit_val = raw_pixel_val / 255.0
    return (unit_val - CHEST_MEAN) / CHEST_STD

def denormalize_intensity(norm_val: float) -> int:
    """
    Inverse normalization back to [0, 255] integer range for display.
    """
    unit_val = (norm_val * CHEST_STD) + CHEST_MEAN
    clamped = max(0.0, min(1.0, unit_val))
    return int(clamped * 255.0)

def resize_bilinear(pixels: List[int], orig_w: int, orig_h: int, target_w: int, target_h: int) -> List[float]:
    """
    Bilinear interpolation to resize image buffers to standardized target dimensions (224x224).
    """
    if orig_w == target_w and orig_h == target_h:
        return [normalize_intensity(p) for p in pixels]

    resized = [0.0] * (target_w * target_h)
    x_ratio = float(orig_w - 1) / target_w if target_w > 1 else 0
    y_ratio = float(orig_h - 1) / target_h if target_h > 1 else 0

    for i in range(target_h):
        y = int(y_ratio * i)
        y_diff = (y_ratio * i) - y
        for j in range(target_w):
            x = int(x_ratio * j)
            x_diff = (x_ratio * j) - x
            index = y * orig_w + x

            a = pixels[index]
            b = pixels[index + 1] if x + 1 < orig_w else a
            c = pixels[index + orig_w] if y + 1 < orig_h else a
            d = pixels[index + orig_w + 1] if (x + 1 < orig_w and y + 1 < orig_h) else a

            # Interpolate
            pixel_interp = (
                a * (1.0 - x_diff) * (1.0 - y_diff)
                + b * (x_diff) * (1.0 - y_diff)
                + c * (y_diff) * (1.0 - x_diff)
                + d * (x_diff * y_diff)
            )
            resized[i * target_w + j] = normalize_intensity(pixel_interp)

    return resized

def apply_mild_augmentation(pixels_norm: List[float], width: int, height: int, contrast_factor: float = 1.05, brightness_shift: float = 0.02) -> List[float]:
    """
    Mild radiologic augmentation preserving anatomical morphology.
    Slight contrast adjustment and mild intensity shift simulating varied X-ray tube mAs exposure.
    """
    augmented = []
    for val in pixels_norm:
        adj = val * contrast_factor + brightness_shift
        augmented.append(adj)
    return augmented
