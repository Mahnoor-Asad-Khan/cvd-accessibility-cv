"""
Image I/O helpers. This is the OpenCV "plumbing" — load/save/normalize —
kept separate so the research logic (color_spaces.py, simulate.py) stays
readable and isn't cluttered with cv2 quirks (e.g. BGR vs RGB ordering).
"""

import cv2
import numpy as np


def load_image_rgb_normalized(path: str) -> np.ndarray:
    """
    Load an image from disk as RGB, normalized to [0, 1] float values.

    OpenCV loads images as BGR by default, and as uint8 (0-255) — this
    function handles both conversions so the rest of the pipeline can
    assume RGB + [0, 1] float.
    """
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb.astype(np.float64) / 255.0


def save_image_rgb_normalized(img_rgb_normalized: np.ndarray, path: str) -> None:
    """
    Save a [0, 1] float RGB image back to disk.

    Clips values to [0, 1] first (simulation/correction math can push
    values slightly out of range), converts back to uint8 BGR for cv2.
    """
    img_clipped = np.clip(img_rgb_normalized, 0.0, 1.0)
    img_uint8 = (img_clipped * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)
