"""
Differentiable (torch-native) versions of color_spaces.py and simulate.py's
logic. Needed because PyTorch autograd only tracks gradients through
torch.Tensor operations, not numpy - so the U-Net's self-supervised loss
(which depends on LMS conversion + CVD simulation) needs this math ported
to torch so gradients can flow from the loss back into the network's weights.

Same math as color_spaces.py / simulate.py - just torch instead of numpy,
and PyTorch's (B, C, H, W) channel-first image convention instead of
OpenCV/numpy's (H, W, C) channel-last convention (hence the .permute() calls).
"""

import torch

RGB_TO_LMS_TORCH = torch.tensor([
    [17.8824,    43.5161,  4.11935],
    [3.45565,    27.1554,  3.86714],
    [0.0299566,  0.184309, 1.46709],
], dtype=torch.float32)

LMS_TO_RGB_TORCH = torch.linalg.inv(RGB_TO_LMS_TORCH)

PROTANOPIA_MATRIX_TORCH = torch.tensor([
    [0.0, 2.02344, -2.52581],
    [0.0, 1.0,      0.0],
    [0.0, 0.0,      1.0],
], dtype=torch.float32)

DEUTERANOPIA_MATRIX_TORCH = torch.tensor([
    [1.0,      0.0, 0.0],
    [0.494207, 0.0, 1.24827],
    [0.0,      0.0, 1.0],
], dtype=torch.float32)

TRITANOPIA_MATRIX_TORCH = torch.tensor([
    [1.0,       0.0,      0.0],
    [0.0,       1.0,      0.0],
    [-0.395913, 0.801109, 0.0],
], dtype=torch.float32)


def rgb_to_lms_torch(img: torch.Tensor) -> torch.Tensor:
    """img: (B, 3, H, W), values in [0,1]. Returns (B, 3, H, W) in LMS."""
    img_perm = img.permute(0, 2, 3, 1)          # -> (B, H, W, 3)
    lms = img_perm @ RGB_TO_LMS_TORCH.T.to(img.device)
    return lms.permute(0, 3, 1, 2)               # -> (B, 3, H, W)


def lms_to_rgb_torch(img_lms: torch.Tensor) -> torch.Tensor:
    """img_lms: (B, 3, H, W). Returns (B, 3, H, W) in RGB (unclipped)."""
    img_perm = img_lms.permute(0, 2, 3, 1)
    rgb = img_perm @ LMS_TO_RGB_TORCH.T.to(img_lms.device)
    return rgb.permute(0, 3, 1, 2)


def simulate_cvd_torch(img_rgb: torch.Tensor, cvd_matrix: torch.Tensor) -> torch.Tensor:
    """Full RGB -> simulated RGB pipeline, torch-native and differentiable."""
    lms = rgb_to_lms_torch(img_rgb)
    lms_perm = lms.permute(0, 2, 3, 1)
    sim_lms = lms_perm @ cvd_matrix.T.to(img_rgb.device)
    sim_lms = sim_lms.permute(0, 3, 1, 2)
    return lms_to_rgb_torch(sim_lms)


def simple_delta_e_torch(img1_rgb: torch.Tensor, img2_rgb: torch.Tensor) -> torch.Tensor:
    """
    Simplified, fully-differentiable perceptual distance, used as the
    accessibility-gain term in training. NOTE: this is a lightweight
    proxy, not a full RGB->Lab->CIE76 pipeline (Lab conversion involves
    non-linear cube-root operations that are differentiable but slower
    and not worth porting for training-loop speed). It operates directly
    on RGB pixel differences, weighted to roughly emphasize perceptual
    sensitivity (green weighted highest, matching luminance perception -
    see the WCAG luminance formula from Phase 3).

    This means metrics.py's accessibility_score() (numpy, CIE76 Delta-E)
    remains your "true"/reported evaluation metric - this torch version
    is ONLY used internally during training, as an approximation optimized
    for speed and differentiability. Worth noting as a design choice/
    limitation in the research log.
    """
    diff = img1_rgb - img2_rgb
    weights = torch.tensor([0.30, 0.59, 0.11], device=img1_rgb.device).view(1, 3, 1, 1)
    weighted_sq_diff = (diff ** 2) * weights
    return torch.sqrt(weighted_sq_diff.sum(dim=1) + 1e-8)  # (B, H, W), per-pixel distance