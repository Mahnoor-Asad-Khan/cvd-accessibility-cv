"""
Self-supervised training loop for the U-Net CVD-correction model.

No labeled dataset needed - the loss function is built directly from
this project's own simulation math (differentiable.py), rewarding the
network for producing outputs that are MORE color-distinguishable under
CVD simulation than the original, while staying close to the original
image (so it doesn't just output noise/random colors to "win" on the
accessibility term).

loss = distortion_from_original - lambda * accessibility_gain
"""

import torch
import torch.nn as nn
import torch.optim as optim

from src.Unet import SmallUNet
from src.differentiable import simulate_cvd_torch, simple_delta_e_torch, PROTANOPIA_MATRIX_TORCH


def compute_loss(model_output: torch.Tensor, original_input: torch.Tensor,
                  cvd_matrix: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """
    Args:
        model_output: (B, 3, H, W) - the U-Net's corrected output.
        original_input: (B, 3, H, W) - the original image, unchanged.
        cvd_matrix: which CVD type to train the correction for (e.g. PROTANOPIA_MATRIX_TORCH).
        lam: weight balancing distortion penalty vs accessibility reward.

    Returns:
        A single scalar loss (lower = better).
    """
    # Distortion term: keep output close to the original image.
    distortion = nn.functional.mse_loss(model_output, original_input)

    # Accessibility term: how distinguishable are corresponding pixels
    # from each other, under CVD simulation, output vs input?
    # We compare each image against a 'shifted' version of itself (simple
    # proxy for "pairs of different colors") - specifically, we measure
    # how much CVD simulation compresses local color variation, and reward
    # the model for reducing that compression relative to the input.
    sim_output = simulate_cvd_torch(model_output, cvd_matrix)
    sim_input = simulate_cvd_torch(original_input, cvd_matrix)

    # Local color spread: compare each pixel to its neighbor (shift by 1
    # pixel horizontally) as a cheap stand-in for "pairs of nearby colors
    # that should stay distinguishable."
    def local_spread(img):
        return simple_delta_e_torch(img[:, :, :, :-1], img[:, :, :, 1:]).mean()

    spread_before = local_spread(sim_input)
    spread_after = local_spread(sim_output)
    accessibility_gain = spread_after - spread_before  # want this positive (more spread after correction)

    loss = distortion - lam * accessibility_gain
    return loss, distortion.item(), accessibility_gain.item()


def train(image_tensor: torch.Tensor, cvd_matrix: torch.Tensor = PROTANOPIA_MATRIX_TORCH,
          epochs: int = 100, lr: float = 1e-3, lam: float = 0.1) -> SmallUNet:
    """
    Train the U-Net on a SINGLE image (this is intentional for now - given
    the timeline, we're doing per-image optimization rather than training
    on a full dataset of images. This is a real, documented scope choice:
    the model overfits to correcting THIS image well, rather than learning
    a generalizable correction for arbitrary images. Noted as future work
    to train across a dataset for generalization.

    Args:
        image_tensor: (1, 3, H, W) tensor, RGB values in [0,1].
        cvd_matrix: CVD type to train correction for.
        epochs: training iterations.
        lr: learning rate.
        lam: accessibility-gain loss weight.

    Returns:
        The trained SmallUNet model.
    """
    model = SmallUNet()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_distortion = float('inf')
    best_state = None

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(image_tensor)
        loss, distortion, gain = compute_loss(output, image_tensor, cvd_matrix, lam)
        loss.backward()
        optimizer.step()

        if distortion < best_distortion:
            best_distortion = distortion
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d}  loss={loss.item():.4f}  "
                  f"distortion={distortion:.4f}  accessibility_gain={gain:.4f}")

    model.load_state_dict(best_state)
    print(f"\nLoaded checkpoint with lowest distortion: {best_distortion:.4f}")
    return model