"""
CVD (color vision deficiency) simulation.

Simplified Brettel-style projection: reconstructs the missing cone's
LMS value as a linear combination of the other two cones, using
coefficients derived from confusion-line geometry. See research_log.md
for the reasoning (why this isn't just "zeroing out" the missing cone).

Note: this is the commonly-implemented simplified single-plane version.
Brettel's original (1997) paper uses a piecewise projection with two
anchor points for higher accuracy.
"""

import numpy as np
from src.color_spaces import rgb_to_lms, lms_to_rgb

# Each matrix operates in LMS space: [L_sim, M_sim, S_sim] = MATRIX @ [L, M, S]

PROTANOPIA_MATRIX = np.array([
    [0.0,       2.02344, -2.52581],
    [0.0,       1.0,      0.0],
    [0.0,       0.0,      1.0],
])

DEUTERANOPIA_MATRIX = np.array([
    [1.0,       0.0,      0.0],
    [0.494207,  0.0,      1.24827],
    [0.0,       0.0,      1.0],
])

TRITANOPIA_MATRIX = np.array([
    [1.0,        0.0,      0.0],
    [0.0,        1.0,      0.0],
    [-0.395913,  0.801109, 0.0],
])


def simulate_cvd(img_rgb_normalized: np.ndarray, cvd_matrix: np.ndarray) -> np.ndarray:
    """
    Simulating how an image would appear to someone with a given dichromacy.

    Args:
        img_rgb_normalized: (H, W, 3) RGB image, values in [0, 1].
        cvd_matrix: one of PROTANOPIA_MATRIX / DEUTERANOPIA_MATRIX / TRITANOPIA_MATRIX.

    Returns:
        (H, W, 3) RGB image, values in [0, 1] (may need clipping - see utils.py's
        save function, which already clips).
    """
    lms_img = rgb_to_lms(img_rgb_normalized)
    sim_lms = lms_img @ cvd_matrix.T
    return lms_to_rgb(sim_lms)

    # TODO (Kalon): three steps, all functions/pattern you already have:
    # 1. Convert img_rgb_normalized to LMS (you already wrote this function)
    # 2. Apply cvd_matrix to the LMS image (same @ matrix.T pattern as before)
    # 3. Convert the simulated LMS back to RGB (you already wrote this function too)
    raise NotImplementedError