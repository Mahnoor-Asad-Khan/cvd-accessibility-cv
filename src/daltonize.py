"""
Rule-based daltonization (color correction for CVD accessibility).

Concept: simulate the image as a CVD viewer would see it, compute the
"error" (perceptual information lost in simulation), and redistribute
that error into color channels the CVD viewer CAN perceive. This trades
exact color fidelity for distinguishability, the correct trade-off for
accessibility purposes.

Classic reference: Fidaner et al. daltonization approach.
"""

import numpy as np
from src.color_spaces import rgb_to_lms, lms_to_rgb
from src.simulate import PROTANOPIA_MATRIX, DEUTERANOPIA_MATRIX, TRITANOPIA_MATRIX

# Error redistribution matrices: push "lost" signal from the compromised
# cone into the two cones the CVD viewer still has working.
# Applied as: correction = error_LMS @ matrix.T

PROTANOPIA_CORRECTION_MATRIX = np.array([
    [0.0, 0.0, 0.0],
    [0.7, 1.0, 0.0],
    [0.7, 0.0, 1.0],
])

DEUTERANOPIA_CORRECTION_MATRIX = np.array([
    [1.0, 0.7, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.7, 1.0],
])

TRITANOPIA_CORRECTION_MATRIX = np.array([
    [1.0, 0.0, 0.7],
    [0.0, 1.0, 0.7],
    [0.0, 0.0, 0.0],
])


def daltonize(img_rgb_normalized: np.ndarray, cvd_matrix: np.ndarray,
              correction_matrix: np.ndarray) -> np.ndarray:
    """
    Apply rule-based daltonization to correct an image for a given CVD type.

    Args:
        img_rgb_normalized: (H, W, 3) RGB image, values in [0, 1].
        cvd_matrix: the CVD simulation matrix (e.g. PROTANOPIA_MATRIX from simulate.py) -
            used here to compute what the CVD viewer would see, NOT to modify the
            output directly.
        correction_matrix: the matching *_CORRECTION_MATRIX from this file.

    Returns:
        (H, W, 3) corrected RGB image, values in [0, 1] (unclipped - clip on save).
    """

    original_lms = rgb_to_lms(img_rgb_normalized)
    simulated_lms = original_lms @ cvd_matrix.T
    error_lms = original_lms - simulated_lms
    correction_lms = error_lms @ correction_matrix.T
    corrected_lms = original_lms + correction_lms
    return lms_to_rgb(corrected_lms)

    # TODO (Kalon):
    # 1. Convert img_rgb_normalized to LMS -> call it original_lms
    #    (reuse rgb_to_lms - you already wrote this)
    # 2. Simulate CVD in LMS space directly
    #    (this is the SAME operation simulate_cvd does internally, but we need
    #    the LMS intermediate here, not the final RGB, so don't call simulate_cvd()
    #    as a black box this time - replicate its middle step)
    # 3. Compute the error
    # 4. Redistribute the error
    # 5. Add correction back to the ORIGINAL (not the simulated) LMS
    # 6. Convert corrected_lms back to RGB and return it
