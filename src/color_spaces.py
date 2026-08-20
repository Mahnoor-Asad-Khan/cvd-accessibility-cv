"""
Color space conversion utilities: RGB <-> LMS.

LMS space represents cone cell responses directly (L, M, S cones),
which is why CVD (color vision deficiency) simulation happens here
rather than in RGB. See research_log.md for the reasoning.

Matrix source: Hunt-Pointer-Estevez transform, sRGB-adapted.
This is the same matrix used as a base by Brettel et al. (1997)
and Machado et al. (2009).
"""

import numpy as np

# RGB -> LMS matrix (Hunt-Pointer-Estevez, sRGB-adapted)
RGB_TO_LMS = np.array([
    [17.8824,   43.5161,  4.11935],
    [3.45565,   27.1554,  3.86714],
    [0.0299566,  0.184309, 1.46709],
])

# LMS -> RGB is just the matrix inverse
LMS_TO_RGB = np.linalg.inv(RGB_TO_LMS)

"""
    Convert an image from RGB to LMS space.

    Args:
        img_rgb_normalized: (H, W, 3) array, RGB values in [0, 1] range.

    Returns:
        (H, W, 3) array of LMS values.
    """

def rgb_to_lms(img_rgb_normalized: np.ndarray) -> np.ndarray:

    return img_rgb_normalized @ RGB_TO_LMS.T
    # TODO (Kalon): apply RGB_TO_LMS to every pixel.


"""
    Convert an image from LMS back to RGB space.

    Args:
        img_lms: (H, W, 3) array of LMS values.

    Returns:
        (H, W, 3) array of RGB values in [0, 1] range (NOT yet clipped/rounded).
    """
def lms_to_rgb(img_lms: np.ndarray) -> np.ndarray:

    return img_lms @ LMS_TO_RGB.T
    # TODO (Kalon): same idea as rgb_to_lms, but with LMS_TO_RGB.
