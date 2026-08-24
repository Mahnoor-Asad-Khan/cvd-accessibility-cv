"""
Accessibility quantification.

Pipeline:
1. Extract dominant colors from an image via k-means clustering in Lab space
   (Lab, not RGB, because Euclidean distance in Lab approximates perceptual
   difference - same reasoning as Delta-E).
2. For every pair of dominant colors, compute:
   - normal_diff: Delta-E between the two colors as normally seen
   - cvd_diff: Delta-E between the two colors AFTER running both through
     CVD simulation (src/simulate.py)
3. Aggregate pairwise cvd_diff scores into a single accessibility score.
   Low cvd_diff on a pair that has high normal_diff = accessibility risk
   (colors that look different normally, collapse together for a CVD viewer).
"""

import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb

from src.simulate import simulate_cvd

from itertools import combinations

import matplotlib.pyplot as plt


def extract_dominant_colors(img_rgb_normalized: np.ndarray, k: int = 6) -> np.ndarray:
    """
    Extract k dominant colors from an image via k-means clustering in Lab space.

    Args:
        img_rgb_normalized: (H, W, 3) RGB image, values in [0, 1].
        k: number of dominant colors to extract.

    Returns:
        (k, 3) array of dominant colors, in RGB [0, 1] space (converted back
        from Lab after clustering, so they're usable by simulate_cvd directly).
    """
    lab_img = rgb2lab(img_rgb_normalized)
    lab_img_2d = lab_img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=42) #defining the model
    kmeans.fit(lab_img_2d)
    dom_colors = kmeans.cluster_centers_
    return lab2rgb(dom_colors)
    # TODO (Kalon):
    # 1. Convert img_rgb_normalized to Lab using rgb2lab (expects [0,1] RGB input)
    # 2. Reshape the (H, W, 3) Lab image into (H*W, 3) - a flat list of Lab pixels
    #    (KMeans expects a 2D array of samples, not an image)
    # 3. Fit KMeans(n_clusters=k) on that flat array, get cluster centers
    #    (cluster centers = the k dominant colors, still in Lab)
    # 4. Convert the k cluster centers back to RGB (skimage.color.lab2rgb)
    # 5. Return the (k, 3) RGB array


def delta_e_cie76(color1_lab: np.ndarray, color2_lab: np.ndarray) -> float:
    """
    CIE76 Delta-E: perceptual color difference, Euclidean distance in Lab space.

    Args:
        color1_lab, color2_lab: (3,) arrays, single colors in Lab space.

    Returns:
        A single float distance value.
    """
    return np.linalg.norm(color1_lab - color2_lab)
    # TODO (Kalon): Euclidean distance between the two Lab vectors.



def accessibility_score(img_rgb_normalized: np.ndarray, cvd_matrix: np.ndarray, k: int = 4) -> dict:
    """
    Compute a CVD-accessibility score for an image, for a given CVD type.

    Returns a dict with per-pair details and an overall score, so you can
    inspect *which* color pairs are the problem, not just a single number.
    """
    dom_colors = extract_dominant_colors(img_rgb_normalized, k)
    cvd_dom_colors = simulate_cvd(dom_colors.reshape(1, k, 3), cvd_matrix).reshape(k, 3)

    pairs = []

    for i, j in combinations(range(k), 2):
        normal_lab_i = rgb2lab(dom_colors[i].reshape(1, 1, 3)).reshape(3)
        normal_lab_j = rgb2lab(dom_colors[j].reshape(1, 1, 3)).reshape(3)
        normal_diff = delta_e_cie76(normal_lab_i, normal_lab_j)

        cvd_lab_i = rgb2lab(cvd_dom_colors[i].reshape(1, 1, 3)).reshape(3)
        cvd_lab_j = rgb2lab(cvd_dom_colors[j].reshape(1, 1, 3)).reshape(3)
        cvd_diff = delta_e_cie76(cvd_lab_i, cvd_lab_j)

        pairs.append({
            "colors": (dom_colors[i], dom_colors[j]),
            "normal_diff": normal_diff,
            "cvd_diff": cvd_diff,
        })

    overall_score = min(p["cvd_diff"] for p in pairs) #your system is only as strong as its weakest link, not its average link (some socrates or smth)
    #getting the scores of pairs under a threshold would be a more complete metric, for future expansion of work 
    return {"overall_score": overall_score, "pairs": pairs}


    # TODO (Kalon), once extract_dominant_colors and delta_e_cie76 work:
    # 1. Get dominant colors (RGB) via extract_dominant_colors
    # 2. Simulate CVD on those dominant colors via simulate_cvd (reuse Phase 2!)
    # 3. For every pair (i, j) of dominant colors:
    #    - convert both original colors to Lab, compute normal_diff (delta_e_cie76)
    #    - convert both simulated colors to Lab, compute cvd_diff (delta_e_cie76)
    # 4. Aggregate: e.g. overall_score = min(cvd_diff across all pairs)
    #    (worst-case confusability - the single most dangerous pair)
    # 5. Return {"overall_score": ..., "pairs": [list of per-pair dicts]}

def print_accessibility_report(result: dict, cvd_type: str = "") -> None:
    """
    Pretty-print an accessibility_score() result for human reading.
    """
    print(f"--- Accessibility Report: {cvd_type} ---")
    print(f"Overall score (worst-case pair Delta-E): {result['overall_score']:.2f}")
    print("DANGER: likely indistinguishable pair present" if result['overall_score'] < 3
          else "No critically confusable pair found")
    print()
    sorted_pairs = sorted(result["pairs"], key=lambda p: p["cvd_diff"])
    for p in sorted_pairs:
        print(f"  normal_diff={p['normal_diff']:6.2f}  cvd_diff={p['cvd_diff']:6.2f}  "
              f"colors={np.round(p['colors'][0], 2)} vs {np.round(p['colors'][1], 2)}")

def plot_dominant_colors(dom_colors):
    fig, axes = plt.subplots(1, len(dom_colors), figsize=(2*len(dom_colors), 2))
    if len(dom_colors) == 1:
        axes = [axes]
    for ax, color in zip(axes, dom_colors):
        ax.imshow([[color]])
        ax.set_title(f"{np.round(color, 2)}", fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.show()