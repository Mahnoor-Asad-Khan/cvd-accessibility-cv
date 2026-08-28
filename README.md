# Colorblind Accessibility — CV Research Project

A computer vision research project that simulates color vision deficiency
(CVD/colorblindness), quantifies how accessible an image is for CVD
viewers, and tries two different correction approaches: classical
daltonization and a self-supervised U-Net (PyTorch).

This builds on an earlier Java-based vision-test project I did with
teammates. This version is a proper CV research take on the same idea —
actually quantifying the accessibility problem and testing different ways
to solve it, instead of just building a tool.

## Results (protanopia, Ishihara test image)

| Method | Accessibility Score (worst-case Delta-E) | Verified functional? |
|---|---|---|
| Original | 8.96 | — (confirmed inaccessible) |
| Rule-based daltonization | 29.17 | Yes — number clearly visible post-correction |
| U-Net (self-supervised) | 31.48 | No — highest score, but number still indistinguishable; metric was gamed |

The U-Net actually scores higher than daltonization, but don't be fooled
by that — visual verification (simulating CVD on its output) showed the
number was still not distinguishable from the background. Turns out the
score was inflated by artifacts the model introduced, not a real fix.
Full writeup of this and other findings/limitations in
[`research_log.md`](research_log.md).

See [`notebooks/results.ipynb`](notebooks/results.ipynb) for the full
walkthrough with visuals.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Structure

- `src/color_spaces.py` — RGB <-> LMS conversion
- `src/simulate.py` — Brettel-style CVD simulation (protanopia, deuteranopia, tritanopia)
- `src/metrics.py` — accessibility scoring (k-means dominant colors + CVD-aware Delta-E)
- `src/daltonize.py` — rule-based correction (daltonization)
- `src/unet.py` — U-Net architecture for learned correction
- `src/differentiable.py` — torch-native (differentiable) simulation math, used in training
- `src/train.py` — self-supervised U-Net training loop
- `src/utils.py` — image I/O helpers
- `notebooks/results.ipynb` — end-to-end results walkthrough
- `research_log.md` — full research notes, experiments, and findings