# Research Log

## Problem Framing
Building a CV-based system to (1) simulate how images/UIs appear to users
with color vision deficiency (CVD), (2) quantify accessibility via a
scoring metric, and (3) apply conditional correction (daltonization) when
accessibility falls below a threshold. Builds on an earlier Java-based
vision-test project; this version reframes it as a CV research project
with quantification and experimental comparison of methods.

## Phase 1: Color Vision Theory
- Trichromatic vision: L/M/S cones, overlapping sensitivity curves
  (L and M overlap heavily — why red-green CVD is most common/severe)
- CVD types: dichromacy (missing cone) vs anomalous trichromacy (shifted cone)
- Why RGB is the wrong space for simulation (display space, not biological space)

## Phase 2: Simulation
- RGB -> LMS via Hunt-Pointer-Estevez matrix (sRGB-adapted)
- Confusion lines: direction of the lost cone axis in LMS space; colors along
  a confusion line are indistinguishable to a given dichromat
- Brettel et al. (1997): projects colors onto the dichromat's perceivable
  plane using confusion-line geometry
- Machado et al. (2009): extends to a severity continuum (0-1), enabling
  simulation of anomalous trichromacy, not just full dichromacy
- Design decision to note: simplified single-plane projection (implemented
  here) vs. Brettel's full piecewise two-anchor-point version

### Validation & Findings
- Implemented protanopia, deuteranopia, tritanopia via the simplified single-plane
  LMS projection matrices.
- Validated protanopia against an independent online CVD simulator using an
  Ishihara-style test image: output matched (red/orange background became
  indistinguishable from the green number), confirming correct implementation.
- Validated tritanopia the same way and found a discrepancy: our simulation
  turned the orange background yellow/green, while the reference simulator
  showed a pinkish shift.
- Root cause (diagnosed via single-pixel test): the tritanopia projection
  produces an out-of-gamut LMS result (negative S value) for saturated
  orange. Converting back to RGB gives a negative blue channel, which naive
  per-channel clipping floors to 0 — destroying the blue component entirely
  instead of proportionally dimming it, which produces the wrong hue.
- Two separable causes: (1) the simplified single-plane Brettel projection is
  known in the literature to be less accurate for tritanopia specifically,
  and (2) naive per-channel clipping is a crude gamut-mapping strategy;
  proportional/hue-preserving gamut mapping would likely fix this
  independently of the projection method used.
- Decision: documenting as a known limitation rather than fixing now, given
  timeline. Protanopia and deuteranopia (>99% of real-world CVD cases) are
  verified correct against a reference simulator. A full Brettel two-plane
  projection and/or proper gamut mapping are noted as future improvements.

## Phase 3: Accessibility Quantification

### Dominant Color Extraction: k Selection
- k=6: overall_score=7.43. Investigated further - dominant colors didn't
  include green (the plate's number color) at all, despite it being visually
  present. K-means selects colors by pixel frequency, and the number
  occupies far fewer pixels than the background, so it was absorbed into
  background-color clusters rather than forming its own.
- k=3, k=2: same issue, worse (scores rose to 15.86 and ~32, i.e. *more*
  "safe," while visual inspection confirmed the CVD-simulated image is
  clearly inaccessible - orange background and green number both simulate
  to near-identical green). This is a false negative in the metric, not
  an actual improvement.
- k=4-5: green cluster starts appearing. k=4: overall_score=6.92 (still
  above the ~3 JND danger threshold, worth further checking whether the
  green-containing pair specifically is the low scorer).
- Finding: frequency-based dominant color extraction can miss small but
  semantically critical regions (thin text/figures vs. large background
  areas) — a known limitation of naive palette extraction, not fixable by
  k tuning alone since increasing k mainly changes how background colors
  get subdivided, not whether minority colors get detected.

### Root cause of missed-green issue (refined)
- Visualized k=4 dominant colors directly against the source image. The
  "green" the metric found is not the number's true saturated green - it's
  a blended/averaged color (green mixed with background/edge pixels from
  anti-aliasing), diluted toward orange. This is why its cvd_diff (6.92)
  still landed above the ~3 JND danger threshold, despite visual inspection
  showing the simulated image is clearly inaccessible.
- Root cause: k-means centroid averaging has no concept of "meaningful
  object boundary" - a cluster containing partly-number, partly-background
  edge pixels produces a centroid that's a wash of both, not the true
  number color. This is distinct from (and more precise than) the earlier
  "minority pixels underrepresented" framing - the issue isn't just pixel
  count, it's that mixed/edge pixels corrupt the cluster's centroid color.
- Implication: naive whole-image k-means is not well-suited to detecting
  thin, edge-heavy shapes (text, icons) against a background - this is a
  real, citable limitation of frequency/centroid-based palette extraction
  for this use case.

## Phase 4: Correction

### Step 1: Rule-Based Daltonization
- Implemented classic daltonization: simulate CVD in LMS space, compute the
  error (original_LMS - simulated_LMS), redistribute that error into the
  cone channels the CVD viewer retains, add correction back to the
  ORIGINAL (not simulated) LMS, convert back to RGB.
- Validated on the Ishihara test image (protanopia): accessibility_score
  improved from 6.92 to 27.93 (~4x), both numerically and via visual
  inspection (number becomes clearly distinguishable from background under
  simulated protanopia post-correction).
- Observed and explained an interesting behavior: correction was
  concentrated on the orange background (high L-cone/red contribution,
  therefore high "error") while the green number changed minimally (low
  L-cone dependence, therefore low error) - correction magnitude scales
  correctly with how much a color relies on the compromised cone, which is
  the expected, principled behavior of daltonization.
- Result: reliable, interpretable, fast. No dependency on the Phase 3
  k-means pipeline (operates per-pixel, not on extracted palettes) -
  intentional design choice made after discovering k-means dominant-color
  extraction's limitations (see Phase 3 notes).

### Step 2: U-Net Learned Correction (PyTorch)
- Motivation: demonstrate PyTorch / neural network competency for
  internship applications (many relevant projects list U-Net/ResNet/GAN
  familiarity as a requirement), while keeping rule-based daltonization as
  the primary, reliable correction method.
- Architecture: small U-Net (3 encoder levels: 16->32->64 filters,
  bottleneck 128, symmetric decoder with skip connections via channel
  concatenation).
- Training approach: self-supervised (no labeled "correct" image dataset
  exists or is practical to create). Loss = distortion_from_original
  (MSE) - lambda * accessibility_gain, where accessibility_gain uses a
  differentiable, torch-native port of this project's own simulation math
  (src/differentiable.py).
- Two documented simplifications in the training signal, both explicitly
  chosen for timeline/tractability, not oversights:
  1. Used a luminance-weighted RGB-space distance instead of full
     Lab/CIE76 Delta-E (avoids porting Lab's non-linear cube-root
     conversion to torch). NOT luminance-only/hue-blind like WCAG contrast
     - still captures per-channel color difference, just less perceptually
     accurate than true Delta-E. Final evaluation still uses the real,
     accurate metric (metrics.py).
  2. Used local (neighboring-pixel) color spread as a differentiable proxy
     for "distinguishability," rather than the Phase 3 metric's global
     dominant-color-pair comparison. These are genuinely different
     signals - flagged as a training/evaluation metric mismatch.
- Trained on a single image (per-image optimization, not a generalizable
  model across a dataset) - explicit scope decision given timeline.
- Result (initial run, lambda=0.5, 200 epochs): training loss decreased
  and the training-time accessibility_gain metric rose steadily
  (0.22 -> 2.08), confirming the training loop and differentiable pipeline
  work correctly end-to-end. However, distortion (MSE from original) rose
  in parallel rather than staying bounded - a warning sign.
- Evaluating the trained output against the REAL Phase 3 metric gave
  overall_score = 39.55 (higher than both the original [6.92] and
  rule-based daltonization [27.93]) - numerically the "best" result.
  Visual inspection told a different story: the corrected image showed
  checkerboard artifacts (a known ConvTranspose2d upsampling issue) and
  globally distorted, oversaturated color (e.g. dominant colors shifting
  to magenta/pale-yellow, not present in the original image).
- Root cause: the local-spread training loss has no notion of "meaningful"
  vs. "arbitrary" contrast, so the model could inflate the proxy metric by
  distorting the whole image (maximizing pixel-to-pixel contrast
  indiscriminately) rather than performing a targeted, faithful color
  correction. This is a genuine reward-hacking-style failure mode, and it
  was correctly hypothesized BEFORE training (via a thought experiment
  about within-object contrast, e.g. leaf shading) and then confirmed via
  both the diverging distortion metric and visual inspection - a case
  study in why visual/qualitative validation is necessary even when a
  numeric proxy looks favorable.
- Mitigation attempted: reduced accessibility-loss weight (lambda 0.5 -> 0.1),
  reduced epochs (200 -> 100), and added a best-checkpoint safeguard that
  keeps the lowest-distortion model state seen during training rather than
  the final epoch's state.

  | Version | Overall Score | Visual Quality |
|---|---|---|
| Original | 6.92 | — |
| Rule-based daltonization | 27.93 | Clean, faithful |
| U-Net (lam=0.5, 200 epochs) | 39.55 | Severely distorted, checkerboard, oversaturated |
| U-Net (lam=0.1, 100 epochs) | 16.75 | Mild checkerboard, one localized artifact patch, otherwise closer to original |

### Conclusion
- Rule-based daltonization is the reliable, production-viable correction
  method delivered by this project (validated ~4x accessibility
  improvement).
- The U-Net demonstrates a full self-supervised PyTorch pipeline
  (custom differentiable loss built on this project's own metric code,
  encoder-decoder architecture with skip connections, training loop with
  Adam optimization) and genuine, rigorously-diagnosed research findings
  about its current limitations - but is not currently a viable
  correction method as implemented.

### Future Work
- Full Lab/CIE76 Delta-E ported to torch for a more perceptually-accurate
  training signal.
- Spatially-aware or region-aware loss (e.g. incorporating the Phase 3
  dominant-color-pair comparison directly, or a segmentation-aware term)
  instead of naive local pixel-neighbor spread, to avoid indiscriminate
  contrast maximization.
- Upsample+conv instead of ConvTranspose2d to address checkerboard
  artifacts.
- Train across a dataset of many images rather than per-image, for a
  generalizable correction model.
- Extend rule-based daltonization and U-Net training to deuteranopia and
  tritanopia (currently protanopia-only).

## Phase 5: Final Verification & Comparison

### Numeric scores aren't enough — verifying "does it actually work"
Ran a final consistent comparison (same image, same resize, same k=4, U-Net
with seed=42 and 200 epochs) across all three states: original, rule-based
daltonization, U-Net correction.

Final scores: Original = 8.96, Daltonization = 29.17, U-Net = 31.48

U-Net scored highest, so before calling it a win, I did the same visual
check we did for daltonization earlier: simulate CVD on the corrected
output and actually look at whether the number is visible. Turns out it's
not — the number's still stuck in the background, invisible to a protanope,
despite the higher score. Kind of a letdown, not gonna lie, but also the
most useful check I did the whole project.

So the score itself was misleading — U-Net's higher number seems to come
from the artifacts it introduces (the checkerboard/purple-line patterns)
creating new "dominant colors" that happen to score well pairwise, not
from actually fixing the real problem. If I'd trusted the number alone,
I'd have wrongly concluded U-Net was the better method.

**Conclusion:** rule-based daltonization is the only method in this
project that's verified both numerically and visually to solve the
accessibility problem. U-Net demonstrates the PyTorch pipeline works
end-to-end (training converges, integrates with the project's own metric
as a differentiable loss) but does not produce a functionally correct
result as implemented — a good example of
