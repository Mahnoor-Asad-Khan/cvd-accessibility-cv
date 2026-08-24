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
(TBD)

## Phase 5: Experiments & Comparison
(TBD)
