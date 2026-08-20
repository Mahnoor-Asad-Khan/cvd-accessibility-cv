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

## Phase 3: Accessibility Quantification
(TBD)

## Phase 4: Correction
(TBD)

## Phase 5: Experiments & Comparison
(TBD)
