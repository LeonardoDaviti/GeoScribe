# GeoScribe — Parametric Georgian Handwriting Generator

A single-page, zero-dependency-server synthetic handwriting generator for the Georgian
(Mkhedruli) script. Everything runs client-side in Canvas 2D, so it can be hosted on
**GitHub Pages** as-is.

**Live demo:** enable GitHub Pages on this repo (Settings → Pages → deploy from `main`, root)
and open `https://<user>.github.io/GeoScribe/`.

## What it does (Stage A: glyph-space deformations)

Instead of rendering whole strings atomically, each character is rendered separately and
composed with a chain of parametric distortions — a miniature SynthTIGER for Georgian:

1. **Per-glyph random affine** — rotation, scale and shear jitter, plus a global slant.
2. **Kerning jitter with negative spacing** — occasional overlaps produce fake ligatures,
   crucial for handwriting realism.
3. **Baseline drift** — slow sine wave + mean-reverting random walk.
4. **Elastic distortion** — a smooth random displacement field (Gaussian-filtered noise),
   applied via backward mapping with bilinear sampling.
5. **Stroke-width variation** — morphological erode/dilate, globally and per-glyph.
6. **Ink & paper model** — per-glyph darkness variation, edge bleed, ink hues,
   paper texture noise, optional ruled lines.

Every knob is a distribution parameter (`elastic α/σ`, `baseline_amp`, `overlap_prob`, …).
Randomness is fully **seeded** (mulberry32), so every sample is reproducible.

## Dataset export

The **Dataset .zip** button generates N samples (random Georgian word sequences, random
seeds, current knob settings) and downloads:

```
images/00000.png …
labels.jsonl        # {"file": "images/00000.png", "text": "...", "seed": ...}
params.json         # the generator configuration used
```

Ready to feed into HTR training or upload to Hugging Face.

## Roadmap

- **Stage B** — stroke-based letters: each of the 33 Mkhedruli letters as Bézier control
  points; correlated control-point noise (motor variability), joining splines (true cursive),
  speed-dependent pen thickness (sigma-lognormal-inspired).
- **Stage C** — small trajectory RNN/diffusion trained on parametric + real scanned samples.

## Fonts

Uses Noto Sans/Serif Georgian via Google Fonts (OFL-licensed).
