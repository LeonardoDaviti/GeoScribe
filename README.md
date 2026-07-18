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

## Drawing mode (Stage B: your hand as a generative model)

The **drawing board** captures each of the 33 Mkhedruli letters as pointer-event
trajectories — real (x, y, t) online handwriting data, not pixels:

- strokes are arc-length resampled (44 pts) with **real pen speed** carried per point,
  normalized against the median velocity;
- draw each letter ≥2× and generation **morphs between random pairs of your executions**
  plus smooth correlated noise — a cheap measured motor-variability model;
- rendering thins strokes on fast segments (sigma-lognormal-flavored pen model),
  with `pen width`, `speed thinning`, and `motor jitter` knobs;
- hand glyphs flow through the *same* Stage A composition pipeline (baseline drift,
  overlaps, elastic field, ink model); letters without samples fall back to fonts;
- the writer profile persists in localStorage and can be exported/imported as a small
  JSON file — one file ≈ one writer.

### Multi-writer & word-level capture

- **Word mode** on the drawing board captures whole handwritten words (wider canvas,
  any word you type or a random corpus word). When generated text contains a drawn
  word, a morphed instance of the *whole word* is used — real co-articulation and
  spacing — with letter-composition fallback otherwise.
- **Writer profiles**: create/delete/merge named writers; any subset can be *active*
  for generation. Imports **add** writers (auto-renamed on collision), never overwrite —
  so community-contributed profile JSONs can be pooled into a writer population.
- **Hand/font mixing**: `hand probability` sets the per-sample chance of a hand-written
  vs font-rendered sample. The source is decided per sample, not per glyph — a real
  scanned line is one writer with one pen; the mixture lives across the batch.
- `labels.jsonl` records `source` (hand/font) and `writer` per sample, enabling
  **writer-disjoint splits** downstream — required for a credible benchmark.

## Roadmap

- **Stage B+** — joining splines between letter exit/entry points (true cursive);
  per-letter mean-shape + covariance estimation once ≥5 samples exist.
- **Stage C** — small trajectory RNN/diffusion trained on parametric + real scanned samples.

## Fonts

19 verified Mkhedruli styles: 13 handwriting/calligraphic hands (DM writer-hand fonts,
3D Unicode, Calligraphy, MCh Vardi ×2) + 6 print styles (BPG Glaho, BPG Nino Elite Caps
(Mtavruli), Gancxadebebi, Kolkheti, Noto Sans/Serif Georgian via Google Fonts).

Every local font in `fonts/` was validated with fontTools (full U+10D0–U+10F0 cmap
coverage + non-empty glyph outlines) plus a visual render check. Three fonts are legacy
ASCII-mapped Georgian fonts; the generator transliterates each character to its Latin
keyboard slot before rendering (standard Georgian keyboard mapping). Rejected from the
source collection: 4 fonts containing only Latin glyphs, 1 with a scrambled mapping,
2 with nonstandard symbol cmaps (tofu).

**Font modes** (research-motivated — style *diversity* beats per-sample realism for
downstream HTR pretraining, cf. "Quo Vadis HTG for HTR?", ICCV 2025):
- fixed font,
- random handwriting font per sample (default; the font id is recorded in `labels.jsonl`),
- random handwriting font per glyph (maximum diversity, unrealistic but useful).

⚠️ **License note:** Noto is OFL. The licenses of the DM / MCh / BPG / legacy fonts in
`fonts/` have not been verified — check before publishing rendered datasets or hosting
the fonts publicly.
