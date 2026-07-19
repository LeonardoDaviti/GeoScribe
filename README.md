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

The **Dataset .zip** button generates N samples (sentences + random word lines, random
seeds, current knob settings) and downloads an HF-imagefolder-style archive:

```
images/00000.webp …
metadata.jsonl      # {"file_name": "images/00000.webp", "text", "split", "seed", "source", "writer", "font"}
params.json         # generator + export configuration used
```

**Export options** (training-optimized): WebP/JPEG/PNG with quality control, downscale
to final training height (48–64 px — full-res RGBA PNGs are ~5× wasted bytes when the
loader resizes anyway), and grayscale. WebP q0.85 @ 64 px ≈ 10–25 KB/image, so
50k samples ≈ 0.7–1 GB instead of ~15 GB of PNGs.

**Leakage-proof splits**: every sample gets a `split` field computed as a hash of its
*text* — identical text always lands in the same split, so no sentence can appear in
both train and val. (Text-overlapping splits let a decoder memorize the corpus instead
of learning to read; with a small corpus this dominates the loss descent.)

## External corpus & local bulk generation

`corpus/words.txt` and `corpus/sentences.txt` (one item per line) replace the embedded
mini-corpus at runtime — served both by GitHub Pages and any local static server, so
the corpus can grow to book scale without touching or bloating the HTML.
Corpus size should exceed dataset image count to prevent text memorization.

For datasets beyond the browser-zip range (100k+ samples), the headless driver reuses
the exact same page:

```
python3 -m http.server        # local dev: same app at localhost:8000
npm install puppeteer         # one-time
node tools/generate.mjs --n 50000 --out dataset --set handprob=0.5
```

It writes `images/` + `metadata.jsonl` straight to disk (no zip memory limit) via the
in-page `window.GEOSCRIBE.renderSample` hook — the same code path as the batch button.
No framework, no build step: the "local app" and the public site are the same file.

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

### Stage B+: cursive joining (implemented)

Adjacent hand-drawn letters within a word can connect: a Bernoulli connect-vs-lift
decision per letter pair — `P(connect) = knob × exp(−1.3·|Δy| − 2·max(0, gap−0.7))`,
penalizing vertical travel (Mkhedruli has 13 ascenders + 13 descenders) — then a
G1-continuous cubic Bézier bridge along the exit/entry tangents (handles = gap/3),
drawn thinner (fast transit stroke) before elastic/ink processing so joins age with
the rest of the line. The `cursive joining` knob sweeps formal print-hand (0) to
flowing diary-hand (1) — the two registers documented for Georgian handwriting.

## Roadmap

- Per-letter mean-shape + covariance estimation once ≥5 samples exist.
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
