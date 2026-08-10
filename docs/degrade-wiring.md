# Wiring `js/degrade.js` into index.html

Four edits to `index.html` + one to `tools/generate.mjs`. `js/degrade.js` is a plain
global (`window.DEGRADE`), no module/import needed.

## 1. Script tag

Next to the JSZip tag in `<head>` (index.html ~line 10):

```html
<script src="js/degrade.js"></script>
```

## 2. UI select — inside the "Export options" `<details>` body

Put it after the `exportgray` checkline (index.html ~line 298), before the trailing
`<p class="status">`:

```html
<div class="knob"><label>Degradation</label>
  <select id="degradepreset">
    <option value="clean" selected>clean — canvas render as-is</option>
    <option value="scan">scan — light noise, faint bands, jpeg q0.8</option>
    <option value="phone">phone — perspective, uneven light, blur, sensor noise</option>
    <option value="phone-hard">phone-hard — dim room, shaky hand, worn pen</option>
    <option value="mix">mix — random preset per sample (25% each)</option>
  </select></div>
```

## 3. `exportOpts()` — carry the choice

```js
    gray: document.getElementById('exportgray').checked,
    degrade: document.getElementById('degradepreset').value,   // <-- add
```

## 4. Export path — `generateSample()`

Degradation goes **after `exportTransform`, before `toBlob`**. Add the helper next to
`exportTransform` and swap two lines in `generateSample`:

```js
// Degradation runs on its own rand stream derived from p.seed, so tweaking the
// render code never shifts the degradation draws (and vice versa).
async function applyDegrade(cv, opts, seed) {
  const name = opts.degrade || 'clean';
  if (name === 'clean' || !window.DEGRADE) return { canvas: cv, preset: 'clean' };
  const r = mulberry32((seed ^ 0x9e3779b9) >>> 0);
  const pool = ['clean', 'scan', 'phone', 'phone-hard'];
  const preset = name === 'mix' ? pool[Math.floor(r() * pool.length) % pool.length] : name;
  if (preset === 'clean') return { canvas: cv, preset };
  let out = await DEGRADE.apply(cv, DEGRADE.presets[preset], r);
  if (opts.gray) {                       // chroma noise would undo the grayscale export
    const g = document.createElement('canvas');
    g.width = out.width; g.height = out.height;
    const gc = g.getContext('2d');
    gc.filter = 'grayscale(100%)';
    gc.drawImage(out, 0, 0);
    out = g;
  }
  return { canvas: out, preset };
}
```

```js
async function generateSample(base, off, opts) {
  const p = { ...base };
  p.seed = Math.floor(Math.random() * 1e9);
  p.text = sampleBatchText(base);
  renderLine(p, off);
  const out = exportTransform(off, opts);
  const deg = await applyDegrade(out, opts, p.seed);              // <-- add
  const blob = await new Promise(res => deg.canvas.toBlob(res, opts.mime, opts.q));  // <-- deg.canvas
  return {
    blob,
    meta: {
      text: p.text, split: textSplit(p.text), seed: p.seed,
      source: p._source || 'font', writer: p._writer || null, font: p._usedFont,
      degrade: deg.preset,                                        // <-- add
    },
  };
}
```

That single call site covers the batch button, the zip export and
`window.GEOSCRIBE.renderSample` (used by `tools/generate.mjs`) — they all funnel
through `generateSample`.

## 5. `tools/generate.mjs` — MIME map needs `.js`

Its static server has no `.js` entry, so `js/degrade.js` would 404 in the headless
run and `window.DEGRADE` would be undefined (silently falling back to clean):

```js
const MIME = { '.html': 'text/html', '.js': 'text/javascript', /* …rest unchanged… */ };
```

Optionally add `--set degrade=phone` support — it already works, since `renderSample`
overrides go through `readParams()`; if `degrade` is read from `exportOpts()` only,
pass it via the select instead, e.g.
`await page.select('#degradepreset', 'phone')` before the loop.

## Notes

- **Reproducibility.** `DEGRADE.apply(canvas, opts, rand)` draws every magnitude from
  the supplied `rand`, so `(seed, preset)` fully determines the output (JPEG encoding
  aside). Never pass `Math.random`.
- **Ordering tradeoff.** Placing degradation after `exportTransform` degrades at the
  final training height (64 px), which is fast and keeps artifacts at the scale the
  model sees. Degrading at full render resolution *then* downscaling is physically
  truer (real noise/blur is averaged away by the resize) but ~5–10x slower; if you
  want that, call `applyDegrade(off, …)` before `exportTransform` instead.
- **Preview.** To show the effect live, call the same helper on the preview canvas
  after `render()`; it is async, so `await` it or `.then`-chain.
