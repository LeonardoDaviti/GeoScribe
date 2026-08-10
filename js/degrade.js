/* ============================================================================
   GeoScribe · degrade.js — camera/scan degradation pipeline for synthetic text
   ----------------------------------------------------------------------------
   Self-contained browser module, no deps, vanilla JS (same style as index.html).

     await DEGRADE.apply(canvas, opts, rand) -> Promise<canvas>   (new canvas)
     DEGRADE.presets = { clean, scan, phone, 'phone-hard' }

   `rand` is a seeded ()=>[0,1) supplied by the caller (index.html uses
   mulberry32); every magnitude below is drawn from it, so the same seed +
   same preset always yields the same image (JPEG encoding aside, which is the
   browser's own deterministic-but-opaque encoder).

   `opts` is either a preset name, or an object keyed by stage. Each stage value
   is one of:
       false / 0 / undefined     stage off
       0.7                       intensity 0..1
       [0.3, 0.8]                intensity sampled uniformly per call
       { amount: [.3,.8], k: 2 } intensity + stage params (arrays sampled too)
   plus two non-stage keys:
       oneOf: [['defocus','motionblur']]   keep exactly one of each group
       background: '#fdfcf8'               used to flatten alpha before JPEG

   Stages run in physically sensible order — what happens to the page first
   happens to the pixels first:
       lowink -> paper -> perspective -> illumination -> defocus -> motionblur
              -> downup -> noise -> banding -> jpeg
   ========================================================================== */
(function (global) {
'use strict';

/* ---------------- small helpers ---------------- */
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return c;
}
function copyOnto(src, bg) {
  const c = makeCanvas(src.width, src.height);
  const x = c.getContext('2d');
  if (bg) { x.fillStyle = bg; x.fillRect(0, 0, c.width, c.height); }
  x.drawImage(src, 0, 0);
  return c;
}
const uni = (rand, a, b) => a + rand() * (b - a);
function sampleVal(rand, v) {
  return Array.isArray(v) ? uni(rand, v[0], v[1]) : v;
}
const gauss = rand => {
  let u = 0, v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};

/* Resolve one stage config into concrete params, or null if off. */
function stageParams(rand, cfg) {
  if (cfg === undefined || cfg === null || cfg === false) return null;
  if (typeof cfg === 'number') return cfg > 0 ? { amount: cfg } : null;
  if (Array.isArray(cfg)) {
    const a = uni(rand, cfg[0], cfg[1]);
    return a > 0 ? { amount: a } : null;
  }
  if (typeof cfg === 'object') {
    const o = {};
    for (const k in cfg) o[k] = sampleVal(rand, cfg[k]);
    if (o.amount === undefined) o.amount = 1;
    return o.amount > 0 ? o : null;
  }
  return null;
}

/* Low-frequency value-noise field on a coarse grid + bilinear sampler.
   Two smoothing passes make the blobs look like lighting/ink variation rather
   than salt-and-pepper. Returns f(x01, y01) -> [-1, 1]-ish. */
function noiseField(rand, gw, gh, smooth) {
  gw = Math.max(2, gw | 0); gh = Math.max(2, gh | 0);
  let g = new Float32Array(gw * gh);
  for (let i = 0; i < g.length; i++) g[i] = rand() * 2 - 1;
  for (let pass = 0; pass < (smooth === undefined ? 2 : smooth); pass++) {
    const t = new Float32Array(gw * gh);
    for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) {
      let s = 0, n = 0;
      for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) {
        const yy = y + j, xx = x + i;
        if (yy < 0 || xx < 0 || yy >= gh || xx >= gw) continue;
        s += g[yy * gw + xx]; n++;
      }
      t[y * gw + x] = s / n;
    }
    g = t;
  }
  // re-normalize to ~[-1,1] (smoothing shrinks the range a lot)
  let m = 1e-6;
  for (let i = 0; i < g.length; i++) m = Math.max(m, Math.abs(g[i]));
  for (let i = 0; i < g.length; i++) g[i] /= m;

  return function (u, v) {
    const fx = clamp(u, 0, 1) * (gw - 1), fy = clamp(v, 0, 1) * (gh - 1);
    const x0 = fx | 0, y0 = fy | 0;
    const x1 = Math.min(gw - 1, x0 + 1), y1 = Math.min(gh - 1, y0 + 1);
    const tx = fx - x0, ty = fy - y0;
    const a = g[y0 * gw + x0] * (1 - tx) + g[y0 * gw + x1] * tx;
    const b = g[y1 * gw + x0] * (1 - tx) + g[y1 * gw + x1] * tx;
    return a * (1 - ty) + b * ty;
  };
}

/* Solve an n×n dense system with partial pivoting (n = 8 here). */
function solveLinear(A, b, n) {
  const M = [];
  for (let i = 0; i < n; i++) M.push(A[i].slice().concat([b[i]]));
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    const tmp = M[col]; M[col] = M[piv]; M[piv] = tmp;
    const d = M[col][col];
    if (Math.abs(d) < 1e-12) return null;
    for (let c = col; c <= n; c++) M[col][c] /= d;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col];
      if (!f) continue;
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map(r => r[n]);
}

/* Homography mapping the unit-ish rect (0,0),(w,0),(w,h),(0,h) onto `quad`. */
function homography(w, h, quad) {
  const src = [[0, 0], [w, 0], [w, h], [0, h]];
  const A = [], b = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i], [X, Y] = quad[i];
    A.push([x, y, 1, 0, 0, 0, -x * X, -y * X]); b.push(X);
    A.push([0, 0, 0, x, y, 1, -x * Y, -y * Y]); b.push(Y);
  }
  const s = solveLinear(A, b, 8);
  return s ? [s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], 1] : null;
}

/* ---------------- stages ---------------- */

/* 1. lowink — streaky ink fading: multiply ink darkness by a low-freq field.
      Horizontally-stretched cells mimic a drying pen / worn ballpoint. */
function lowink(cv, p, rand) {
  const w = cv.width, h = cv.height;
  const ctx = cv.getContext('2d', { willReadFrequently: true });
  const id = ctx.getImageData(0, 0, w, h), d = id.data;
  const cells = Math.max(3, Math.round(w / (p.cell || 40)));
  const f = noiseField(rand, cells, Math.max(2, Math.round(h / 12)));
  const amt = p.amount * 0.85;
  // paper level estimated from the image corners
  const bgL = (d[0] * 0.299 + d[1] * 0.587 + d[2] * 0.114) || 250;
  for (let y = 0; y < h; y++) {
    const v = y / (h - 1 || 1);
    for (let x = 0; x < w; x++) {
      const o = (y * w + x) * 4;
      const l = d[o] * 0.299 + d[o + 1] * 0.587 + d[o + 2] * 0.114;
      if (l >= bgL - 6) continue;               // paper, nothing to fade
      const n = f(x / (w - 1 || 1), v);          // -1..1
      const fade = clamp(amt * (0.5 + 0.5 * n), 0, 0.95);
      d[o]     += (bgL - d[o])     * fade;
      d[o + 1] += (bgL - d[o + 1]) * fade;
      d[o + 2] += (bgL - d[o + 2]) * fade;
      if (d[o + 3] < 255) d[o + 3] *= 1 - fade;  // honour alpha inks too
    }
  }
  ctx.putImageData(id, 0, 0);
  return cv;
}

/* 2. paper — global tint shift + coarse texture (fibres / cheap notebook stock). */
function paper(cv, p, rand) {
  const w = cv.width, h = cv.height;
  const ctx = cv.getContext('2d', { willReadFrequently: true });
  const id = ctx.getImageData(0, 0, w, h), d = id.data;
  const a = p.amount;
  // warm (yellowed) or cool (fluorescent) shift
  const warm = rand() < 0.65;
  const tr = (warm ? uni(rand, 2, 14) : uni(rand, -10, -2)) * a;
  const tg = (warm ? uni(rand, 0, 6) : uni(rand, -4, 2)) * a;
  const tb = (warm ? uni(rand, -16, -4) : uni(rand, 4, 14)) * a;
  const coarse = noiseField(rand, Math.max(4, Math.round(w / 6)), Math.max(4, Math.round(h / 6)), 1);
  const grain = (p.grain === undefined ? uni(rand, 4, 12) : p.grain) * a;
  for (let y = 0; y < h; y++) {
    const v = y / (h - 1 || 1);
    for (let x = 0; x < w; x++) {
      const o = (y * w + x) * 4;
      const n = coarse(x / (w - 1 || 1), v) * grain;
      d[o]     = clamp(d[o] + tr + n, 0, 255);
      d[o + 1] = clamp(d[o + 1] + tg + n, 0, 255);
      d[o + 2] = clamp(d[o + 2] + tb + n, 0, 255);
    }
  }
  ctx.putImageData(id, 0, 0);
  return cv;
}

/* 3. perspective — subtle 4-corner homography, inverse-mapped with bilinear
      sampling (same technique as elasticWarp in index.html). The sampled quad
      is inset by the max corner offset so no destination pixel falls outside
      the source: no black wedges, just a mild zoom. */
function perspective(cv, p, rand) {
  const w = cv.width, h = cv.height;
  const a = p.amount;
  const kx = (p.kx === undefined ? 0.020 : p.kx) * a * w;   // keystone, x
  const ky = (p.ky === undefined ? 0.070 : p.ky) * a * h;   // keystone, y
  const pad = Math.max(kx, ky) + 1;
  const padx = Math.min(pad, w * 0.12), pady = Math.min(pad, h * 0.2);
  const base = [[padx, pady], [w - padx, pady], [w - padx, h - pady], [padx, h - pady]];
  const quad = base.map(c => [
    clamp(c[0] + gauss(rand) * kx * 0.5, 0, w - 1),
    clamp(c[1] + gauss(rand) * ky * 0.5, 0, h - 1),
  ]);
  const H = homography(w, h, quad);
  if (!H) return cv;

  const ctx = cv.getContext('2d', { willReadFrequently: true });
  const src = ctx.getImageData(0, 0, w, h).data;
  const out = new Uint8ClampedArray(src.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const den = H[6] * x + H[7] * y + H[8];
      let sx = (H[0] * x + H[1] * y + H[2]) / den;
      let sy = (H[3] * x + H[4] * y + H[5]) / den;
      sx = clamp(sx, 0, w - 1.001); sy = clamp(sy, 0, h - 1.001);
      const x0 = sx | 0, y0 = sy | 0, tx = sx - x0, ty = sy - y0;
      const o = (y * w + x) * 4;
      const i00 = (y0 * w + x0) * 4, i10 = i00 + 4;
      const i01 = i00 + w * 4, i11 = i01 + 4;
      for (let c = 0; c < 4; c++) {
        const t = (src[i00 + c] * (1 - tx) + src[i10 + c] * tx) * (1 - ty)
                + (src[i01 + c] * (1 - tx) + src[i11 + c] * tx) * ty;
        out[o + c] = t;
      }
    }
  }
  ctx.putImageData(new ImageData(out, w, h), 0, 0);
  return cv;
}

/* 4. illumination — directional gradient + soft shadow blobs + vignette,
      evaluated on a coarse grid and bilinearly upsampled. */
function illumination(cv, p, rand) {
  const w = cv.width, h = cv.height;
  const a = p.amount;
  const ang = rand() * Math.PI * 2;
  const grad = (p.grad === undefined ? uni(rand, 0.10, 0.32) : p.grad) * a;
  const vig = (p.vignette === undefined ? uni(rand, 0.06, 0.22) : p.vignette) * a;
  const nb = Math.round(p.blobs === undefined ? uni(rand, 1, 3.4) : p.blobs);
  const blobs = [];
  for (let i = 0; i < nb; i++) {
    blobs.push({
      x: uni(rand, -0.1, 1.1), y: uni(rand, -0.2, 1.2),
      rx: uni(rand, 0.15, 0.55), ry: uni(rand, 0.3, 1.2),
      s: uni(rand, 0.10, 0.38) * a * (rand() < 0.25 ? -1 : 1), // mostly shadows, sometimes a highlight
    });
  }
  const gx = Math.cos(ang), gy = Math.sin(ang);

  // coarse multiplier grid
  const GW = 33, GH = 17;
  const grid = new Float32Array(GW * GH);
  for (let j = 0; j < GH; j++) {
    const v = j / (GH - 1);
    for (let i = 0; i < GW; i++) {
      const u = i / (GW - 1);
      let m = 1 + grad * ((u - 0.5) * gx + (v - 0.5) * gy) * 2;
      for (const b of blobs) {
        const dx = (u - b.x) / b.rx, dy = (v - b.y) / b.ry;
        m -= b.s * Math.exp(-(dx * dx + dy * dy) * 2);
      }
      const rx = (u - 0.5) * 2, ry = (v - 0.5) * 2;
      m *= 1 - vig * clamp(rx * rx * 0.6 + ry * ry * 0.4, 0, 1);
      grid[j * GW + i] = clamp(m, 0.25, 1.6);
    }
  }

  const ctx = cv.getContext('2d', { willReadFrequently: true });
  const id = ctx.getImageData(0, 0, w, h), d = id.data;
  for (let y = 0; y < h; y++) {
    const fy = (y / (h - 1 || 1)) * (GH - 1), y0 = Math.min(GH - 2, fy | 0), ty = fy - y0;
    for (let x = 0; x < w; x++) {
      const fx = (x / (w - 1 || 1)) * (GW - 1), x0 = Math.min(GW - 2, fx | 0), tx = fx - x0;
      const m = (grid[y0 * GW + x0] * (1 - tx) + grid[y0 * GW + x0 + 1] * tx) * (1 - ty)
              + (grid[(y0 + 1) * GW + x0] * (1 - tx) + grid[(y0 + 1) * GW + x0 + 1] * tx) * ty;
      const o = (y * w + x) * 4;
      d[o] = clamp(d[o] * m, 0, 255);
      d[o + 1] = clamp(d[o + 1] * m, 0, 255);
      d[o + 2] = clamp(d[o + 2] * m, 0, 255);
    }
  }
  ctx.putImageData(id, 0, 0);
  return cv;
}

/* 5. defocus — gaussian-ish blur via ctx.filter. */
function defocus(cv, p, rand) {
  const px = p.px === undefined ? uni(rand, 0.3, 1.6) * p.amount : p.px;
  if (px < 0.05) return cv;
  const t = makeCanvas(cv.width, cv.height);
  const c = t.getContext('2d');
  c.filter = `blur(${px.toFixed(2)}px)`;
  c.drawImage(cv, 0, 0);
  return t;
}

/* 6. motionblur — N shifted copies along a random angle, averaged (hand shake). */
function motionblur(cv, p, rand) {
  const len = p.len === undefined ? uni(rand, 1.0, 4.5) * p.amount : p.len;
  if (len < 0.3) return cv;
  const ang = rand() * Math.PI;           // direction is symmetric
  const n = Math.max(3, Math.min(12, Math.round(len * 3)));
  const dx = Math.cos(ang) * len, dy = Math.sin(ang) * len * 0.5;
  const t = makeCanvas(cv.width, cv.height);
  const c = t.getContext('2d');
  c.globalAlpha = 1;
  c.drawImage(cv, -dx / 2, -dy / 2);      // first pass opaque, keeps edges filled
  for (let i = 1; i < n; i++) {
    const f = i / (n - 1) - 0.5;
    c.globalAlpha = 1 / (i + 1);
    c.drawImage(cv, dx * f, dy * f);
  }
  c.globalAlpha = 1;
  return t;
}

/* 7. downup — resolution round-trip (phone crop / re-upload). */
function downup(cv, p, rand) {
  const s = p.scale === undefined ? uni(rand, 0.42, 0.80) : p.scale;
  const dw = Math.max(2, Math.round(cv.width * s)), dh = Math.max(2, Math.round(cv.height * s));
  const small = makeCanvas(dw, dh);
  const sc = small.getContext('2d');
  sc.imageSmoothingQuality = 'high';
  sc.drawImage(cv, 0, 0, dw, dh);
  const t = makeCanvas(cv.width, cv.height);
  const c = t.getContext('2d');
  c.imageSmoothingEnabled = p.smooth === undefined ? rand() < 0.8 : !!p.smooth;
  c.imageSmoothingQuality = 'high';
  c.drawImage(small, 0, 0, cv.width, cv.height);
  return t;
}

/* 8. noise — sensor noise: luma-dominant, weaker chroma, amplified in shadows. */
function noise(cv, p, rand) {
  const w = cv.width, h = cv.height;
  const sig = (p.sigma === undefined ? uni(rand, 2, 14) : p.sigma) * p.amount;
  const chroma = p.chroma === undefined ? uni(rand, 0.25, 0.6) : p.chroma;
  const shadow = p.shadow === undefined ? uni(rand, 0.6, 1.8) : p.shadow;
  const ctx = cv.getContext('2d', { willReadFrequently: true });
  const id = ctx.getImageData(0, 0, w, h), d = id.data;
  for (let i = 0; i < d.length; i += 4) {
    const l = d[i] * 0.299 + d[i + 1] * 0.587 + d[i + 2] * 0.114;
    const gain = sig * (1 + shadow * (1 - l / 255));
    const nl = gauss(rand) * gain;
    const nr = gauss(rand) * gain * chroma;
    const nb = gauss(rand) * gain * chroma;
    d[i]     = clamp(d[i] + nl + nr, 0, 255);
    d[i + 1] = clamp(d[i + 1] + nl - (nr + nb) * 0.5, 0, 255);
    d[i + 2] = clamp(d[i + 2] + nl + nb, 0, 255);
  }
  ctx.putImageData(id, 0, 0);
  return cv;
}

/* 9. banding — faint horizontal scanner / rolling-shutter stripes. */
function banding(cv, p, rand) {
  const w = cv.width, h = cv.height;
  const amp = (p.amp === undefined ? uni(rand, 1.5, 9) : p.amp) * p.amount;
  const nWaves = 2 + Math.floor(rand() * 2);
  const waves = [];
  for (let i = 0; i < nWaves; i++) {
    waves.push({
      f: uni(rand, 0.6, 9) * Math.PI * 2,
      ph: rand() * Math.PI * 2,
      a: uni(rand, 0.3, 1),
    });
  }
  const jitter = noiseField(rand, 2, Math.max(4, Math.round(h / 2)));
  const ctx = cv.getContext('2d', { willReadFrequently: true });
  const id = ctx.getImageData(0, 0, w, h), d = id.data;
  for (let y = 0; y < h; y++) {
    const v = y / (h - 1 || 1);
    let s = 0, tot = 0;
    for (const wv of waves) { s += Math.sin(v * wv.f + wv.ph) * wv.a; tot += wv.a; }
    const off = (s / (tot || 1)) * amp + jitter(0.5, v) * amp * 0.4;
    let o = y * w * 4;
    for (let x = 0; x < w; x++, o += 4) {
      d[o] = clamp(d[o] + off, 0, 255);
      d[o + 1] = clamp(d[o + 1] + off, 0, 255);
      d[o + 2] = clamp(d[o + 2] + off, 0, 255);
    }
  }
  ctx.putImageData(id, 0, 0);
  return cv;
}

/* 10. jpeg — 1..2 recompression cycles (messaging apps, re-saves). async. */
async function jpegCycles(cv, p, rand, bg) {
  const rounds = Math.max(1, Math.round(p.rounds === undefined ? uni(rand, 1, 2.4) : p.rounds));
  const qlo = p.qlo === undefined ? 0.5 : p.qlo;
  const qhi = p.qhi === undefined ? 0.8 : p.qhi;
  let cur = cv;
  for (let i = 0; i < rounds; i++) {
    const q = p.q === undefined ? uni(rand, qlo, qhi) : p.q;
    const blob = await new Promise(res => cur.toBlob(res, 'image/jpeg', q));
    if (!blob) break;
    const bmp = await createImageBitmap(blob);
    const t = makeCanvas(cur.width, cur.height);
    const c = t.getContext('2d');
    c.fillStyle = bg; c.fillRect(0, 0, t.width, t.height);
    c.drawImage(bmp, 0, 0);
    if (bmp.close) bmp.close();
    cur = t;
  }
  return cur;
}

/* ---------------- presets ----------------
   Values are per-call ranges; a batch of samples with one preset still varies.
   `clean` is a no-op passthrough (returns an untouched copy).                 */
const presets = {
  clean: {},

  // flatbed / phone-scanner app output: already deskewed and white-balanced
  scan: {
    illumination: { amount: [0.15, 0.4], blobs: [0, 1.4], vignette: [0.1, 0.4] },
    noise:        { amount: [0.15, 0.4], sigma: [2, 6], shadow: [0.4, 1.0] },
    banding:      { amount: [0.2, 0.5], amp: [1.5, 4] },
    jpeg:         { rounds: 1, qlo: 0.75, qhi: 0.85 },
    paper:        { amount: [0, 0.3], grain: [3, 7] },
  },

  // handheld photo of a notebook page: geometry + lighting + sensor + re-encode
  phone: {
    oneOf:        [['defocus', 'motionblur']],
    perspective:  [0.35, 0.9],
    illumination: { amount: [0.4, 0.85], blobs: [1, 3.4], grad: [0.12, 0.3], vignette: [0.1, 0.3] },
    defocus:      { amount: 1, px: [0.4, 1.4] },
    motionblur:   { amount: 1, len: [1.0, 3.2] },
    noise:        { amount: [0.3, 0.7], sigma: [4, 11], shadow: [0.7, 1.6] },
    banding:      [0, 0.35],
    downup:       { scale: [0.5, 0.8] },
    paper:        { amount: [0.2, 0.6], grain: [4, 10] },
    jpeg:         { rounds: [1, 2.4], qlo: 0.55, qhi: 0.8 },
  },

  // worst realistic case: dim room, shaky hand, worn pen, re-shared image
  'phone-hard': {
    oneOf:        [['defocus', 'motionblur']],
    lowink:       { amount: [0.3, 0.75], cell: [25, 70] },
    perspective:  [0.7, 1.0],
    illumination: { amount: [0.75, 1.0], blobs: [2, 4.4], grad: [0.2, 0.35], vignette: [0.2, 0.45] },
    defocus:      { amount: 1, px: [1.0, 2.6] },
    motionblur:   { amount: 1, len: [2.5, 5.5] },
    noise:        { amount: [0.6, 1.0], sigma: [9, 18], shadow: [1.0, 2.0] },
    banding:      { amount: [0.3, 0.8], amp: [3, 9] },
    downup:       { scale: [0.35, 0.6] },
    paper:        { amount: [0.4, 1.0], grain: [7, 16] },
    jpeg:         { rounds: [1.6, 2.4], qlo: 0.4, qhi: 0.65 },
  },
};

/* Stage table in application order. `fn` may return a new canvas. */
const ORDER = [
  ['lowink', lowink],
  ['paper', paper],
  ['perspective', perspective],
  ['illumination', illumination],
  ['defocus', defocus],
  ['motionblur', motionblur],
  ['downup', downup],
  ['noise', noise],
  ['banding', banding],
];

/* ---------------- public API ---------------- */
async function apply(canvas, opts, rand) {
  if (typeof rand !== 'function') rand = Math.random;
  if (typeof opts === 'string') opts = presets[opts] || {};
  opts = opts || {};
  const bg = opts.background || '#fdfcf8';

  // resolve every stage up-front so the rand() draw order is stable regardless
  // of which stages end up running
  const resolved = {};
  for (const [name] of ORDER) resolved[name] = stageParams(rand, opts[name]);
  resolved.jpeg = stageParams(rand, opts.jpeg);

  // "one of" groups: keep exactly one member (e.g. defocus XOR motionblur)
  for (const group of (opts.oneOf || [])) {
    const live = group.filter(n => resolved[n]);
    if (live.length > 1) {
      const keep = live[Math.floor(rand() * live.length) % live.length];
      for (const n of live) if (n !== keep) resolved[n] = null;
    }
  }

  let cv = copyOnto(canvas, bg);          // flatten alpha, never touch the input
  for (const [name, fn] of ORDER) {
    const p = resolved[name];
    if (p) cv = fn(cv, p, rand) || cv;
  }
  if (resolved.jpeg) cv = await jpegCycles(cv, resolved.jpeg, rand, bg);
  return cv;
}

global.DEGRADE = {
  version: 1,
  apply,
  presets,
  presetNames: Object.keys(presets),
  // exposed for tests / bespoke pipelines
  _stages: { lowink, paper, perspective, illumination, defocus, motionblur, downup, noise, banding, jpegCycles },
};

})(typeof window !== 'undefined' ? window : globalThis);
