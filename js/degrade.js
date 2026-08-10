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

/* Invert a 3x3 homography (row-major, 9 elements). */
function invert3(H) {
  const [a, b, c, d, e, f, g, h, i] = H;
  const A = e * i - f * h, B = c * h - b * i, C = b * f - c * e;
  const det = a * A + d * B + g * C;
  if (Math.abs(det) < 1e-12) return null;
  return [
    A / det, B / det, C / det,
    (f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det,
    (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det,
  ];
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

/* 2. paper — global tint shift + coarse texture (fibres / cheap notebook stock).
      When real photographed blank-paper textures are supplied, one is multiply-
      blended over the page first — closes texture gaps no noise model can. */
function paper(cv, p, rand, images) {
  const w = cv.width, h = cv.height;
  if (images && images.paper && images.paper.length) {
    const img = images.paper[Math.floor(rand() * images.paper.length) % images.paper.length];
    const iw = img.width || img.naturalWidth, ih = img.height || img.naturalHeight;
    const sc2 = Math.max(w / iw, h / ih) * uni(rand, 1.0, 1.4);
    const dx = uni(rand, 0, Math.max(0, iw * sc2 - w)), dy = uni(rand, 0, Math.max(0, ih * sc2 - h));
    const x2 = cv.getContext('2d');
    x2.save();
    x2.globalCompositeOperation = 'multiply';
    x2.globalAlpha = clamp(p.amount, 0, 1) * 0.55;
    x2.drawImage(img, -dx, -dy, iw * sc2, ih * sc2);
    x2.restore();
  }
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

/* 2b. scene — "photographed on a desk": the page becomes a rotated/keystoned quad
       composited over a textured desk surface with a drop shadow along its edges.
       Unlike `perspective` (which insets to hide wedges), this WANTS the surround
       visible — that's what a real phone photo of a notebook page looks like. */
function scene(cv, p, rand, images) {
  const w = cv.width, h = cv.height;
  const a = p.amount;

  // page quad: scaled toward center, slightly rotated, per-corner keystone jitter
  const s = p.scale === undefined ? uni(rand, 0.80, 0.95) : p.scale;
  const th = gauss(rand) * 0.035 * a;                      // ±~2-4° camera roll
  const kx = 0.05 * a * w, ky = 0.09 * a * h;
  const cx = w / 2, cy = h / 2, cs = Math.cos(th), sn = Math.sin(th);
  const base = [[-cx * s, -cy * s], [cx * s, -cy * s], [cx * s, cy * s], [-cx * s, cy * s]];
  const quad = base.map(([x, y]) => [
    clamp(cx + x * cs - y * sn + gauss(rand) * kx * 0.5, 1, w - 2),
    clamp(cy + x * sn + y * cs + gauss(rand) * ky * 0.5, 1, h - 2),
  ]);
  const H = homography(w, h, quad);
  const Hi = H && invert3(H);
  if (!Hi) return cv;

  // desk: a real photographed surface when available (cover-scaled, random flip),
  // otherwise a procedural dark surface with coarse streaky texture
  const desk = makeCanvas(w, h);
  const dctx = desk.getContext('2d', { willReadFrequently: true });
  const deskImgs = images && images.desk && images.desk.length ? images.desk : null;
  if (deskImgs) {
    const img = deskImgs[Math.floor(rand() * deskImgs.length) % deskImgs.length];
    const iw = img.width || img.naturalWidth, ih = img.height || img.naturalHeight;
    const sc2 = Math.max(w / iw, h / ih) * uni(rand, 1.0, 1.3);   // slight over-zoom = crop variety
    const dx = uni(rand, 0, Math.max(0, iw * sc2 - w)), dy = uni(rand, 0, Math.max(0, ih * sc2 - h));
    dctx.save();
    if (rand() < 0.5) { dctx.translate(w, 0); dctx.scale(-1, 1); }
    dctx.drawImage(img, -dx, -dy, iw * sc2, ih * sc2);
    dctx.restore();
  } else {
    const kinds = [[92, 62, 38], [70, 70, 74], [34, 32, 30], [120, 104, 84]];
    const [br, bg2, bb] = kinds[Math.floor(rand() * kinds.length) % kinds.length];
    dctx.fillStyle = `rgb(${br},${bg2},${bb})`;
    dctx.fillRect(0, 0, w, h);
    const tex = noiseField(rand, Math.max(4, Math.round(w / 30)), Math.max(3, Math.round(h / 90)), 1);
    const did = dctx.getImageData(0, 0, w, h), dd = did.data;
    for (let y = 0; y < h; y++) {
      const v = y / (h - 1 || 1);
      for (let x = 0; x < w; x++) {
        const o = (y * w + x) * 4;
        const n = tex(x / (w - 1 || 1), v) * 14 + (rand() - 0.5) * 6;
        dd[o] = clamp(dd[o] + n, 0, 255);
        dd[o + 1] = clamp(dd[o + 1] + n, 0, 255);
        dd[o + 2] = clamp(dd[o + 2] + n, 0, 255);
      }
    }
    dctx.putImageData(did, 0, 0);
  }

  // page drop shadow: the quad offset a few px, blurred, before the page goes on
  const sx0 = uni(rand, 3, 10), sy0 = uni(rand, 4, 14);
  dctx.save();
  dctx.filter = `blur(${uni(rand, 5, 12).toFixed(1)}px)`;
  dctx.fillStyle = 'rgba(0,0,0,0.35)';
  dctx.beginPath();
  quad.forEach(([x, y], i) => i ? dctx.lineTo(x + sx0, y + sy0) : dctx.moveTo(x + sx0, y + sy0));
  dctx.closePath(); dctx.fill();
  dctx.restore();

  // inverse-map: dest pixel inside the quad samples the page, outside keeps desk
  const src = cv.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, w, h).data;
  const out = dctx.getImageData(0, 0, w, h), od = out.data;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const den = Hi[6] * x + Hi[7] * y + Hi[8];
      const sx = (Hi[0] * x + Hi[1] * y + Hi[2]) / den;
      const sy = (Hi[3] * x + Hi[4] * y + Hi[5]) / den;
      if (sx < 0 || sy < 0 || sx > w - 1.001 || sy > h - 1.001) continue;
      const x0 = sx | 0, y0 = sy | 0, tx = sx - x0, ty = sy - y0;
      const o = (y * w + x) * 4;
      const i00 = (y0 * w + x0) * 4, i10 = i00 + 4, i01 = i00 + w * 4, i11 = i01 + 4;
      for (let c = 0; c < 3; c++) {
        od[o + c] = (src[i00 + c] * (1 - tx) + src[i10 + c] * tx) * (1 - ty)
                  + (src[i01 + c] * (1 - tx) + src[i11 + c] * tx) * ty;
      }
      od[o + 3] = 255;
    }
  }
  dctx.putImageData(out, 0, 0);
  return desk;
}

/* 2c. colortemp — white-balance shift: warm tungsten lamp or cool daylight/LED. */
function colortemp(cv, p, rand) {
  const t = (p.t === undefined ? uni(rand, 0.04, 0.18) : p.t) * p.amount;
  const warm = rand() < 0.6;
  const rGain = warm ? 1 + t : 1 - t * 0.7;
  const bGain = warm ? 1 - t : 1 + t * 0.8;
  const ctx = cv.getContext('2d', { willReadFrequently: true });
  const id = ctx.getImageData(0, 0, cv.width, cv.height), d = id.data;
  for (let i = 0; i < d.length; i += 4) {
    d[i] = clamp(d[i] * rGain, 0, 255);
    d[i + 2] = clamp(d[i + 2] * bGain, 0, 255);
  }
  ctx.putImageData(id, 0, 0);
  return cv;
}

/* 4b. shadowcast — a hard-ish cast shadow (the photographing hand/phone, a shelf):
       one defined dark shape with a soft edge, much stronger than illumination
       blobs. Either a bar entering from an edge or a rotated ellipse. */
function shadowcast(cv, p, rand) {
  const w = cv.width, h = cv.height;
  const a = p.amount;
  const t = makeCanvas(w, h);
  const c = t.getContext('2d');
  c.drawImage(cv, 0, 0);
  c.save();
  c.filter = `blur(${uni(rand, 6, Math.max(10, h * 0.06)).toFixed(1)}px)`;
  c.globalAlpha = uni(rand, 0.15, 0.4) * a;
  c.fillStyle = 'rgb(10,8,6)';
  if (rand() < 0.5) {
    // bar from a random edge (phone silhouette at the frame border)
    const edge = Math.floor(rand() * 4) % 4;
    const depth = uni(rand, 0.12, 0.4);
    c.beginPath();
    if (edge === 0) c.rect(0, 0, w, h * depth);
    else if (edge === 1) c.rect(0, h * (1 - depth), w, h * depth);
    else if (edge === 2) c.rect(0, 0, w * depth, h);
    else c.rect(w * (1 - depth), 0, w * depth, h);
    c.fill();
  } else {
    // rotated ellipse mid-frame (hand/arm shadow)
    c.translate(uni(rand, 0.2, 0.8) * w, uni(rand, 0.1, 0.9) * h);
    c.rotate(rand() * Math.PI);
    c.beginPath();
    c.ellipse(0, 0, uni(rand, 0.2, 0.5) * w, uni(rand, 0.1, 0.35) * h, 0, 0, Math.PI * 2);
    c.fill();
  }
  c.restore();
  return t;
}

/* 4c. glare — specular highlight: glossy paper under a lamp / window reflection. */
function glare(cv, p, rand) {
  const w = cv.width, h = cv.height;
  const a = p.amount;
  const t = makeCanvas(w, h);
  const c = t.getContext('2d');
  c.drawImage(cv, 0, 0);
  const gx = uni(rand, 0.15, 0.85) * w, gy = uni(rand, 0.1, 0.9) * h;
  const r = uni(rand, 0.18, 0.55) * Math.max(w, h);
  const g = c.createRadialGradient(gx, gy, 0, gx, gy, r);
  const peak = uni(rand, 0.25, 0.6) * a;
  g.addColorStop(0, `rgba(255,253,244,${peak.toFixed(3)})`);
  g.addColorStop(0.55, `rgba(255,253,244,${(peak * 0.35).toFixed(3)})`);
  g.addColorStop(1, 'rgba(255,253,244,0)');
  c.globalCompositeOperation = 'screen';
  c.save();
  c.translate(gx, gy);
  c.scale(1, uni(rand, 0.4, 1));            // elongated sheen more often than round
  c.rotate(gauss(rand) * 0.4);
  c.translate(-gx, -gy);
  c.fillStyle = g;
  c.fillRect(0, 0, w, h);
  c.restore();
  c.globalCompositeOperation = 'source-over';
  return t;
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

  // simulated photo shoot: page on a desk at a real angle, cast shadows, glare,
  // lamp/daylight white balance — the closest to "I photographed my notebook"
  photo: {
    oneOf:        [['defocus', 'motionblur']],
    scene:        { amount: [0.5, 1.0], scale: [0.80, 0.95] },
    colortemp:    [0.3, 1.0],
    illumination: { amount: [0.3, 0.75], blobs: [0, 2.4], grad: [0.1, 0.28], vignette: [0.12, 0.35] },
    shadowcast:   [0.35, 1.0],
    glare:        [0, 0.8],
    defocus:      { amount: 1, px: [0.4, 1.3] },
    motionblur:   { amount: 1, len: [1.0, 2.8] },
    noise:        { amount: [0.3, 0.7], sigma: [4, 10], shadow: [0.7, 1.6] },
    downup:       { scale: [0.5, 0.8] },
    jpeg:         { rounds: [1, 2], qlo: 0.6, qhi: 0.82 },
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
  ['scene', scene],
  ['perspective', perspective],
  ['colortemp', colortemp],
  ['illumination', illumination],
  ['shadowcast', shadowcast],
  ['glare', glare],
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
  const images = opts.images || null;     // real background photos: { desk: [..], paper: [..] }
  for (const [name, fn] of ORDER) {
    const p = resolved[name];
    if (p) cv = fn(cv, p, rand, images) || cv;
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
  _stages: { lowink, paper, scene, perspective, colortemp, illumination, shadowcast, glare, defocus, motionblur, downup, noise, banding, jpegCycles },
};

})(typeof window !== 'undefined' ? window : globalThis);
