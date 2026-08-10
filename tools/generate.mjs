#!/usr/bin/env node
/**
 * GeoScribe headless bulk generator.
 *
 * Drives the same index.html that runs on GitHub Pages, via Puppeteer, and writes
 * an HF-imagefolder dataset (images/ + metadata.jsonl) straight to disk — no browser
 * zip memory limit, so 100k+ samples are fine.
 *
 * Usage:
 *   npm install puppeteer          # one-time, downloads its own Chromium
 *   node tools/generate.mjs --n 5000 --out dataset [--port 8137] [--workers 8]
 *
 * Generator settings (font mode, deformation knobs, export format/height/grayscale)
 * are the DEFAULTS of the page. To bake in different settings, pass overrides:
 *   node tools/generate.mjs --n 5000 --out dataset --set elasticamp=9 --set handprob=0
 *
 * Parallelism: --workers N runs N independent headless browsers, each owning a disjoint
 * slice of the index space and streaming rows to metadata.part-K.jsonl. After all
 * workers finish the parts are merged into metadata.jsonl in index order (the merge is
 * a full rewrite keyed on the index embedded in file_name, so it is idempotent and
 * resume-safe). Rendering is per-sample independent (each sample draws its own random
 * seed in-page), so sharding changes nothing about the output distribution.
 *
 * --batch B renders B samples per page.evaluate round-trip (same in-page code path,
 * just looped) to amortise CDP overhead.
 *
 * --profile prints an in-page ms breakdown (renderLine vs applyDegrade vs encode) per
 * degradation preset and exits without writing a dataset.
 */
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

// ---- args ----
const args = process.argv.slice(2);
const flag = (name, def) => {
  const i = args.indexOf('--' + name);
  return i >= 0 ? args[i + 1] : def;
};
const N = parseInt(flag('n', '1000'));
const OUT = path.resolve(flag('out', 'dataset'));
const PORT = parseInt(flag('port', '8137'));
const WORKERS = Math.max(1, parseInt(flag('workers', '1')));
// --batch measured at B=8: 5.88 vs 5.94 img/s single-worker, i.e. no gain — the CDP
// round-trip is <2% of a sample's cost. Left in as a knob, defaulted off (B=1) so a
// crash never costs more than one sample of retried work.
const BATCH = Math.max(1, parseInt(flag('batch', '1')));
const RECYCLE = Math.max(1, parseInt(flag('recycle', '10000')));
const PROFILE = args.includes('--profile');
const overrides = {};
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--set') {
    const [k, v] = args[i + 1].split('=');
    overrides[k] = v === 'true' ? true : v === 'false' ? false : isNaN(+v) ? v : +v;
  }
}

// --writers profile.json: inject drawn hand profiles (drawing-board export) into the
// headless page's localStorage so generation can use real hands, and enable handmode
const writersPath = flag('writers', null);
let writersPayload = null;
if (writersPath) {
  const raw = JSON.parse(fs.readFileSync(writersPath, 'utf8'));
  const writers = raw.writers || raw;
  writersPayload = JSON.stringify({ writers, active: Object.keys(writers) });
  if (overrides.handmode === undefined) overrides.handmode = true;
  console.log(`writers: ${Object.keys(writers).join(', ')} (handmode on, handprob ${overrides.handprob ?? 'slider default 0.6'})`);
}

// ---- tiny static server for the repo dir ----
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.ttf': 'font/ttf', '.otf': 'font/otf', '.txt': 'text/plain; charset=utf-8', '.md': 'text/plain', '.json': 'application/json', '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp' };
const server = http.createServer((req, res) => {
  const p = path.join(root, decodeURIComponent(req.url.split('?')[0]) === '/' ? 'index.html' : decodeURIComponent(req.url.split('?')[0]));
  if (!p.startsWith(root) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'Content-Type': MIME[path.extname(p)] || 'application/octet-stream' });
  fs.createReadStream(p).pipe(res);
});

const { default: puppeteer } = await import('puppeteer');

await new Promise(r => server.listen(PORT, r));
fs.mkdirSync(path.join(OUT, 'images'), { recursive: true });

// Chromium flags. Measured A/B at 4 workers: no throughput difference (the render path
// is 2D-canvas software raster either way — headless "GPU" here is SwiftShader, another
// software stack). Kept only because --disable-gpu drops one GPU process (~180 MB) per
// worker, which is what actually limits how many workers fit in RAM.
const LAUNCH_ARGS = [
  '--disable-gpu',
  '--disable-dev-shm-usage',
];

// Ctrl-C / SIGTERM would otherwise leave N headless Chromes reparented to init.
const liveBrowsers = new Set();
for (const sig of ['SIGINT', 'SIGTERM']) {
  process.on(sig, () => {
    for (const b of liveBrowsers) { try { b.process()?.kill('SIGKILL'); } catch (e) {} }
    process.exit(130);
  });
}

async function launchWorkerBrowser() {
  const browser = await puppeteer.launch({ args: LAUNCH_ARGS });
  liveBrowsers.add(browser);
  browser.on('disconnected', () => liveBrowsers.delete(browser));
  const p = await browser.newPage();
  if (writersPayload) {
    await p.evaluateOnNewDocument(payload => {
      localStorage.setItem('geoscribe_hand_profiles_v2', payload);
    }, writersPayload);
  }
  await p.goto(`http://localhost:${PORT}/`, { waitUntil: 'networkidle0' });
  await p.waitForFunction(() => window.GEOSCRIBE && window.GEOSCRIBE.ready(), { timeout: 60000 });
  // batched entry point: literally a loop over the same renderSample the single-shot
  // path uses, so per-sample semantics (and per-seed determinism) are unchanged
  await p.evaluate(() => {
    window.__gsBatch = async (o, b) => {
      const out = [];
      for (let i = 0; i < b; i++) out.push(await window.GEOSCRIBE.renderSample(o));
      return out;
    };
  });
  return { browser, page: p };
}

// ---------------------------------------------------------------- profile mode
if (PROFILE) {
  const { browser, page } = await launchWorkerBrowser();
  const presets = ['clean', 'scan', 'phone', 'photo'];
  const perPreset = parseInt(flag('n', '40'));
  console.log(`profiling ${perPreset} samples per preset (in-page ms)…\n`);
  for (const preset of presets) {
    const r = await page.evaluate(async (o, preset, n) => {
      // wrap the globals generateSample() resolves at call time; restore afterwards
      const realRender = window.renderLine, realDeg = window.applyDegrade;
      const t = { render: 0, degrade: 0, total: 0, encode: 0 };
      window.renderLine = function (...a) { const s = performance.now(); const v = realRender.apply(this, a); t.render += performance.now() - s; return v; };
      window.applyDegrade = async function (...a) { const s = performance.now(); const v = await realDeg.apply(this, a); t.degrade += performance.now() - s; return v; };
      const t0 = performance.now();
      for (let i = 0; i < n; i++) {
        const e0 = performance.now();
        await window.GEOSCRIBE.renderSample({ ...o, degrade: preset });
        t.encode += performance.now() - e0;
      }
      t.total = performance.now() - t0;
      window.renderLine = realRender; window.applyDegrade = realDeg;
      return t;
    }, overrides, preset, perPreset);
    const per = x => (x / perPreset).toFixed(2);
    const rest = r.total - r.render - r.degrade;
    console.log(`  ${preset.padEnd(6)}  total ${per(r.total)} ms/img  ·  renderLine ${per(r.render)}  ·  applyDegrade ${per(r.degrade)}  ·  export+encode+b64 ${per(rest)}  ·  ${(perPreset / (r.total / 1000)).toFixed(1)} img/s in-page`);
  }
  await browser.close();
  server.close();
  process.exit(0);
}

// ---------------------------------------------------------------- work planning
const metaPath = path.join(OUT, 'metadata.jsonl');
const partPath = k => path.join(OUT, `metadata.part-${k}.jsonl`);
const resume = args.includes('--resume');

const idxOf = row => {
  const m = /(\d+)\.[a-z]+$/.exec(row.file_name || '');
  return m ? parseInt(m[1], 10) : null;
};
// tolerant reader: a killed run can leave a half-flushed final line in a part file
function readRows(file) {
  if (!fs.existsSync(file)) return [];
  const out = [];
  for (const line of fs.readFileSync(file, 'utf8').split('\n')) {
    if (!line) continue;
    try { const r = JSON.parse(line); if (idxOf(r) !== null) out.push(r); } catch (e) { /* torn tail */ }
  }
  return out;
}

// every row already on disk — merged metadata plus any part files a killed run left behind
const done = new Map();
if (resume) {
  for (const r of readRows(metaPath)) done.set(idxOf(r), r);
  for (const f of fs.readdirSync(OUT)) {
    if (/^metadata\.part-\d+\.jsonl$/.test(f)) for (const r of readRows(path.join(OUT, f))) done.set(idxOf(r), r);
  }
  console.log(`resuming: ${done.size} rows already on disk, target ${N}`);
} else {
  for (const f of fs.readdirSync(OUT)) {
    if (/^metadata\.part-\d+\.jsonl$/.test(f)) fs.unlinkSync(path.join(OUT, f));
  }
}

const todo = [];
for (let i = 0; i < N; i++) if (!done.has(i)) todo.push(i);
if (!todo.length) {
  console.log('nothing to do');
}

// contiguous split: every sample costs the same, so equal-sized slices stay balanced
const slices = Array.from({ length: WORKERS }, () => []);
{
  const per = Math.ceil(todo.length / WORKERS);
  for (let k = 0; k < WORKERS; k++) slices[k] = todo.slice(k * per, (k + 1) * per);
}

// ---------------------------------------------------------------- run
let completed = 0, bytes = 0;
const t0 = Date.now();
let lastLog = 0;
let tFirst = 0; // first completed sample — lets us report steady-state rate net of browser launch

async function runWorker(k, indices) {
  if (!indices.length) return;
  let { browser, page } = await launchWorkerBrowser();
  const stream = fs.createWriteStream(partPath(k), { flags: 'a' });
  const recycle = async () => {
    await browser.close().catch(() => {});
    ({ browser, page } = await launchWorkerBrowser());
  };
  let sinceRecycle = 0;
  for (let pos = 0; pos < indices.length; pos += BATCH) {
    const chunk = indices.slice(pos, pos + BATCH);
    // the renderer accumulates memory over tens of thousands of evaluate round-trips
    // and eventually crashes — recycle proactively, and relaunch + retry on failure
    if (sinceRecycle >= RECYCLE) {
      console.log(`worker ${k}: recycling browser after ${sinceRecycle} samples (renderer memory hygiene)`);
      await recycle();
      sinceRecycle = 0;
    }
    let samples;
    for (let attempt = 0; ; attempt++) {
      try {
        samples = await page.evaluate((o, b) => window.__gsBatch(o, b), overrides, chunk.length);
        break;
      } catch (e) {
        if (attempt >= 2) throw e;
        console.warn(`worker ${k}: batch at ${chunk[0]} failed (${String(e.message).split('\n')[0]}); relaunching browser`);
        await recycle();
        sinceRecycle = 0;
      }
    }
    for (let j = 0; j < chunk.length; j++) {
      const s = samples[j];
      const fname = `${String(chunk[j]).padStart(6, '0')}.${s.ext}`;
      const buf = Buffer.from(s.b64, 'base64');
      fs.writeFileSync(path.join(OUT, 'images', fname), buf);
      bytes += buf.length;
      stream.write(JSON.stringify({ file_name: 'images/' + fname, ...s.meta }) + '\n');
    }
    sinceRecycle += chunk.length;
    if (!tFirst) tFirst = Date.now();
    completed += chunk.length;
    if (completed - lastLog >= 100) {
      lastLog = completed;
      const rate = completed / ((Date.now() - t0) / 1000);
      const left = todo.length - completed;
      console.log(`${done.size + completed}/${N} · ${(bytes / completed / 1024).toFixed(1)} KB/img · ${rate.toFixed(1)} img/s · ETA ${(left / rate / 60).toFixed(1)} min`);
    }
  }
  await new Promise(r => stream.end(r));
  await browser.close().catch(() => {});
}

// allSettled, not all: if one worker dies we still let the others drain before merging,
// otherwise the merge would race against workers still appending to their part files.
const results = await Promise.allSettled(slices.map((s, k) => runWorker(k, s)));
const failed = results.filter(r => r.status === 'rejected');
{
  // merge parts (+ anything already in metadata.jsonl) into one index-ordered file.
  // full rewrite keyed on index, so re-running merge is idempotent and never duplicates.
  const all = new Map(done);
  const parts = fs.readdirSync(OUT).filter(f => /^metadata\.part-\d+\.jsonl$/.test(f));
  for (const f of parts) for (const r of readRows(path.join(OUT, f))) all.set(idxOf(r), r);
  const ordered = [...all.keys()].sort((a, b) => a - b);
  const tmp = metaPath + '.tmp';
  fs.writeFileSync(tmp, ordered.map(i => JSON.stringify(all.get(i))).join('\n') + (ordered.length ? '\n' : ''));
  fs.renameSync(tmp, metaPath);
  for (const f of parts) fs.unlinkSync(path.join(OUT, f));
  console.log(`merged ${ordered.length} rows → metadata.jsonl`);
}

server.close();
const secs = (Date.now() - t0) / 1000;
const steady = tFirst ? completed / ((Date.now() - tFirst) / 1000) : 0;
console.log(`done: ${completed} new samples in ${secs.toFixed(1)}s (${(completed / secs).toFixed(2)} img/s wall, ` +
  `${steady.toFixed(2)} img/s steady-state, ${WORKERS} workers)` +
  (completed ? `, ${(bytes / 1048576).toFixed(1)} MB → ${OUT}` : ''));
if (failed.length) {
  for (const f of failed) console.error(`worker failed: ${f.reason?.message || f.reason}`);
  console.error(`${failed.length}/${WORKERS} workers failed — rerun with --resume to fill the gaps`);
  process.exitCode = 1;
}
