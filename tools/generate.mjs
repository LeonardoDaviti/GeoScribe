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
 *   node tools/generate.mjs --n 5000 --out dataset [--port 8137]
 *
 * Generator settings (font mode, deformation knobs, export format/height/grayscale)
 * are the DEFAULTS of the page. To bake in different settings, pass overrides:
 *   node tools/generate.mjs --n 5000 --out dataset --set elasticamp=9 --set handprob=0
 *
 * NOTE: written on a machine without a browser installed, so this script is untested;
 * the in-page hook (window.GEOSCRIBE.renderSample) is the same code path as the tested
 * batch button. If something breaks, it will be here, not in the render math.
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

let browser, page;
async function setupPage() {
  const p = await browser.newPage();
  if (writersPayload) {
    await p.evaluateOnNewDocument(payload => {
      localStorage.setItem('geoscribe_hand_profiles_v2', payload);
    }, writersPayload);
  }
  await p.goto(`http://localhost:${PORT}/`, { waitUntil: 'networkidle0' });
  await p.waitForFunction(() => window.GEOSCRIBE && window.GEOSCRIBE.ready(), { timeout: 30000 });
  return p;
}
async function recycleBrowser() {
  if (browser) await browser.close().catch(() => {});
  browser = await puppeteer.launch();
  page = await setupPage();
}
await recycleBrowser();

// --resume: continue an interrupted run — count finished images, append to metadata.
// (Without it a restart would truncate metadata.jsonl and orphan every existing image.)
const resume = args.includes('--resume');
let start = 0;
if (resume) {
  const metaPath = path.join(OUT, 'metadata.jsonl');
  // trust metadata line count (an image may exist whose metadata row was never flushed;
  // it just gets regenerated under the same name)
  start = fs.existsSync(metaPath)
    ? fs.readFileSync(metaPath, 'utf8').split('\n').filter(Boolean).length : 0;
  console.log(`resuming at ${start}/${N}`);
}
const metaStream = fs.createWriteStream(path.join(OUT, 'metadata.jsonl'), { flags: resume ? 'a' : 'w' });
let bytes = 0;
const t0 = Date.now();
for (let i = start; i < N; i++) {
  // the renderer accumulates memory over tens of thousands of evaluate round-trips
  // and eventually crashes — recycle proactively, and relaunch + retry on failure
  if (i > start && (i - start) % 10000 === 0) {
    console.log(`recycling browser at ${i} (renderer memory hygiene)`);
    await recycleBrowser();
  }
  let s;
  for (let attempt = 0; ; attempt++) {
    try {
      s = await page.evaluate(o => window.GEOSCRIBE.renderSample(o), overrides);
      break;
    } catch (e) {
      if (attempt >= 2) throw e;
      console.warn(`sample ${i} failed (${String(e.message).split('\n')[0]}); relaunching browser`);
      await recycleBrowser();
    }
  }
  const fname = `${String(i).padStart(6, '0')}.${s.ext}`;
  const buf = Buffer.from(s.b64, 'base64');
  fs.writeFileSync(path.join(OUT, 'images', fname), buf);
  bytes += buf.length;
  metaStream.write(JSON.stringify({ file_name: 'images/' + fname, ...s.meta }) + '\n');
  if ((i + 1) % 100 === 0) {
    const done = i + 1 - start;
    const rate = done / ((Date.now() - t0) / 1000);
    console.log(`${i + 1}/${N} · ${(bytes / done / 1024).toFixed(1)} KB/img · ${rate.toFixed(1)} img/s · ETA ${((N - i - 1) / rate / 60).toFixed(1)} min`);
  }
}
metaStream.end();
await browser.close();
server.close();
console.log(`done: ${N} samples, ${(bytes / 1048576).toFixed(1)} MB total → ${OUT}`);
