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
    overrides[k] = isNaN(+v) ? v : +v;
  }
}

// ---- tiny static server for the repo dir ----
const MIME = { '.html': 'text/html', '.ttf': 'font/ttf', '.otf': 'font/otf', '.txt': 'text/plain; charset=utf-8', '.md': 'text/plain', '.json': 'application/json' };
const server = http.createServer((req, res) => {
  const p = path.join(root, decodeURIComponent(req.url.split('?')[0]) === '/' ? 'index.html' : decodeURIComponent(req.url.split('?')[0]));
  if (!p.startsWith(root) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'Content-Type': MIME[path.extname(p)] || 'application/octet-stream' });
  fs.createReadStream(p).pipe(res);
});

const { default: puppeteer } = await import('puppeteer');

await new Promise(r => server.listen(PORT, r));
fs.mkdirSync(path.join(OUT, 'images'), { recursive: true });

const browser = await puppeteer.launch();
const page = await browser.newPage();
await page.goto(`http://localhost:${PORT}/`, { waitUntil: 'networkidle0' });
await page.waitForFunction(() => window.GEOSCRIBE && window.GEOSCRIBE.ready(), { timeout: 30000 });

const metaStream = fs.createWriteStream(path.join(OUT, 'metadata.jsonl'));
let bytes = 0;
const t0 = Date.now();
for (let i = 0; i < N; i++) {
  const s = await page.evaluate(o => window.GEOSCRIBE.renderSample(o), overrides);
  const fname = `${String(i).padStart(6, '0')}.${s.ext}`;
  const buf = Buffer.from(s.b64, 'base64');
  fs.writeFileSync(path.join(OUT, 'images', fname), buf);
  bytes += buf.length;
  metaStream.write(JSON.stringify({ file_name: 'images/' + fname, ...s.meta }) + '\n');
  if ((i + 1) % 100 === 0) {
    const rate = (i + 1) / ((Date.now() - t0) / 1000);
    console.log(`${i + 1}/${N} · ${(bytes / (i + 1) / 1024).toFixed(1)} KB/img · ${rate.toFixed(1)} img/s · ETA ${((N - i - 1) / rate / 60).toFixed(1)} min`);
  }
}
metaStream.end();
await browser.close();
server.close();
console.log(`done: ${N} samples, ${(bytes / 1048576).toFixed(1)} MB total → ${OUT}`);
