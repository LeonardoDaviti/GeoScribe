#!/usr/bin/env node
/**
 * Test harness for js/degrade.js.
 *
 * Serves the repo over a tiny static server (same pattern as tools/generate.mjs),
 * loads tools/degrade_test.html in Puppeteer, and for every preset checks:
 *   · output dimensions == input dimensions
 *   · non-clean presets actually change pixels (diff fraction over threshold)
 *   · clean is a pure passthrough
 *   · output is not blank — ink is still visible (dark-pixel fraction + contrast)
 *   · determinism: same seed twice -> byte-identical PNG (jpeg stage disabled)
 *   · with the jpeg stage on, repeated runs stay within 15% output size
 *   · different seeds -> different images (a batch varies)
 *   · the input canvas is never mutated
 *
 * Writes /tmp/degrade_<preset>.png for eyeballing.
 *
 * Usage: node tools/test_degrade.mjs [--port 8139] [--out /tmp]
 */
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const flag = (n, d) => { const i = args.indexOf('--' + n); return i >= 0 ? args[i + 1] : d; };
const PORT = parseInt(flag('port', '8139'));
const OUTDIR = flag('out', '/tmp');

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.ttf': 'font/ttf', '.otf': 'font/otf', '.txt': 'text/plain; charset=utf-8',
  '.json': 'application/json',
};
const server = http.createServer((req, res) => {
  const rel = decodeURIComponent(req.url.split('?')[0]);
  const p = path.join(root, rel === '/' ? 'index.html' : rel);
  if (!p.startsWith(root) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'Content-Type': MIME[path.extname(p)] || 'application/octet-stream' });
  fs.createReadStream(p).pipe(res);
});

const { default: puppeteer } = await import('puppeteer');
await new Promise(r => server.listen(PORT, r));

const browser = await puppeteer.launch();
const page = await browser.newPage();
page.on('pageerror', e => { console.error('PAGE ERROR:', e.message); });
page.on('console', m => { if (m.type() === 'error') console.error('console:', m.text()); });
await page.goto(`http://localhost:${PORT}/tools/degrade_test.html`, { waitUntil: 'networkidle0' });
await page.waitForFunction(() => window.T && window.DEGRADE);

let failures = 0;
const check = (name, ok, detail) => {
  if (!ok) failures++;
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${name}${detail ? ' · ' + detail : ''}`);
};
const n3 = v => (typeof v === 'number' ? v.toFixed(3) : String(v));

const presets = await page.evaluate(() => window.T.presets());
const written = [];

for (const preset of presets) {
  console.log(`\n[${preset}]`);
  const r = await page.evaluate(p => window.T.run(p, 12345), preset);

  check('dimensions preserved', r.w === r.sw && r.h === r.sh, `${r.w}x${r.h}`);

  const clean = preset === 'clean';
  if (clean) {
    check('clean is a passthrough', r.diff.frac === 0 && r.diff.meanAbs === 0);
  } else {
    check('pixels changed', r.diff.frac > 0.02,
      `changed=${n3(r.diff.frac)} meanAbs=${n3(r.diff.meanAbs)}`);
  }

  // ink still legible: dark-pixel fraction within a sane band of the source, and
  // the image still has real contrast (not washed flat / not crushed to black)
  const ratio = r.out.darkFrac / r.src.darkFrac;
  check('ink still visible', ratio > 0.25 && ratio < 4 && r.out.std > 8,
    `darkFrac ${n3(r.src.darkFrac)}->${n3(r.out.darkFrac)} (x${n3(ratio)}) std=${n3(r.out.std)} mean=${n3(r.out.mean)}`);
  check('not blank', r.out.std > 6 && r.out.mean > 30 && r.out.mean < 250, `std=${n3(r.out.std)}`);

  const det = await page.evaluate(p => window.T.determinism(p, 777, true), preset);
  check('deterministic (no jpeg)', det.same, `bytes ${det.bytesA}/${det.bytesB}`);

  const detJ = await page.evaluate(p => window.T.determinism(p, 777, false), preset);
  const sizeDelta = Math.abs(detJ.bytesA - detJ.bytesB) / detJ.bytesA;
  check('jpeg path stable in size', sizeDelta < 0.15, `Δ=${(sizeDelta * 100).toFixed(1)}%`);

  if (!clean) {
    const v = await page.evaluate(p => window.T.variety(p), preset);
    check('different seeds differ', !v.same);
  }

  const pur = await page.evaluate(p => window.T.purity(p), preset);
  check('input canvas untouched', pur.untouched);

  const file = path.join(OUTDIR, `degrade_${preset}.png`);
  fs.writeFileSync(file, Buffer.from(r.png.split(',')[1], 'base64'));
  written.push(file);
}

await browser.close();
server.close();

console.log('\nPNGs written:');
for (const f of written) console.log('  ' + f);
console.log(failures ? `\n${failures} check(s) FAILED` : '\nall checks passed');
process.exit(failures ? 1 : 0);
