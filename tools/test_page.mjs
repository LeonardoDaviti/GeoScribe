#!/usr/bin/env node
/**
 * Headless smoke test for page composition + bounding-box ground truth.
 *
 *   node tools/test_page.mjs [--port 8139]
 *
 * Renders 2 line samples and 2 page samples through the same in-page hook the bulk
 * generator uses, then checks the geometry invariants (boxes positive, inside the image,
 * words inside their line). Writes one page PNG to /tmp/geoscribe_page.png.
 */
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const flag = (n, d) => { const i = args.indexOf('--' + n); return i >= 0 ? args[i + 1] : d; };
const PORT = parseInt(flag('port', '8139'));
const PNG_OUT = flag('png', '/tmp/geoscribe_page.png');

const MIME = { '.html': 'text/html', '.ttf': 'font/ttf', '.otf': 'font/otf', '.txt': 'text/plain; charset=utf-8', '.md': 'text/plain', '.json': 'application/json', '.js': 'text/javascript' };
const server = http.createServer((req, res) => {
  const rel = decodeURIComponent(req.url.split('?')[0]);
  const p = path.join(root, rel === '/' ? 'index.html' : rel);
  if (!p.startsWith(root) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'Content-Type': MIME[path.extname(p)] || 'application/octet-stream' });
  fs.createReadStream(p).pipe(res);
});

let failures = 0;
const check = (cond, msg) => {
  if (!cond) { failures++; console.error('FAIL: ' + msg); }
};

const { default: puppeteer } = await import('puppeteer');
await new Promise(r => server.listen(PORT, r));

const browser = await puppeteer.launch();
const page = await browser.newPage();
page.on('pageerror', e => { failures++; console.error('FAIL: page error: ' + e.message); });
await page.goto(`http://localhost:${PORT}/`, { waitUntil: 'networkidle0' });
await page.waitForFunction(() => window.GEOSCRIBE && window.GEOSCRIBE.ready(), { timeout: 30000 });

// ---- line mode (unchanged contract) ----
for (let i = 0; i < 2; i++) {
  const s = await page.evaluate(() => window.GEOSCRIBE.renderSample({ layout: 'lines' }));
  const buf = Buffer.from(s.b64, 'base64');
  check(buf.length > 200, `line sample ${i}: image is empty (${buf.length} bytes)`);
  check(typeof s.meta.text === 'string' && s.meta.text.length > 0, `line sample ${i}: missing text`);
  check(['train', 'val'].includes(s.meta.split), `line sample ${i}: bad split ${s.meta.split}`);
  check(s.meta.lines === undefined, `line sample ${i}: line metadata must stay box-free`);
  console.log(`line ${i}: ${buf.length} B · "${s.meta.text.slice(0, 40)}"`);
}

// ---- page mode ----
const contains = (outer, inner, m) =>
  inner[0] >= outer[0] - m && inner[1] >= outer[1] - m &&
  inner[0] + inner[2] <= outer[0] + outer[2] + m && inner[1] + inner[3] <= outer[1] + outer[3] + m;

for (let i = 0; i < 2; i++) {
  const s = await page.evaluate(() => window.GEOSCRIBE.renderSample({ layout: 'pages' }));
  const buf = Buffer.from(s.b64, 'base64');
  const m = s.meta;
  check(buf.length > 2000, `page ${i}: image is empty (${buf.length} bytes)`);
  check(m.lines.length >= 6 && m.lines.length <= 14, `page ${i}: ${m.lines.length} lines, expected 6–14`);
  check(m.text.split('\n').length === m.lines.length, `page ${i}: text/lines mismatch`);
  check(['train', 'val'].includes(m.split), `page ${i}: bad split ${m.split}`);
  check(m.words.length > 0, `page ${i}: no word boxes`);
  const W = m.width, H = m.height;
  check(W > 0 && H > 0, `page ${i}: bad page size ${W}x${H}`);
  for (const [li, l] of m.lines.entries()) {
    const b = l.box;
    check(b.every(Number.isInteger), `page ${i} line ${li}: non-integer box ${b}`);
    check(b[2] > 0 && b[3] > 0, `page ${i} line ${li}: non-positive size ${b}`);
    check(contains([0, 0, W, H], b, 0), `page ${i} line ${li}: box ${b} outside ${W}x${H}`);
    check(l.words.length > 0, `page ${i} line ${li}: no words`);
    for (const w of l.words) {
      check(w.box[2] > 0 && w.box[3] > 0, `page ${i} line ${li}: word "${w.text}" non-positive size ${w.box}`);
      check(contains([0, 0, W, H], w.box, 0), `page ${i} line ${li}: word box ${w.box} outside image`);
      // words are merged from the same placements as the line box, so containment is exact;
      // 2px of slack absorbs the independent rounding of each box
      check(contains(b, w.box, 2), `page ${i} line ${li}: word "${w.text}" box ${w.box} escapes line box ${b}`);
    }
  }
  const flat = m.lines.flatMap(l => l.words.map(w => w.text)).join(' ');
  check(flat === m.words.map(w => w.text).join(' '), `page ${i}: flat word list out of sync with lines`);
  console.log(`page ${i}: ${buf.length} B · ${W}x${H} · ${m.lines.length} lines · ${m.words.length} words · ${m.source}/${m.font}`);
}

// ---- one PNG on disk for eyeballing ----
const png = await page.evaluate(async () => {
  const cv = document.createElement('canvas');
  const pg = renderPage({ ...readParams(), seed: 424242, ruled: true }, cv);
  return { b64: cv.toDataURL('image/png').split(',')[1], n: pg.lines.length };
});
fs.writeFileSync(PNG_OUT, Buffer.from(png.b64, 'base64'));
console.log(`sample page (${png.n} lines) → ${PNG_OUT}`);

await browser.close();
server.close();
if (failures) { console.error(`\n${failures} check(s) failed`); process.exit(1); }
console.log('\nall checks passed');
