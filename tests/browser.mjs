// node tests/browser.mjs [url] [song.f32]
// Drives the real page in headless Chromium (playwright-core + chromium-headless-shell): worker, onnxruntime-web,
// model fetches and the UI. Without a url it serves this folder on :8765. With a raw mono 22.05 kHz float32
// file (ffmpeg -ac 1 -ar 22050 -f f32le) it analyses that song; otherwise a synthetic 124 BPM track with a drop.
import { createServer } from 'node:http';
import { readFileSync, existsSync, statSync } from 'node:fs';
import { join, dirname, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
let url = process.argv[2] && process.argv[2] !== '-' ? process.argv[2] : null;
const raw = process.argv[3];
let server;
if (!url) {
  const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.onnx': 'application/octet-stream', '.bin': 'application/octet-stream', '.f32': 'application/octet-stream' };
  server = createServer((req, res) => {
    let p = decodeURIComponent(req.url.split('?')[0]);
    const file = p === '/__test.f32' && raw ? raw : join(root, p === '/' ? 'index.html' : p);
    if (!existsSync(file) || statSync(file).isDirectory()) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { 'content-type': types[extname(file)] || 'application/octet-stream' }); res.end(readFileSync(file));
  }).listen(8765);
  url = 'http://localhost:8765/';
}

const browser = await chromium.launch({ args: ['--enable-unsafe-webgpu'] });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') console.log('[page]', m.type(), m.text().slice(0, 300)); });
page.on('pageerror', e => console.log('[pageerror]', e.message));
await page.goto(url, { waitUntil: 'load' });
const out = await page.evaluate(async useRaw => {
  let x; const sr = 22050;
  if (useRaw) x = new Float32Array(await (await fetch('/__test.f32')).arrayBuffer());
  else {
    const secs = 40, step = 60 / 124; x = new Float32Array(sr * secs);
    for (let t = 0.5; t < secs - 0.2; t += step) {
      const loud = t > 20 ? 1 : 0.25, i0 = Math.round(t * sr);
      for (let i = 0; i < 0.12 * sr && i0 + i < x.length; i++) x[i0 + i] += loud * (Math.exp(-i / (0.03 * sr)) * Math.sin(2 * Math.PI * (50 + 80 * Math.exp(-i / (0.01 * sr))) * i / sr) * 0.9 + Math.exp(-i / (0.003 * sr)) * (Math.random() * 2 - 1) * 0.3);
    }
  }
  const t = performance.now();
  const r = await window.cutgrid.analyzeSamples(x, sr, useRaw ? 'song' : 'synthetic 124');
  return { ms: Math.round(performance.now() - t), bpm: r.bpm, meter: r.meter, beats: r.beats.length, model: r.model, mlError: r.mlError || null, mlMs: r.mlMs, scores: r.scores, metrics: { confMean: r.metrics.confMean, calmShare: r.metrics.calmShare }, drops: r.drops.map(d => d.t), status: document.getElementById('modelStatus').textContent, feel: document.getElementById('feel').innerText.replace(/\s+/g, ' ').slice(0, 240), webgpu: !!navigator.gpu };
}, !!raw);
console.log(JSON.stringify(out, null, 1));
await page.screenshot({ path: join(root, 'tests', 'shot.jpg'), type: 'jpeg', quality: 70, fullPage: true });
await browser.close();
if (server) server.close();
