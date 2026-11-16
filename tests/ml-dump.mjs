// node tests/ml-dump.mjs <out-dir> <audio files...>
// Runs both models once per song and caches the result as <out-dir>/<name>.ml.json (beat and downbeat
// probabilities quantised to 0..255 at 50 fps, plus the feel). tests/run.mjs --ml <out-dir> reuses it,
// so the engine can be tuned without rerunning the networks.
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { join, dirname, basename, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as ort from 'onnxruntime-node';
import { createML } from '../ml.js';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const [outDir, ...files] = process.argv.slice(2);
mkdirSync(outDir, { recursive: true });
const load = name => readFileSync(join(root, 'models', name));
const ml = await createML(ort, load, { sessionOptions: { executionProviders: ['cpu'], intraOpNumThreads: +(process.env.THREADS || 4) } });
const q = a => Buffer.from(Uint8Array.from(a, v => Math.round(Math.max(0, Math.min(1, v)) * 255))).toString('base64');

for (const f of files) {
  const out = join(outDir, basename(f, extname(f)) + '.ml.json');
  if (existsSync(out)) continue;
  try {
    const t = Date.now();
    const buf = execFileSync('ffmpeg', ['-v', 'error', '-i', f, '-ac', '1', '-ar', '22050', '-f', 'f32le', '-'], { maxBuffer: 1 << 30 });
    const x = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
    const r = await ml.run(x);
    const feel = r.feel && !r.feel.error ? { mood: r.feel.mood, styles: r.feel.styles, curves: Object.fromEntries(Object.entries(r.feel.curves).map(([k, v]) => [k, Array.from(v, z => +(+z).toFixed(3))])) } : r.feel;
    writeFileSync(out, JSON.stringify({ fps: r.fps, beat: q(r.beat), down: q(r.down), feel }));
    console.log(`${((Date.now() - t) / 1000).toFixed(0)} s  ${basename(f)}`);
  } catch (e) { console.log('FAIL', basename(f), String(e.message || e).slice(0, 200)); }
}
