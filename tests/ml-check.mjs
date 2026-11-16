// node tests/ml-check.mjs song.mp3 [tools/ref/<song>.json]
// Checks the JS Beat This front end + chunked inference against the PyTorch reference, then prints the feel.
import { execFileSync } from 'node:child_process';
import { readFileSync, existsSync } from 'node:fs';
import { join, dirname, basename, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as ort from 'onnxruntime-node';
import { beatThisSpect, createML } from '../ml.js';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const f = process.argv[2];
const refPath = process.argv[3] || join(root, 'tools', 'ref', basename(f, extname(f)) + '.json');
const buf = execFileSync('ffmpeg', ['-v', 'error', '-i', f, '-ac', '1', '-ar', '22050', '-f', 'f32le', '-'], { maxBuffer: 1 << 30 });
const x = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const load = name => readFileSync(join(root, 'models', name));

if (existsSync(refPath)) {
  const ref = JSON.parse(readFileSync(refPath, 'utf8'));
  const fbB = load('beatthis-mel.bin'), fb = new Float32Array(fbB.buffer.slice(fbB.byteOffset, fbB.byteOffset + fbB.byteLength));
  const { data, T } = beatThisSpect(x, fb);
  let maxD = 0; for (const [i, row] of Object.entries(ref.spectRows)) for (let b = 0; b < 128; b++) maxD = Math.max(maxD, Math.abs(row[b] - data[i * 128 + b]));
  console.log(`spect frames js ${T} py ${ref.frames}, max |diff| on sample rows ${maxD.toFixed(5)}`);
}
const t = Date.now();
let lastStage = '';
const onProgress = s => { if (s !== lastStage) { console.log(`  ${((Date.now() - t) / 1000).toFixed(1)} s  ${s}`); lastStage = s; } };
const ml = await createML(ort, load, { onProgress, chunk: +(process.env.CHUNK || 1500), patchHop: +(process.env.PHOP || 62), sessionOptions: { executionProviders: ['cpu'], intraOpNumThreads: +(process.env.THREADS || 1) } });
const r = await ml.run(x);
console.log(`ml ${Date.now() - t} ms, frames ${r.beat.length}`);
if (existsSync(refPath)) {
  const ref = JSON.parse(readFileSync(refPath, 'utf8'));
  let maxL = 0; const sig = v => 1 / (1 + Math.exp(-v));
  for (let i = 0; i < ref.beatLogits.length; i++) maxL = Math.max(maxL, Math.abs(sig(ref.beatLogits[i]) - r.beat[i]));
  console.log(`beat prob max |diff| vs python (first 60 s) ${maxL.toFixed(4)}`);
}
console.log('mood', JSON.stringify(Object.fromEntries(Object.entries(r.feel.mood || {}).map(([k, v]) => [k, +v.toFixed(3)]))), 'styles', JSON.stringify(r.feel.styles), r.feel.error || '');
