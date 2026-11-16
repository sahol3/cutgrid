// node tests/corpus.mjs <cache-dir> <out.json> <audio files...>
// Runs the engine on each file with its cached model output (tests/ml-dump.mjs) and writes one row of
// metrics, scores and feel per song. The parent folder name is kept as the label.
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { join, basename, extname, dirname } from 'node:path';
import { analyze, TARGET_SR } from '../engine.js';

export function loadMl(cacheDir, file) {
  const p = join(cacheDir, basename(file, extname(file)) + '.ml.json');
  if (!existsSync(p)) return null;
  const j = JSON.parse(readFileSync(p, 'utf8'));
  const deq = s => Float32Array.from(Buffer.from(s, 'base64'), v => v / 255);
  return { fps: j.fps, beat: deq(j.beat), down: deq(j.down), feel: j.feel };
}

const [cacheDir, outPath, ...files] = process.argv.slice(2);
if (cacheDir) {
  const rows = [];
  for (const f of files) {
    const ml = loadMl(cacheDir, f);
    if (!ml) { console.log('no cache', basename(f)); continue; }
    try {
      const buf = execFileSync('ffmpeg', ['-v', 'error', '-i', f, '-ac', '1', '-ar', String(TARGET_SR), '-f', 'f32le', '-'], { maxBuffer: 1 << 30 });
      const x = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
      const r = analyze(x, TARGET_SR, { ml }), r0 = analyze(x, TARGET_SR);
      if (!r.ok) continue;
      rows.push({
        name: basename(f), label: basename(dirname(f)), dur: Math.round(r.duration), bpm: r.bpm, bpmDsp: r0.bpm, meter: r.meter,
        scores: r.scores, scoresDsp: r0.scores, metrics: r.metrics, mood: ml.feel && ml.feel.mood, styles: ml.feel && ml.feel.styles && ml.feel.styles.slice(0, 2),
        drops: r.drops.length, sections: r.sections.map(s => s.label + (s.pulse < 0.25 ? '~' : '')).join(' '),
        kicks: r.hits.kick.length, kicksDsp: r0.hits.kick.length, beats: r.beats.length,
        win: r.windows[15] && { start: r.windows[15].start, ...r.windows[15].feat },
      });
      console.log(`${String(r.scores.overall).padStart(3)} (dsp ${String(r0.scores.overall).padStart(3)})  ${basename(dirname(f)).padEnd(18)} ${basename(f).slice(0, 50)}`);
    } catch (e) { console.log('FAIL', basename(f), String(e.message || e).slice(0, 160)); }
  }
  writeFileSync(outPath, JSON.stringify(rows, null, 1));
}
