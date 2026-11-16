// Node test harness: node tests/run.mjs [audio files...]
// Decodes with ffmpeg to mono 22.05 kHz float, runs the engine, prints the editor summary.
// With no files it runs the synthetic click-track check (timing offset + tempo).
import { execFileSync } from 'node:child_process';
import { analyze, TARGET_SR } from '../engine.js';

function decode(path) {
  const buf = execFileSync('ffmpeg', ['-v', 'error', '-i', path, '-ac', '1', '-ar', String(TARGET_SR), '-f', 'f32le', '-'], { maxBuffer: 1 << 30 });
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

function clickTrack(bpm, secs, start) {
  const sr = TARGET_SR, x = new Float32Array(sr * secs), step = 60 / bpm, times = [];
  for (let t = start; t < secs - 0.1; t += step) {
    times.push(t);
    const i0 = Math.round(t * sr);
    for (let i = 0; i < 0.03 * sr; i++) x[i0 + i] += Math.exp(-i / (0.004 * sr)) * Math.sin(2 * Math.PI * 1500 * i / sr) * 0.8 + Math.exp(-i / (0.02 * sr)) * Math.sin(2 * Math.PI * 60 * i / sr) * 0.8;
  }
  return { x, times };
}

const files = process.argv.slice(2);
if (!files.length) {
  for (const bpm of [90, 128, 150]) {
    const { x, times } = clickTrack(bpm, 30, 0.5);
    const t = Date.now(), r = analyze(x, TARGET_SR), ms = Date.now() - t;
    const errs = r.beats.map(b => { let m = 1e9; for (const tt of times) if (Math.abs(b - tt) < Math.abs(m)) m = b - tt; return m; });
    const meanErr = errs.reduce((a, b) => a + b, 0) / errs.length;
    console.log(`click ${bpm}: got ${r.bpm} BPM, ${r.beats.length}/${times.length} beats, mean offset ${(meanErr * 1000).toFixed(1)} ms, max |err| ${(Math.max(...errs.map(Math.abs)) * 1000).toFixed(1)} ms, ${ms} ms`);
  }
} else {
  for (const f of files) {
    const x = decode(f), t = Date.now(), r = analyze(x, TARGET_SR), ms = Date.now() - t;
    const name = f.split(/[\\/]/).pop();
    if (!r.ok) { console.log(name, r.reason); continue; }
    console.log(`\n== ${name}  (${r.duration.toFixed(0)} s, ${ms} ms)`);
    console.log(`BPM ${r.bpm} alts ${r.bpmAlts} conf ${r.tempoConf} drift ${r.tempoDrift} | beats ${r.beats.length} first ${r.metrics.firstBeat}s | grid ${r.grid.name}`);
    console.log('scores', JSON.stringify(r.scores), 'fits', JSON.stringify(r.fits));
    console.log('metrics', JSON.stringify(r.metrics));
    console.log('hits', Object.fromEntries(Object.entries(r.hits).map(([k, v]) => [k, v.length])), 'drops', JSON.stringify(r.drops.map(d => [d.t, d.jumpDb, d.lowJumpDb])), 'gaps', r.gaps.length);
    console.log('sections', r.sections.map(s => `${s.label}@${s.start}`).join(' '));
    console.log('windows', JSON.stringify(Object.fromEntries(Object.entries(r.windows).map(([k, w]) => [k, w && [w.start, w.end, w.bars, w.dropAt]]))));
  }
}
