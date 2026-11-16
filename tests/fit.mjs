// node tests/fit.mjs corpus.json
// Fits the edit-power head: L2 logistic regression, "your go-to edit songs" (my-go-to) vs calm library folders,
// on a handful of interpretable features. Prints leave-one-out AUC, held-out folder means, anchors and the
// constants to paste into engine.js (EDIT_HEAD).
import { readFileSync } from 'node:fs';
const rows = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const POS = new Set(['my-go-to']);
const NEG = new Set(['lofi-chill', 'aesthetic-rnb', 'soul-jazz-funk', 'rnb-deep', 'ambient-electronic']);
const logit = p => Math.log(Math.max(1e-3, p) / Math.max(1e-3, 1 - p));
export const FEATS = {
  intensity: r => logit(r.mood.aggressive),
  edge: r => -logit(r.mood.approachability),
  drops: r => Math.min(4, r.drops),
  dropSize: r => Math.min(20, r.metrics.strongestDrop),
  density: r => r.metrics.midBusy,
  sustain: r => -r.metrics.decayDb,
  hook: r => r.scores.hook / 100,
};
const names = process.env.FEATS ? process.env.FEATS.split(',') : Object.keys(FEATS);
const X = rows.filter(r => r.mood).map(r => ({ r, x: names.map(n => FEATS[n](r)), y: POS.has(r.label) ? 1 : NEG.has(r.label) ? 0 : null }));
const train = X.filter(d => d.y !== null);
const mu = names.map((_, j) => train.reduce((s, d) => s + d.x[j], 0) / train.length);
const sd = names.map((_, j) => Math.sqrt(train.reduce((s, d) => s + (d.x[j] - mu[j]) ** 2, 0) / train.length) || 1);
const z = x => x.map((v, j) => (v - mu[j]) / sd[j]);
const sig = v => 1 / (1 + Math.exp(-v));
function fit(data, lambda = 2) {
  let w = new Array(names.length).fill(0), b = 0;
  for (let it = 0; it < 3000; it++) {
    const gw = w.map(v => lambda * v), lr = 0.05; let gb = 0;
    for (const d of data) { const zz = z(d.x), p = sig(b + zz.reduce((s, v, j) => s + v * w[j], 0)), e = p - d.y; zz.forEach((v, j) => (gw[j] += e * v)); gb += e; }
    w = w.map((v, j) => v - (lr * gw[j]) / data.length); b -= (lr * gb) / data.length;
  }
  return { w, b };
}
const pred = (m, x) => sig(m.b + z(x).reduce((s, v, j) => s + v * m.w[j], 0));
const auc = (p, n) => { let s = 0; for (const a of p) for (const c of n) s += a > c ? 1 : a === c ? 0.5 : 0; return s / (p.length * n.length); };
const lambda = +(process.argv[3] || 2);
const loo = train.map((d, i) => ({ y: d.y, p: pred(fit(train.filter((_, k) => k !== i), lambda), d.x) }));
const m = fit(train, lambda);
console.log(`lambda ${lambda}  train AUC ${auc(train.filter(d => d.y).map(d => pred(m, d.x)), train.filter(d => !d.y).map(d => pred(m, d.x))).toFixed(3)}  leave-one-out AUC ${auc(loo.filter(d => d.y).map(d => d.p), loo.filter(d => !d.y).map(d => d.p)).toFixed(3)}`);
console.log('weights (per std):', names.map((n, j) => `${n} ${m.w[j].toFixed(2)}`).join(', '), `bias ${m.b.toFixed(2)}`);
const byLabel = {};
X.forEach(d => (byLabel[d.r.label] ||= []).push(pred(m, d.x)));
console.log('mean p by folder:', Object.entries(byLabel).map(([k, v]) => `${k} ${(v.reduce((a, b) => a + b, 0) / v.length).toFixed(2)} (n${v.length})`).join(' | '));
const neg = X.filter(d => d.y === 0).map(d => pred(m, d.x));
for (const lab of ['trending-edits', 'drift-phonk', 'dark-edit-viral', 'cinematic-epic']) { const p = X.filter(d => d.r.label === lab).map(d => pred(m, d.x)); if (p.length) console.log(`held-out ${lab} vs calm AUC ${auc(p, neg).toFixed(2)}`); }
for (const n of ['windburn', 'moonlight (official']) { const d = X.find(d => d.r.name.toLowerCase().includes(n)); if (d) console.log(`anchor ${d.r.name}: p ${pred(m, d.x).toFixed(3)}`); }
console.log('\nEDIT_HEAD = ' + JSON.stringify({ names, mu: mu.map(v => +v.toFixed(4)), sd: sd.map(v => +v.toFixed(4)), w: m.w.map(v => +v.toFixed(4)), b: +m.b.toFixed(4) }));
if (process.argv[4]) X.slice().sort((a, b) => pred(m, b.x) - pred(m, a.x)).forEach(d => console.log((pred(m, d.x) * 100).toFixed(0).padStart(4), d.r.label.padEnd(20), d.r.name.slice(0, 55)));
