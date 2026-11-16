// node tests/stats.mjs corpus.json
// For every numeric feature: AUC of "your go-to edit songs" vs calm genre folders, plus the two anchor songs.
import { readFileSync } from 'node:fs';
const rows = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const POS = new Set(['my-go-to']);
const NEG = new Set(['lofi-chill', 'aesthetic-rnb', 'soul-jazz-funk', 'rnb-deep', 'ambient-electronic']);
const flat = r => ({
  ...Object.fromEntries(Object.entries(r.scores).map(([k, v]) => ['s.' + k, v])),
  ...Object.fromEntries(Object.entries(r.scoresDsp).map(([k, v]) => ['dsp.' + k, v])),
  ...Object.fromEntries(Object.entries(r.metrics).filter(([, v]) => typeof v === 'number').map(([k, v]) => ['m.' + k, v])),
  ...Object.fromEntries(Object.entries(r.mood || {}).map(([k, v]) => ['mood.' + k, v])),
  ...Object.fromEntries(Object.entries(r.win || {}).filter(([k, v]) => typeof v === 'number' && k !== 'start').map(([k, v]) => ['w.' + k, v])),
  bpm: r.bpm, drops: r.drops, kicksPerMin: r.kicks / (r.dur / 60),
});
const auc = (p, n) => { let s = 0; for (const a of p) for (const b of n) s += a > b ? 1 : a === b ? 0.5 : 0; return s / (p.length * n.length); };
const F = rows.map(r => ({ r, f: flat(r) }));
const pos = F.filter(x => POS.has(x.r.label)), neg = F.filter(x => NEG.has(x.r.label));
const keys = Object.keys(F[0].f);
const anchor = n => F.find(x => x.r.name.toLowerCase().includes(n));
const wb = anchor('windburn'), ml = anchor('moonlight (official');
console.log(`pos ${pos.length} neg ${neg.length} other ${F.length - pos.length - neg.length}`);
console.log('feature'.padEnd(22), 'AUC ', 'pos med', 'neg med', 'windburn', 'moonlight');
const med = a => { const b = a.slice().sort((x, y) => x - y); return b[b.length >> 1]; };
for (const k of keys) {
  const p = pos.map(x => x.f[k]).filter(v => v !== null && v !== undefined), n = neg.map(x => x.f[k]).filter(v => v !== null && v !== undefined);
  if (!p.length || !n.length) continue;
  console.log(k.padEnd(22), auc(p, n).toFixed(2), String(+med(p).toFixed(3)).padStart(7), String(+med(n).toFixed(3)).padStart(7), String(wb ? +(+wb.f[k]).toFixed(3) : '-').padStart(8), String(ml ? +(+ml.f[k]).toFixed(3) : '-').padStart(9));
}
if (process.argv[3]) {
  const k = process.argv[3];
  console.log('\nranked by', k);
  F.slice().sort((a, b) => b.f[k] - a.f[k]).forEach(x => console.log(String(+(+x.f[k]).toFixed(2)).padStart(7), x.r.label.padEnd(20), x.r.name.slice(0, 60)));
}
