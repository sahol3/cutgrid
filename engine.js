// CutGrid analysis engine.
// Pure JavaScript with no dependencies. Runs in a browser, a Web Worker or Node.
// Input: mono Float32Array + sample rate. Output: a plain object an editor can read.

export const TARGET_SR = 22050;
// fitted by tests/fit.mjs on the labelled corpus (lambda 2); features are standardised with mu and sd
const EDIT_HEAD = { names: ['intensity', 'edge', 'drops', 'dropSize', 'density', 'sustain', 'hook'], mu: [-3.3446, 0.1228, 2.2048, 12.2651, 0.7457, -2.0217, 0.892], sd: [1.5499, 0.8373, 1.4036, 7.152, 0.3435, 1.4182, 0.0884], w: [1.0268, 0.5149, 0.4707, 0.0014, 0.4614, 0.26, 0.3672], b: 0.0648 };
const EDIT_HEAD_DSP = { names: ['drops', 'dropSize', 'density', 'sustain', 'hook'], mu: [2.2048, 12.2651, 0.7457, -2.0217, 0.892], sd: [1.4036, 7.152, 0.3435, 1.4182, 0.0884], w: [0.4768, -0.1832, 0.4773, 0.4341, 0.2981], b: 0.0891 };
const N = 1024;          // FFT size (46 ms at 22.05 kHz)
const HOP = 256;         // hop (11.6 ms) -> about 86 frames per second
const ONSET_SHIFT = 0.013; // extra seconds added so markers land on the transient (measured on a click track)

// ---------- basic helpers ----------
const clamp01 = v => (v < 0 ? 0 : v > 1 ? 1 : v);
const lerp01 = (v, lo, hi) => clamp01((v - lo) / (hi - lo));
function mean(a, s = 0, e = a.length) { let t = 0; for (let i = s; i < e; i++) t += a[i]; return e > s ? t / (e - s) : 0; }
function std(a, m = mean(a)) { let t = 0; for (let i = 0; i < a.length; i++) t += (a[i] - m) ** 2; return Math.sqrt(t / Math.max(1, a.length)); }
function median(arr) { if (!arr.length) return 0; const b = Array.from(arr).sort((x, y) => x - y); const m = b.length >> 1; return b.length % 2 ? b[m] : (b[m - 1] + b[m]) / 2; }
function percentile(arr, p) { if (!arr.length) return 0; const b = Array.from(arr).sort((x, y) => x - y); const i = Math.min(b.length - 1, Math.max(0, Math.round((p / 100) * (b.length - 1)))); return b[i]; }
function movingMean(a, w) {
  const n = a.length, out = new Float32Array(n), ps = new Float64Array(n + 1), h = w >> 1;
  for (let i = 0; i < n; i++) ps[i + 1] = ps[i] + a[i];
  for (let i = 0; i < n; i++) { const s = Math.max(0, i - h), e = Math.min(n, i - h + w); out[i] = (ps[e] - ps[s]) / Math.max(1, e - s); }
  return out;
}
function localMax(a, i, r) { let m = -Infinity; for (let j = Math.max(0, i - r); j <= Math.min(a.length - 1, i + r); j++) if (a[j] > m) m = a[j]; return m; }

export function mixToMono(channels) {
  if (channels.length === 1) return channels[0];
  const n = channels[0].length, out = new Float32Array(n), k = 1 / channels.length;
  for (const c of channels) for (let i = 0; i < n; i++) out[i] += c[i] * k;
  return out;
}

export function resample(x, srIn, srOut = TARGET_SR) {
  if (Math.abs(srIn - srOut) < 1) return x;
  const ratio = srIn / srOut;
  let y = x;
  if (ratio > 1) { const w = Math.max(1, Math.round(ratio)); y = movingMean(movingMean(x, w), w); }
  const nOut = Math.floor(x.length / ratio), out = new Float32Array(nOut);
  for (let i = 0; i < nOut; i++) {
    const p = i * ratio, j = p | 0, f = p - j, a = y[j], b = j + 1 < y.length ? y[j + 1] : a;
    out[i] = a + (b - a) * f;
  }
  return out;
}

function makeFFT(n) {
  const levels = Math.log2(n), cos = new Float64Array(n / 2), sin = new Float64Array(n / 2), rev = new Uint32Array(n);
  for (let i = 0; i < n / 2; i++) { cos[i] = Math.cos((2 * Math.PI * i) / n); sin[i] = Math.sin((2 * Math.PI * i) / n); }
  for (let i = 0; i < n; i++) { let r = 0; for (let b = 0; b < levels; b++) r = (r << 1) | ((i >>> b) & 1); rev[i] = r; }
  return (re, im) => {
    for (let i = 0; i < n; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = (n / size) | 0;
      for (let i = 0; i < n; i += size) {
        for (let j = i, k = 0; j < i + half; j++, k += step) {
          const l = j + half, tre = re[l] * cos[k] + im[l] * sin[k], tim = -re[l] * sin[k] + im[l] * cos[k];
          re[l] = re[j] - tre; im[l] = im[j] - tim; re[j] += tre; im[j] += tim;
        }
      }
    }
  };
}

// ---------- 1. spectral features ----------
function features(x, sr) {
  const nF = x.length >= N ? Math.floor((x.length - N) / HOP) + 1 : 0;
  const fft = makeFFT(N), win = new Float64Array(N), re = new Float64Array(N), im = new Float64Array(N);
  for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N);
  const nb = N / 2 + 1, binHz = sr / N;
  const bin = hz => Math.min(nb - 1, Math.max(1, Math.round(hz / binHz)));
  const LOW = [bin(30), bin(150)], MID = [bin(150), bin(2500)], HIGH = [bin(5000), bin(10500)];
  const TB = 12, edges = [];
  for (let i = 0; i <= TB; i++) edges.push(bin(40 * Math.pow(10500 / 40, i / TB)));
  const flux = new Float32Array(nF), fLow = new Float32Array(nF), fMid = new Float32Array(nF), fHigh = new Float32Array(nF);
  const rms = new Float32Array(nF), eLow = new Float32Array(nF), eHigh = new Float32Array(nF), peak = new Float32Array(nF);
  const timbre = new Float32Array(nF * TB);
  let prev = new Float32Array(nb), cur = new Float32Array(nb);
  const top = HIGH[1], ampA = new Float64Array(nb), ampB = new Float64Array(nb), sc = 1 / (N / 4);
  const frameStats = t => { const off = t * HOP; let s2 = 0, pk = 0; for (let i = 0; i < N; i++) { const v = x[off + i]; s2 += v * v; } for (let i = 0; i < HOP; i++) { const v = Math.abs(x[off + i]); if (v > pk) pk = v; } rms[t] = Math.sqrt(s2 / N); peak[t] = pk; };
  for (let t = 0; t < nF; t++) {
    // even frame: pack frames t and t+1 into one complex FFT, then split the two spectra
    if ((t & 1) === 0) {
      const offA = t * HOP, offB = (t + 1) * HOP, hasB = t + 1 < nF;
      for (let i = 0; i < N; i++) { re[i] = x[offA + i] * win[i]; im[i] = hasB ? x[offB + i] * win[i] : 0; }
      fft(re, im);
      for (let k = 0; k < nb; k++) {
        const nk = (N - k) % N;
        const ar = 0.5 * (re[k] + re[nk]), ai = 0.5 * (im[k] - im[nk]);
        const br = 0.5 * (im[k] + im[nk]), bi = -0.5 * (re[k] - re[nk]);
        ampA[k] = Math.sqrt(ar * ar + ai * ai) * sc; ampB[k] = Math.sqrt(br * br + bi * bi) * sc;
      }
    }
    const amps = (t & 1) === 0 ? ampA : ampB;
    frameStats(t);
    let el = 0, eh = 0;
    for (let k = 0; k < nb; k++) {
      const amp = amps[k];
      cur[k] = Math.log1p(1000 * amp);
      if (k >= LOW[0] && k <= LOW[1]) el += amp * amp;
      if (k >= HIGH[0] && k <= HIGH[1]) eh += amp * amp;
    }
    eLow[t] = el / (LOW[1] - LOW[0] + 1); eHigh[t] = eh / (HIGH[1] - HIGH[0] + 1);
    if (t > 0) {
      let sAll = 0, sL = 0, sM = 0, sH = 0;
      for (let k = 1; k <= top; k++) {
        const pm = Math.max(prev[k - 1], prev[k], prev[k + 1] ?? prev[k]); // SuperFlux style max filter
        const d = cur[k] - pm;
        if (d > 0) { sAll += d; if (k >= LOW[0] && k <= LOW[1]) sL += d; else if (k >= MID[0] && k <= MID[1]) sM += d; else if (k >= HIGH[0]) sH += d; }
      }
      flux[t] = sAll / top; fLow[t] = sL / (LOW[1] - LOW[0] + 1); fMid[t] = sM / (MID[1] - MID[0] + 1); fHigh[t] = sH / (HIGH[1] - HIGH[0] + 1);
    }
    for (let b = 0; b < TB; b++) { let s = 0; const a = edges[b], e = Math.max(edges[b] + 1, edges[b + 1]); for (let k = a; k < e; k++) s += cur[k]; timbre[t * TB + b] = s / (e - a); }
    const tmp = prev; prev = cur; cur = tmp;
  }
  return { nF, fps: sr / HOP, flux, fLow, fMid, fHigh, rms, eLow, eHigh, peak, timbre, TB };
}

function zNorm(a) {
  const m = mean(a), s = std(a, m) || 1, out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = (a[i] - m) / s + m / s; // scale by std, keep it positive
  return out;
}

// ---------- 2. tempo ----------
function estimateTempo(env, fps) {
  const minBpm = 55, maxBpm = 215;
  const minLag = Math.floor((fps * 60) / maxBpm), maxLag = Math.ceil((fps * 60) / minBpm), L = 2 * maxLag + 2;
  const n = env.length, m = mean(env), e = new Float32Array(n);
  for (let i = 0; i < n; i++) e[i] = env[i] - m;
  const ac = new Float64Array(L + 1);
  for (let lag = 0; lag <= L; lag++) { let s = 0; for (let t = 0; t + lag < n; t++) s += e[t] * e[t + lag]; ac[lag] = s / Math.max(1, n - lag); }
  const a0 = ac[0] || 1;
  const acAt = l => { const i = Math.floor(l), f = l - i; return i + 1 <= L ? (ac[i] * (1 - f) + ac[i + 1] * f) / a0 : 0; };
  const score = new Float64Array(maxLag + 2);
  for (let lag = minLag; lag <= maxLag; lag++) {
    const bpm = (60 * fps) / lag, w = Math.exp(-0.5 * (Math.log2(bpm / 120) / 0.9) ** 2);
    const s = acAt(lag) + 0.5 * acAt(2 * lag);
    score[lag] = Math.max(0, s) * w;
  }
  let best = minLag;
  for (let lag = minLag; lag <= maxLag; lag++) if (score[lag] > score[best]) best = lag;
  let lagF = best;
  if (best > minLag && best < maxLag) { const a = score[best - 1], b = score[best], c = score[best + 1], d = a - 2 * b + c; if (d < 0) lagF = best + (0.5 * (a - c)) / d; }
  let pos = 0, cnt = 0; for (let lag = minLag; lag <= maxLag; lag++) { pos += score[lag]; cnt++; }
  const conf = score[best] / (pos / cnt || 1);
  const peaks = [];
  for (let lag = minLag + 1; lag < maxLag; lag++) if (score[lag] > score[lag - 1] && score[lag] >= score[lag + 1]) peaks.push([lag, score[lag]]);
  peaks.sort((p, q) => q[1] - p[1]);
  return { bpm: (60 * fps) / lagF, conf, peakAc: acAt(lagF), alts: peaks.slice(0, 4).map(p => +((60 * fps) / p[0]).toFixed(1)) };
}

// ---------- 3. beat tracking (dynamic programming, after Ellis 2007) ----------
function trackBeats(env, period, tightness = 100) {
  const n = env.length; if (n < period * 3) return [];
  const P = Math.round(period), local = new Float32Array(n), wnd = [];
  for (let t = -P; t <= P; t++) wnd.push(Math.exp(-0.5 * ((t * 32) / period) ** 2));
  for (let i = 0; i < n; i++) { let s = 0; for (let k = 0; k < wnd.length; k++) { const j = i + k - P; if (j >= 0 && j < n) s += env[j] * wnd[k]; } local[i] = s; }
  const lo = Math.round(-2 * period), hi = Math.min(-1, Math.round(-period / 2)), tx = [];
  for (let d = lo; d <= hi; d++) tx.push(-tightness * Math.log(-d / period) ** 2);
  const cum = new Float32Array(n), back = new Int32Array(n).fill(-1);
  const startThr = 0.01 * Math.max(...local);
  let started = false;
  for (let i = 0; i < n; i++) {
    let best = -Infinity, arg = -1;
    for (let d = lo; d <= hi; d++) { const j = i + d; if (j < 0) continue; const s = cum[j] + tx[d - lo]; if (s > best) { best = s; arg = j; } }
    if (!started && local[i] > startThr) started = true;
    if (arg >= 0 && started) { cum[i] = local[i] + best; back[i] = arg; } else cum[i] = local[i];
  }
  const maxima = [];
  for (let i = 1; i < n - 1; i++) if (cum[i] > cum[i - 1] && cum[i] >= cum[i + 1]) maxima.push(i);
  if (!maxima.length) return [];
  const med = median(maxima.map(i => cum[i]));
  let last = maxima[maxima.length - 1];
  for (let k = maxima.length - 1; k >= 0; k--) if (cum[maxima[k]] >= 0.5 * med) { last = maxima[k]; break; }
  const beats = []; for (let i = last; i >= 0; i = back[i]) beats.push(i);
  beats.reverse();
  // trim weak beats at both ends (silence, fades)
  const vals = beats.map(b => local[b]), sm = vals.map((_, i) => { let s = 0, c = 0; for (let k = -2; k <= 2; k++) if (vals[i + k] !== undefined) { s += vals[i + k]; c++; } return s / c; });
  const thr = 0.5 * Math.sqrt(mean(sm.map(v => v * v)));
  let a = 0, b = beats.length - 1;
  while (a < b && sm[a] < thr) a++;
  while (b > a && sm[b] < thr) b--;
  return beats.slice(a, b + 1);
}

// ---------- 4. peak picking for hits ----------
function pickPeaks(sig, fps, { win = 0.5, delta = 1, minGap = 0.07, radius = 0.03 }) {
  const avg = movingMean(sig, Math.max(3, Math.round(win * fps))), r = Math.max(1, Math.round(radius * fps)), gap = Math.round(minGap * fps), out = [];
  let lastI = -1e9;
  for (let i = 1; i < sig.length - 1; i++) {
    const v = sig[i];
    if (v < avg[i] + delta) continue;
    if (v < localMax(sig, i, r)) continue;
    if (i - lastI < gap) { if (out.length && v > out[out.length - 1][1]) out[out.length - 1] = [i, v]; continue; }
    out.push([i, v]); lastI = i;
  }
  return out;
}

// ---------- main ----------
export function analyze(samples, sr, opts = {}) {
  const x = sr === TARGET_SR ? samples : resample(samples, sr, TARGET_SR);
  const SR = TARGET_SR, duration = x.length / SR;
  const F = features(x, SR), fps = F.fps, nF = F.nF;
  const t0 = N / 2 / SR + ONSET_SHIFT;
  const fT = f => Math.max(0, f / fps + t0);   // frame -> seconds
  const tF = t => Math.max(0, Math.min(nF - 1, Math.round((t - t0) * fps)));
  if (nF < fps * 4) return { ok: false, reason: 'Need at least 4 seconds of audio.', duration };

  // loudness curves
  const pow = new Float32Array(nF); for (let i = 0; i < nF; i++) pow[i] = F.rms[i] * F.rms[i];
  const db = a => { const o = new Float32Array(a.length); for (let i = 0; i < a.length; i++) o[i] = 10 * Math.log10(a[i] + 1e-12); return o; };
  const loud = db(movingMean(pow, Math.round(0.4 * fps)));
  const loudFast = db(movingMean(pow, Math.round(0.1 * fps)));
  const loud3 = db(movingMean(pow, Math.round(3 * fps)));
  const lowDb = db(movingMean(F.eLow, Math.round(0.4 * fps)));
  const audible = Array.from(loud).filter(v => v > -60);
  const lp2 = percentile(audible, 2), lp99 = percentile(audible, 99), loudMed = median(audible);
  const loud1 = db(movingMean(pow, Math.round(1.0 * fps)));
  const a1 = Array.from(loud1).filter(v => v > -60), l1lo = percentile(a1, 2), l1hi = percentile(a1, 99);
  const energy = new Float32Array(nF); for (let i = 0; i < nF; i++) energy[i] = clamp01((loud1[i] - l1lo) / (l1hi - l1lo || 1));

  // onset envelope
  const env = zNorm(F.flux);
  const envD = new Float32Array(nF); { const mm = movingMean(env, Math.round(0.5 * fps)); for (let i = 0; i < nF; i++) envD[i] = Math.max(0, env[i] - mm[i]); }
  const envStd = std(env), envMean = mean(env);

  // neural beat activation (Beat This!), resampled onto the engine frames. Without it the DSP onset curve is used.
  const ml = opts.ml && opts.ml.beat && opts.ml.beat.length ? opts.ml : null;
  let act = null, dact = null;
  if (ml) {
    act = new Float32Array(nF); dact = new Float32Array(nF);
    const m = ml.beat.length;
    for (let i = 0; i < nF; i++) {
      const p = (fT(i) - ONSET_SHIFT) * ml.fps, j = Math.floor(p), f = p - j;
      if (j >= 0 && j + 1 < m) { act[i] = ml.beat[j] * (1 - f) + ml.beat[j + 1] * f; dact[i] = ml.down[j] * (1 - f) + ml.down[j + 1] * f; }
    }
  }
  const confAt = f => { if (!act) return 1; let v = 0; for (let i = Math.max(0, f - 4); i <= Math.min(nF - 1, f + 4); i++) if (act[i] > v) v = act[i]; return v; };

  // tempo + beats
  const tEnv = act ? (() => { const o = new Float32Array(nF); for (let i = 0; i < nF; i++) o[i] = act[i] + 0.15 * clamp01(env[i] / (envMean + 4 * envStd)); return o; })() : env;
  const tempo = estimateTempo(movingMean(movingMean(tEnv, 3), 3), fps); // smoothed so sharp clicks still correlate at fractional lags
  let beatsF = trackBeats(tEnv, (60 * fps) / tempo.bpm);
  let bpm = tempo.bpm;
  if (beatsF.length >= 8) {
    // refine tempo by regression of beat frames on index (middle 80%)
    const a = Math.floor(beatsF.length * 0.1), b = Math.ceil(beatsF.length * 0.9);
    let sx = 0, sy = 0, sxx = 0, sxy = 0, c = 0;
    for (let i = a; i < b; i++) { sx += i; sy += beatsF[i]; sxx += i * i; sxy += i * beatsF[i]; c++; }
    const slope = (c * sxy - sx * sy) / (c * sxx - sx * sx);
    if (slope > 0) bpm = (60 * fps) / slope;
  }
  const beatDur = 60 / bpm;
  let barDur = beatDur * 4;
  const detected = beatsF.length ? [fT(beatsF[0]), fT(beatsF[beatsF.length - 1])] : [0, 0];
  // extend the grid at the same period through beatless intros and outros (drawn dimmer in the UI)
  let prepend = 0;
  if (beatsF.length >= 8) {
    const pf = (60 * fps) / bpm, head = [], tail = [];
    for (let f = beatsF[0] - pf; f >= 0; f -= pf) head.unshift(Math.round(f));
    for (let f = beatsF[beatsF.length - 1] + pf; f < nF - 1; f += pf) tail.push(Math.round(f));
    prepend = head.length; beatsF = [...head, ...beatsF, ...tail];
  }
  const beats = beatsF.map(fT);
  // how sure the network is that each grid line is a felt beat (1 without the model)
  const beatConf = beatsF.map(f => +confAt(f).toFixed(2));

  // beat strength + pulse clarity. With the model, only beats it can hear count; calm stretches no longer pass as pulse.
  let coreF = beatsF.filter(f => fT(f) >= detected[0] - 0.01 && fT(f) <= detected[1] + 0.01);
  if (act) { const sure = beatsF.filter((_, k) => beatConf[k] >= 0.35); if (sure.length >= 8) coreF = sure; }
  const atBeat = coreF.map(f => localMax(env, f, 2));
  const hitThr = envMean + 0.5 * envStd;
  const hitRatio = atBeat.length ? atBeat.filter(v => v > hitThr).length / atBeat.length : 0;
  const beatContrast = atBeat.length ? mean(atBeat) / (envMean || 1) : 1;
  const lowN = zNorm(F.fLow), midN = zNorm(F.fMid), highN = zNorm(F.fHigh);
  const lowAtBeat = coreF.map(f => localMax(lowN, f, 2));
  const lowContrast = lowAtBeat.length ? mean(lowAtBeat) / (mean(lowN) || 1) : 1;
  // absolute hit strength: how far the spectrum jumps at each beat (log-amplitude rise per bin), and how deep the level dips after the hit
  const rawRise = median(coreF.map(f => localMax(F.flux, f, 2)));
  const rawLowRise = median(coreF.map(f => localMax(F.fLow, f, 2)));
  const decayDb = median(coreF.map(f => {
    let pk = 0; for (let i = Math.max(0, f - 2); i <= Math.min(nF - 1, f + 3); i++) pk = Math.max(pk, F.rms[i]);
    let s = 0, c = 0; for (let i = f + Math.round(0.06 * fps); i <= Math.min(nF - 1, f + Math.round(0.15 * fps)); i++) { s += F.rms[i] ** 2; c++; }
    return c ? 20 * Math.log10((pk + 1e-9) / (Math.sqrt(s / c) + 1e-9)) : 0;
  }));

  // downbeats (bar starts). With the model: Viterbi over bar position on its downbeat probability, meter 3 or 4,
  // so a dropped or extra beat does not shift every later bar line.
  let phase = 0, meter = 4, beatPos = null;
  if (act && beatsF.length >= 8) {
    const dp = beatsF.map(f => { let v = 0; for (let i = Math.max(0, f - 4); i <= Math.min(nF - 1, f + 4); i++) if (dact[i] > v) v = dact[i]; return v; });
    const decode = S => {
      const n = dp.length, stay = Math.log(0.998), jump = Math.log(0.002 / (S - 1));
      const em = (k, s) => (s === 0 ? 2 * Math.log(dp[k] + 0.02) : Math.log(1.02 - dp[k]));
      let sc = Array.from({ length: S }, (_, s) => em(0, s)); const back = [];
      for (let k = 1; k < n; k++) {
        const nx = new Array(S), bk = new Array(S);
        for (let s = 0; s < S; s++) { let best = -Infinity, arg = 0; for (let p = 0; p < S; p++) { const v = sc[p] + (s === (p + 1) % S ? stay : jump); if (v > best) { best = v; arg = p; } } nx[s] = best + em(k, s); bk[s] = arg; }
        sc = nx; back.push(bk);
      }
      let s = sc.indexOf(Math.max(...sc)); const pos = new Array(n); pos[n - 1] = s;
      for (let k = n - 2; k >= 0; k--) { s = back[k][s]; pos[k] = s; }
      const on = [], off = []; pos.forEach((q, k) => (q === 0 ? on : off).push(dp[k]));
      return { pos, contrast: mean(on) - mean(off) };
    };
    const d4 = decode(4), d3 = decode(3), use3 = d3.contrast > d4.contrast + 0.12;
    meter = use3 ? 3 : 4; beatPos = (use3 ? d3 : d4).pos;
    phase = beatPos.findIndex(q => q === 0); if (phase < 0) phase = 0;
    barDur = beatDur * meter;
  } else if (beatsF.length >= 8) {
    const L = [0, 0, 0, 0], M = [0, 0, 0, 0], J = [0, 0, 0, 0], C = [0, 0, 0, 0];
    beatsF.forEach((f, i) => {
      const q = i % 4; L[q] += localMax(lowN, f, 2); M[q] += localMax(midN, f, 2) + localMax(highN, f, 2); C[q]++;
      const a = Math.max(0, f - Math.round(fps * beatDur)), b = Math.min(nF - 1, f + Math.round(fps * beatDur));
      J[q] += Math.abs(loud[b] - loud[a]);
    });
    for (let q = 0; q < 4; q++) { L[q] /= C[q] || 1; M[q] /= C[q] || 1; J[q] /= C[q] || 1; }
    let bestS = -Infinity;
    for (let p0 = 0; p0 < 4; p0++) { const p = p0;
      const s = (L[p] + L[(p + 2) % 4]) - (L[(p + 1) % 4] + L[(p + 3) % 4]) + (M[(p + 1) % 4] + M[(p + 3) % 4]) - (M[p] + M[(p + 2) % 4]) + 0.5 * (L[p] - L[(p + 2) % 4]) + 0.3 * (J[p] - mean(J));
      if (s > bestS) { bestS = s; phase = p; }
    }
  }
  void prepend;
  if (!beatPos) beatPos = beatsF.map((_, i) => (((i - phase) % 4) + 4) % 4);
  const barNo = []; { let b = 0; beatPos.forEach((q, i) => { if (q === 0 && i > 0) b++; barNo.push(b + (beatPos[0] === 0 ? 1 : 0)); }); }
  const downF = beatsF.filter((_, i) => beatPos[i] === 0), downbeats = downF.map(fT);

  // hits per band
  const loM = mean(lowN), loS = std(lowN, loM);
  const kickPk = pickPeaks(lowN, fps, { delta: 1.2, minGap: 0.1 }).filter(([, v]) => v > loM + 0.75 * loS);
  const mh = new Float32Array(nF); for (let i = 0; i < nF; i++) mh[i] = 0.5 * (midN[i] + highN[i]);
  const hiM = mean(highN), hiS = std(highN, hiM), miM = mean(midN), miS = std(midN, miM);
  const snarePk = pickPeaks(mh, fps, { delta: 1.1, minGap: 0.1 }).filter(([i]) => localMax(highN, i, 2) > hiM + 0.8 * hiS && localMax(midN, i, 2) > miM + 0.5 * miS);
  const hatPk = pickPeaks(highN, fps, { delta: 0.8, minGap: 0.05 }).filter(([i]) => localMax(midN, i, 1) < miM + 1.0 * miS);
  const envP95 = percentile(Array.from(env), 95);
  const accentPk = pickPeaks(env, fps, { delta: 3 * envStd, win: 2, minGap: 0.25 }).filter(([, v]) => v > envP95)
    .sort((a, b) => b[1] - a[1]).slice(0, Math.max(4, Math.ceil(duration / 3))).sort((a, b) => a[0] - b[0]);
  const allOnsets = pickPeaks(env, fps, { delta: 0.6 * envStd, minGap: 0.05 });
  // where the network hears a pulse (0..1, smoothed over two bars). Calm stretches drop all but their biggest hits.
  let activity = null;
  if (act) {
    const r = Math.max(1, Math.round(0.6 * beatDur * fps)), mx = new Float32Array(nF);
    for (let i = 0; i < nF; i++) mx[i] = localMax(act, i, r);
    activity = movingMean(mx, Math.round(2 * barDur * fps));
  }
  const gate = pk => { if (!activity || !pk.length) return pk; const hi = percentile(pk.map(p => p[1]), 90); return pk.filter(([i, v]) => activity[i] >= 0.2 || v >= hi); };
  const toHits = pk => pk.map(([i, v]) => ({ t: +fT(i).toFixed(3), s: +v.toFixed(2) }));
  const hits = { kick: toHits(gate(kickPk)), snare: toHits(gate(snarePk)), hat: toHits(gate(hatPk)), accent: toHits(accentPk) };

  // rhythm grid: inter-onset intervals in beats
  const ioiHist = new Array(49).fill(0); // 0..2 beats in 1/24 steps
  for (let k = 1; k < allOnsets.length; k++) {
    const d = (allOnsets[k][0] - allOnsets[k - 1][0]) / fps / beatDur;
    if (d > 0.05 && d <= 2) ioiHist[Math.round(d * 24)]++;
  }
  const ioiTotal = ioiHist.reduce((a, b) => a + b, 0) || 1;
  const near = (c, w = 1) => { let s = 0; for (let k = Math.round(c * 24) - w; k <= Math.round(c * 24) + w; k++) s += ioiHist[k] || 0; return s / ioiTotal; };
  const grid = { sixteenth: near(0.25), eighth: near(0.5), triplet: near(1 / 3) + near(2 / 3), quarter: near(1), half: near(2, 1) };
  const gridName = grid.triplet > Math.max(grid.sixteenth, grid.eighth) * 0.9 ? 'triplet / swing feel' : grid.sixteenth > grid.eighth ? 'straight 16ths (busy)' : grid.eighth > grid.quarter ? 'straight 8ths' : 'quarter-note pulse (sparse)';

  // tempo map from beats
  const tempoMap = [];
  for (let i = 4; i < beats.length - 4; i += 2) {
    const ibis = []; for (let k = i - 4; k < i + 4; k++) ibis.push(beats[k + 1] - beats[k]);
    tempoMap.push({ t: +beats[i].toFixed(2), bpm: +(60 / median(ibis)).toFixed(1) });
  }
  const tm = tempoMap.map(p => p.bpm), tempoDrift = tm.length ? (percentile(tm, 95) - percentile(tm, 5)) : 0;

  // bars for structure (fallback: 2 s blocks)
  let barStarts = downF.slice();
  if (barStarts.length < 6) { barStarts = []; for (let f = 0; f < nF; f += Math.round(2 * fps)) barStarts.push(f); }
  if (barStarts[0] > fps * 0.5) barStarts.unshift(0);
  const nBars = barStarts.length;
  const barEnd = b => (b + 1 < nBars ? barStarts[b + 1] : nF);
  const barLoud = [], barLow = [], barHigh = [], barT = [];
  for (let b = 0; b < nBars; b++) {
    const s = barStarts[b], e = Math.max(s + 1, barEnd(b));
    let p = 0, l = 0, h = 0; for (let i = s; i < e; i++) { p += pow[i]; l += F.eLow[i]; h += F.fHigh[i]; }
    barLoud.push(10 * Math.log10(p / (e - s) + 1e-12)); barLow.push(10 * Math.log10(l / (e - s) + 1e-12)); barHigh.push(h / (e - s));
    const v = new Float32Array(F.TB); for (let i = s; i < e; i++) for (let k = 0; k < F.TB; k++) v[k] += F.timbre[i * F.TB + k] / (e - s);
    barT.push(v);
  }
  // section novelty (checkerboard over a cosine self-similarity of z-scored timbre + loudness)
  const D = F.TB + 1, feat = barT.map((v, b) => [...v, barLoud[b] / 6]);
  for (let k = 0; k < D; k++) { const col = feat.map(r => r[k]), m = mean(col), s = std(col, m) || 1; feat.forEach(r => (r[k] = (r[k] - m) / s)); }
  const cos = (a, b) => { let s = 0, na = 0, nb2 = 0; for (let k = 0; k < D; k++) { s += a[k] * b[k]; na += a[k] * a[k]; nb2 += b[k] * b[k]; } return s / (Math.sqrt(na * nb2) || 1); };
  const K = nBars >= 24 ? 4 : 2, nov = new Float32Array(nBars);
  for (let b = K; b <= nBars - K; b++) {
    let s = 0;
    for (let i = -K; i < K; i++) for (let j = -K; j < K; j++) {
      const sign = (i < 0) === (j < 0) ? 1 : -1, g = Math.exp(-((i + 0.5) ** 2 + (j + 0.5) ** 2) / (2 * (K / 1.5) ** 2));
      s += sign * g * cos(feat[b + i], feat[b + j]);
    }
    nov[b] = Math.max(0, s);
  }
  const novM = mean(nov), novS = std(nov, novM), minSec = nBars >= 48 ? 8 : nBars >= 24 ? 4 : 2, bounds = [0];
  for (let b = 1; b < nBars; b++) {
    if (nov[b] < novM + 0.7 * novS) continue;
    let isMax = true; for (let k = Math.max(0, b - 2); k <= Math.min(nBars - 1, b + 2); k++) if (nov[k] > nov[b]) isMax = false;
    if (isMax && b - bounds[bounds.length - 1] >= minSec && nBars - b >= 2) bounds.push(b);
  }
  const secRaw = bounds.map((b, i) => {
    const e = i + 1 < bounds.length ? bounds[i + 1] : nBars;
    return { b0: b, b1: e, start: fT(barStarts[b]), end: e < nBars ? fT(barStarts[e]) : duration, loud: mean(barLoud.slice(b, e)), low: mean(barLow.slice(b, e)) };
  });
  const eMax = Math.max(...secRaw.map(s => s.loud)), eMin = Math.min(...secRaw.map(s => s.loud));
  const lvl = s => (eMax - s.loud <= 2.5 ? 'Peak' : eMax - s.loud <= 6 ? 'Mid' : 'Low');
  const labelled = secRaw.map((s, i) => {
    let label = lvl(s);
    const next = secRaw[i + 1], prevS = secRaw[i - 1];
    if (i === 0 && label !== 'Peak') label = 'Intro';
    else if (i === secRaw.length - 1 && label !== 'Peak' && secRaw.length > 2) label = 'Outro';
    else if (label !== 'Peak' && next && lvl(next) === 'Peak') label = 'Build';
    else if (label === 'Low' && prevS && lvl(prevS) === 'Peak') label = 'Breakdown';
    return { start: s.start, end: s.end, label, loud: s.loud };
  });
  const sections = [];
  for (const s of labelled) {
    const last = sections[sections.length - 1];
    if (last && last.label === s.label) { last.end = +s.end.toFixed(2); last.loudDb = +Math.max(last.loudDb, s.loud).toFixed(1); }
    else sections.push({ start: +s.start.toFixed(2), end: +s.end.toFixed(2), label: s.label, level: +clamp01(1 - (eMax - s.loud) / 12).toFixed(2), loudDb: +s.loud.toFixed(1) });
  }

  // drops: loudness and low-end jumps at bar starts
  const dropsRaw = [];
  for (let b = 2; b < nBars - 1; b++) {
    const before = (barLoud[b - 2] + barLoud[b - 1]) / 2, after = (barLoud[b] + barLoud[Math.min(nBars - 1, b + 1)]) / 2;
    const lowB = (barLow[b - 2] + barLow[b - 1]) / 2, lowA = (barLow[b] + barLow[Math.min(nBars - 1, b + 1)]) / 2;
    const jump = after - before, lowJump = lowA - lowB;
    if (jump >= 3 && lowJump >= 2.5 && (lowJump >= 4 || jump >= 5) && after > loudMed - 3) dropsRaw.push({ b, jump, lowJump });
  }
  const drops = dropsRaw.filter(d => !dropsRaw.some(o => o !== d && Math.abs(o.b - d.b) <= 3 && o.jump + 0.5 * o.lowJump > d.jump + 0.5 * d.lowJump)).map(d => {
    let f = barStarts[d.b];
    { const lo = f - Math.round(0.5 * beatDur * fps), hi = f + Math.round(barDur * 1.5 * fps), quiet = Math.round(1.5 * beatDur * fps);
      const ks = kickPk.filter(([i]) => i >= lo && i <= hi), kmax = Math.max(0, ...ks.map(k => k[1]));
      const strong = kickPk.filter(([, v]) => v >= 0.5 * kmax);
      const fresh = strong.find(([i]) => i >= lo && i <= hi && !strong.some(([j]) => j < i && j >= i - quiet));
      if (fresh) { f = fresh[0]; const nb = beatsF.reduce((m, b) => (Math.abs(b - f) < Math.abs(m - f) ? b : m), beatsF[0] ?? f); if (Math.abs(nb - f) < 0.06 * fps) f = nb; } }
    const buildRise = d.b >= 4 ? barHigh[d.b - 1] - barHigh[d.b - 4] : 0;
    const lastBeat = Math.max(0, f - Math.round(fps * beatDur));
    let minL = Infinity; for (let i = lastBeat; i < f; i++) minL = Math.min(minL, loudFast[i]);
    const before = (barLoud[d.b - 2] + barLoud[d.b - 1]) / 2;
    return { t: +fT(f).toFixed(2), jumpDb: +d.jump.toFixed(1), lowJumpDb: +d.lowJump.toFixed(1), build: buildRise > 0, gapBefore: minL < before - 6 };
  });

  // silence gaps (good spots for a freeze, a text beat or a hard stop)
  const gaps = []; { const thr = Math.max(-55, loudMed - 24); let s = -1;
    for (let i = 0; i < nF; i++) { const q = loudFast[i] < thr; if (q && s < 0) s = i; if ((!q || i === nF - 1) && s >= 0) { if ((i - s) / fps >= 0.2 && fT(s) > 0.3 && fT(i) < duration - 0.3) gaps.push({ t: +fT(s).toFixed(2), dur: +((i - s) / fps).toFixed(2) }); s = -1; } } }

  // global metrics
  const loud3a = Array.from(loud3).filter(v => v > -60);
  const lra = percentile(loud3a, 95) - percentile(loud3a, 10);
  const onsetRate = allOnsets.length / duration;
  const accentRate = (accentPk.length / duration) * 60;
  const firstHit = allOnsets.find(([i, v]) => v > envMean + 1.5 * envStd);
  const firstHitT = firstHit ? fT(firstHit[0]) : null;
  const firstBeat = beats[0] ?? null;
  const midBusy = mean(midN);  // relative, used only inside fits

  // best windows for reels (bar-quantised so they loop and end on a bar line)
  const accentSet = accentPk.map(p => fT(p[0]));
  const windowFor = sec => {
    if (duration < sec + 1) return null;
    const useBars = downbeats.length >= 6, barsN = useBars ? Math.max(1, Math.round(sec / barDur)) : 0, len = useBars ? barsN * barDur : sec;
    const starts = [...(useBars ? downbeats : beats.length ? beats : Array.from({ length: Math.floor(duration * 2) }, (_, i) => i / 2)), ...drops.map(d => d.t)].sort((p, q) => p - q);
    let best = null;
    for (const s of starts) {
      const e = s + len; if (e > duration - 0.2) continue;
      const a = tF(s), b = tF(e), h = tF(s + Math.min(1, len)), firstSec = mean(energy, a, Math.max(a + 1, h)), eMean = mean(energy, a, b);
      const startHit = clamp01(localMax(envD, a, Math.round(0.06 * fps)) / (envMean + 3 * envStd));
      const nAcc = accentSet.filter(t => t >= s && t < e).length, accN = clamp01(nAcc / (len / 2));
      let dropB = 0, dropAt = null;
      for (const d of drops) { if (Math.abs(d.t - s) < 0.2) { dropB = Math.max(dropB, 1); dropAt = d.t; } else if (d.t > s + 0.5 && d.t < s + len * 0.6) { const v = 0.85 - (d.t - s) / len; if (v > dropB) { dropB = v; dropAt = d.t; } } }
      const gapPen = gaps.some(g => g.t >= s && g.t < s + 2) ? 0.15 : 0;
      const fadePen = clamp01((mean(energy, Math.max(a, b - Math.round(fps)), b) < 0.25 ? 0.15 : 0));
      const score = 0.32 * eMean + 0.2 * firstSec + 0.15 * startHit + 0.18 * dropB + 0.15 * accN - gapPen - fadePen;
      if (!best || score > best.score) best = { start: +s.toFixed(2), end: +e.toFixed(2), len: +len.toFixed(2), bars: barsN, score: +score.toFixed(3), firstSec, startHit, dropAt, accents: nAcc };
    }
    return best;
  };
  const windows = { 7: windowFor(7), 15: windowFor(15), 30: windowFor(30) };
  // what the reel actually uses: features inside a window (beat confidence, hit strength, model mood over its patches)
  const mc = ml && ml.feel && ml.feel.curves && ml.feel.curves.t ? ml.feel.curves : null;
  const winFeat = w => {
    if (!w) return null;
    const ks = []; beats.forEach((b, k) => { if (b >= w.start && b < w.end) ks.push(k); });
    const fr = ks.map(k => beatsF[k]), o = { energy: mean(energy, tF(w.start), tF(w.end)) };
    o.rise = fr.length ? median(fr.map(f => localMax(F.flux, f, 2))) : 0;
    o.lowRise = fr.length ? median(fr.map(f => localMax(F.fLow, f, 2))) : 0;
    if (act) { o.conf = ks.length ? mean(ks.map(k => beatConf[k])) : 0; o.strong = ks.length ? ks.filter(k => beatConf[k] >= 0.5).length / ks.length : 0; }
    if (mc) for (const key of ['aggressive', 'party', 'relaxed', 'danceable', 'engagement', 'approachability']) {
      if (!mc[key]) continue; let s = 0, c = 0;
      for (let p = 0; p < mc.t.length; p++) if (mc.t[p] >= w.start - 1 && mc.t[p] <= w.end + 1) { s += mc[key][p]; c++; }
      o[key] = c ? s / c : null;
    }
    for (const k in o) if (typeof o[k] === 'number') o[k] = +o[k].toFixed(3);
    return o;
  };
  for (const L of [7, 15, 30]) if (windows[L]) windows[L].feat = winFeat(windows[L]);

  // ---- scores (heuristics, 0-100; each one explained in the UI) ----
  const punch = 100 * (0.45 * lerp01(rawRise, 0.05, 0.6) + 0.35 * lerp01(rawLowRise, 0.03, 0.2) + 0.2 * lerp01(beatContrast, 1.2, 8));
  // model-side pulse: how sure the network is about the grid, over the audible part of the song
  const audibleK = beatsF.map((f, k) => k).filter(k => loud[beatsF[k]] > lp2 + 3);
  const confMean = act && audibleK.length ? mean(audibleK.map(k => beatConf[k])) : null;
  const strongShare = act && audibleK.length ? audibleK.filter(k => beatConf[k] >= 0.5).length / audibleK.length : null;
  const calmShare = activity ? (() => { let c = 0, n = 0; for (let i = 0; i < nF; i++) if (loud[i] > lp2 + 3) { n++; if (activity[i] < 0.2) c++; } return n ? c / n : 0; })() : null;
  const pulse = act
    ? 100 * (0.55 * lerp01(confMean, 0.2, 0.8) + 0.45 * strongShare) * (tempoDrift > 8 ? 0.9 : 1)
    : 100 * (0.55 * hitRatio + 0.45 * lerp01(tempo.peakAc, 0.05, 0.45)) * (tempoDrift > 8 ? 0.85 : 1);
  const strongestDrop = drops.reduce((m, d) => Math.max(m, d.jumpDb + 0.3 * d.lowJumpDb), 0);
  const payoff = 100 * (0.65 * lerp01(strongestDrop, 2.5, 10) + 0.35 * lerp01(lra, 2, 10));
  const hitsPerBeat = coreF.length ? (hits.kick.length + hits.snare.length) / coreF.length : 0;
  const drive = 100 * (0.6 * lerp01(onsetRate, 1.5, 6) + 0.4 * lerp01(hitsPerBeat, 0.5, 2.5));
  const w15 = windows[15] || windows[7];
  const hook = w15 ? 100 * clamp01(0.45 * w15.firstSec + 0.35 * w15.startHit + 0.2 * (w15.dropAt !== null && Math.abs(w15.dropAt - w15.start) < 0.2 ? 1 : 0.4)) : 0;
  // Edit power: a small logistic head fitted on a labelled corpus (tests/fit.mjs): 43 songs an editor keeps in a
  // go-to folder for edits vs 40 calm library tracks (lofi, R&B, soul, ambient). Leave-one-out AUC 0.83 with the
  // neural features, 0.69 for the DSP-only fallback; held-out phonk, cinematic and trending-edit folders all rank
  // above the calm ones. The old hand-weighted mix scored 0.43 on the same test, so it is gone.
  const mood = ml && ml.feel && ml.feel.mood ? ml.feel.mood : null;
  const lg = p => Math.log(Math.max(1e-3, p) / Math.max(1e-3, 1 - p));
  const featVal = {
    intensity: mood ? lg(mood.aggressive) : null, edge: mood ? -lg(mood.approachability) : null,
    drops: Math.min(4, drops.length), dropSize: Math.min(20, strongestDrop), density: midBusy, sustain: -decayDb, hook: hook / 100,
  };
  const head = mood ? EDIT_HEAD : EDIT_HEAD_DSP;
  let zsum = head.b; const contrib = {};
  head.names.forEach((n, j) => { const c = head.w[j] * ((featVal[n] - head.mu[j]) / head.sd[j]); contrib[n] = +c.toFixed(2); zsum += c; });
  const overall = 100 / (1 + Math.exp(-zsum));
  const intensityScore = mood ? 100 * clamp01(0.55 * lerp01(lg(mood.aggressive), -5, 0) + 0.25 * (1 - mood.relaxed) + 0.2 * mood.party) : null;
  const bpmFast = bpm < 100 ? bpm * 2 : bpm;
  const tempoFitFast = clamp01(1 - Math.abs(bpmFast - 140) / 60);
  const tempoFitSlow = clamp01(1 - Math.abs((bpm > 120 ? bpm / 2 : bpm) - 80) / 40);
  const fits = {
    montage: 100 * (0.25 * tempoFitFast + 0.35 * punch / 100 + 0.25 * pulse / 100 + 0.15 * drive / 100),
    cinematic: 100 * (0.45 * payoff / 100 + 0.25 * lerp01(lra, 3, 12) + 0.15 * (1 - drive / 100) + 0.15 * punch / 100),
    voiceover: 100 * (0.3 * pulse / 100 + 0.3 * (1 - lerp01(lra, 3, 10)) + 0.25 * (1 - drive / 100) + 0.15 * clamp01(1 - Math.abs(bpm - 100) / 40)),
    story: 100 * (0.3 * tempoFitSlow + 0.3 * payoff / 100 + 0.2 * (1 - drive / 100) + 0.2 * lerp01(lra, 2, 8)),
  };
  for (const k in fits) fits[k] = Math.round(fits[k]);

  // cut plan: cadence per section level
  for (const s of sections) { const ks = beats.map((b, k) => k).filter(k => beats[k] >= s.start && beats[k] < s.end); s.pulse = ks.length ? +mean(ks.map(k => beatConf[k])).toFixed(2) : 0; }
  const cutPlan = sections.map(s => {
    let every = s.label === 'Peak' ? (bpm < 100 ? 0.5 : 1) : s.label === 'Mid' || s.label === 'Build' ? 2 : 4;
    const calm = act && s.pulse < 0.25;
    if (calm) every = meter * 2; // nothing to cut on: hold shots for two bars and let the big hits do the work
    return { start: s.start, end: s.end, label: s.label, everyBeats: every, shotSec: +(every * beatDur).toFixed(2), calm: !!calm };
  });

  // compact curves for drawing (one value per frame)
  const wave = new Float32Array(nF); for (let i = 0; i < nF; i++) wave[i] = F.peak[i];
  const lowCurve = new Float32Array(nF); { const lp2l = percentile(Array.from(lowDb), 5), lp99l = percentile(Array.from(lowDb), 99); for (let i = 0; i < nF; i++) lowCurve[i] = clamp01((lowDb[i] - lp2l) / (lp99l - lp2l || 1)); }
  const onsetCurve = new Float32Array(nF); { const m = percentile(Array.from(env), 99.5) || 1; for (let i = 0; i < nF; i++) onsetCurve[i] = clamp01(env[i] / m); }
  const curves = { wave, energy, low: lowCurve, onset: onsetCurve };
  if (act) { curves.beatProb = act; curves.activity = activity; }

  return {
    ok: true, duration, fps, t0,
    bpm: +bpm.toFixed(1), bpmAlts: tempo.alts, tempoConf: +tempo.peakAc.toFixed(3), tempoDrift: +tempoDrift.toFixed(1),
    beatDur, barDur, meter, beats, beatConf, beatPos, barNo, downbeats, phase, tempoMap, detected,
    model: act ? { used: true, feel: ml.feel && !ml.feel.error ? { mood: ml.feel.mood, styles: ml.feel.styles } : null, feelError: ml.feel && ml.feel.error || null } : { used: false },
    hits, gaps, drops, sections, cutPlan, windows,
    grid: { ...Object.fromEntries(Object.entries(grid).map(([k, v]) => [k, +v.toFixed(3)])), name: gridName }, ioiHist,
    metrics: {
      beatContrast: +beatContrast.toFixed(2), lowContrast: +lowContrast.toFixed(2), hitRatio: +hitRatio.toFixed(2),
      lra: +lra.toFixed(1), loudMedDb: +loudMed.toFixed(1), onsetRate: +onsetRate.toFixed(2), accentRate: +accentRate.toFixed(1),
      firstHit: firstHitT === null ? null : +firstHitT.toFixed(2), firstBeat: firstBeat === null ? null : +firstBeat.toFixed(2),
      strongestDrop: +strongestDrop.toFixed(1), midBusy: +midBusy.toFixed(2), rawRise: +rawRise.toFixed(3), rawLowRise: +rawLowRise.toFixed(3), decayDb: +decayDb.toFixed(2),
      confMean: confMean === null ? null : +confMean.toFixed(3), strongShare: strongShare === null ? null : +strongShare.toFixed(3), calmShare: calmShare === null ? null : +calmShare.toFixed(3),
    },
    scores: { overall: Math.round(overall), punch: Math.round(punch), pulse: Math.round(pulse), payoff: Math.round(payoff), drive: Math.round(drive), hook: Math.round(hook), ...(intensityScore === null ? {} : { intensity: Math.round(intensityScore) }) },
    editPower: { head: mood ? 'neural' : 'dsp', contrib },
    fits,
    curves,
  };
}
