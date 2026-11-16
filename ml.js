// CutGrid model layer. Two small neural nets run through onnxruntime (web in the browser, node in tests):
//  1. Beat This! small0 (CPJKU, ISMIR 2024, MIT): frame-wise beat and downbeat probability at 50 fps.
//     It is what tells a real pulse from a calm passage where nothing is hitting.
//  2. Discogs-EffNet (MTG Essentia, CC BY-NC-SA 4.0) + six heads: aggressive, party, relaxed, danceable,
//     engagement and approachability, plus 400 Discogs styles. They give the "feel" a DSP meter cannot.
// The front ends below reproduce torchaudio LogMelSpect (Beat This) and Essentia TensorflowInputMusiCNN
// (EffNet) so the models see what they were trained on. tests/run.mjs checks the Beat This one against PyTorch.

export const MODEL_FILES = {
  beat: 'beatthis-small0.onnx', beatMel: 'beatthis-mel.bin', effnet: 'effnet-discogs.onnx', styles: 'effnet-classes.json',
  heads: { aggressive: ['head-mood_aggressive.onnx', 0], party: ['head-mood_party.onnx', 1], relaxed: ['head-mood_relaxed.onnx', 1],
    danceable: ['head-danceability.onnx', 0], engagement: ['head-engagement_regression.onnx', 0], approachability: ['head-approachability_regression.onnx', 0] },
};
export const BEAT_FPS = 50;

// ---------- FFT (radix 2, in place) ----------
function makeFFT(n) {
  const levels = Math.log2(n), cos = new Float64Array(n / 2), sin = new Float64Array(n / 2), rev = new Uint32Array(n);
  for (let i = 0; i < n / 2; i++) { cos[i] = Math.cos((2 * Math.PI * i) / n); sin[i] = Math.sin((2 * Math.PI * i) / n); }
  for (let i = 0; i < n; i++) { let r = 0; for (let b = 0; b < levels; b++) r = (r << 1) | ((i >>> b) & 1); rev[i] = r; }
  return (re, im) => {
    for (let i = 0; i < n; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = (n / size) | 0;
      for (let i = 0; i < n; i += size) for (let j = i, k = 0; j < i + half; j++, k += step) {
        const l = j + half, tre = re[l] * cos[k] + im[l] * sin[k], tim = -re[l] * sin[k] + im[l] * cos[k];
        re[l] = re[j] - tre; im[l] = im[j] - tim; re[j] += tre; im[j] += tim;
      }
    }
  };
}

// Magnitude spectra of many frames, two real frames per complex FFT. frame(t, buf) fills buf with windowed samples.
function magFrames(nFrames, n, frame, onRow) {
  const fft = makeFFT(n), re = new Float64Array(n), im = new Float64Array(n), nb = n / 2 + 1, bufA = new Float64Array(n), bufB = new Float64Array(n);
  const a = new Float64Array(nb), b = new Float64Array(nb);
  for (let t = 0; t < nFrames; t += 2) {
    frame(t, bufA); const hasB = t + 1 < nFrames; if (hasB) frame(t + 1, bufB);
    for (let i = 0; i < n; i++) { re[i] = bufA[i]; im[i] = hasB ? bufB[i] : 0; }
    fft(re, im);
    for (let k = 0; k < nb; k++) {
      const nk = (n - k) % n, ar = 0.5 * (re[k] + re[nk]), ai = 0.5 * (im[k] - im[nk]), br = 0.5 * (im[k] + im[nk]), bi = -0.5 * (re[k] - re[nk]);
      a[k] = Math.sqrt(ar * ar + ai * ai); b[k] = Math.sqrt(br * br + bi * bi);
    }
    onRow(t, a); if (hasB) onRow(t + 1, b);
  }
}

// ---------- band-limited resampler (windowed sinc) ----------
export function resampleSinc(x, srIn, srOut) {
  if (Math.abs(srIn - srOut) < 1) return x;
  const ratio = srOut / srIn, cutoff = Math.min(1, ratio) * 0.95, half = 16, nOut = Math.floor(x.length * ratio), out = new Float32Array(nOut);
  const scale = Math.min(1, ratio), taps = 2 * half;
  for (let i = 0; i < nOut; i++) {
    const p = i / ratio, c = Math.floor(p); let s = 0, w = 0;
    for (let j = c - half + 1; j <= c + half; j++) {
      if (j < 0 || j >= x.length) continue;
      const d = (p - j) * scale * cutoff, sinc = d === 0 ? 1 : Math.sin(Math.PI * d) / (Math.PI * d);
      const win = 0.5 + 0.5 * Math.cos((Math.PI * (p - j)) / (half + 0.5)), k = sinc * win;
      s += x[j] * k; w += k;
    }
    out[i] = w ? s / w : 0;
  }
  void taps;
  return out;
}

// ---------- Beat This front end: torchaudio MelSpectrogram(22050, n_fft 1024, hop 441, 30-11000 Hz, 128 slaney, power 1,
// normalized="frame_length", centre + reflect pad), then log1p(1000 x). Returns T x 128 row major.
export function beatThisSpect(x, fb) {
  const n = 1024, hop = 441, pad = n / 2, nb = n / 2 + 1, T = 1 + Math.floor(x.length / hop);
  const win = new Float64Array(n); let wss = 0;
  for (let i = 0; i < n; i++) { win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n); wss += win[i] * win[i]; }
  const norm = 1 / Math.sqrt(n), L = x.length; // torch.stft(normalized=True) scales by frame_length ** -0.5
  void wss;
  const at = j => { let k = j - pad; if (k < 0) k = -k; if (k >= L) k = 2 * (L - 1) - k; return x[Math.max(0, Math.min(L - 1, k))]; };
  const out = new Float32Array(T * 128);
  magFrames(T, n, (t, buf) => { const o = t * hop; for (let i = 0; i < n; i++) buf[i] = at(o + i) * win[i]; }, (t, mag) => {
    const row = t * 128;
    for (let k = 0; k < nb; k++) { const m = mag[k] * norm; if (m === 0) continue; const fo = k * 128; for (let b = 0; b < 128; b++) { const w = fb[fo + b]; if (w) out[row + b] += m * w; } }
    for (let b = 0; b < 128; b++) out[row + b] = Math.log1p(1000 * out[row + b]);
  });
  return { data: out, T };
}

// ---------- EffNet front end: Essentia TensorflowInputMusiCNN (16 kHz, frame 512, hop 256, symmetric Hann,
// power spectrum, 96 slaney mel bands 0-8000 Hz, linear weighting, unit_tri), then log10(1 + 10000 x). frames x 96.
const hz2mel = hz => (hz < 1000 ? hz / (200 / 3) : 15 + Math.log(hz / 1000) / (Math.log(6.4) / 27));
const mel2hz = m => (m < 15 ? m * (200 / 3) : 1000 * Math.exp((Math.log(6.4) / 27) * (m - 15)));
function essentiaMelBank() {
  const nb = 257, nBands = 96, sr = 16000, fScale = sr / 2 / (nb - 1), lo = hz2mel(0), hi = hz2mel(8000), inc = (hi - lo) / (nBands + 1);
  const f = []; for (let i = 0; i < nBands + 2; i++) f.push(mel2hz(lo + i * inc));
  const bank = [];
  for (let i = 0; i < nBands; i++) {
    const s1 = f[i + 1] - f[i], s2 = f[i + 2] - f[i + 1], j0 = Math.ceil(f[i] / fScale), j1 = Math.min(nb - 1, Math.floor(f[i + 2] / fScale)), w = [];
    const area = (s1 + s2) / 2;
    for (let j = j0; j <= j1; j++) { const hz = j * fScale, v = hz < f[i + 1] ? (hz - f[i]) / s1 : (f[i + 2] - hz) / s2; w.push(v / area); }
    bank.push({ j0, w });
  }
  return bank;
}
export function effnetMel(x16) {
  const n = 512, hop = 256, L = x16.length, nFrames = 1 + Math.ceil(Math.max(0, L - n / 2) / hop), bank = essentiaMelBank();
  const win = new Float64Array(n); for (let i = 0; i < n; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
  const out = new Float32Array(nFrames * 96);
  magFrames(nFrames, n, (t, buf) => { const o = t * hop - n / 2; for (let i = 0; i < n; i++) { const j = o + i; buf[i] = j >= 0 && j < L ? x16[j] * win[i] : 0; } }, (t, mag) => {
    for (let b = 0; b < 96; b++) { const { j0, w } = bank[b]; let s = 0; for (let k = 0; k < w.length; k++) { const m = mag[j0 + k]; s += m * m * w[k]; } out[t * 96 + b] = Math.log10(1 + 10000 * s); }
  });
  return { data: out, frames: nFrames };
}

// ---------- runner ----------
// ort: the onnxruntime module (onnxruntime-web or onnxruntime-node). load(name) -> Promise<ArrayBuffer|Uint8Array>.
export async function createML(ort, load, { onProgress = () => {}, chunk = 1500, patchHop = 62, sessionOptions = {} } = {}) {
  const bytes = async name => { const b = await load(name); return b instanceof Uint8Array ? b : new Uint8Array(b); };
  const opts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all', ...sessionOptions };
  const sess = async name => ort.InferenceSession.create(await bytes(name), opts);
  onProgress('Loading beat model', 0);
  const fbRaw = await bytes(MODEL_FILES.beatMel), fb = new Float32Array(fbRaw.buffer.slice(fbRaw.byteOffset, fbRaw.byteOffset + fbRaw.byteLength));
  const beatSess = await sess(MODEL_FILES.beat);
  onProgress('Loading music model', 0.15);
  const effSess = await sess(MODEL_FILES.effnet);
  const heads = {};
  for (const [k, [file, idx]] of Object.entries(MODEL_FILES.heads)) heads[k] = { s: await sess(file), idx };
  const styles = JSON.parse(new TextDecoder().decode(await bytes(MODEL_FILES.styles)));
  const yieldNow = () => new Promise(r => setTimeout(r, 0));

  async function beatsFrom(x22) {
    const { data, T } = beatThisSpect(x22, fb), border = 6, step = chunk - 2 * border;
    const beat = new Float32Array(T).fill(-1000), down = new Float32Array(T).fill(-1000);
    const starts = []; for (let s = -border; s < T - border; s += step) starts.push(s);
    if (T > step) starts[starts.length - 1] = T - (chunk - border);
    // keep_first: run in reverse so earlier chunks overwrite later ones
    for (let c = starts.length - 1; c >= 0; c--) {
      const s = starts[c], a = Math.max(s, 0), e = Math.min(s + chunk, T), left = Math.max(0, -s), right = Math.max(0, Math.min(border, s + chunk - T));
      const len = left + (e - a) + right, buf = new Float32Array(len * 128);
      buf.set(data.subarray(a * 128, e * 128), left * 128);
      const out = await beatSess.run({ [beatSess.inputNames[0]]: new ort.Tensor('float32', buf, [1, len, 128]) });
      const b = out.beat.data, d = out.downbeat.data;
      for (let i = border; i < len - border; i++) { const g = s + i; if (g >= 0 && g < T) { beat[g] = b[i]; down[g] = d[i]; } }
      onProgress('Finding beats', 0.2 + 0.45 * (starts.length - c) / starts.length); await yieldNow();
    }
    const sig = v => 1 / (1 + Math.exp(-v));
    for (let i = 0; i < T; i++) { beat[i] = sig(beat[i]); down[i] = sig(down[i]); }
    return { beat, down, fps: BEAT_FPS };
  }

  async function feelFrom(x22) {
    const x16 = resampleSinc(x22, 22050, 16000), { data, frames } = effnetMel(x16), P = 128;
    const starts = []; for (let s = 0; s + P <= frames; s += patchHop) starts.push(s);
    if (!starts.length) return null;
    const B = 16, emb = new Float32Array(starts.length * 1280), act = new Float64Array(400);
    for (let i0 = 0; i0 < starts.length; i0 += B) {
      const nB = Math.min(B, starts.length - i0), buf = new Float32Array(nB * P * 96);
      for (let k = 0; k < nB; k++) buf.set(data.subarray(starts[i0 + k] * 96, (starts[i0 + k] + P) * 96), k * P * 96);
      const out = await effSess.run({ [effSess.inputNames[0]]: new ort.Tensor('float32', buf, [nB, P, 96]) });
      emb.set(out.embeddings.data, i0 * 1280);
      const a = out.activations.data; for (let k = 0; k < nB * 400; k++) act[k % 400] += a[k];
      onProgress('Reading the feel', 0.65 + 0.3 * (i0 + nB) / starts.length); await yieldNow();
    }
    const nP = starts.length, curves = { t: starts.map(s => +((s + P / 2) * 256 / 16000).toFixed(2)) }, mood = {};
    for (const [k, { s, idx }] of Object.entries(heads)) {
      const out = await s.run({ [s.inputNames[0]]: new ort.Tensor('float32', emb, [nP, 1280]) });
      const o = out[s.outputNames[0]].data, w = o.length / nP, c = new Float32Array(nP);
      for (let p = 0; p < nP; p++) c[p] = o[p * w + idx];
      curves[k] = c; let m = 0; for (const v of c) m += v; mood[k] = m / nP;
    }
    const top = Array.from(act, (v, i) => [styles[i], v / nP]).sort((p, q) => q[1] - p[1]).slice(0, 5).map(([n, p]) => [n, +p.toFixed(3)]);
    return { mood, curves, styles: top, patches: nP };
  }

  return {
    async run(x22) {
      const beats = await beatsFrom(x22);
      let feel = null; try { feel = await feelFrom(x22); } catch (e) { feel = { error: String(e && e.message || e) }; }
      onProgress('Scoring', 0.97);
      return { ...beats, feel };
    },
  };
}
