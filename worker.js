// Runs the analysis off the main thread so playback and the cut preview stay smooth.
// Two passes: the DSP read comes back in a second or two, then the neural pass (Beat This! + EffNet) upgrades it.
// Modules are imported with the same ?v= query as this worker so a deploy never mixes versions.
const Q = self.location.search;
const enginePromise = import(`./engine.js${Q}`);
const mlPromise = import(`./ml.js${Q}`);
const ORT_VER = '1.22.0';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;
let mlReady = null, backend = null, forceWasm = false;

async function getML(onProgress) {
  if (!mlReady) mlReady = (async () => {
    const hasGPU = !forceWasm && !!(self.navigator && navigator.gpu && (await navigator.gpu.requestAdapter().catch(() => null)));
    const ort = await import(ORT_BASE + (hasGPU ? 'ort.webgpu.min.mjs' : 'ort.wasm.min.mjs'));
    ort.env.wasm.wasmPaths = ORT_BASE;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
    const { createML } = await mlPromise;
    const load = async name => { const r = await fetch(`models/${name}`); if (!r.ok) throw new Error(`model ${name}: HTTP ${r.status}`); return new Uint8Array(await r.arrayBuffer()); };
    const tryCreate = eps => createML(ort, load, { onProgress: (s, p) => current && current(s, p), sessionOptions: { executionProviders: eps } });
    if (hasGPU) { try { const m = await tryCreate(['webgpu', 'wasm']); backend = 'WebGPU'; return m; } catch (e) { console.warn('WebGPU failed, using WASM', e); } }
    const m = await tryCreate(['wasm']); backend = 'WASM'; return m;
  })();
  current = onProgress;
  try { return await mlReady; } catch (e) { mlReady = null; throw e; }
}
let current = null;

self.onmessage = async e => {
  const { id, samples, sr, ml } = e.data;
  const post = (msg, transfer) => self.postMessage({ id, ...msg }, transfer || []);
  try {
    const { analyze } = await enginePromise, { resampleSinc } = await mlPromise;
    const x = resampleSinc(samples, sr, 22050);
    let t = performance.now();
    const r = analyze(x, 22050);
    r.ms = Math.round(performance.now() - t);
    if (!ml || !r.ok) { post({ result: r, final: true }, Object.values(r.curves || {}).map(a => a.buffer)); return; }
    post({ result: r, final: false });
    t = performance.now();
    let model;
    const onP = (stage, p) => post({ progress: { stage, p, backend } });
    try { model = await (await getML(onP)).run(x); }
    catch (err) {
      if (backend === 'WebGPU') { // some GPUs fail at run time, not at load: retry once on WASM
        console.warn('WebGPU run failed, retrying on WASM', err);
        forceWasm = true; mlReady = null;
        try { model = await (await getML(onP)).run(x); } catch (err2) { err = err2; model = null; }
      }
      if (!model) { post({ result: r, final: true, mlError: String((err && err.message) || err) }); return; }
    }
    const r2 = analyze(x, 22050, { ml: model });
    r2.ms = r.ms; r2.mlMs = Math.round(performance.now() - t); r2.model.backend = backend;
    post({ result: r2, final: true }, Object.values(r2.curves || {}).map(a => a.buffer));
  } catch (err) {
    post({ error: String((err && err.stack) || err) });
  }
};
