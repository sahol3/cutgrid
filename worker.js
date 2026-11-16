// Runs the analysis off the main thread so playback and the cut preview stay smooth.
// The engine is imported with the same ?v= query as this worker so a deploy never mixes versions.
const enginePromise = import(`./engine.js${self.location.search}`);

self.onmessage = async e => {
  const { id, samples, sr } = e.data;
  try {
    const { analyze } = await enginePromise;
    const t = performance.now();
    const r = analyze(samples, sr);
    r.ms = Math.round(performance.now() - t);
    const transfer = r.curves ? Object.values(r.curves).map(a => a.buffer) : [];
    self.postMessage({ id, result: r }, transfer);
  } catch (err) {
    self.postMessage({ id, error: String((err && err.stack) || err) });
  }
};
