// CutGrid UI: loading (YouTube capture or file), playback clock, timeline, cut preview, readouts, exports.
const $ = id => document.getElementById(id);
const PLATES = ['#2b2825', '#3a3530', '#252a2e', '#34302b', '#2b302c', '#3b322f', '#2f2c33'];
const FIT_NAMES = { montage: 'Fast beat-cut montage', cinematic: 'Cinematic or slow motion', voiceover: 'Bed under a voiceover', story: 'Story or emotional reel' };

const S = {
  mode: null, result: null, title: '', total: 0, playing: false,
  ctx: null, buffer: null, src: null, startedAt: 0, offset: 0,          // file playback
  player: null, videoId: null, ytAnchor: { t: 0, at: 0 }, ytState: -1,  // YouTube
  cap: null, analysing: false, pendingLive: false,
  view: { t0: 0, t1: 1 }, hover: null, nudge: 0, cadence: 'auto', cuts: [], cutIdx: -1, loop: null,
  uiCtx: null, clickOn: false, staticLayer: null,
};

// ---------- formatting ----------
const pad = (n, w = 2) => String(n).padStart(w, '0');
function fmt(t, dec = 1) { if (!isFinite(t)) return '--'; const m = Math.floor(t / 60), s = t - m * 60; return `${m}:${pad(s.toFixed(dec), dec ? 3 + dec : 2)}`; }
function tc30(t) { const f = Math.round(t * 30), ff = f % 30, s = Math.floor(f / 30) % 60, m = Math.floor(f / 1800); return `${pad(m)}:${pad(s)}:${pad(ff)}`; }
const secs = v => `${v.toFixed(v < 1 ? 3 : 2)} s`;

// ---------- worker ----------
const VERSION = '5';
const worker = new Worker(`worker.js?v=${VERSION}`, { type: 'module' });
let jobId = 0; const jobs = new Map();
worker.onmessage = e => {
  const d = e.data, j = jobs.get(d.id); if (!j) return;
  if (d.error) { jobs.delete(d.id); j.reject(new Error(d.error)); return; }
  if (d.progress) { j.onProgress?.(d.progress); return; }
  if (!d.final) { j.onPartial?.(d.result); return; }
  jobs.delete(d.id); if (d.mlError) d.result.mlError = d.mlError; j.resolve(d.result);
};
// ml: also run the neural pass (Beat This! + EffNet). onPartial gets the quick DSP read first.
function runAnalysis(samples, sr, { ml = false, onPartial, onProgress } = {}) {
  const id = ++jobId;
  return new Promise((resolve, reject) => { jobs.set(id, { resolve, reject, onPartial, onProgress }); worker.postMessage({ id, samples, sr, ml }, [samples.buffer]); });
}
function setModelStatus(text, cls = '') { const el = $('modelStatus'); if (el) { el.textContent = text; el.className = 'modelStatus ' + cls; } }
const modelOpts = () => ({
  ml: true,
  onPartial: r => { showResult(r, false); setModelStatus('Quick read shown. The neural pass is listening for the real beat and feel…', 'busy'); },
  onProgress: p => setModelStatus(`${p.stage}${p.backend ? ` (${p.backend})` : ''}… ${Math.round(p.p * 100)}%`, 'busy'),
});
function modelDone(r) {
  if (r.mlError) setModelStatus(`The neural pass could not run here (${r.mlError}). Showing the quick read.`, 'err');
  else if (r.model?.used) setModelStatus(`Neural pass done on ${r.model.backend} in ${(r.mlMs / 1000).toFixed(1)} s.`, 'ok');
}

// ---------- loading ----------
function parseYouTube(s) {
  s = s.trim();
  if (/^[\w-]{11}$/.test(s)) return s;
  try {
    const u = new URL(s.startsWith('http') ? s : 'https://' + s);
    if (u.hostname.endsWith('youtu.be')) return u.pathname.slice(1, 12);
    if (u.searchParams.get('v')) return u.searchParams.get('v').slice(0, 11);
    const m = u.pathname.match(/\/(shorts|embed|live|v)\/([\w-]{11})/);
    if (m) return m[2];
  } catch { /* not a URL */ }
  return null;
}

$('linkForm').addEventListener('submit', e => {
  e.preventDefault();
  const id = parseYouTube($('link').value);
  if (!id) { $('link').setCustomValidity('That does not look like a YouTube link.'); $('link').reportValidity(); return; }
  $('link').setCustomValidity('');
  loadYouTube(id);
});
$('link').addEventListener('input', () => $('link').setCustomValidity(''));

const drop = $('drop'), fileIn = $('file');
drop.addEventListener('click', () => fileIn.click());
drop.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileIn.click(); } });
fileIn.addEventListener('change', () => fileIn.files[0] && loadFile(fileIn.files[0]));
['dragenter', 'dragover'].forEach(ev => document.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
document.addEventListener('dragleave', e => { if (!e.relatedTarget) drop.classList.remove('over'); });
document.addEventListener('drop', e => { e.preventDefault(); drop.classList.remove('over'); });
document.addEventListener('drop', e => { const f = e.dataTransfer?.files?.[0]; if (f) loadFile(f); });
$('resetBtn').addEventListener('click', () => { stopAll(); history.replaceState(null, '', location.pathname); location.reload(); });

function resetStage(kind, title) {
  stopAll();
  S.result = null; S.cuts = []; S.cutIdx = -1; S.loop = null; S.offset = 0;
  document.body.classList.add('compact');
  $('stage').hidden = false; showRes(false);
  $('srcKind').textContent = kind; $('srcTitle').textContent = title;
  $('ytWrap').hidden = kind !== 'YouTube'; $('capturePanel').hidden = kind !== 'YouTube';
  $('transport').hidden = false;
}

async function loadFile(f) {
  resetStage('Audio file', f.name.replace(/\.[^.]+$/, ''));
  S.mode = 'file'; S.title = $('srcTitle').textContent;
  setHeadline('Decoding…', '');
  showRes(true);
  try {
    S.ctx = S.ctx || new AudioContext();
    const buf = await S.ctx.decodeAudioData(await f.arrayBuffer());
    useBuffer(buf);
  } catch (err) {
    setHeadline('This file could not be decoded', 'Try an MP3, WAV or M4A export of the track.');
  }
}

async function useBuffer(buf) {
  S.buffer = buf; S.total = buf.duration; S.view = { t0: 0, t1: buf.duration };
  const chans = []; for (let c = 0; c < buf.numberOfChannels; c++) chans.push(buf.getChannelData(c));
  const mono = new Float32Array(buf.length);
  for (const ch of chans) for (let i = 0; i < mono.length; i++) mono[i] += ch[i] / chans.length;
  setHeadline('Analysing…', `${fmt(buf.duration, 0)} of audio.`);
  try { const r = await runAnalysis(mono, buf.sampleRate, modelOpts()); showResult(r, true); modelDone(r); }
  catch (err) { setHeadline('Analysis failed', String(err.message || err)); }
}

// ---------- YouTube ----------
let ytApi;
function loadYTApi() {
  if (ytApi) return ytApi;
  ytApi = new Promise(res => { window.onYouTubeIframeAPIReady = () => res(window.YT); const s = document.createElement('script'); s.src = 'https://www.youtube.com/iframe_api'; document.head.appendChild(s); });
  return ytApi;
}
async function loadYouTube(id) {
  resetStage('YouTube', 'Loading video…');
  S.mode = 'yt'; S.videoId = id;
  history.replaceState(null, '', '#v=' + id);
  $('capStatus').textContent = ''; $('capStatus').className = 'status';
  const YT = await loadYTApi();
  if (S.player) { S.player.destroy(); S.player = null; }
  $('ytWrap').innerHTML = '<div id="yt"></div>';
  S.player = new YT.Player('yt', {
    videoId: id, playerVars: { playsinline: 1, rel: 0, modestbranding: 1 },
    events: {
      onReady: () => {
        const d = S.player.getVideoData(); S.title = d?.title || 'YouTube video';
        $('srcTitle').textContent = S.title; S.total = S.player.getDuration() || 0; S.view = { t0: 0, t1: Math.max(1, S.total) };
      },
      onStateChange: e => {
        S.ytState = e.data; S.playing = e.data === 1; syncPlayIcon();
        S.ytAnchor = { t: S.player.getCurrentTime(), at: performance.now() };
        if (!S.total) S.total = S.player.getDuration() || 0;
        if (e.data === 0 && S.cap) stopListen('Song finished. Final analysis below.');
      },
      onError: e => {
        const blocked = e.data === 101 || e.data === 150;
        $('capStatus').className = 'status err';
        $('capStatus').textContent = blocked ? 'The owner blocks this video from playing outside YouTube. Use the file route for it.' : 'YouTube could not play this video.';
      },
    },
  });
}

// ---------- tab-audio capture ----------
const TAP = `class Tap extends AudioWorkletProcessor{constructor(){super();this.b=new Float32Array(2048);this.n=0}
process(inp){const ch=inp[0];if(!ch||!ch.length)return true;const L=ch[0].length;for(let i=0;i<L;i++){let v=0;for(let c=0;c<ch.length;c++)v+=ch[c][i];this.b[this.n++]=v/ch.length;
if(this.n===2048){this.port.postMessage(this.b);this.b=new Float32Array(2048);this.n=0}}return true}}registerProcessor('tap',Tap);`;
const tapURL = URL.createObjectURL(new Blob([TAP], { type: 'application/javascript' }));

$('listenBtn').addEventListener('click', startListen);
$('stopBtn').addEventListener('click', () => stopListen('Stopped. Analysis of everything heard is below.'));

async function startListen() {
  const st = $('capStatus'); st.className = 'status';
  if (!navigator.mediaDevices?.getDisplayMedia) { st.className = 'status err'; st.textContent = 'This browser cannot share tab audio. Use Chrome or Edge on desktop, or drop a file.'; return; }
  if (!S.player?.getDuration) { st.textContent = 'The player is still loading.'; return; }
  let stream;
  try {
    stream = await navigator.mediaDevices.getDisplayMedia({
      video: { frameRate: 1 }, audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, suppressLocalAudioPlayback: false },
      preferCurrentTab: true, selfBrowserSurface: 'include', systemAudio: 'include', surfaceSwitching: 'exclude',
    });
  } catch { st.className = 'status err'; st.textContent = 'Sharing was cancelled.'; return; }
  const track = stream.getAudioTracks()[0];
  if (!track) { stream.getTracks().forEach(t => t.stop()); st.className = 'status err'; st.textContent = 'No audio came through. Choose this tab and keep Share tab audio switched on.'; return; }
  let ctx, src;
  try { ctx = new AudioContext({ sampleRate: 22050 }); src = ctx.createMediaStreamSource(new MediaStream([track])); }
  catch { ctx = new AudioContext(); src = ctx.createMediaStreamSource(new MediaStream([track])); }
  await ctx.audioWorklet.addModule(tapURL);
  const node = new AudioWorkletNode(ctx, 'tap', { numberOfInputs: 1, numberOfOutputs: 1 });
  const mute = ctx.createGain(); mute.gain.value = 0;
  src.connect(node); node.connect(mute); mute.connect(ctx.destination);
  const sr = ctx.sampleRate, total = S.player.getDuration() || 600;
  S.total = total; S.view = { t0: 0, t1: total };
  S.cap = { stream, ctx, node, sr, buf: new Float32Array(Math.ceil((total + 3) * sr)), pos: 0, max: 0, resync: true, heard: 0, timer: 0 };
  node.port.onmessage = e => onChunk(e.data);
  track.onended = () => S.cap && stopListen('Sharing stopped. Analysis of everything heard is below.');
  $('listenBtn').hidden = true; $('stopBtn').hidden = false;
  showRes(true); setHeadline('Listening…', 'The first read appears after a few seconds of audio.');
  S.player.seekTo(0, true); S.player.playVideo();
  setTimeout(() => { if (S.cap && S.ytState !== 1) st.textContent = 'Press play on the video to start listening.'; }, 1800);
  S.cap.timer = setInterval(liveAnalyse, 4000);
}

function onChunk(chunk) {
  const c = S.cap; if (!c) return;
  if (S.ytState !== 1) { c.resync = true; return; }
  const pt = S.player.getCurrentTime(), expected = Math.round(pt * c.sr) - chunk.length;
  if (c.resync || Math.abs(expected - c.pos) > 0.6 * c.sr) { c.pos = Math.max(0, expected); c.resync = false; }
  if (c.pos + chunk.length > c.buf.length) return;
  c.buf.set(chunk, c.pos); c.pos += chunk.length; c.max = Math.max(c.max, c.pos); c.heard += chunk.length;
  $('capStatus').className = 'status';
  $('capStatus').textContent = `Listening. ${fmt(c.heard / c.sr, 0)} heard of ${fmt(S.total, 0)}.`;
}

async function liveAnalyse(final = false) {
  const c = S.cap || S.lastCap; if (!c) return;
  if (S.analysing) { S.pendingLive = S.pendingLive || final; return; }
  if (c.max < c.sr * 5) return;
  S.analysing = true;
  try {
    // live reads during capture stay DSP only; the final one runs the neural pass
    const r = await runAnalysis(c.buf.slice(0, c.max), c.sr, final ? modelOpts() : {});
    showResult(r, final); if (final) modelDone(r);
  }
  catch (err) { console.error(err); }
  S.analysing = false;
  if (S.pendingLive) { S.pendingLive = false; liveAnalyse(true); }
}

function stopListen(msg, analyse = true) {
  const c = S.cap; if (!c) return;
  clearInterval(c.timer); c.stream.getTracks().forEach(t => t.stop()); c.ctx.close();
  S.cap = null; S.lastCap = c;
  $('listenBtn').hidden = false; $('listenBtn').textContent = 'Listen again'; $('stopBtn').hidden = true;
  $('capStatus').className = 'status'; $('capStatus').textContent = msg;
  if (!analyse) { S.lastCap = null; return; }
  if (c.max < c.sr * 5) { setHeadline('Not enough audio heard', 'CutGrid needs at least 5 seconds. Check that Share tab audio was on.'); return; }
  liveAnalyse(true);
}

// ---------- playback ----------
function clock() {
  if (S.mode === 'file') return S.playing ? S.ctx.currentTime - S.startedAt : S.offset;
  if (S.mode === 'yt' && S.player?.getCurrentTime) {
    const now = performance.now(), t = S.player.getCurrentTime();
    if (t !== S.ytAnchor.t) S.ytAnchor = { t, at: now };
    return S.playing ? S.ytAnchor.t + Math.min(0.5, (now - S.ytAnchor.at) / 1000) : t;
  }
  return 0;
}
function play(from) {
  if (S.mode === 'file' && S.buffer) {
    if (S.ctx.state === 'suspended') S.ctx.resume();
    stopFileSource();
    const t = Math.max(0, Math.min(S.buffer.duration - 0.05, from ?? S.offset));
    S.src = S.ctx.createBufferSource(); S.src.buffer = S.buffer; S.src.connect(S.ctx.destination);
    S.src.onended = () => { if (S.src && S.playing && clock() >= S.buffer.duration - 0.1) { S.playing = false; S.offset = 0; syncPlayIcon(); } };
    S.src.start(0, t); S.startedAt = S.ctx.currentTime - t; S.playing = true;
  } else if (S.mode === 'yt' && S.player) {
    if (from !== undefined) S.player.seekTo(from, true);
    S.player.playVideo();
  }
  S.cutIdx = -2; syncPlayIcon();
}
function pause() {
  if (S.mode === 'file') { S.offset = clock(); stopFileSource(); S.playing = false; }
  else if (S.player) S.player.pauseVideo();
  syncPlayIcon();
}
function seek(t) {
  t = Math.max(0, Math.min(S.total || t, t));
  if (S.mode === 'file') { if (S.playing) play(t); else S.offset = t; }
  else if (S.player) S.player.seekTo(t, true);
  S.cutIdx = -2;
}
function stopFileSource() { if (S.src) { S.src.onended = null; try { S.src.stop(); } catch { /* already stopped */ } S.src = null; } }
function stopAll() { if (S.cap) stopListen('', false); if (S.mode === 'file') pause(); else if (S.player?.pauseVideo) S.player.pauseVideo(); S.playing = false; }
function syncPlayIcon() { $('playBtn').classList.toggle('playing', S.playing); $('playBtn').setAttribute('aria-label', S.playing ? 'Pause' : 'Play'); }
$('playBtn').addEventListener('click', () => (S.playing ? pause() : play()));
document.addEventListener('keydown', e => { if (e.code === 'Space' && !$('stage').hidden && !/INPUT|SELECT|TEXTAREA|BUTTON/.test(e.target.tagName)) { e.preventDefault(); S.playing ? pause() : play(); } });
$('nudge').addEventListener('input', e => { S.nudge = +e.target.value; $('nudgeOut').textContent = `${S.nudge > 0 ? '+' : ''}${S.nudge} ms`; });
$('clickOn').addEventListener('change', e => { S.clickOn = e.target.checked; if (S.clickOn && !S.uiCtx) S.uiCtx = new AudioContext(); });
$('cadence').addEventListener('change', e => { S.cadence = e.target.value; buildCuts(); S.cutIdx = -2; });

function tick() {
  if (!S.uiCtx) return;
  const c = S.uiCtx, o = c.createOscillator(), g = c.createGain(), t = c.currentTime;
  o.frequency.value = 1700; g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.35, t + 0.002); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.03);
  o.connect(g); g.connect(c.destination); o.start(t); o.stop(t + 0.04);
}

// ---------- cuts ----------
// bar and beat of grid line i (1-based). The engine gives bar positions; older results fall back to phase.
function barBeat(r, i) {
  if (i < 0) return { bar: 0, beat: 1 };
  if (r.beatPos) return { bar: r.barNo[i], beat: r.beatPos[i] + 1 };
  const rel = i - r.phase; return { bar: Math.floor(rel / 4) + 1, beat: ((rel % 4) + 4) % 4 + 1 };
}
// is grid line i on a cut when cutting every k beats (k a fraction of a bar, a bar, or whole bars)
function onEvery(r, i, k) {
  const { bar, beat } = barBeat(r, i), m = r.meter || 4;
  if (k >= m) { const bars = Math.max(1, Math.round(k / m)); return beat === 1 && bar >= 1 && (bar - 1) % bars === 0; }
  return (beat - 1) % k === 0;
}
function buildCuts() {
  const r = S.result; if (!r?.ok) { S.cuts = []; return; }
  const B = r.beats, out = [0];
  const every = (k, from = 0, to = Infinity) => {
    if (k === 0.5) { for (let i = 0; i < B.length; i++) { if (B[i] >= from && B[i] < to) out.push(B[i]); const m = i + 1 < B.length ? (B[i] + B[i + 1]) / 2 : B[i] + r.beatDur / 2; if (m >= from && m < to) out.push(m); } return; }
    for (let i = 0; i < B.length; i++) if (B[i] >= from && B[i] < to && onEvery(r, i, k)) out.push(B[i]);
  };
  const c = S.cadence;
  if (c === 'accent') r.hits.accent.forEach(h => out.push(h.t));
  else if (c === 'kick') r.hits.kick.forEach(h => out.push(h.t));
  else if (c === 'auto') r.cutPlan.forEach(p => every(p.everyBeats, p.start, p.end));
  else every(+c);
  r.drops.forEach(d => out.push(d.t));
  out.sort((a, b) => a - b);
  S.cuts = out.filter((t, i) => i === 0 || t - out[i - 1] > 0.08);
}
const near = (list, t, w) => list.some(x => Math.abs((x.t ?? x) - t) < w);
function upperIdx(arr, t) { let lo = 0, hi = arr.length - 1, ans = -1; while (lo <= hi) { const m = (lo + hi) >> 1; if (arr[m] <= t) { ans = m; lo = m + 1; } else hi = m - 1; } return ans; }
function sectionAt(t) { const sc = S.result?.sections; if (!sc?.length) return null; if (t < sc[0].start) return sc[0]; return sc.find(s => t >= s.start && t < s.end) || sc[sc.length - 1]; }

function updatePreview(t) {
  const r = S.result; if (!r?.ok || !S.cuts.length) return;
  const i = upperIdx(S.cuts, t);
  if (i !== S.cutIdx) {
    const stepped = i === S.cutIdx + 1;
    S.cutIdx = i;
    const ct = S.cuts[Math.max(0, i)], isDrop = near(r.drops, ct, 0.1), isAcc = near(r.hits.accent, ct, 0.07);
    const shot = $('shot');
    shot.style.background = isDrop ? 'var(--accent)' : PLATES[Math.max(0, i) % PLATES.length];
    $('shotNo').textContent = pad(Math.max(1, i + 1));
    $('shotSec').textContent = sectionAt(ct)?.label || '';
    $('hudCuts').textContent = `${Math.max(1, i + 1)} / ${S.cuts.length} cuts`;
    if (stepped && S.playing) {
      if (isAcc || isDrop) { shot.classList.remove('punch'); void shot.offsetWidth; shot.classList.add('punch'); }
      if (isDrop) { const f = $('flash'); f.classList.remove('on'); void f.offsetWidth; f.classList.add('on'); }
      if (S.clickOn) tick();
    }
  }
  const bi = upperIdx(r.beats, t);
  if (bi >= 0) { const { bar, beat } = barBeat(r, bi), calm = r.beatConf && r.beatConf[bi] < 0.2; $('hudBeat').textContent = calm ? 'no beat here' : `bar ${Math.max(0, bar)} · beat ${beat}`; }
}

// ---------- render loop ----------
function frame() {
  if (!$('stage').hidden) {
    const t = clock();
    $('clock').textContent = fmt(t);
    const st = t + S.nudge / 1000;
    if (S.loop && S.playing && t >= S.loop.end) seek(S.loop.start);
    updatePreview(st);
    drawTimelineFrame(t);
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// ---------- timeline ----------
const tl = $('timeline');
const TL = { secTop: 0, secH: 18, mainTop: 26, mainH: 136, laneTop: 170, axisY: 254 };
function tlSize() { const dpr = Math.min(2, devicePixelRatio || 1), w = tl.clientWidth, h = tl.clientHeight; if (tl.width !== Math.round(w * dpr) || tl.height !== Math.round(h * dpr)) { tl.width = Math.round(w * dpr); tl.height = Math.round(h * dpr); S.staticLayer = null; } return { w, h, dpr }; }
const xOf = (t, w) => ((t - S.view.t0) / (S.view.t1 - S.view.t0)) * w;
const tOf = (x, w) => S.view.t0 + (x / w) * (S.view.t1 - S.view.t0);

function drawStatic() {
  const { w, h, dpr } = tlSize(), r = S.result;
  const layer = document.createElement('canvas'); layer.width = tl.width; layer.height = tl.height;
  const g = layer.getContext('2d'); g.scale(dpr, dpr);
  g.font = '600 11px Satoshi, system-ui, sans-serif'; g.textBaseline = 'middle';
  const { t0, t1 } = S.view, span = t1 - t0, ppS = w / span;
  if (!r?.ok) { S.staticLayer = layer; return; }
  const mid = TL.mainTop + TL.mainH / 2, amp = TL.mainH / 2 - 4, fps = r.fps, cv = r.curves;
  const fOf = t => Math.max(0, Math.round((t - r.t0) * fps));
  // best 15 s window
  const win = r.windows?.[15] || r.windows?.[7];
  if (win) { const a = xOf(win.start, w), b = xOf(win.end, w); g.fillStyle = 'rgba(238,233,224,.07)'; g.fillRect(a, TL.mainTop, b - a, TL.mainH); g.strokeStyle = 'rgba(238,233,224,.5)'; g.lineWidth = 1; g.beginPath(); g.moveTo(a, TL.mainTop + 6); g.lineTo(a, TL.mainTop); g.lineTo(b, TL.mainTop); g.lineTo(b, TL.mainTop + 6); g.stroke(); }
  // sections band
  r.sections.forEach(s => {
    const a = Math.max(0, xOf(s.start, w)), b = Math.min(w, xOf(s.end, w)); if (b <= 0 || a >= w) return;
    g.fillStyle = s.label === 'Peak' ? 'rgba(255,91,46,.85)' : `rgba(238,233,224,${0.06 + 0.22 * s.level})`;
    g.fillRect(a + 1, TL.secTop, b - a - 2, TL.secH);
    if (b - a > 44) { g.fillStyle = s.label === 'Peak' ? '#140a06' : '#eee9e0'; g.fillText(s.label, a + 7, TL.secTop + TL.secH / 2 + 1); }
  });
  // waveform (max per pixel column) + energy line
  g.fillStyle = '#3d3a35';
  for (let x = 0; x < w; x++) {
    const fa = fOf(tOf(x, w)), fb = Math.max(fa + 1, fOf(tOf(x + 1, w)));
    if (fa >= cv.wave.length) break;
    let m = 0; for (let f = fa; f < Math.min(fb, cv.wave.length); f++) if (cv.wave[f] > m) m = cv.wave[f];
    const hh = Math.max(0.5, Math.sqrt(m) * amp); g.fillRect(x, mid - hh, 1, hh * 2);
  }
  g.strokeStyle = 'rgba(238,233,224,.85)'; g.lineWidth = 1.5; g.beginPath();
  for (let x = 0; x < w; x += 2) { const f = fOf(tOf(x, w)); if (f >= cv.energy.length) break; const y = TL.mainTop + TL.mainH - 6 - cv.energy[f] * (TL.mainH - 14); x ? g.lineTo(x, y) : g.moveTo(x, y); }
  g.stroke();
  // beat grid
  // calm stretches: where the neural tracker hears no pulse, shade the area and say so
  if (cv.activity) {
    let runA = -1;
    const flush = (a, b) => { if (b - a < 3) return; g.fillStyle = 'rgba(15,14,13,.55)'; g.fillRect(a, TL.mainTop, b - a, TL.laneTop + 72 - TL.mainTop); if (b - a > 70) { g.fillStyle = '#8d877c'; g.fillText('no beat', a + 6, TL.mainTop + TL.mainH - 16); } };
    for (let x = 0; x <= w; x++) {
      const f = fOf(tOf(x, w)), calm = f < cv.activity.length && cv.activity[f] < 0.2 && cv.energy[f] > 0.02;
      if (calm && runA < 0) runA = x; if ((!calm || x === w) && runA >= 0) { flush(runA, x); runA = -1; }
    }
  }
  const det = r.detected || [0, r.duration], beatPx = r.beatDur * ppS, M = r.meter || 4;
  r.beats.forEach((b, i) => {
    if (b < t0 || b > t1) return;
    const isBar = barBeat(r, i).beat === 1, conf = r.beatConf ? r.beatConf[i] : (b >= det[0] - 0.05 && b <= det[1] + 0.05 ? 1 : 0.3);
    if (!isBar && beatPx < 5) return; if (isBar && beatPx * M < 4) return;
    const x = Math.round(xOf(b, w)) + 0.5, k = 0.3 + 0.7 * Math.min(1, conf / 0.6);
    g.strokeStyle = isBar ? `rgba(238,233,224,${(0.34 * k).toFixed(3)})` : `rgba(238,233,224,${(0.11 * k).toFixed(3)})`;
    g.lineWidth = 1; g.beginPath(); g.moveTo(x, TL.mainTop); g.lineTo(x, TL.laneTop + 72); g.stroke();
  });
  // lanes
  const lanes = [['kick', '#e2b04a'], ['snare', '#8db4cc'], ['hat', '#7d786f']];
  lanes.forEach(([k, col], li) => {
    const y = TL.laneTop + 10 + li * 22;
    g.fillStyle = '#6f6a62'; g.fillText(k === 'snare' ? 'snare' : k, 4, y);
    g.fillStyle = col;
    r.hits[k].forEach(hh => { if (hh.t < t0 || hh.t > t1) return; const x = xOf(hh.t, w); if (x < 40) return; g.beginPath(); g.arc(x, y, Math.min(3.2, 1.6 + hh.s * 0.25), 0, 7); g.fill(); });
  });
  // silence gaps
  g.fillStyle = 'rgba(141,180,204,.35)';
  r.gaps.forEach(gp => { const a = xOf(gp.t, w), b = xOf(gp.t + gp.dur, w); g.fillRect(a, TL.mainTop + TL.mainH - 3, Math.max(2, b - a), 3); });
  // big hits
  g.fillStyle = '#ff5b2e';
  r.hits.accent.forEach(a => { if (a.t < t0 || a.t > t1) return; const x = xOf(a.t, w); g.beginPath(); g.moveTo(x - 4, TL.mainTop + 1); g.lineTo(x + 4, TL.mainTop + 1); g.lineTo(x, TL.mainTop + 8); g.fill(); });
  // drops
  r.drops.forEach(d => { if (d.t < t0 || d.t > t1) return; const x = Math.round(xOf(d.t, w)); g.fillStyle = '#ff5b2e'; g.fillRect(x - 1, TL.secTop + TL.secH + 2, 2, TL.laneTop + 72 - TL.secH); g.font = '800 11px Satoshi, system-ui, sans-serif'; g.fillText('DROP', x + 5, TL.mainTop + 22); g.font = '600 11px Satoshi, system-ui, sans-serif'; });
  // unheard (live capture)
  if (S.total && r.duration < S.total - 0.5) {
    const a = Math.max(0, xOf(r.duration, w));
    g.fillStyle = 'rgba(15,14,13,.7)'; g.fillRect(a, 0, w - a, TL.laneTop + 72);
    g.strokeStyle = 'rgba(238,233,224,.08)'; g.lineWidth = 1;
    for (let x = a - 300; x < w; x += 10) { g.beginPath(); g.moveTo(x, TL.laneTop + 72); g.lineTo(x + 80, 0); g.stroke(); }
    if (w - a > 90) { g.fillStyle = '#a39d92'; g.fillText('not heard yet', a + 10, TL.mainTop + TL.mainH / 2); }
  }
  // axis
  const steps = [1, 2, 5, 10, 15, 30, 60, 120]; let step = steps.find(s => s * ppS >= 70) || 240;
  g.fillStyle = '#6f6a62'; g.textAlign = 'left';
  for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) { const x = xOf(t, w); g.fillRect(x, TL.axisY - 10, 1, 4); g.fillText(fmt(t, 0), x + 3, TL.axisY - 2); }
  S.staticLayer = layer;
}

function drawTimelineFrame(t) {
  if ($('resultsTop').hidden || !tl.clientWidth) return;
  const { w, h, dpr } = tlSize();
  if (!S.staticLayer) drawStatic();
  const g = tl.getContext('2d');
  g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, tl.width, tl.height); g.drawImage(S.staticLayer, 0, 0);
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (S.loop) { const a = xOf(S.loop.start, w), b = xOf(S.loop.end, w); g.strokeStyle = '#ff5b2e'; g.lineWidth = 1; g.strokeRect(a + 0.5, TL.mainTop + 0.5, b - a - 1, TL.mainH - 1); }
  const x = xOf(t, w);
  if (x >= 0 && x <= w) { g.fillStyle = '#ff5b2e'; g.fillRect(x - 1, 0, 2, TL.laneTop + 72); }
  if (S.hover !== null && S.result?.ok) {
    const ht = tOf(S.hover, w), r = S.result, bi = upperIdx(r.beats, ht), bb = barBeat(r, bi);
    const label = `${fmt(ht, 2)}  ·  bar ${Math.max(0, bb.bar)}.${bb.beat}${r.beatConf && bi >= 0 ? `  ·  beat ${Math.round(r.beatConf[bi] * 100)}% sure` : ''}`;
    g.strokeStyle = 'rgba(238,233,224,.5)'; g.lineWidth = 1; g.beginPath(); g.moveTo(S.hover + 0.5, TL.mainTop); g.lineTo(S.hover + 0.5, TL.laneTop + 72); g.stroke();
    g.font = '700 12px Satoshi, system-ui, sans-serif'; const tw = g.measureText(label).width + 14, bx = Math.min(w - tw, S.hover + 8);
    g.fillStyle = 'rgba(15,14,13,.92)'; g.fillRect(bx, TL.mainTop + 22, tw, 22); g.fillStyle = '#eee9e0'; g.textBaseline = 'middle'; g.fillText(label, bx + 7, TL.mainTop + 33);
  }
}

function setView(t0, t1) {
  const total = Math.max(1, S.total || S.result?.duration || 1), minSpan = 2;
  let span = Math.max(minSpan, Math.min(total, t1 - t0));
  t0 = Math.max(0, Math.min(total - span, t0)); S.view = { t0, t1: t0 + span }; S.staticLayer = null;
}
function zoomAt(factor, x) { const w = tl.clientWidth, c = tOf(x ?? w / 2, w), { t0, t1 } = S.view; setView(c - (c - t0) * factor, c + (t1 - c) * factor); }
tl.addEventListener('wheel', e => { e.preventDefault(); const rect = tl.getBoundingClientRect(); if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) { const dt = ((e.deltaX || e.deltaY) / rect.width) * (S.view.t1 - S.view.t0); setView(S.view.t0 + dt, S.view.t1 + dt); } else zoomAt(Math.pow(1.0015, e.deltaY), e.clientX - rect.left); }, { passive: false });
let drag = null;
tl.addEventListener('pointerdown', e => { drag = { x: e.clientX, v: { ...S.view }, moved: false }; tl.setPointerCapture(e.pointerId); });
tl.addEventListener('pointermove', e => {
  const rect = tl.getBoundingClientRect(); S.hover = e.clientX - rect.left;
  if (!drag) return; const dx = e.clientX - drag.x; if (Math.abs(dx) > 4) drag.moved = true;
  if (drag.moved) { const dt = (-dx / rect.width) * (drag.v.t1 - drag.v.t0); setView(drag.v.t0 + dt, drag.v.t1 + dt); }
});
tl.addEventListener('pointerup', e => { const rect = tl.getBoundingClientRect(); if (drag && !drag.moved) { S.loop = null; seek(tOf(e.clientX - rect.left, rect.width)); } drag = null; });
tl.addEventListener('pointerleave', () => { S.hover = null; });
$('zoomIn').addEventListener('click', () => zoomAt(0.5)); $('zoomOut').addEventListener('click', () => zoomAt(2)); $('zoomFit').addEventListener('click', () => setView(0, S.total || S.result?.duration || 1));
addEventListener('resize', () => { S.staticLayer = null; drawSmallCharts(); });

// ---------- readouts ----------
function showRes(on) { $('results').hidden = !on; $('resultsTop').hidden = !on; }
function setHeadline(h, p) { $('headline').textContent = h; $('summary').textContent = p; }
const bar = (v, hot) => `<div class="bar${hot ? ' hot' : ''}"><span style="width:${Math.max(2, Math.min(100, v))}%"></span></div>`;

function showResult(r, final) {
  if (!r.ok) { setHeadline('Not enough audio yet', r.reason); return; }
  const firstTime = !S.result;
  S.result = r;
  if (!S.total) S.total = r.duration;
  if (firstTime || S.view.t1 <= 1) S.view = { t0: 0, t1: Math.max(r.duration, S.total || 0) };
  buildCuts(); S.cutIdx = -2; S.staticLayer = null;
  showRes(true);
  const s = r.scores, m = r.metrics, beat = r.beatDur;
  $('overall').textContent = s.overall;
  $('bpm').textContent = Math.round(r.bpm);
  const alt = r.bpm < 100 ? r.bpm * 2 : r.bpm > 140 ? r.bpm / 2 : null;
  $('bpmAlt').textContent = alt ? `BPM · or ${Math.round(alt)}` : 'BPM';
  const fits = Object.entries(r.fits).sort((a, b) => b[1] - a[1]);
  const head = s.overall >= 75 ? 'Strong edit track' : s.overall >= 55 ? 'Cuts well with the right pacing' : s.overall >= 35 ? 'Usable for the right section' : 'Better as a background bed';
  const w15 = r.windows[15] || r.windows[7];
  const bigDrop = r.drops.slice().sort((a, b) => b.jumpDb + 0.3 * b.lowJumpDb - (a.jumpDb + 0.3 * a.lowJumpDb))[0];
  const parts = [
    `${r.bpm} BPM on a ${s.pulse >= 75 ? 'clear, steady' : s.pulse >= 55 ? 'mostly steady' : 'loose'} grid, so a cut on every beat is ${beat.toFixed(2)} s (${(beat * 30).toFixed(1)} frames at 30 fps).`,
    bigDrop ? `The biggest drop lands at ${fmt(bigDrop.t)}.` : 'There is no clear drop, so the lift has to come from your cuts.',
    w15 ? `For a ${Math.round(w15.len)} second reel, start at ${fmt(w15.start)}${w15.dropAt !== null && Math.abs(w15.dropAt - w15.start) < 0.2 ? ', right on the drop' : ''}.` : '',
    `Best use: ${FIT_NAMES[fits[0][0]].toLowerCase()}.`,
  ];
  setHeadline(S.cap ? `First read: ${head.toLowerCase()}` : head, parts.filter(Boolean).join(' '));

  const pct = v => Math.round(v * 100);
  const rows = [
    ['Punch', s.punch, `How hard the hits land. Each beat lifts the spectrum by ${m.rawRise} on average and the bass by ${m.rawLowRise}; soft piano sits near 0.03 and a hard club kick near 0.6.`],
    ['Pulse', s.pulse, m.confMean !== null && m.confMean !== undefined
      ? `How clearly the beat can be felt. The beat model is sure of ${pct(m.strongShare)}% of beats${m.calmShare > 0.05 ? `, and ${pct(m.calmShare)}% of the song has no beat to cut on` : ''}. The tempo ${r.tempoDrift < 3 ? 'holds steady' : `moves by about ${r.tempoDrift} BPM`}.`
      : `How easy the grid is to cut to. ${pct(m.hitRatio)}% of beats carry an audible hit and the tempo ${r.tempoDrift < 3 ? 'holds steady' : `moves by about ${r.tempoDrift} BPM`}.`],
    ...(s.intensity !== undefined ? [['Intensity', s.intensity, `How hard and edgy the music feels to the mood model: aggressive ${pct(r.model.feel.mood.aggressive)}%, relaxed ${pct(r.model.feel.mood.relaxed)}%. It is the strongest single sign of a song editors reach for.`]] : []),
    ['Payoff', s.payoff, r.drops.length ? `${r.drops.length} drop${r.drops.length > 1 ? 's' : ''}, the strongest lifting the level ${m.strongestDrop} dB. Loud and quiet parts sit ${m.lra} dB apart.` : `No drop found. Loud and quiet parts sit ${m.lra} dB apart.`],
    ['Drive', s.drive, `${m.onsetRate} hits a second with a ${r.grid.name} feel. Higher means more material to cut on.`],
    ['Hook', s.hook, w15 ? `The best window opens at ${pct(w15.firstSec)}% of peak energy${w15.startHit > 0.5 ? ' on a clear hit' : ''}${w15.dropAt !== null && Math.abs(w15.dropAt - w15.start) < 0.2 ? ' and right on a drop' : ''}. The first second decides the scroll.` : 'The song is too short to pick a window.'],
  ];
  const CN = { intensity: 'intensity', edge: 'an unusual, edgy sound', drops: 'the number of drops', dropSize: 'drop size', density: 'a dense, busy texture', sustain: 'hits that hold their weight', hook: 'a strong opening window' };
  const cs = Object.entries(r.editPower?.contrib || {}).filter(([, v]) => Math.abs(v) >= 0.15).sort((a, b) => b[1] - a[1]);
  const CNN = { intensity: 'a soft, relaxed mood', edge: 'a smooth, familiar sound', drops: 'few drops', dropSize: 'small drops', density: 'a sparse texture', sustain: 'hits that fade fast', hook: 'a weak opening window' };
  const ups = cs.filter(([, v]) => v > 0).slice(0, 2).map(([k]) => CN[k]), downs = cs.filter(([, v]) => v < 0).slice(-2).reverse().map(([k]) => CNN[k]);
  const powerWhy = `Edit power comes from a small model fitted on songs picked for edits against calm library tracks${r.editPower?.head === 'dsp' ? ' (quick-read version, less sure)' : ''}.${ups.length ? ` Lifted by ${ups.join(' and ')}.` : ''}${downs.length ? ` Held back by ${downs.join(' and ')}.` : ''}`;
  $('scores').innerHTML = `<p class="why powerWhy">${powerWhy}</p>` + rows.map(([n, v, why]) => `<div class="score"><span class="name">${n}</span>${bar(v, v >= 75)}<span class="val">${v}</span><p class="why">${why}</p></div>`).join('');
  $('fits').innerHTML = fits.map(([k, v], i) => `<div class="score"><span class="name" style="grid-column:1/3">${FIT_NAMES[k]}</span><span class="val">${v}</span><div style="grid-column:1/-1">${bar(v, i === 0)}</div></div>`).join('');

  $('windows').innerHTML = [7, 15, 30].map(L => {
    const wv = r.windows[L]; if (!wv) return '';
    const why = [];
    if (wv.dropAt !== null) why.push(Math.abs(wv.dropAt - wv.start) < 0.2 ? 'starts on the drop' : `drop lands ${(wv.dropAt - wv.start).toFixed(1)} s in`);
    if (wv.startHit > 0.5) why.push('opens on a hit');
    why.push(`${wv.accents} big hit${wv.accents === 1 ? '' : 's'}`);
    if (wv.bars) why.push(`${wv.bars} bars, ends on a bar line`);
    return `<div class="win"><span class="len">${L}s</span><span class="range">${fmt(wv.start)} to ${fmt(wv.end)}</span><p class="why">${why.join(', ')}</p><button class="btn ghost small" type="button" data-win="${L}">Preview</button></div>`;
  }).join('');

  const fr = (v, f) => (v * f).toFixed(1);
  const sp = [['Beat', beat], ['Half beat', beat / 2], ['Bar (4 beats)', beat * 4], ['2 bars', beat * 8]];
  $('spacing').innerHTML = `<tr><th></th><th>seconds</th><th>30 fps</th><th style="text-align:right">60 fps</th></tr>` + sp.map(([n, v]) => `<tr><td>${n}</td><td>${secs(v)}</td><td>${fr(v, 30)} f</td><td>${fr(v, 60)} f</td></tr>`).join('')
    + `<tr><td>First beat</td><td colspan="3">${m.firstBeat !== null ? fmt(m.firstBeat, 2) : '--'}</td></tr><tr><td>Tempo candidates</td><td colspan="3">${r.bpmAlts.join(', ')} BPM</td></tr>`;
  $('gridName').textContent = `Most hits are spaced like ${r.grid.name}. Peaks at 1/4 mean 16th notes, 1/3 and 2/3 mean triplets or swing.`;
  $('tempoNote').textContent = r.tempoDrift < 3 ? 'Flat line: the grid holds, so a fixed cut rate stays on beat.' : `The tempo moves by about ${r.tempoDrift} BPM. Re-check cuts against the grid in the later parts.`;

  const M = r.meter || 4, every = k => (k === 0.5 ? 'half beat' : k === 1 ? 'beat' : k === M ? 'bar' : k % M === 0 ? `${k / M} bars` : k + ' beats');
  $('sections').innerHTML = `<tr><th>Start</th><th>Part</th><th>Cut every</th><th>Shot length</th></tr>` + r.cutPlan.map(p => `<tr><td><button class="jump" data-t="${p.start}">${fmt(p.start, 0)}</button></td><td><span class="pill${p.label === 'Peak' ? ' peak' : ''}">${p.label}</span>${p.calm ? ' <span class="pill calm">no beat</span>' : ''}</td><td>${every(p.everyBeats)}${p.calm ? ', or on big hits' : ''}</td><td>${secs(p.shotSec)}</td></tr>`).join('');
  renderFeel(r);
  const moments = [
    ...r.drops.map(d => [d.t, 'Drop', `level +${d.jumpDb} dB, bass +${d.lowJumpDb} dB${d.build ? ', riser before it' : ''}${d.gapBefore ? ', pause right before' : ''}`]),
    ...r.hits.accent.slice().sort((a, b) => b.s - a.s).slice(0, 8).map(a => [a.t, 'Big hit', 'punch in or flash here']),
    ...r.gaps.slice(0, 6).map(gp => [gp.t, 'Gap', `${gp.dur} s of near silence, room for a freeze or a text beat`]),
  ].sort((a, b) => a[0] - b[0]);
  $('moments').innerHTML = moments.length ? moments.map(([t, k, d]) => `<tr><td><button class="jump" data-t="${t}">${fmt(t)}</button></td><td>${k}</td><td>${d}</td></tr>`).join('') : '<tr><td>None found</td></tr>';
  drawSmallCharts();
  if (final && S.mode === 'yt') $('capStatus').textContent += ` Analysed ${fmt(r.duration, 0)} in ${r.ms} ms.`;
}

const MOOD_NAMES = { aggressive: 'Aggressive', party: 'Party', danceable: 'Danceable', relaxed: 'Relaxed', engagement: 'Holds attention', approachability: 'Easy on the ear' };
function renderFeel(r) {
  const el = $('feel'); if (!el) return;
  const f = r.model?.feel;
  if (!f) { el.innerHTML = `<p class="fine">${r.model?.used === false && !r.mlError ? 'Waiting for the neural pass…' : 'The feel needs the neural pass, which did not run.'}</p>`; return; }
  const styles = f.styles.slice(0, 3).map(([n, p]) => `<span class="pill">${n.split('---').pop()}</span>`).join(' ');
  el.innerHTML = `<p class="styles">${styles}</p>` + Object.entries(MOOD_NAMES).filter(([k]) => f.mood[k] !== undefined).map(([k, n]) => {
    const v = Math.round(f.mood[k] * 100);
    return `<div class="score"><span class="name">${n}</span>${bar(v, false)}<span class="val">${v}</span></div>`;
  }).join('') + `<p class="fine">From Discogs-EffNet (MTG, Essentia) and its mood heads, averaged over the song. Styles are the top Discogs labels.</p>`;
}

document.addEventListener('click', e => {
  const j = e.target.closest('[data-t]'); if (j) { S.loop = null; seek(+j.dataset.t); if (!S.playing) play(+j.dataset.t); return; }
  const wbtn = e.target.closest('[data-win]'); if (wbtn && S.result) { const wv = S.result.windows[wbtn.dataset.win]; S.loop = { start: wv.start, end: wv.end }; setView(Math.max(0, wv.start - 2), wv.end + 2); play(wv.start); return; }
  const ex = e.target.closest('[data-export]'); if (ex && S.result) doExport(ex.dataset.export);
});

function drawSmallCharts() {
  const r = S.result; if (!r?.ok) return;
  // inter-onset histogram
  const c = $('ioi'), dpr = Math.min(2, devicePixelRatio || 1), w = c.clientWidth, h = c.clientHeight;
  c.width = w * dpr; c.height = h * dpr; const g = c.getContext('2d'); g.scale(dpr, dpr);
  const hist = r.ioiHist, max = Math.max(1, ...hist), bw = (w - 10) / hist.length, base = h - 22;
  hist.forEach((v, i) => { const bh = (v / max) * (base - 8); g.fillStyle = [6, 8, 12, 16, 24].includes(i) ? '#ff5b2e' : '#4a4640'; g.fillRect(5 + i * bw + 1, base - bh, Math.max(1, bw - 2), bh); });
  g.fillStyle = '#a39d92'; g.font = '600 11px Satoshi, system-ui, sans-serif'; g.textAlign = 'center';
  [[6, '1/4'], [8, '1/3'], [12, '1/2'], [16, '2/3'], [24, '1 beat'], [48, '2']].forEach(([i, l]) => g.fillText(l, 5 + i * bw + bw / 2, h - 6));
  // tempo map
  const c2 = $('tempo'), w2 = c2.clientWidth, h2 = c2.clientHeight; c2.width = w2 * dpr; c2.height = h2 * dpr;
  const g2 = c2.getContext('2d'); g2.scale(dpr, dpr);
  const pts = r.tempoMap; if (pts.length < 2) return;
  const lo = Math.min(...pts.map(p => p.bpm)) - 4, hi = Math.max(...pts.map(p => p.bpm)) + 4, T = r.duration;
  g2.strokeStyle = '#2c2a26'; g2.lineWidth = 1; g2.beginPath(); const yb = h2 - 20 - ((r.bpm - lo) / (hi - lo)) * (h2 - 30); g2.moveTo(30, yb); g2.lineTo(w2, yb); g2.stroke();
  g2.strokeStyle = '#eee9e0'; g2.lineWidth = 1.5; g2.beginPath();
  pts.forEach((p, i) => { const x = 30 + (p.t / T) * (w2 - 32), y = h2 - 20 - ((p.bpm - lo) / (hi - lo)) * (h2 - 30); i ? g2.lineTo(x, y) : g2.moveTo(x, y); });
  g2.stroke();
  g2.fillStyle = '#a39d92'; g2.font = '600 11px Satoshi, system-ui, sans-serif'; g2.textAlign = 'left';
  g2.fillText(Math.round(hi), 0, 12); g2.fillText(Math.round(lo), 0, h2 - 22); g2.fillText('0:00', 30, h2 - 4); g2.textAlign = 'right'; g2.fillText(fmt(T, 0), w2, h2 - 4);
}

// ---------- exports ----------
function download(name, text, type) { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000); }
const slug = () => (S.title || 'song').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 50) || 'song';
function markerRows() {
  const r = S.result, rows = [];
  r.beats.forEach((b, i) => { if (r.beatConf && r.beatConf[i] < 0.2) return; const { bar, beat: bt } = barBeat(r, i); rows.push([bt === 1 ? 'bar' : 'beat', b, `bar ${bar} beat ${bt}`]); });
  r.drops.forEach(d => rows.push(['drop', d.t, `drop +${d.jumpDb} dB`]));
  r.sections.forEach(s => rows.push(['section', s.start, s.label]));
  r.hits.accent.forEach(a => rows.push(['big_hit', a.t, 'big hit']));
  r.gaps.forEach(g => rows.push(['gap', g.t, `gap ${g.dur} s`]));
  ['kick', 'snare', 'hat'].forEach(k => r.hits[k].forEach(h => rows.push([k, h.t, k])));
  return rows.sort((a, b) => a[1] - b[1]);
}
function doExport(kind) {
  const r = S.result, name = slug();
  if (kind === 'csv') download(`${name}-markers.csv`, 'type,time_s,timecode_30fps,label\n' + markerRows().map(([k, t, l]) => `${k},${t.toFixed(3)},${tc30(t)},"${l}"`).join('\n'), 'text/csv');
  if (kind === 'audacity') download(`${name}-labels.txt`, markerRows().filter(x => ['bar', 'drop', 'section', 'big_hit', 'gap'].includes(x[0])).map(([k, t, l]) => `${t.toFixed(3)}\t${t.toFixed(3)}\t${l}`).join('\n'), 'text/plain');
  if (kind === 'json') { const { curves, ...rest } = r; void curves; download(`${name}-cutgrid.json`, JSON.stringify({ title: S.title, source: S.mode === 'yt' ? `https://youtu.be/${S.videoId}` : 'file', ...rest }, null, 1), 'application/json'); }
  $('exportStatus').textContent = 'Saved.';
}
$('copyCuts').addEventListener('click', async () => {
  if (!S.cuts.length) return;
  const text = S.cuts.map((t, i) => `${pad(i + 1, 3)}  ${tc30(t)}  ${t.toFixed(3)} s`).join('\n');
  try { await navigator.clipboard.writeText(text); $('exportStatus').textContent = `${S.cuts.length} cuts copied.`; } catch { $('exportStatus').textContent = 'Copy failed. Use Markers CSV instead.'; }
});

// ---------- start ----------
const hashV = new URLSearchParams(location.hash.slice(1)).get('v');
if (hashV && /^[\w-]{11}$/.test(hashV)) { $('link').value = `https://youtu.be/${hashV}`; loadYouTube(hashV); }

// test hook: analyse raw samples without a file picker (used for automated checks)
window.cutgrid = {
  async analyzeSamples(samples, sr, title = 'Test signal') {
    resetStage('Audio file', title); S.mode = 'file'; S.title = title; showRes(true);
    S.ctx = S.ctx || new AudioContext();
    const buf = S.ctx.createBuffer(1, samples.length, sr); buf.copyToChannel(samples, 0);
    await useBuffer(buf); return S.result;
  },
  state: S,
};
