# CutGrid

An editor's preview of a song. Paste a YouTube link or drop an audio file and CutGrid shows the BPM, the beat and bar grid, kicks, snares and hats, drops, sections, the best 7, 15 and 30 second reel windows, and a 9:16 cut preview that plays placeholder shots on the beat so you can feel the pacing before the song goes near your timeline.

Live: https://sahol3.github.io/cutgrid/

## How it gets the audio

- **Audio file.** Decoded in the browser and analysed there. Nothing is uploaded.
- **YouTube link.** A static page cannot download YouTube audio, so CutGrid embeds the player and listens to it through tab audio sharing (Chrome or Edge on desktop: choose this tab and keep "Share tab audio" on). The readout fills in while the song plays, and you can stop early.

## Two reads

1. **Quick read (DSP, about a second).** `engine.js` runs a 1024-point STFT with an 11.6 ms hop, takes SuperFlux-style spectral flux in bass, mid and treble bands, finds the tempo from the autocorrelation of the onset curve and tracks beats with dynamic programming (Ellis 2007).
2. **Neural read (`ml.js`, onnxruntime-web, WebGPU or WASM).**
   - [Beat This!](https://github.com/CPJKU/beat_this) small0 (Foscarin, Schlüter and Widmer, ISMIR 2024; MIT; ONNX conversion from [ashudesai/songbird-models](https://huggingface.co/ashudesai/songbird-models)) gives beat and downbeat probability at 50 fps. The grid follows it, every beat carries a confidence, bar lines come from a Viterbi pass over bar position (3 or 4 beats), and stretches with no pulse are marked as calm instead of being filled with beats.
   - [Discogs-EffNet](https://essentia.upf.edu/models.html) with the mood_aggressive, mood_party, mood_relaxed, danceability, engagement and approachability heads (MTG, Universitat Pompeu Fabra; CC BY-NC-SA 4.0) gives the style and mood readout.
   - The front ends reproduce torchaudio `LogMelSpect` and Essentia `TensorflowInputMusiCNN`; `tests/ml-check.mjs` compares the JS spectrogram and beat probabilities with the PyTorch reference (max difference 1e-5).

Models live in `models/` (about 32 MB) and download once per browser. The Essentia models are licensed for non-commercial use, which is how CutGrid uses them.

## Scores

Scores (0 to 100) are explained in the page. Edit power weights Punch, Pulse, Payoff, Drive and Hook; the weights and the model features were checked against a labelled set of songs (see `tests/corpus.mjs`).

## Exports

Markers CSV (seconds and 30 fps timecode), Audacity label track, full JSON, and the cut list for the chosen cadence. Beats the model marks as calm are left out of the markers.

## Run locally

Any static server works, for example `py -m http.server 8000`.

- `npm i` (dev only: onnxruntime-node and onnxruntime-web 1.22.0)
- `node tests/run.mjs` runs the click-track timing check; `node tests/run.mjs song.mp3` prints the summary for real files (needs ffmpeg).
- `node tests/ml-check.mjs song.mp3` runs both networks in Node; `tests/ml-dump.mjs` caches their output for `tests/corpus.mjs`.
- `node tests/browser.mjs [url] [song.f32]` drives the page in headless Chromium (playwright-core).
- `py tools/export_beatthis.py song.mp3` fetches the beat model and writes the PyTorch reference used by `ml-check`.
