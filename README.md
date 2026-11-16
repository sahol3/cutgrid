# CutGrid

An editor's preview of a song. Paste a YouTube link or drop an audio file and CutGrid shows the BPM, the beat and bar grid, kicks, snares and hats, drops, sections, the best 7, 15 and 30 second reel windows, and a 9:16 cut preview that plays placeholder shots on the beat so you can feel the pacing before the song goes near your timeline.

Live: https://kronx.me/cutgrid/

## How it gets the audio

- **Audio file.** Decoded in the browser and analysed in a few seconds. Nothing is uploaded.
- **YouTube link.** A static page cannot download YouTube audio, so CutGrid embeds the player and listens to it through tab audio sharing (Chrome or Edge on desktop: choose this tab and keep "Share tab audio" on). The readout fills in while the song plays, and you can stop early.

## What it measures

The engine (`engine.js`, no dependencies) runs a 1024-point STFT with an 11.6 ms hop, takes the SuperFlux-style spectral flux in bass, mid and treble bands, finds the tempo from the autocorrelation of the onset curve, tracks beats with dynamic programming (Ellis 2007), and picks bar starts from where the kicks and snares fall. Sections come from a checkerboard novelty curve over bar-level tone and loudness. A drop is a bar where the level and the low end both jump, snapped to the first strong kick.

Scores (0 to 100) are heuristics tuned on EDM, phonk, lofi, trap, ambient techno, piano and film score:

| Score | What it reads |
|---|---|
| Punch | How far the spectrum and the bass leap on each beat |
| Pulse | Share of beats with an audible hit, plus tempo clarity and drift |
| Payoff | Strongest drop and the dynamic range |
| Drive | Hits per second and kick/snare density |
| Hook | Opening energy and hit of the best 15 second window |

Edit power weights them 30, 20, 20, 15 and 15.

## Exports

Markers CSV (seconds and 30 fps timecode), Audacity label track, full JSON, and the cut list for the chosen cadence.

## Run locally

Any static server works, for example `py -m http.server 8000`. Tests: `node tests/run.mjs` runs the click-track timing check; `node tests/run.mjs song.mp3` prints the summary for real files (needs ffmpeg).
