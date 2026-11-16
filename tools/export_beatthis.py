"""Fetch the Beat This! small0 ONNX model for CutGrid, dump the mel filterbank, and make a reference run.

py tools/export_beatthis.py [reference.mp3 ...]

Writes models/beatthis-small0.onnx (MIT, converted from CPJKU/beat_this by ashudesai/songbird-models)
and models/beatthis-mel.bin (513 x 128 float32, row major, torchaudio slaney mel 30-11000 Hz).
For each reference file it writes tools/ref/<name>.json with spect samples, logits and beats from the
PyTorch front end + onnxruntime, so tests/run.mjs can check the JS front end matches.
"""
import json
import os
import subprocess
import sys
import urllib.request

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from beat_this.preprocessing import LogMelSpect
from beat_this.inference import split_predict_aggregate
from beat_this.model.postprocessor import Postprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "models")
REF = os.path.join(HERE, "ref")
os.makedirs(OUT, exist_ok=True)
os.makedirs(REF, exist_ok=True)

onnx_path = os.path.join(OUT, os.environ.get("BT_MODEL", "beatthis-small0.onnx"))
if not os.path.exists(onnx_path):
    urllib.request.urlretrieve("https://huggingface.co/ashudesai/songbird-models/resolve/main/small0.onnx", onnx_path)
print("onnx", os.path.getsize(onnx_path))

lm = LogMelSpect()
fb = lm.spect_class.mel_scale.fb.numpy().astype(np.float32)  # (513, 128)
fb.tofile(os.path.join(OUT, "beatthis-mel.bin"))
print("fb", fb.shape)

sess = ort.InferenceSession(onnx_path)
iname = sess.get_inputs()[0].name


def model(chunk):
    b, d = sess.run(None, {iname: chunk.numpy().astype(np.float32)})
    return {"beat": torch.from_numpy(b), "downbeat": torch.from_numpy(d)}


for ref in sys.argv[1:]:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", ref, "-ac", "1", "-ar", "22050", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    x = torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy())
    spect = lm(x)
    pred = split_predict_aggregate(spect, 1500, 6, "keep_first", model)
    beats, downs = Postprocessor("minimal")(pred["beat"], pred["downbeat"])
    s = spect.numpy()
    name = os.path.splitext(os.path.basename(ref))[0]
    json.dump({
        "frames": int(s.shape[0]),
        "spectRows": {str(i): s[i].tolist() for i in (100, 1000, 3000) if i < len(s)},
        "spectMean": float(s.mean()),
        "beatLogits": pred["beat"].numpy()[:3000].tolist(),
        "beats": beats.tolist(), "downbeats": downs.tolist(),
    }, open(os.path.join(REF, name + ".json"), "w"))
    d = np.diff(beats)
    print(name, "beats", len(beats), "downbeats", len(downs), "bpm~", round(60 / np.median(d), 1))
