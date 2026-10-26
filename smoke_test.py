"""Check a running backend: POST synthetic strokes to /forecast and print the shapes.

Start the server first: uv run backend/server.py
"""
import json
import sys
import urllib.request

import numpy as np

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

t = np.arange(0, 2, 1 / 60)  # 2 s at 60 Hz, already uniform like the frontend sends
strokes = {
    "arc": (0.5 + 0.3 * np.cos(2.5 * t), 0.5 + 0.3 * np.sin(2.5 * t)),
    "sine": (0.1 + 0.3 * t, 0.5 + 0.15 * np.sin(6 * t)),
}

for name, (x, y) in strokes.items():
    req = urllib.request.Request(
        f"{URL}/forecast",
        data=json.dumps({"x": x.tolist(), "y": y.tolist(), "horizon": 120}).encode(),
        headers={"Content-Type": "application/json"},
    )
    body = json.load(urllib.request.urlopen(req))
    shapes = {q: np.shape(v) for q, v in body["quantiles"].items()}
    print(f"{name}: {shapes}, {body['ms']:.0f} ms")
