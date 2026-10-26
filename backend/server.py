"""Inference server: POST /forecast with a uniform (x, y) context, get quantile paths back.

Also serves the static frontend so one process is enough for local use.
"""
import argparse
import json
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from timestral import TimestralForecaster

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
MAX_HORIZON = 1024


class Predictor:
    def __init__(self, model):
        self.model = model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fc = TimestralForecaster.from_pretrained(model, device=device)
        self.lock = threading.Lock()  # one forecast at a time

    def predict(self, x, y, horizon):
        """Uniformly sampled x, y of length T -> {quantile: [[x...], [y...]]} of length horizon."""
        ctx = np.asarray([[x, y]], dtype=np.float32)  # (1, 2, T)
        with self.lock:
            f = self.fc.forecast(ctx, prediction_length=horizon)
        # All 21 model levels, keyed like "0.05"; rounding hides float noise such as 0.15000000000000002.
        return {str(round(q, 4)): f.quantile(q)[0].cpu().numpy().tolist() for q in f.quantile_levels}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, predictor, **kwargs):
        self.predictor = predictor
        super().__init__(*args, directory=str(FRONTEND), **kwargs)

    def do_GET(self):
        if self.path == "/info":
            return self.send_json(200, {"model": self.predictor.model})
        super().do_GET()

    def do_POST(self):
        if self.path != "/forecast":
            return self.send_json(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            x, y = body["x"], body["y"]
            horizon = int(body.get("horizon", 120))
            if len(x) != len(y) or len(x) < 2 or not 1 <= horizon <= MAX_HORIZON:
                raise ValueError("x and y need equal length >= 2, horizon in 1..1024")
        except KeyError as e:
            return self.send_json(400, {"error": f"missing field {e}"})
        except (TypeError, ValueError) as e:
            return self.send_json(400, {"error": str(e)})
        t0 = time.perf_counter()
        quantiles = self.predictor.predict(x, y, horizon)
        ms = (time.perf_counter() - t0) * 1000
        self.send_json(200, {"quantiles": quantiles, "ms": ms})

    def send_json(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Timestral-28M")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    predictor = Predictor(args.model)
    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, predictor=predictor))
    print(f"{args.model} ready at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
