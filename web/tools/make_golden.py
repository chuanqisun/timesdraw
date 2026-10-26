"""Dump PyTorch reference outputs for the Rust/wasm port.

Run: uv run web/tools/make_golden.py [out_dir]   (default web/tools/golden)

For each fixture writes `<name>.in.f32` (x then y, little-endian float32, T each) and `<name>.out.f32`
(forecast quantiles, shape (2, H, 21)), with the server's forecast() defaults. For the `arc` fixture
it also writes per-stage intermediates of the first (non-mirrored) pass as `stage_*.f32`.
`manifest.json` describes everything.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from timestral import TimestralForecaster
from timestral.loading import resolve_checkpoint

REPO = "mistralai/Timestral-5M"
out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "golden")
out.mkdir(parents=True, exist_ok=True)


def arc(n):
    t = np.arange(n) / 60
    return 0.5 + 0.3 * np.cos(2.5 * t), 0.5 + 0.3 * np.sin(2.5 * t)


def sine(n):
    t = np.arange(n) / 60
    return 0.1 + 0.3 * t, 0.5 + 0.15 * np.sin(6 * t)


def wobble(n):  # irregular hand-like stroke
    rng = np.random.default_rng(0)
    steps = rng.normal(0, 0.004, (n, 2)).cumsum(0)
    p = 0.5 + 0.2 * np.sin(np.arange(n)[:, None] / 40 + np.array([0, 1.3])) + steps * 0.3
    return p[:, 0], p[:, 1]


FIXTURES = [  # name, (x, y), horizon, max_context
    ("arc", arc(120), 120, None),
    ("sine", sine(120), 120, None),
    ("short40", arc(40), 120, None),
    ("long600", wobble(600), 120, None),
    ("line", (np.linspace(0.1, 0.9, 120), np.linspace(0.2, 0.7, 120)), 120, None),
    ("xconst", (np.full(120, 0.5), sine(120)[1]), 120, None),
    ("h32", arc(120), 32, None),
    ("h1500", arc(120), 1500, None),
    ("ctx256", wobble(600), 120, 256),
]

fcs = {}


def forecaster(mc):
    if mc not in fcs:
        fcs[mc] = TimestralForecaster.from_pretrained(REPO, max_context=mc)
    return fcs[mc]


def dump(name, arr, dtype=np.float32):
    np.ascontiguousarray(arr, dtype=dtype).tofile(out / name)


manifest = {"model": REPO, "checkpoint": str(resolve_checkpoint(REPO)), "fixtures": [], "stages": {}}
for name, (x, y), h, mc in FIXTURES:
    ctx = np.asarray([[x, y]], dtype=np.float32)
    f = forecaster(mc).forecast(ctx, prediction_length=h)
    dump(f"{name}.in.f32", ctx[0])
    dump(f"{name}.out.f32", f.quantiles[0].numpy())
    manifest["fixtures"].append({"name": name, "t": ctx.shape[-1], "horizon": h, "max_context": mc})
    print(name, ctx.shape, f.quantiles.shape)

# Per-stage intermediates for the first pass of `arc`.
fc = forecaster(None)
model = fc.model
cap = {}


def grab(key):
    def hook(_m, _i, o):
        cap.setdefault(key, o.detach())  # first call = forward (non-mirrored) pass
    return hook


hs = [model.representation.patch_proj.register_forward_hook(grab("embed"))]
for i, layer in enumerate(model.transformer.layers):
    hs.append(layer.register_forward_hook(grab(f"layer{i}")))
hs.append(model.transformer.register_forward_hook(grab("transformer")))
hs.append(model.head.register_forward_hook(grab("head")))
orig_scale, orig_refine, orig_block = model.scale, fc._refine_block, model.predict_block


def scale_wrap(*a, **k):
    r = orig_scale(*a, **k)
    cap.setdefault("scaled", r[0])
    cap.setdefault("loc", r[1])
    cap.setdefault("scale", r[2])
    return r


def block_wrap(*a, **k):
    r = orig_block(*a, **k)
    cap.setdefault("denorm", r)
    return r


def refine_wrap(q, *a, **k):
    r = orig_refine(q, *a, **k)
    cap.setdefault("refined", r)
    return r


model.scale, model.predict_block, fc._refine_block = scale_wrap, block_wrap, refine_wrap
x, y = arc(120)
fc.forecast(np.asarray([[x, y]], dtype=np.float32), prediction_length=120)
for k, v in cap.items():
    dump(f"stage_{k}.f32", v.double().numpy())
    manifest["stages"][k] = list(v.shape)
    print("stage", k, tuple(v.shape))
(out / "manifest.json").write_text(json.dumps(manifest, indent=1))
