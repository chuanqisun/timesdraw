"""Headless check: feed synthetic strokes through the pipeline and save a PNG."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.argv = sys.argv[:1] + sys.argv[1:2]
import main

# Fake irregular mouse timestamps over ~2 s for a circle arc and a sine wave.
rng = np.random.default_rng(0)
t = np.cumsum(rng.uniform(0.008, 0.03, 120))
strokes = {
    "arc": (0.5 + 0.3 * np.cos(2.5 * t), 0.5 + 0.3 * np.sin(2.5 * t)),
    "sine": (0.1 + 0.3 * t, 0.5 + 0.15 * np.sin(6 * t)),
}

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, (name, (x, y)) in zip(axes, strokes.items()):
    ctx = main.resample(list(zip(t, x, y)))
    p = main.predict(ctx)
    print(name, "ctx", ctx.shape, "median", p[0.5].shape, type(p[0.5]))
    ax.plot(ctx[0, 0], ctx[0, 1], "k-")
    for q in (0.1, 0.9):
        ax.plot(p[q][0], p[q][1], "r-", alpha=0.25)
    ax.plot(p[0.5][0], p[0.5][1], "r-", lw=2)
    ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal"); ax.set_title(name)
fig.savefig("smoke.png")
print("saved smoke.png")
