# Prototype plan: Timestral stroke continuation

The plan uses the official `timestral` package API. I haven't run it, so the points marked "verify" are assumptions to check.

## 1. Goal and scope

- You draw one stroke with the mouse. When you release, the model forecasts the continuation and draws it in a different colour.
- There is no pen-up handling, no multiple strokes and no training.
- Stack: Python, `timestral`, and matplotlib for the UI. Matplotlib's mouse events are enough, so there's no web server or JS.

## 2. How the model API maps to this

-

The documented call is `TimestralForecaster.from_pretrained("mistralai/Timestral-209M")`, followed by `forecaster.forecast(context, prediction_length=96)`. Context has shape `(series, variates, time)`, and `NaN` marks missing values.

- Treat the stroke as **one series with two variates (x, y)**, so the shape is `(1, 2, T)`.

The variates of a series are forecast jointly

, which should keep x and y coherent.

-

The result gives `quantiles` with shape `(series, variates, horizon, 21)`, plus `.median` and `.quantile(q)`.

- Mistral's own CPU latency figures are

about 10 ms for the 5M model and about 140 ms for the 209M model on an 8-thread server CPU

. That's fast enough to forecast on mouse release, or even live while drawing.

-

Inference runs in PyTorch on CPU or GPU.

## 3. Setup with uv

```bash
uv init stroke-forecast && cd stroke-forecast
uv add matplotlib numpy torch
uv add "timestral @ git+https://github.com/mistralai/timestral.git"
```

-

The package needs Python ≥ 3.10 and torch ≥ 2.4.

- For fast iteration, start with `mistralai/Timestral-28M` or `5M`, then switch to 209M for quality. The 209M checkpoint is about 837 MB (

All four checkpoints, from Timestral-5M (21 MB) to Timestral-209M (837 MB)

).

## 4. Pipeline

1. **Capture.** Record `(t, x, y)` using `button_press_event`, `motion_notify_event` and `button_release_event`. Use `time.perf_counter()` for `t`.
2. **Resample to a uniform 60 Hz grid.** Mouse events arrive at irregular intervals, but the model assumes evenly spaced steps. Use `np.interp` over `np.arange(t0, t1, 1/60)` for x and y separately.
3. **Build the context.** Stack into `ctx = np.stack([x, y])[None]`, which has shape `(1, 2, T)`.
4. **Forecast.** Call `forecaster.forecast(ctx, prediction_length=H)`, with H = 90–180 steps (1.5–3 s).
5. **Render.**
   - Draw the user stroke in black.
   - Draw `forecast.median` as the main continuation in a second colour.
   - For the artistic effect, also draw the `quantile(0.1)` and `quantile(0.9)` paths as faint "ghost" strokes.
6. **Reset.** Pressing the mouse again clears the canvas.

## 5. Skeleton (`main.py`)

```python
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from timestral import TimestralForecaster

HZ, HORIZON = 60, 120
device = "cuda" if torch.cuda.is_available() else "cpu"
fc = TimestralForecaster.from_pretrained("mistralai/Timestral-28M", device=device)

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")  # y down, like a canvas
pts, drawing = [], False

def on_press(e):
    global pts, drawing
    if e.inaxes != ax: return
    ax.clear(); ax.set_xlim(0, 1); ax.set_ylim(1, 0)
    pts, drawing = [(time.perf_counter(), e.xdata, e.ydata)], True

def on_move(e):
    if drawing and e.inaxes == ax:
        pts.append((time.perf_counter(), e.xdata, e.ydata))
        xs, ys = zip(*[(p[1], p[2]) for p in pts])
        ax.plot(xs[-2:], ys[-2:], "k-"); fig.canvas.draw_idle()

def on_release(e):
    global drawing
    if not drawing or len(pts) < 8: drawing = False; return
    drawing = False
    t, x, y = map(np.array, zip(*pts))
    grid = np.arange(t[0], t[-1], 1 / HZ)
    ctx = np.stack([np.interp(grid, t, x), np.interp(grid, t, y)])[None]  # (1,2,T)
    f = fc.forecast(ctx.astype(np.float32), prediction_length=HORIZON)
    for q, a in [(0.1, .25), (0.9, .25)]:
        qq = f.quantile(q)[0]; ax.plot(qq[0], qq[1], "r-", alpha=a)
    m = f.median[0];           ax.plot(m[0], m[1], "r-", lw=2)
    fig.canvas.draw_idle()

for ev, cb in [("button_press_event", on_press),
               ("motion_notify_event", on_move),
               ("button_release_event", on_release)]:
    fig.canvas.mpl_connect(ev, cb)
plt.show()
```

Run it with `uv run main.py`.

## 6. Things to verify or tune

| Item                                                                                   | Why it matters                                                                                                                                                                                              |
| -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Short context.** A 1-second stroke is only about 60 steps, or about 2 patches of 32. | Check how the forecaster pads short inputs. The tiny-random example in the repo uses a context of 96, so try ≥ 96 steps. Slow down your drawing, or tile or repeat the stroke if forecasts look degenerate. |
| **Normalisation.**                                                                     | Check whether the forecaster scales inputs internally, as most foundation models do. If not, normalise x and y yourself.                                                                                    |
| **Return type.**                                                                       | `forecast.median` may be a numpy array or a tensor. Check the shape against what the example prints, `(series, variates, horizon)`.                                                                         |
| **Quantile paths are marginal.**                                                       | The 0.1 and 0.9 paths aren't coherent trajectories, so treat them as visual flourishes.                                                                                                                     |
| **Longer horizons.**                                                                   |

The model emits up to 1,024 steps per forward pass

. The repo exposes options like `per_shot_horizon_patches`, which the tiny-random example sets to 32. |
| **Model size.** | Compare 5M, 28M and 209M outputs on the same stroke. Smaller models may give wigglier or more "organic" continuations. |

## 7. Optional extensions

- Forecast live while drawing, throttled to about 10 Hz, to get a "ghost hand" that draws ahead of you.
- Roll the forecast forward by feeding the median back as context, which gives an endless continuation.
- Replace matplotlib with Gradio or a small web canvas if you want to share the demo.

## 8. License note

The code is Apache 2.0, but the weights are released separately under a Modified MIT license.

One write-up says that license

stops at $20 million a month in revenue

. Read the weights' LICENSE file before you build on it commercially.

## References

1. [GitHub - mistralai/timestral: Repository for Timestral, the time-series foundation model by Mistral AI · GitHub](https://github.com/mistralai/timestral)
2. [Mistral AI Releases Timestral: Open-Weight Time-Series Models That Run on a CPU, Not a GPU](https://techjacksolutions.com/ai-brief/mistral-timestral-cpu-time-series-foundation-models/)
3. [mistralai/Timestral-209M · Hugging Face](https://huggingface.co/mistralai/Timestral-209M)
4. [Timestral — Mistral's first open models for…](https://ai-tldr.dev/releases/mistral-timestral/)
5. [Timestral 1.0 — Mistral Time-Series Forecasting Models](https://ai-tldr.dev/models/timestral-1-0/)
6. [Mistral's Timestral Forecasting Models Run on a CPU, No GPU Needed. The 'Modified MIT' Licence Stops at $20 Million a Month in Revenue.](https://medium.com/@andy.a.g/mistrals-timestral-forecasting-models-run-on-a-cpu-no-gpu-needed-649c75075558)
