"""Draw a stroke; on release, Timestral forecasts its continuation."""
import sys
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from timestral import TimestralForecaster

HZ, HORIZON = 60, 120
MODEL = sys.argv[1] if len(sys.argv) > 1 else "mistralai/Timestral-28M"
device = "cuda" if torch.cuda.is_available() else "cpu"
fc = TimestralForecaster.from_pretrained(MODEL, device=device)


def resample(pts):
    """Irregular (t, x, y) mouse samples -> uniform HZ grid, shape (1, 2, T)."""
    t, x, y = map(np.asarray, zip(*pts))
    grid = np.arange(t[0], t[-1], 1 / HZ)
    return np.stack([np.interp(grid, t, x), np.interp(grid, t, y)])[None]


def predict(ctx, horizon=HORIZON):
    """Returns numpy (2, H) arrays for the median and the 0.1 / 0.9 quantiles."""
    f = fc.forecast(ctx, prediction_length=horizon)
    return {q: f.quantile(q)[0].cpu().numpy() for q in (0.1, 0.5, 0.9)}


def reset_axes(ax):
    ax.clear()
    ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")  # y down, like a canvas
    ax.set_title(f"{MODEL} - draw a stroke, release to forecast")


def main():
    fig, ax = plt.subplots(figsize=(8, 8))
    reset_axes(ax)
    state = {"pts": [], "drawing": False}

    def on_press(e):
        if e.inaxes != ax:
            return
        reset_axes(ax)
        state["pts"] = [(time.perf_counter(), e.xdata, e.ydata)]
        state["drawing"] = True
        fig.canvas.draw_idle()

    def on_move(e):
        if not state["drawing"] or e.inaxes != ax:
            return
        pts = state["pts"]
        pts.append((time.perf_counter(), e.xdata, e.ydata))
        ax.plot([pts[-2][1], pts[-1][1]], [pts[-2][2], pts[-1][2]], "k-")
        fig.canvas.draw_idle()

    def on_release(e):
        drawing, state["drawing"] = state["drawing"], False
        if not drawing or len(state["pts"]) < 8:
            return
        ctx = resample(state["pts"])
        t0 = time.perf_counter()
        p = predict(ctx)
        dt = (time.perf_counter() - t0) * 1000
        for q in (0.1, 0.9):
            ax.plot(p[q][0], p[q][1], "r-", alpha=0.25)
        ax.plot(p[0.5][0], p[0.5][1], "r-", lw=2)
        ax.set_title(f"{MODEL} - context {ctx.shape[-1]} steps, forecast {dt:.0f} ms")
        fig.canvas.draw_idle()

    for ev, cb in [("button_press_event", on_press),
                   ("motion_notify_event", on_move),
                   ("button_release_event", on_release)]:
        fig.canvas.mpl_connect(ev, cb)
    plt.show()


if __name__ == "__main__":
    main()
