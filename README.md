# timesdraw

Draw a stroke with the mouse. When you release the button, [Timestral](https://github.com/mistralai/timestral) predicts how the stroke continues and draws it in red. The thick line is the median forecast; the faint lines are the 10% and 90% quantiles.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.14 or later (uv can install it for you)
- A display, because the app opens a matplotlib window

No GPU is needed. The app uses CUDA if it finds a GPU and the CPU otherwise.

## Getting started

```bash
uv sync               # install dependencies
uv run smoke_test.py  # test without a window; writes smoke.png
uv run main.py        # open the drawing window
```

The first run downloads the model weights from the Hugging Face Hub and caches them.

In the window:

1. Press the mouse button and draw a stroke. Draw for at least 1–2 seconds so the model has enough context.
2. Release the button to see the forecast. The title shows the context length and how long the forecast took.
3. Press the button again to clear the canvas and start over.

## Choosing a model

The default model is `mistralai/Timestral-28M`. To use another one, pass it as an argument:

```bash
uv run main.py mistralai/Timestral-5M    # 21 MB, fastest
uv run main.py mistralai/Timestral-209M  # 837 MB, best quality
```

## Settings

`HZ` (resampling rate, default 60) and `HORIZON` (forecast length in steps, default 120, about 2 s) are set at the top of [main.py](main.py).

## Known issues

- Curved strokes can jump abruptly in the forecast, possibly every 32 steps (the model's patch size).
- The quantile lines are computed separately at each step, so they are not real paths. Treat them as decoration.

See [docs.md](docs.md) for the full plan and next steps.

## License

The Timestral weights use a Modified MIT license, which differs from the Apache 2.0 code license. Read the [weights' license](https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE) before any commercial use.
