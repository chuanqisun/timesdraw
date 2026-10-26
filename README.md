# timesdraw

Draw a stroke with the mouse. When you release the button, [Timestral](https://github.com/mistralai/timestral) predicts how the stroke continues and draws it in red. It draws all 21 of the model's quantiles, from 0.01 to 0.99. The median (0.5) is the thick, solid line; the other lines fade evenly the further they are from it, down to 10% for 0.01 and 0.99.

## Layout

- [frontend/](frontend/) is a plain HTML canvas page. It records the stroke, resamples it to a uniform 60 Hz grid, sends it to the backend, and draws the forecast.
- [backend/server.py](backend/server.py) loads the model and serves `POST /forecast`. It also serves the frontend, so you only run one process.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.14 or later (uv can install it for you)
- A web browser

No GPU is needed. The backend uses CUDA if it finds a GPU and the CPU otherwise.

## Getting started

```bash
uv sync                      # install dependencies
uv run backend/server.py     # start the server
```

The first run downloads the model weights from the Hugging Face Hub and caches them. When the server prints `ready`, open <http://127.0.0.1:8000>.

1. Press the mouse button and draw a stroke. Draw for at least 1–2 seconds so the model has enough context.
2. Release the button to see the forecast. The status line shows the context length and how long the forecast took.
3. Press the button again to clear the canvas and start over.

To check the backend without a browser, run `uv run smoke_test.py` while the server is running.

## Choosing a model

The default model is `mistralai/Timestral-28M`. To use another one, pass `--model`:

```bash
uv run backend/server.py --model mistralai/Timestral-5M    # 21 MB, fastest
uv run backend/server.py --model mistralai/Timestral-209M  # 837 MB, best quality
```

`--host` and `--port` (default `127.0.0.1:8000`) set where the server listens.

## API

`POST /forecast` takes uniformly sampled coordinates and a horizon in steps (1–1024):

```json
{"x": [0.1, 0.11, ...], "y": [0.5, 0.52, ...], "horizon": 120}
```

It returns `[xs, ys]` paths for all 21 quantile levels (0.01, 0.05, 0.1, 0.15, …, 0.9, 0.95, 0.99), and the inference time:

```json
{"quantiles": {"0.01": [[...], [...]], "0.05": [[...], [...]], ..., "0.99": [[...], [...]]}, "ms": 41.2}
```

`GET /info` returns the loaded model name.

## Settings

`HZ` (resampling rate, default 60) and `HORIZON` (forecast length in steps, default 120, about 2 s) and `MIN_ALPHA` (opacity of the outermost quantiles, default 0.1) are set at the top of [frontend/app.js](frontend/app.js).

## Known issues

- Curved strokes can jump abruptly in the forecast, possibly every 32 steps (the model's patch size).
- The quantile lines are computed separately at each step, so they are not real paths. Treat them as decoration.

See [docs.md](docs.md) for the full plan and next steps.

## License

The Timestral weights use a Modified MIT license, which differs from the Apache 2.0 code license. Read the [weights' license](https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE) before any commercial use.
