# timesdraw

![Demo Video](./demo.mp4)

Draw strokes with the mouse. As you draw, [Timestral](https://github.com/mistralai/timestral) keeps predicting how the stroke continues and draws it ahead of your pen. It draws all 21 of the model's quantiles, from 0.01 to 0.99, each in its own rainbow colour from red (0.01) to violet (0.99). Each line starts at the end of your stroke and fades to transparent toward the end of the forecast. Lines also start fainter the further their quantile is from the median: the median (0.5) starts solid, and 0.01 and 0.99 start at 10%.

Your strokes stay on the canvas and build up. Each stroke starts violet, the colour of the highest quantile. As you draw, it shifts toward the colour of whichever forecast line your pen is closest to: following the median turns it green, and veering toward the low or high side turns it red or violet.

Every forecast is also kept: as each one arrives it is drawn faintly into the canvas, underneath your strokes, so the alternatives the model considered build up into a glow around each stroke.

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
2. Once the stroke has a few points, the forecast appears and updates as you move. A new forecast is requested as soon as the previous one returns, so faster models update more often. The page shows no text: the model name, each forecast's context length and latency, and any errors are logged to the browser console (F12).
3. Release the button to end the stroke. The live forecast disappears; the stroke and the faint forecasts stay. Press again to start another stroke on top.
4. Press Escape to clear the canvas, including the faint forecasts.

### Recording

Open <http://127.0.0.1:8000/?record=true> to record a video of the canvas. Recording starts when you begin your first stroke. Pressing Escape stops it, downloads the video and clears the canvas; the next stroke starts a new recording. The video is 1080×1920 (9:16 portrait), in MP4 (H.264) where the browser supports recording it, and WebM otherwise.

To check the backend without a browser, run `uv run smoke_test.py` while the server is running.

## Choosing a model

The default model is `mistralai/Timestral-28M`. To use another one, pass `--model`:

```bash
uv run backend/server.py --model mistralai/Timestral-5M    # 21 MB, fastest
uv run backend/server.py --model mistralai/Timestral-209M  # 837 MB, best quality
```

`--host` and `--port` (default `127.0.0.1:8000`) set where the server listens.

## Context window

The forecast uses the whole stroke until it's longer than the model's context window, which is 32,768 steps (about 9 minutes at 60 Hz). After that, only the most recent 32,768 steps are used, as a rolling window. Longer contexts make each forecast slower. To trade accuracy for speed, shrink the window:

```bash
uv run backend/server.py --max-context 600   # last 10 s of the stroke
```

The frontend reads the window from `/info` and trims the stroke before sending it.

## API

`POST /forecast` takes uniformly sampled coordinates and a horizon in steps (1–1024):

```json
{"x": [0.1, 0.11, ...], "y": [0.5, 0.52, ...], "horizon": 120}
```

It returns `[xs, ys]` paths for all 21 quantile levels (0.01, 0.05, 0.1, 0.15, …, 0.9, 0.95, 0.99), and the inference time:

```json
{"quantiles": {"0.01": [[...], [...]], "0.05": [[...], [...]], ..., "0.99": [[...], [...]]}, "ms": 41.2}
```

`GET /info` returns the loaded model name and its context window in steps (`max_context`).

## Settings

These are set at the top of [frontend/app.js](frontend/app.js):

- `HZ`: resampling rate, default 60
- `HORIZON`: forecast length in steps, default 120 (about 2 s)
- `MIN_ALPHA`: starting opacity of the outermost quantiles, default 0.1
- `TIME_FADE`: how quickly the lines fade along the forecast, default 1.5 (1 is linear; higher fades sooner)
- `GHOST_ALPHA`: opacity of the kept forecasts relative to the live one, default 0.08
- `LINE_WIDTH`: width of the strokes and forecast lines in canvas pixels, default 4
- `HUE_LOW`, `HUE_HIGH`: the rainbow hue range, default 0 (red, 0.01) to 270 (violet, 0.99)
- `HUE_SMOOTHING`: how quickly a stroke's colour moves toward the line it's following, default 0.2 per pointer event (lower is more gradual)

## Known issues

- Curved strokes can jump abruptly in the forecast, possibly every 32 steps (the model's patch size).
- The quantile lines are computed separately at each step, so they are not real paths. Treat them as decoration.

See [docs.md](docs.md) for the full plan and next steps.

## License

The Timestral weights use a Modified MIT license, which differs from the Apache 2.0 code license. Read the [weights' license](https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE) before any commercial use.
