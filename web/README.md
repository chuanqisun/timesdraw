# timesdraw, in the browser

The same drawing app as [frontend/](../frontend/), but the model runs in the page: no Python, no server. A Web Worker loads [Timestral-5M](https://huggingface.co/mistralai/Timestral-5M) (21 MB, downloaded from the Hugging Face Hub on the first visit and then kept in the browser's Cache Storage) and runs it as WebAssembly with SIMD128. The inference code is a from-scratch Rust port of the model in [timestral-wasm/](timestral-wasm/), about 200 KB of wasm with no dependencies at runtime.

A forecast takes about 15 ms for a 2 s stroke and about 45 ms for a 10 s one (Node 26 on a 1.3 GHz laptop CPU), and its cost grows linearly with the stroke length.

## Build and run

```bash
rustup target add wasm32-unknown-unknown
web/build.sh                          # writes web/pkg/timestral.wasm (uses wasm-opt if installed)
python3 -m http.server -d web 8001    # any static file server works
```

Open <http://127.0.0.1:8001>. The page shows a progress line while the weights load; the drawing controls are the same as in the main README (`?record=true` works too).

Query parameters:

- `?model=` is a Hugging Face repo id (default `mistralai/Timestral-5M`) or a URL prefix holding `config.json` and `model.safetensors`, for example a self-hosted copy. Other Timestral sizes load too, but the 28M, 123M and 209M checkpoints are 112 to 837 MB.
- `?max_context=` is the rolling window in steps (default 600, 10 s at 60 Hz). Unlike the Python server, the browser can't afford the model's full 32,768-step window.

The app is static files plus the wasm, so it deploys to any static host (GitHub Pages included). It needs no special headers because inference is single-threaded.

## Checking the port

The Rust code is tested against PyTorch outputs for several strokes (short and left-padded, 600 steps, straight lines, a 1500-step horizon that takes two decode blocks, a trimmed context), plus the intermediate activations of every layer.

```bash
uv run web/tools/make_golden.py                  # write the PyTorch fixtures to web/tools/golden/
(cd web/timestral-wasm && cargo test --release)  # native, scalar kernels
web/build.sh && node web/tools/check_wasm.mjs    # the same fixtures through the SIMD wasm, plus latency
node web/tools/browser_smoke.mjs                 # headless Chromium: load, draw a stroke, screenshot
```

Quantiles agree with PyTorch to about 1e-5 for ordinary forecasts. Very long horizons agree to about 3e-5 relative to the magnitude of the values, which there can reach 1e3 because the refine frame amplifies float32 rounding.

## License

The code is the same as the rest of this repo. The model weights are not included: they are fetched from Mistral AI's repository and are under its Modified MIT license, which among other terms does not grant rights to companies whose monthly revenue exceeds $20 million. Read the license in the weights repository before building on it commercially.
