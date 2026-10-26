# 01 — Web port: Timestral-5M running in the browser

## Status: implemented

Phases 1 to 5 are done (see [web/](web/) and [web/README.md](web/README.md)). Deviations from the plan below and measured results:

- **C ABI instead of wasm-bindgen.** `lib.rs` exports plain `extern "C"` functions (`alloc`, `model_new`, `model_forecast`, ...) and [web/timestral.js](web/timestral.js) wraps them. There is no wasm-bindgen CLI dependency, and the wasm has zero imports. The build is `web/build.sh`; SIMD is switched on by `timestral-wasm/.cargo/config.toml`, which cargo only reads from the working directory, so the script runs cargo from the crate directory.
- **Size:** 200 KB of wasm after `wasm-opt -O3`. The only crate dependencies are serde and serde_json.
- **Accuracy:** all nine golden fixtures agree with PyTorch to under 1e-5 (relative to `1 + |reference|`) at the 5 to 95% quantiles and under 4e-5 at 1% and 99%, both natively (scalar kernels) and through the SIMD wasm. The `h1500` fixture is the loosest, at about 4e-5, because the advancing refine frame amplifies float32 noise on values that reach 1e3.
- **Latency** (Node 26, i7-1065G7 at 1.3 GHz base, horizon 120, median of 15 runs): 16 ms at a 120-step context, 44 ms at 600, 137 ms at 1920, 311 ms at 4096. In headless Chromium the page measured about 25 ms per forecast at 130 steps, including the worker round trip.
- **Default `max_context` is 600 steps (10 s), not 1920.** At 1920 a forecast takes about 140 ms, which makes the live fan visibly stutter. `?max_context=` overrides it.
- **A bug the SIMD benchmark caught:** without `+simd128` the build silently fell back to scalar code at about 3 GFLOP/s, with the same output. `grep -c f32x4.mul` on the disassembly confirms that SIMD is really in the binary.
- Weight fetch and Cache Storage work as planned: a cold load of the 21 MB checkpoint took about 1.4 s here and a cached load about 0.2 s.
- Not done: Phase 6 (relaxed SIMD, f16 weights, trying 28M), and testing in Firefox and Safari. `weights.rs` already reads F16 and BF16 tensors.

## Context

Right now timesdraw needs a Python process. [backend/server.py](backend/server.py) loads `mistralai/Timestral-5M/28M/209M` with PyTorch, and [frontend/app.js](frontend/app.js) POSTs the resampled stroke to `/forecast` on every update. The goal is a **standalone, static web app in `./web/`**: no server, model weights fetched once and cached, inference running locally in a Web Worker. The existing `frontend/` + `backend/` app stays as it is, as a separate app.

## Research findings

### The model is small and simple enough to hand-write

From the installed `timestral` package (`.venv/.../timestral/{model,layers,forecaster,scaler,masks}.py`) and the 5M `config.json`:

| | Timestral-5M |
|---|---|
| Parameters | 5,208,960 fp32 (`model.safetensors`, 20.8 MB, plain F32 tensors) |
| Width / heads / head_dim | d=192, 3 heads, head_dim 64, RoPE on the first 32 dims (partial factor 0.5, θ=1e4) |
| Layers | 8: `time ×3, time_variate, time ×3, time_variate`, so 10 attention sublayers in total |
| MLP | fused SwiGLU 192→2×512→192, no bias |
| Attention | no bias; sigmoid output gate `out * σ(x·W_g)`; fused QKV weight laid out as `(qkv, head_dim, n_heads)`, **not** `(qkv, heads, head_dim)` |
| Patch embed | ResidualBlock(64→768→192, bias), input = `[32 scaled values, 32 missing flags]` |
| Head | ResidualBlock(192→768→672, no bias) = 32 steps × 21 quantiles, monotone cumsum with softplus, `quantile_width_factor=3.0` |
| Norms | RMSNorm, eps 1e-6 |

Only these ops are needed: linear (matmul), RMSNorm, SiLU, sigmoid, softplus, softmax, RoPE, masked attention, asinh and sinh. Nothing exotic.

### How timesdraw uses it

- One series with 2 variates (x, y), both in 0..1, at 60 Hz. The horizon is 120 steps, which is **4 patches, all in one decode block** (block_size 32). Because there's more than one patch, the "advancing refine frame" post-step applies.
- `forecast()` defaults that the server relies on, which must all be reproduced:
  - symmetric averaging: a second pass on `-context`, then `0.5·(q + flip(-q_mirror))`
  - non-negative clipping, which always fires for canvas coordinates
  - the 1e4·scale quantile cap
  - left-padding to a multiple of 32
  - `max_context` trimming
- Cost estimate: about 7.7 MFLOP per token-row through the transformer, plus about 1.6 MFLOP per decoder token for the head. The forward and mirrored passes run together as one batch of 4 rows in 2 groups (`{x,y}`, `{-x,-y}`). That is exact, because rows only attend within their own group.
  - 2 s stroke: (4 + 4 tokens) × 4 rows ≈ **0.26 GFLOP**
  - 10 s stroke: (19 + 4) × 4 ≈ **0.75 GFLOP**
  - At 10–30 GFLOP/s for single-thread wasm SIMD128, that is roughly **10–70 ms**. Mistral quotes about 10 ms on CPU, and the community ORT-web export measured 8 ms p50 at a 512-step context on an M3. Contexts of several minutes cost seconds, so the web app caps the context (see Phase 5).

### Existing engines and exports (why we are not using them)

- **[onnx-community/Timestral-5M-ONNX](https://huggingface.co/onnx-community/Timestral-5M-ONNX)**: fixed 512-step context and a 32-step horizon (no refine frame). Rows are independent.
- **[mrfakename/Timestral-5M-ONNX](https://huggingface.co/mrfakename/Timestral-5M-ONNX)**: dynamic shapes, but variate attention is collapsed to "one row per series". That **drops the x↔y coupling** that timesdraw's backend uses.
- **[weatherstral](https://github.com/ExplodingCB/weatherstral)**: 28M on ORT-web, single variate. Useful notes on export pitfalls.
- **[onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web) 1.30**: works, but the wasm runtime is about 13 MB (more than half the model's size). It would also need our own export with dynamic R/T/H and mask inputs. All the float64 pre/post-processing has to live outside the graph either way, as both community exports found.
- Candle, Burn and TF.js would mean reimplementing or converting the model anyway, and they bring large runtimes.

**Decision:** hand-write the whole pipeline in **one Rust crate compiled to `wasm32-unknown-unknown` with `+simd128`**.
- It is small (about 200 KB of wasm).
- f64 is native, so the causal statistics match PyTorch exactly.
- It can load any Timestral config, so 28M works for free.
- The same code builds natively, so `cargo test` can check it against PyTorch golden outputs.

### Browser features

- Fixed-width SIMD128 works in every current browser, Safari since 16.4. Relaxed SIMD (FMA) is on by default in Chrome and in Firefox 145+, but stable Safari doesn't have it yet. Ship SIMD128 as the baseline; relaxed SIMD is an optional second build.
- Threads need SharedArrayBuffer, and that needs COOP/COEP headers, which static hosts like GitHub Pages can't set. **We use a single thread inside one Dedicated Worker**, which is enough at these FLOP counts.

### Weights and license

- Fetch `config.json` and `model.safetensors` from `https://huggingface.co/mistralai/Timestral-5M/resolve/main/…`. The Hub sends CORS headers. Store them in the Cache Storage API, so the download happens once.
- We don't redistribute the weights, so the Modified MIT license isn't a problem. The app still links to the weights' license.
- `?model=` can point at a self-hosted copy or at 28M.

## Layout

```
web/
  index.html            # copy of frontend/index.html
  app.js                # copy of frontend/app.js; fetch("/forecast"|"/info") replaced by worker messages
  worker.js             # loads wasm + weights (HF fetch + Cache Storage), answers {x, y, horizon}
  timestral.js          # JS wrapper over the wasm C ABI
  pkg/                  # build output (gitignored): timestral.wasm
  timestral-wasm/       # Rust crate
    Cargo.toml
    src/lib.rs          # C ABI for wasm: model_new(config_json, safetensors_bytes), model_forecast(x, y, horizon)
    src/weights.rs      # minimal safetensors parser (8-byte LE header length + JSON + F32 data), pack weights at load
    src/kernels.rs      # matmul / rmsnorm / attention; scalar fallback + simd128 (std::arch::wasm32)
    src/model.rs        # embedding, transformer (time & variate attention, RoPE, gate, SwiGLU), head, monotone quantiles
    src/forecast.rs     # f64 pre/post: ffill, pad, causal stats, masks/positions, denorm, refine frame, cap, clampsym, clip, multi-block AR
    tests/golden.rs     # compares against fixtures from tools/
  tools/make_golden.py  # dumps PyTorch fixtures (run with `uv run`)
  build.sh              # cargo build --release --target wasm32-unknown-unknown, then wasm-opt
```

## Phases

### Phase 1: Golden fixtures (`web/tools/make_golden.py`)

Use `TimestralForecaster.from_pretrained("mistralai/Timestral-5M")` with the server's defaults to produce:

- **Full forecasts** for these fixtures:
  - the arc and sine from [smoke_test.py](smoke_test.py)
  - a short stroke of 40 steps, which is left-padded
  - a 600-step stroke
  - a perfectly straight line, which hits the scale floor
  - a stroke with x≈const
  - horizon 32 (1 patch, no refine), horizon 120, and horizon 1500 (2 blocks, AR feedback)
  - a `max_context=256` trim case
- **Per-stage intermediates for one fixture**, captured with forward hooks: scaled input, patch embeddings, the output of every layer, head raw output, normalized quantiles, and denormalized quantiles before and after refine.
- Format: raw little-endian `.f32`/`.f64` files plus one `manifest.json`. That keeps the Rust side free of an npy parser.

### Phase 2: Native Rust forward pass, scalar first

- **`weights.rs`**: parse the safetensors header and look tensors up by the exact names (e.g. `transformer.layers.3.attns.1.wQKV.weight [576,192]`). Transpose and pack each Linear weight from `[out,in]` into a k-major layout for the SIMD kernel.
  - De-interleave wQKV at load time, so the kernels see Q, K and V as `[heads][head_dim]`. Source row index = `qkv*192 + d*3 + h`.
- **`model.rs`**: port the code one function at a time. Exactness details:
  - RoPE `rotate_half` splits at head_dim/2 = 32. The inv_freq table is zero past index 16, so dims 16..31 and 48..63 are left unrotated.
  - SwiGLU uses `w12` chunk 0 as the gate and chunk 1 as `up`, computing `gate * silu(up)`.
  - The sigmoid gate is computed from the **normed input** of the sublayer.
  - Time mask: the encoder is causal over valid keys, with the diagonal always open. Leading left-pad patches are invalid keys. Each decoder token reads all valid encoder keys plus a causal chain over the decoder tokens.
  - Positions are `patch_index - first_observed_patch`.
  - Variate attention runs per token across the rows of the same group. It has no RoPE, and the mask is group-equal & key-valid | diagonal.
  - Encoder queries only attend to encoder keys (the TwoStreamMask split), which saves compute and gives identical results.
- Test natively against the per-stage intermediates. The tolerance is 1e-5 abs on the hidden states.

### Phase 3: f64 pre/post-processing (`forecast.rs`)

Port the following, mirroring the PyTorch formulas exactly:

- from `forecaster.py`: `_as_context`, `_assemble_rows`, `pad_to_patch`, `_block_ar_decode`, `_refine_block`, `_cap_quantiles`, the clampsym average, `nonnegative_context` and the clip
- from `scaler.py`: `compute_causal_statistics`, `causal_patch_loc_scale`, `advancing_block_frames`, `welford_combine`, `clamp_nonfinite`

Mirror the PyTorch formulas exactly. In particular, the "shifted mean" m2 increment and Bessel's correction with `max(cnt-1, 1)`. Batch the mirrored pass as rows 2–3 of one 4-row call.

Test against the full-forecast fixtures:
- **≤1e-4 abs** on the 5–95% quantiles.
- **≤1e-2 abs** on 1%/99% far into the horizon, where sinh and the refine frame amplify fp32 noise (the community exports saw about 8e-3 there).

### Phase 4: SIMD128 kernels + wasm build

- **`kernels.rs`**: `#[cfg(target_feature = "simd128")]` matmul `Y[M,N] = X[M,K]·W[K,N]`.
  - Tile 4 rows × 16 columns, with 16 `v128` accumulators and `f32x4_mul`/`f32x4_add`.
  - Also vectorize RMSNorm, the attention dot products (head_dim 64 is 16 lanes), and SiLU/sigmoid with a polynomial `exp`.
  - Keep the scalar path for the native build and for reference.
- **Toolchain:** `rustup target add wasm32-unknown-unknown`. No wasm-bindgen or wasm-pack (the crate exports a plain C ABI).
  - Set `RUSTFLAGS="-C target-feature=+simd128"`, `opt-level=3`, `lto=true` and `panic=abort` for the release build.
  - Optionally run `wasm-opt -O3`.
- Add a Node test (`node web/tools/check_wasm.mjs`) that loads `pkg/` in Node 26 and runs the same golden comparison. This proves that the SIMD path matches the scalar path.
- Benchmark: median ms per forecast at contexts of 120, 600, 1920 and 4096 steps, measured in Node and in Chrome, Firefox and Safari.

### Phase 5: Web app (`web/`)

- **`worker.js`**:
  - On init, it fetches `config.json` and `model.safetensors`, using Cache Storage on later visits. It posts download progress, then builds the `Model`.
  - On `{id, x, y, horizon}`, it returns a transferable `Float32Array` laid out as `[21][2][H]`, plus `ms`.
- **`app.js`**: copy `frontend/app.js` and replace the two `fetch` calls.
  - Replace `/info` with the worker's `ready` message, which carries the model name and `maxContext`.
  - Replace `/forecast` with a promise around `worker.postMessage`. Convert the result to the existing `{ "0.01": [[xs],[ys]], … }` shape, so the drawing, colouring and recording code stays unchanged.
  - Keep the one-request-in-flight logic as it is.
  - Show a small loading line until the model is ready. This is the only new UI.
- **URL params:**
  - `?model=` takes an HF repo id or a URL prefix, and defaults to `mistralai/Timestral-5M`.
  - `?max_context=` defaults to **600 steps (10 s)** (retuned after the Phase 4 benchmark; see Status), because unlike the server, the browser can't afford the 32,768-step window.
  - `?record=true` stays.
- **Serving:** any static server works, e.g. `python -m http.server -d web`. It is deployable to GitHub Pages without special headers.
- Add a `web/README` section, or a section in the root README, covering build, run, and the license note for the weights.

### Phase 6: Optional follow-ups

- A relaxed-SIMD build (`+relaxed-simd`, `f32x4_relaxed_madd`) chosen at runtime with a feature-detect, for Chrome and Firefox.
- f16 weights, converted to f32 at load, to halve the download. This needs a converter script and a self-hosted copy.
- Check that 28M (112 MB) is usable.
- Encoder KV caching is **not** viable as is. Left-padding to a multiple of 32 shifts every patch boundary each time the stroke grows by one step, so no state carries over between calls.

## Verification

1. `uv run web/tools/make_golden.py` writes the fixtures.
2. `cargo test --release` in `web/timestral-wasm` runs the per-stage and full-forecast golden comparisons natively, using the scalar path.
3. `web/build.sh && node web/tools/check_wasm.mjs` runs the same comparison through the SIMD wasm and prints latency.
4. `python -m http.server -d web 8001`, then open http://127.0.0.1:8001:
   - The model downloads once; on reload it loads from the cache (check the Network tab).
   - Drawing shows the 21-line fan.
   - The console logs latency per forecast.
   - Side-by-side with `uv run backend/server.py --model mistralai/Timestral-5M` on :8000, the same kind of stroke gives visually identical behaviour.
5. Run the same check in Chrome, Firefox and Safari.
