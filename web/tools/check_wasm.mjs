// Run the PyTorch golden fixtures through the built wasm and benchmark it.
// Usage: node web/tools/check_wasm.mjs   (after web/build.sh and tools/make_golden.py)
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { Timestral } from "../timestral.js";

const here = dirname(fileURLToPath(import.meta.url));
const golden = join(here, "golden");
const f32 = name => { const b = readFileSync(join(golden, name)); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };
const manifest = JSON.parse(readFileSync(join(golden, "manifest.json")));

const wasm = readFileSync(join(here, "../pkg/timestral.wasm"));
console.log("imports:", WebAssembly.Module.imports(new WebAssembly.Module(wasm)).length, " size:", (wasm.length / 1024).toFixed(0), "KB");
const ckpt = manifest.checkpoint;
const model = await Timestral.create(wasm, readFileSync(join(ckpt, "config.json"), "utf8"), readFileSync(join(ckpt, "model.safetensors")));
const defaultContext = model.maxContext;

let ok = true;
for (const f of manifest.fixtures) {
  model.maxContext = f.max_context ?? defaultContext;
  const ctx = f32(`${f.name}.in.f32`), want = f32(`${f.name}.out.f32`); // want: (2, H, 21)
  const got = model.forecast(ctx.subarray(0, f.t), ctx.subarray(f.t), f.horizon); // got: (21, 2, H)
  let mid = 0, tail = 0;
  for (let v = 0; v < 2; v++) for (let h = 0; h < f.horizon; h++) for (let k = 0; k < 21; k++) {
    const w = want[(v * f.horizon + h) * 21 + k], g = got[(k * 2 + v) * f.horizon + h];
    const d = Math.abs(g - w) / (1 + Math.abs(w));
    if (k >= 1 && k <= 19) mid = Math.max(mid, d); else tail = Math.max(tail, d);
  }
  const pass = mid <= 1e-4 && tail <= 1e-2;
  ok &&= pass;
  console.log(`${pass ? "ok  " : "FAIL"} ${f.name.padEnd(8)} H=${String(f.horizon).padStart(4)}  5-95%: ${mid.toExponential(2)}  1%/99%: ${tail.toExponential(2)}`);
}

// Latency: a wobbly stroke of the given length, horizon 120, median of 15 runs.
model.maxContext = 32768;
for (const steps of [120, 600, 1920, 4096]) {
  const x = Float32Array.from({ length: steps }, (_, i) => 0.5 + 0.3 * Math.cos(i / 40));
  const y = Float32Array.from({ length: steps }, (_, i) => 0.5 + 0.3 * Math.sin(i / 55));
  model.forecast(x, y, 120);
  const times = [];
  for (let i = 0; i < 15; i++) { const t0 = performance.now(); model.forecast(x, y, 120); times.push(performance.now() - t0); }
  times.sort((a, b) => a - b);
  console.log(`context ${String(steps).padStart(5)} steps: median ${times[7].toFixed(1)} ms`);
}
process.exit(ok ? 0 : 1);
