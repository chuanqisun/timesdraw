// Loads the wasm and the Timestral weights (cached after the first visit), then answers forecast requests.
import { Timestral } from "./timestral.js";

const CACHE = "timesdraw-weights-v1";
const DEFAULT_MODEL = "./model";  // shipped next to this file by build.sh
const DEFAULT_MAX_CONTEXT = 600;  // steps (10 s at 60 Hz); cost grows with context, ~45 ms here on a laptop

// A URL prefix (absolute, or relative to this file) holds config.json and model.safetensors; anything else
// ("org/name") is a repo id on the Hugging Face Hub.
const baseUrl = model => /^(https?:)?\/\/|^\.{0,2}\//.test(model)
  ? new URL(model.replace(/\/$/, ""), import.meta.url).href
  : `https://huggingface.co/${model}/resolve/main`;

// Fetch bytes with progress. Files from another origin (the Hub) go through the Cache Storage API so later
// visits skip the download; same-origin files are left to the browser's HTTP cache.
async function load(url, label) {
  const remote = new URL(url).origin !== self.location.origin;
  const cache = remote && "caches" in self ? await caches.open(CACHE).catch(() => null) : null;
  const hit = cache && await cache.match(url);
  if (hit) return new Uint8Array(await hit.arrayBuffer());
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = +res.headers.get("content-length") || 0;
  const chunks = [];
  let loaded = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    postMessage({ type: "progress", label, loaded, total });
  }
  const bytes = new Uint8Array(loaded);
  let at = 0;
  for (const c of chunks) { bytes.set(c, at); at += c.length; }
  if (cache) await cache.put(url, new Response(bytes)).catch(() => {});  // only complete downloads get cached
  return bytes;
}

let model = null;

async function init({ model: repo = DEFAULT_MODEL, maxContext }) {
  const base = baseUrl(repo);
  const [wasm, config, weights] = await Promise.all([
    load(new URL("timestral.wasm", import.meta.url).href, "wasm"),
    load(`${base}/config.json`, "config"),
    load(`${base}/model.safetensors`, "weights"),
  ]);
  const text = new TextDecoder().decode(config);
  model = await Timestral.create(wasm, text, weights);
  model.maxContext = Math.min(maxContext || DEFAULT_MAX_CONTEXT, JSON.parse(text).context_length);
  const levels = JSON.parse(text).quantile_knots.map(q => String(+q.toFixed(4)));
  postMessage({ type: "ready", model: repo, maxContext: model.maxContext, levels });
}

onmessage = async ({ data }) => {
  try {
    if (data.type === "init") return await init(data);
    const t0 = performance.now();
    const out = model.forecast(data.x, data.y, data.horizon);
    postMessage({ type: "result", id: data.id, data: out, ms: performance.now() - t0 }, [out.buffer]);
  } catch (err) {
    postMessage({ type: "error", id: data.id, message: err.message });
  }
};
