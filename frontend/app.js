// Collect strokes on the canvas, forecast live, and colour each stroke by which quantile it follows.
const HZ = 60, HORIZON = 120, MIN_POINTS = 8, MIN_ALPHA = 0.1;
const LINE_WIDTH = 2;
const TIME_FADE = 1.5;  // forecast opacity falls as (1 - progress)^TIME_FADE: higher fades out sooner
const HUE_LOW = 0, HUE_HIGH = 270;  // rainbow from red (lowest quantile) to violet (highest)
const HUE_SMOOTHING = 0.2;  // 0..1 per pointer event: lower changes stroke colour more gradually
const GHOST_ALPHA = 0.08;   // every forecast is kept at this fraction of its live opacity

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
// Offscreen layers that accumulate; the visible canvas redraws them, then the live forecast.
// ghosts: every forecast fan received, faint. traces: the strokes, kept above the ghosts.
const layer = () => Object.assign(document.createElement("canvas"), { width: canvas.width, height: canvas.height });
const ghosts = layer(), traces = layer();
const gctx = ghosts.getContext("2d"), tctx = traces.getContext("2d");
let model = "";
let maxContext = Infinity;  // rolling window in steps, from /info
let pts = [];  // [t (s), x, y] with x, y in 0..1
let drawing = false;
let forecast = null;  // latest response for the current stroke: {levels, paths, anchor, t0}
let strokeId = 0;     // bumped per stroke so late responses for an old stroke are dropped
let inFlight = false;
let sentLength = 0;   // pts.length covered by the last request; more points means a new one is due
let frameQueued = false;
let hue = HUE_HIGH;   // current stroke colour

fetch("/info").then(r => r.json()).then(info => {
  model = info.model;
  maxContext = info.max_context;
  console.log(`model: ${model}, context window: ${maxContext} steps`);
});

function toUnit(e) {
  const r = canvas.getBoundingClientRect();
  return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
}

const hueOf = (i, n) => HUE_LOW + (HUE_HIGH - HUE_LOW) * i / Math.max(n - 1, 1);
const color = h => `hsl(${h} 100% 60%)`;

// Each segment gets its own opacity, fading from `alpha` at the first point to transparent at
// the last. Butt caps so overlapping segment ends don't leave brighter dots.
function drawFadingPath(ctx, xs, ys, color, width, alpha) {
  ctx.save();
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = "butt";
  const n = xs.length - 1;
  for (let i = 0; i < n; i++) {
    ctx.globalAlpha = alpha * (1 - (i + 0.5) / n) ** TIME_FADE;
    ctx.beginPath();
    ctx.moveTo(xs[i] * canvas.width, ys[i] * canvas.height);
    ctx.lineTo(xs[i + 1] * canvas.width, ys[i + 1] * canvas.height);
    ctx.stroke();
  }
  ctx.restore();
}

// Irregular pointer samples -> uniform HZ grid via linear interpolation. Uses the whole stroke
// until it exceeds maxContext steps, then only the most recent maxContext (a rolling window).
// The grid stays anchored to the stroke's start so samples don't shift as the window slides.
// `t` is the time of the last grid point, where the forecast continues from.
function resample(pts) {
  const xs = [], ys = [];
  const start = pts[0][0], steps = Math.ceil((pts[pts.length - 1][0] - start) * HZ);
  let j = 0;
  for (let k = Math.max(0, steps - maxContext); k < steps; k++) {
    const t = start + k / HZ;
    while (pts[j + 1][0] < t) j++;
    const [t0, x0, y0] = pts[j], [t1, x1, y1] = pts[j + 1];
    const a = t1 > t0 ? (t - t0) / (t1 - t0) : 0;
    xs.push(x0 + a * (x1 - x0));
    ys.push(y0 + a * (y1 - y0));
  }
  return { x: xs, y: ys, t: start + (steps - 1) / HZ };
}

// Hue of the forecast line closest to where the pen is at time t, or null with no usable forecast.
function followedHue(t, x, y) {
  if (!forecast) return null;
  const k = Math.round((t - forecast.t0) * HZ) - 1;  // forecast step 0 is one grid step after t0
  if (k < 0 || k >= HORIZON) return null;
  let best = 0, bestDist = Infinity;
  forecast.levels.forEach((level, i) => {
    const [xs, ys] = forecast.paths[level];
    const dist = (xs[k] - x) ** 2 + (ys[k] - y) ** 2;
    if (dist < bestDist) { best = i; bestDist = dist; }
  });
  return hueOf(best, forecast.levels.length);
}

// Draw a forecast fan onto `target`, with every line's opacity scaled by `scale`.
// Two fades multiply. By quantile: rank distance from the median sets the starting opacity
// (median solid, outermost pair MIN_ALPHA; ranks so the packed tails 0.95/0.99 stay distinct).
// By time: each line then fades to transparent along the horizon, starting at the anchor.
function drawFan(target, { levels, paths, anchor }, scale) {
  const mid = (levels.length - 1) / 2;
  const byDistance = levels.map((level, i) => [level, i, Math.abs(i - mid)]).sort((a, b) => b[2] - a[2]);
  for (const [level, i, d] of byDistance) {  // outermost first so the median ends up on top
    const [xs, ys] = paths[level];
    drawFadingPath(target, [anchor[0], ...xs], [anchor[1], ...ys], color(hueOf(i, levels.length)), LINE_WIDTH,
      scale * (1 - (1 - MIN_ALPHA) * d / Math.max(mid, 1)));
  }
}

// Redraw everything: ghost forecasts, strokes, then the live forecast on top.
function render() {
  frameQueued = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(ghosts, 0, 0);
  ctx.drawImage(traces, 0, 0);
  if (forecast) drawFan(ctx, forecast, 1);
}

function scheduleRender() {
  if (!frameQueued) { frameQueued = true; requestAnimationFrame(render); }
}

// Keep at most one request in flight; when it returns, send another if the stroke grew meanwhile.
async function requestForecast() {
  if (inFlight || !drawing || pts.length < MIN_POINTS || pts.length === sentLength) return;
  const context = resample(pts);
  if (context.x.length < 2) return;
  inFlight = true;
  sentLength = pts.length;
  const id = strokeId;
  try {
    const res = await fetch("/forecast", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x: context.x, y: context.y, horizon: HORIZON }),
    });
    const body = await res.json();
    if (id !== strokeId) return;
    if (!res.ok) { console.error(`forecast error: ${body.error}`); return; }
    forecast = {
      levels: Object.keys(body.quantiles).sort((a, b) => a - b),
      paths: body.quantiles,
      anchor: [context.x.at(-1), context.y.at(-1)],
      t0: context.t,
    };
    drawFan(gctx, forecast, GHOST_ALPHA);
    console.log(`context ${context.x.length} steps, forecast ${body.ms.toFixed(0)} ms`);
    scheduleRender();
  } catch (err) {
    console.error(`forecast error: ${err.message}`);
  } finally {
    inFlight = false;
    requestForecast();
  }
}

canvas.addEventListener("pointerdown", e => {
  canvas.setPointerCapture(e.pointerId);
  strokeId++;
  pts = [[e.timeStamp / 1000, ...toUnit(e)]];
  forecast = null;
  sentLength = 0;
  hue = HUE_HIGH;
  drawing = true;
});

canvas.addEventListener("pointermove", e => {
  if (!drawing) return;
  const [, px, py] = pts[pts.length - 1];
  const p = [e.timeStamp / 1000, ...toUnit(e)];
  pts.push(p);
  // Ease toward the colour of the forecast line the pen is following; keep the colour without one.
  const target = followedHue(...p);
  if (target !== null) hue += HUE_SMOOTHING * (target - hue);
  tctx.strokeStyle = color(hue); tctx.lineWidth = LINE_WIDTH; tctx.lineCap = "round";
  tctx.beginPath();
  tctx.moveTo(px * canvas.width, py * canvas.height);
  tctx.lineTo(p[1] * canvas.width, p[2] * canvas.height);
  tctx.stroke();
  scheduleRender();
  requestForecast();
});

canvas.addEventListener("pointerup", () => {
  if (!drawing) return;
  drawing = false;
  strokeId++;  // drop any forecast still in flight
  forecast = null;
  scheduleRender();
});

// Escape clears all saved strokes and ghost forecasts.
document.addEventListener("keydown", e => {
  if (e.key !== "Escape") return;
  gctx.clearRect(0, 0, ghosts.width, ghosts.height);
  tctx.clearRect(0, 0, traces.width, traces.height);
  scheduleRender();
});
