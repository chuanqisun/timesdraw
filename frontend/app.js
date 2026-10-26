// Collect a stroke on the canvas, resample it to a uniform grid, and draw the forecast live.
const HZ = 60, HORIZON = 120, MIN_POINTS = 8, MIN_ALPHA = 0.1;
const LINE_WIDTH = 2, STROKE_COLOR = "#eee", FORECAST_COLOR = "#ff4d4d";

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
let model = "";
let pts = [];  // [t (s), x, y] with x, y in 0..1
let drawing = false;
let forecast = null;  // latest response's quantiles for the current stroke
let strokeId = 0;     // bumped on pointerdown so late responses for an old stroke are dropped
let inFlight = false;
let sentLength = 0;   // pts.length covered by the last request; more points means a new one is due
let frameQueued = false;

fetch("/info").then(r => r.json()).then(info => {
  model = info.model;
  console.log(`model: ${model}`);
});

function toUnit(e) {
  const r = canvas.getBoundingClientRect();
  return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
}

function drawPath(xs, ys, color, width, alpha) {
  ctx.save();
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.globalAlpha = alpha;
  ctx.lineJoin = ctx.lineCap = "round";
  ctx.beginPath();
  xs.forEach((x, i) => {
    const px = x * canvas.width, py = ys[i] * canvas.height;
    i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
  });
  ctx.stroke();
  ctx.restore();
}

// Irregular pointer samples -> uniform HZ grid via linear interpolation.
function resample(pts) {
  const xs = [], ys = [];
  let j = 0;
  for (let t = pts[0][0]; t < pts[pts.length - 1][0]; t += 1 / HZ) {
    while (pts[j + 1][0] < t) j++;
    const [t0, x0, y0] = pts[j], [t1, x1, y1] = pts[j + 1];
    const a = t1 > t0 ? (t - t0) / (t1 - t0) : 0;
    xs.push(x0 + a * (x1 - x0));
    ys.push(y0 + a * (y1 - y0));
  }
  return { x: xs, y: ys };
}

// Redraw everything: forecast underneath, stroke on top.
function render() {
  frameQueued = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (forecast) {
    // Fade by rank distance from the median: median solid, outermost pair MIN_ALPHA, even steps
    // between. Ranks rather than level values, so the tightly packed tails (0.95, 0.99) stay distinct.
    const levels = Object.keys(forecast).sort((a, b) => a - b);
    const mid = (levels.length - 1) / 2;
    const byDistance = levels.map((level, i) => [level, Math.abs(i - mid)]).sort((a, b) => b[1] - a[1]);
    for (const [level, d] of byDistance) {  // outermost first so the median ends up on top
      const [xs, ys] = forecast[level];
      drawPath(xs, ys, FORECAST_COLOR, LINE_WIDTH, 1 - (1 - MIN_ALPHA) * d / Math.max(mid, 1));
    }
  }
  if (pts.length > 1) drawPath(pts.map(p => p[1]), pts.map(p => p[2]), STROKE_COLOR, LINE_WIDTH, 1);
}

function scheduleRender() {
  if (!frameQueued) { frameQueued = true; requestAnimationFrame(render); }
}

// Keep at most one request in flight; when it returns, send another if the stroke grew meanwhile.
async function requestForecast() {
  if (inFlight || pts.length < MIN_POINTS || pts.length === sentLength) return;
  const context = resample(pts);
  if (context.x.length < 2) return;
  inFlight = true;
  sentLength = pts.length;
  const id = strokeId;
  try {
    const res = await fetch("/forecast", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...context, horizon: HORIZON }),
    });
    const body = await res.json();
    if (id !== strokeId) return;
    if (!res.ok) { console.error(`forecast error: ${body.error}`); return; }
    forecast = body.quantiles;
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
  drawing = true;
  scheduleRender();
});

canvas.addEventListener("pointermove", e => {
  if (!drawing) return;
  pts.push([e.timeStamp / 1000, ...toUnit(e)]);
  scheduleRender();
  requestForecast();
});

canvas.addEventListener("pointerup", () => {
  if (!drawing) return;
  drawing = false;
  requestForecast();  // final forecast from the complete stroke, if the last one didn't cover it
});
