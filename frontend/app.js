// Collect a stroke on the canvas, resample it to a uniform grid, and draw the forecast.
const HZ = 60, HORIZON = 120, MIN_POINTS = 8, MIN_ALPHA = 0.1;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const status = document.getElementById("status");
let model = "";
let pts = [];  // [t (s), x, y] with x, y in 0..1
let drawing = false;

fetch("/info").then(r => r.json()).then(info => {
  model = info.model;
  status.textContent = `${model} - draw a stroke, release to forecast`;
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

canvas.addEventListener("pointerdown", e => {
  canvas.setPointerCapture(e.pointerId);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  pts = [[e.timeStamp / 1000, ...toUnit(e)]];
  drawing = true;
});

canvas.addEventListener("pointermove", e => {
  if (!drawing) return;
  const prev = pts[pts.length - 1];
  pts.push([e.timeStamp / 1000, ...toUnit(e)]);
  drawPath([prev[1], pts[pts.length - 1][1]], [prev[2], pts[pts.length - 1][2]], "black", 2, 1);
});

canvas.addEventListener("pointerup", async () => {
  if (!drawing) return;
  drawing = false;
  if (pts.length < MIN_POINTS) return;
  const context = resample(pts);
  if (context.x.length < 2) return;
  status.textContent = "Forecasting…";
  const res = await fetch("/forecast", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...context, horizon: HORIZON }),
  });
  const body = await res.json();
  if (!res.ok) { status.textContent = `Error: ${body.error}`; return; }
  // Fade by rank distance from the median: median solid, outermost pair MIN_ALPHA, even steps
  // between. Ranks rather than level values, so the tightly packed tails (0.95, 0.99) stay distinct.
  const levels = Object.keys(body.quantiles).sort((a, b) => a - b);
  const mid = (levels.length - 1) / 2;
  const byDistance = levels.map((level, i) => [level, Math.abs(i - mid)]).sort((a, b) => b[1] - a[1]);
  for (const [level, d] of byDistance) {  // outermost first so the median ends up on top
    const [xs, ys] = body.quantiles[level];
    drawPath(xs, ys, "red", d === 0 ? 3 : 1.5, 1 - (1 - MIN_ALPHA) * d / Math.max(mid, 1));
  }
  status.textContent = `${model} - context ${context.x.length} steps, forecast ${body.ms.toFixed(0)} ms`;
});
