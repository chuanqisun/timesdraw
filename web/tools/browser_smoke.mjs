// Drive the web app in headless Chromium: load the page, wait for the model, draw a stroke, screenshot.
// Usage: node web/tools/browser_smoke.mjs [url] [screenshot.png]   (serve web/ first: python -m http.server -d web/dist 8001)
import { spawn } from "node:child_process";
import { writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const url = process.argv[2] ?? "http://127.0.0.1:8001/";
const shot = process.argv[3] ?? join(tmpdir(), "timesdraw-web.png");
const port = 9333;
const profile = process.env.PROFILE ?? mkdtempSync(join(tmpdir(), "chrome-profile-"));  // reuse PROFILE to test the weight cache
const chrome = spawn(process.env.CHROME ?? "chromium", [
  "--headless=new", `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, "--no-sandbox", ...(process.env.CHROME_ARGS?.split(" ") ?? []), "--window-size=540,960", "about:blank",
], { stdio: "ignore" });
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function main() {
  let target;
  for (let i = 0; i < 50 && !target; i++) {
    try { target = (await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: "PUT" })).json()); } catch { await sleep(200); }
  }
  if (!target) throw new Error("could not reach Chromium");
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise(r => ws.addEventListener("open", r));
  let id = 0;
  const waiting = new Map(), logs = [];
  ws.addEventListener("message", ({ data }) => {
    const m = JSON.parse(data);
    if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); }
    else if (m.method === "Runtime.consoleAPICalled") logs.push(`[${m.params.type}] ` + m.params.args.map(a => a.value ?? a.description).join(" "));
    else if (m.method === "Runtime.exceptionThrown") logs.push("[exception] " + JSON.stringify(m.params.exceptionDetails.exception?.description ?? m.params.exceptionDetails.text));
  });
  const send = (method, params = {}) => new Promise(r => { waiting.set(++id, r); ws.send(JSON.stringify({ id, method, params })); });

  await send("Runtime.enable");
  await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: 540, height: 960, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url });
  if (process.env.EARLY_SHOT) {  // screenshot the loading screen instead of drawing (EARLY_SHOT = ms to wait)
    await sleep(+process.env.EARLY_SHOT);
    const r = await send("Page.captureScreenshot", {});
    writeFileSync(shot, Buffer.from(r.result.data, "base64"));
    console.log("early screenshot:", shot, logs.filter(l => /error/.test(l)).join(" | "));
    ws.close();
    return;
  }
  const t0 = Date.now();
  while (!logs.some(l => l.includes("model:")) && Date.now() - t0 < 90_000) await sleep(200);
  console.log(`model ready after ${((Date.now() - t0) / 1000).toFixed(1)} s`);

  const button = process.env.BUTTON ?? "left";  // BUTTON=right checks that right-drag draws nothing
  const mouse = (type, x, y) => send("Input.dispatchMouseEvent", { type, x, y, button, buttons: type === "mouseReleased" ? 0 : button === "right" ? 2 : 1, clickCount: 1 });
  await mouse("mousePressed", 150, 450);
  for (let i = 1; i <= 90; i++) {
    const a = i / 90 * 2.6 * Math.PI;
    await mouse("mouseMoved", 270 + 120 * Math.cos(a + 3.1), 450 + 120 * Math.sin(a + 3.1) * 1.3);
    await sleep(16);
  }
  await sleep(300);
  await send("Page.captureScreenshot", {}).then(r => writeFileSync(shot, Buffer.from(r.result.data, "base64")));
  await mouse("mouseReleased", 270, 450);
  const px = await send("Runtime.evaluate", { returnByValue: true, expression: `(() => { const c = document.getElementById("canvas"); const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 0; i < d.length; i += 4) if (d[i] | d[i + 1] | d[i + 2]) n++; return n; })()` });
  console.log(`${button} button: ${px.result.result.value} non-black canvas pixels`);
  console.log(logs.filter(l => !l.includes("recording")).slice(0, 4).join("\n"));
  console.log(`... ${logs.filter(l => l.includes("forecast")).length} forecasts, ${logs.filter(l => /error|exception/.test(l)).length} errors`);
  console.log(logs.filter(l => l.includes("forecast")).slice(-3).join("\n"));
  console.log("screenshot:", shot);
  ws.close();
}
try { await main(); } finally { chrome.kill(); }
