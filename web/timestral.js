// Thin JS wrapper over timestral.wasm (see timestral-wasm/src/lib.rs for the C ABI).

export class Timestral {
  /** `wasm`: bytes (ArrayBuffer/Uint8Array) or a fetch Response; `config`: config.json text; `weights`: safetensors bytes. */
  static async create(wasm, config, weights) {
    const source = wasm instanceof Response
      ? await WebAssembly.instantiateStreaming(wasm, {})
      : await WebAssembly.instantiate(wasm, {});
    const t = new Timestral(source.instance.exports);
    const cfg = new TextEncoder().encode(config);
    const [cp, wp] = [t.#put(cfg), t.#put(weights)];  // either alloc may grow memory, so re-view after each
    t.model = t.x.model_new(cp, cfg.length, wp, weights.byteLength);
    t.x.dealloc(cp, cfg.length);
    t.x.dealloc(wp, weights.byteLength);
    if (!t.model) throw new Error(t.#error());
    return t;
  }

  constructor(exports) { this.x = exports; this.model = 0; }

  #put(bytes) {
    const ptr = this.x.alloc(bytes.byteLength);
    const view = ArrayBuffer.isView(bytes) ? new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength) : new Uint8Array(bytes);
    new Uint8Array(this.x.memory.buffer, ptr, bytes.byteLength).set(view);
    return ptr;
  }

  #error() {
    return new TextDecoder().decode(new Uint8Array(this.x.memory.buffer, this.x.error_ptr(), this.x.error_len()));
  }

  /** Context window in steps; the most recent steps are used. */
  get maxContext() { return this.x.model_max_context(this.model); }
  set maxContext(steps) { this.x.model_set_max_context(this.model, steps); }

  /** Forecast `horizon` steps after the stroke (x, y). Returns a Float32Array laid out [21][2][horizon]. */
  forecast(x, y, horizon) {
    const t = x.length;
    const xp = this.#put(Float32Array.from(x)), yp = this.#put(Float32Array.from(y));
    const out = this.x.model_forecast(this.model, xp, yp, t, horizon);
    this.x.dealloc(xp, t * 4);
    this.x.dealloc(yp, t * 4);
    if (!out) throw new Error(this.#error());
    const n = 42 * horizon;
    const result = new Float32Array(this.x.memory.buffer, out, n).slice();
    this.x.result_free(out, n);
    return result;
  }
}
