//! Timestral inference for the browser.
//!
//! `forecast` + `model` are plain Rust (usable and testable natively); the `extern "C"` functions at the
//! bottom are the wasm API. JS copies the config and weights into wasm memory via `alloc`, builds a model
//! with `model_new`, then calls `model_forecast` per request.

pub mod forecast;
pub mod kernels;
pub mod model;
pub mod weights;

use forecast::Forecaster;
use model::Model;
use std::sync::Mutex;

pub const NUM_QUANTILES_HINT: usize = 21;

impl Forecaster {
    /// Forecast a 2-D stroke. Returns `[Q][2][H]` (quantile, then x/y, then step), flattened.
    pub fn forecast_xy(&mut self, x: &[f32], y: &[f32], horizon: usize) -> Result<Vec<f32>, String> {
        let nq = self.model.cfg.num_quantiles();
        let q = self.forecast(&[x.to_vec(), y.to_vec()], horizon)?; // (2, H, Q)
        let mut out = vec![0.0f32; nq * 2 * horizon];
        for v in 0..2 {
            for h in 0..horizon {
                for k in 0..nq {
                    out[(k * 2 + v) * horizon + h] = q[(v * horizon + h) * nq + k];
                }
            }
        }
        Ok(out)
    }
}

// ---- wasm C ABI -------------------------------------------------------------------------------------------

static LAST_ERROR: Mutex<String> = Mutex::new(String::new());

fn fail<T>(msg: String, ret: T) -> T {
    *LAST_ERROR.lock().unwrap() = msg;
    ret
}

#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(len.max(1));
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// # Safety
/// `ptr` and `len` must come from a matching `alloc` call.
#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: *mut u8, len: usize) {
    drop(Vec::from_raw_parts(ptr, 0, len.max(1)));
}

#[no_mangle]
pub extern "C" fn error_ptr() -> *const u8 {
    LAST_ERROR.lock().unwrap().as_ptr()
}

#[no_mangle]
pub extern "C" fn error_len() -> usize {
    LAST_ERROR.lock().unwrap().len()
}

/// Build a model from `config.json` text and `model.safetensors` bytes; null on error (see `error_ptr`).
///
/// # Safety
/// Both buffers must be valid for the given lengths.
#[no_mangle]
pub unsafe extern "C" fn model_new(cfg_ptr: *const u8, cfg_len: usize, w_ptr: *const u8, w_len: usize) -> *mut Forecaster {
    let cfg = match std::str::from_utf8(std::slice::from_raw_parts(cfg_ptr, cfg_len)) {
        Ok(s) => s,
        Err(e) => return fail(format!("config.json: {e}"), std::ptr::null_mut()),
    };
    match Model::load(cfg, std::slice::from_raw_parts(w_ptr, w_len)) {
        Ok(m) => Box::into_raw(Box::new(Forecaster::new(m))),
        Err(e) => fail(e, std::ptr::null_mut()),
    }
}

/// # Safety
/// `f` must come from `model_new` and not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn model_free(f: *mut Forecaster) {
    drop(Box::from_raw(f));
}

/// # Safety
/// `f` must be a live pointer from `model_new`.
#[no_mangle]
pub unsafe extern "C" fn model_max_context(f: *mut Forecaster) -> usize {
    (*f).max_context
}

/// # Safety
/// `f` must be a live pointer from `model_new`.
#[no_mangle]
pub unsafe extern "C" fn model_set_max_context(f: *mut Forecaster, steps: usize) {
    (*f).max_context = steps.max((*f).model.cfg.patch_size);
}

/// Forecast `horizon` steps after the stroke `x[0..t]`, `y[0..t]`. Returns a pointer to `42 * horizon`
/// floats laid out `[21][2][horizon]`, valid until the next call and released with `result_free`;
/// null on error (see `error_ptr`).
///
/// # Safety
/// `f` must be live; `x` and `y` must each hold `t` floats.
#[no_mangle]
pub unsafe extern "C" fn model_forecast(f: *mut Forecaster, x: *const f32, y: *const f32, t: usize, horizon: usize) -> *mut f32 {
    let (xs, ys) = (std::slice::from_raw_parts(x, t), std::slice::from_raw_parts(y, t));
    match (*f).forecast_xy(xs, ys, horizon) {
        Ok(v) => {
            let mut b = v.into_boxed_slice();
            let p = b.as_mut_ptr();
            std::mem::forget(b);
            p
        }
        Err(e) => fail(e, std::ptr::null_mut()),
    }
}

/// # Safety
/// `ptr` and `len` must be a result from `model_forecast` (`len` = floats).
#[no_mangle]
pub unsafe extern "C" fn result_free(ptr: *mut f32, len: usize) {
    drop(Box::from_raw(std::slice::from_raw_parts_mut(ptr, len)));
}
