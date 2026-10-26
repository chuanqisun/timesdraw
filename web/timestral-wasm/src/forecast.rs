//! Forecast pipeline around the network: padding, causal statistics (f64, as in training), the
//! block-autoregressive decode with the advancing refine frame, quantile cap, symmetric averaging
//! and non-negative clipping. Mirrors `timestral/forecaster.py` and `timestral/scaler.py`.

use crate::model::{Model, Trace};

/// Per-patch causal `(loc, scale)`, each `(r, t / ps)`: the mean and std of every observed value up to
/// the end of each patch (Welford form in f64, Bessel-corrected), rounded to f32 like the model does.
pub fn patch_stats(data: &[f32], observed: &[bool], r: usize, t: usize, ps: usize, min_scale: f64) -> (Vec<f32>, Vec<f32>) {
    let n = t / ps;
    let (mut loc, mut scale) = (vec![0.0f32; r * n], vec![0.0f32; r * n]);
    for row in 0..r {
        let (mut cw, mut cx, mut m2, mut prev_mean) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for i in 0..t {
            let (x, w) = (data[row * t + i] as f64, if observed[row * t + i] { 1.0 } else { 0.0 });
            cw += w;
            cx += w * x;
            let mean = cx / cw.max(1.0);
            m2 += (x - prev_mean) * (x - mean) * w;
            prev_mean = mean;
            if (i + 1) % ps == 0 {
                let p = i / ps;
                loc[row * n + p] = mean as f32;
                scale[row * n + p] = (m2 / (cw - 1.0).max(1.0)).sqrt().max(min_scale) as f32;
            }
        }
    }
    (loc, scale)
}

/// Replace +/-inf by the largest/smallest finite value of the vector (0 if none).
pub fn clamp_nonfinite(v: &mut [f32]) {
    if v.iter().all(|x| x.is_finite() || x.is_nan()) {
        return;
    }
    let max = v.iter().cloned().filter(|x| x.is_finite()).fold(f32::NEG_INFINITY, f32::max);
    let min = v.iter().cloned().filter(|x| x.is_finite()).fold(f32::INFINITY, f32::min);
    for x in v.iter_mut() {
        let y = if *x == f32::INFINITY { max } else if *x == f32::NEG_INFINITY { min } else { *x };
        *x = if y.is_infinite() { 0.0 } else { y };
    }
}

/// Merge two Welford states `(weight, mean, m2)`.
fn welford_combine(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    let w = a.0 + b.0;
    if w == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let w_safe = w.max(f64::MIN_POSITIVE);
    let delta = b.1 - a.1;
    (w, a.1 + delta * b.0 / w_safe, a.2 + b.2 + delta * delta * a.0 * b.0 / w_safe)
}

/// Per-patch `(loc, scale)` of each horizon patch in the advancing refine frame, for one row:
/// patch `j` is normalized with the context plus the block's medians of patches `0..j`.
fn advancing_block_frames(ctx: &[f32], obs: &[bool], ps: usize, sinh_median: &[f64], n_pred: usize, min_scale: f64) -> (Vec<f64>, Vec<f64>) {
    let mut state = (0.0, 0.0, 0.0);
    for (c, o) in ctx.chunks_exact(ps).zip(obs.chunks_exact(ps)) {
        let cnt = o.iter().filter(|&&b| b).count() as f64;
        let mean = c.iter().zip(o).filter(|(_, &b)| b).map(|(&v, _)| v as f64).sum::<f64>() / cnt.max(1.0);
        let m2 = c.iter().zip(o).filter(|(_, &b)| b).map(|(&v, _)| (v as f64 - mean).powi(2)).sum::<f64>();
        let patch = if cnt == 0.0 { (0.0, 0.0, 0.0) } else { (cnt, mean, m2) };
        state = welford_combine(state, patch);
    }
    let offset = state.1;
    let clip = (f32::MAX as f64).sqrt() / 1e3;
    let (mut locs, mut scales) = (Vec::new(), Vec::new());
    for j in 0..n_pred {
        let var = (state.2 / (state.0 - 1.0).max(1.0)).max(0.0);
        let scale = var.sqrt().max(min_scale);
        locs.push(state.1);
        scales.push(scale);
        let folded: Vec<f64> =
            sinh_median[j * ps..(j + 1) * ps].iter().map(|&m| (m * scale + state.1 - offset).clamp(-clip, clip) + offset).collect();
        let b_mean = folded.iter().sum::<f64>() / ps as f64;
        let b_m2 = folded.iter().map(|v| (v - b_mean).powi(2)).sum::<f64>();
        state = welford_combine(state, (ps as f64, b_mean, b_m2));
    }
    (locs, scales)
}

pub struct Forecaster {
    pub model: Model,
    /// Context cap in steps; the most recent ones are kept.
    pub max_context: usize,
    /// Clamp forecasts to `[min - k * s, max + k * s]` of the context (`s` = last patch scale).
    pub cap_k: f32,
    /// When set, the first network call's intermediates are recorded here (for tests).
    pub trace: Option<Trace>,
}

impl Forecaster {
    pub fn new(model: Model) -> Self {
        let max_context = model.cfg.context_length;
        Self { model, max_context, cap_k: 1e4, trace: None }
    }

    /// Forecast `horizon` steps for the variates `rows` (equal length, finite). Returns `(V, H, Q)` flattened,
    /// averaged with the mirrored forecast of the negated context and clipped at 0 for non-negative rows.
    pub fn forecast(&mut self, rows: &[Vec<f32>], horizon: usize) -> Result<Vec<f32>, String> {
        let v = rows.len();
        let t = rows.first().map_or(0, |r| r.len());
        if v == 0 || t == 0 || horizon == 0 || rows.iter().any(|r| r.len() != t) {
            return Err("need equal-length non-empty rows and horizon >= 1".into());
        }
        if rows.iter().any(|r| r.iter().any(|x| !x.is_finite())) {
            return Err("context must be finite".into());
        }
        // Forward and mirrored passes are two series of one batch; rows only attend within their series.
        let mut pass: Vec<Vec<f32>> = rows.to_vec();
        pass.extend(rows.iter().map(|r| r.iter().map(|x| -x).collect()));
        let groups: Vec<usize> = (0..2 * v).map(|i| i / v).collect();
        let nq = self.model.cfg.num_quantiles();
        let q = self.forecast_raw(&pass, &groups, horizon)?;
        let stride = horizon * nq;
        let mut out = vec![0.0f32; v * stride];
        for i in 0..v {
            let eligible = rows[i].iter().all(|&x| x >= 0.0);
            for h in 0..horizon {
                for k in 0..nq {
                    let fwd = q[i * stride + h * nq + k];
                    let mir = -q[(v + i) * stride + h * nq + (nq - 1 - k)];
                    let m = 0.5 * (fwd + mir);
                    out[i * stride + h * nq + k] = if eligible { m.max(0.0) } else { m };
                }
            }
        }
        Ok(out)
    }

    /// One forecast pass: `(R, H, Q)` raw-space quantiles for `R` packed rows.
    fn forecast_raw(&mut self, rows: &[Vec<f32>], groups: &[usize], horizon: usize) -> Result<Vec<f32>, String> {
        let cfg = self.model.cfg.clone();
        let (ps, nq, mid) = (cfg.patch_size, cfg.num_quantiles(), cfg.median_index());
        let r = rows.len();
        let t0 = rows[0].len().min(self.max_context);
        let tp = t0.div_ceil(ps) * ps;
        let (mut context, mut mask) = (vec![0.0f32; r * tp], vec![false; r * tp]);
        for (row, src) in rows.iter().enumerate() {
            context[row * tp + (tp - t0)..(row + 1) * tp].copy_from_slice(&src[src.len() - t0..]);
            mask[row * tp + (tp - t0)..(row + 1) * tp].fill(true);
        }
        // Bounds for the cap come from the original (trimmed, padded) context.
        let (ctx0, mask0, t_ctx0) = (context.clone(), mask.clone(), tp);

        let per_shot = cfg.block_size.min(32);
        let mut remaining = horizon.div_ceil(ps);
        let mut t = tp;
        let mut blocks: Vec<Vec<f32>> = Vec::new();
        loop {
            let block = per_shot.min(remaining);
            let trace = if self.trace.as_ref().is_some_and(|tr| tr.is_empty()) { self.trace.as_mut() } else { None };
            let q = self.model.predict_block(&context, &mask, r, t, groups, block, trace);
            let q = refine_block(&cfg, q, &context, &mask, r, t, block);
            if let Some(tr) = self.trace.as_mut() {
                if tr.iter().all(|(n, _)| n != "refined") {
                    tr.push(("refined".into(), q.clone()));
                }
            }
            remaining -= block;
            if remaining == 0 {
                blocks.push(q);
                break;
            }
            // Feed the block's median back as context.
            let blen = block * ps;
            let mut next = vec![0.0f32; r * (t + blen)];
            for row in 0..r {
                next[row * (t + blen)..row * (t + blen) + t].copy_from_slice(&context[row * t..(row + 1) * t]);
                for i in 0..blen {
                    let m = q[(row * blen + i) * nq + mid];
                    next[row * (t + blen) + t + i] = if m.is_finite() && m.abs() < 1e30 { m } else { 0.0 };
                }
            }
            let mut nmask = vec![true; r * (t + blen)];
            for row in 0..r {
                nmask[row * (t + blen)..row * (t + blen) + t].copy_from_slice(&mask[row * t..(row + 1) * t]);
            }
            blocks.push(q);
            t += blen;
            context = next;
            mask = nmask;
            if t > self.max_context {
                let keep = ps.max(self.max_context / ps * ps);
                context = (0..r).flat_map(|row| context[row * t + t - keep..(row + 1) * t].to_vec()).collect();
                mask = (0..r).flat_map(|row| mask[row * t + t - keep..(row + 1) * t].to_vec()).collect();
                t = keep;
            }
        }

        // Concatenate blocks along time, truncate to the horizon, then cap to the context range.
        let (_, scale0) = patch_stats(&ctx0, &mask0, r, t_ctx0, ps, cfg.minimum_scale);
        let n0 = t_ctx0 / ps;
        let mut out = vec![0.0f32; r * horizon * nq];
        for row in 0..r {
            let obs = (0..t_ctx0).filter(|&i| mask0[row * t_ctx0 + i]).map(|i| ctx0[row * t_ctx0 + i]);
            let (lo, hi) = obs.fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| (a.min(v), b.max(v)));
            let (lo, hi) = if lo.is_finite() { (lo, hi) } else { (0.0, 0.0) };
            let cap = scale0[row * n0 + n0 - 1] * self.cap_k;
            let mut h = 0;
            for blk in &blocks {
                let blen = blk.len() / r / nq;
                for i in 0..blen {
                    if h == horizon {
                        break;
                    }
                    for k in 0..nq {
                        out[(row * horizon + h) * nq + k] = blk[(row * blen + i) * nq + k].clamp(lo - cap, hi + cap);
                    }
                    h += 1;
                }
            }
        }
        if out.iter().any(|x| x.is_nan()) {
            return Err("forecast produced NaN; the input is degenerate".into());
        }
        Ok(out)
    }
}

/// Move a decoded block from the anchor frame to the advancing refine frame it was trained in: the first
/// horizon patch keeps the anchor frame, each later one is re-denormalized with its own frame.
fn refine_block(cfg: &crate::model::Config, q: Vec<f32>, context: &[f32], mask: &[bool], r: usize, t: usize, n_pred: usize) -> Vec<f32> {
    if n_pred <= 1 {
        return q;
    }
    let (ps, nq, mid) = (cfg.patch_size, cfg.num_quantiles(), cfg.median_index());
    let n = t / ps;
    let (loc, scale) = patch_stats(context, mask, r, t, ps, cfg.minimum_scale);
    let blen = n_pred * ps;
    let mut out = q.clone();
    for row in 0..r {
        let (loc0, scale0) = (loc[row * n + n - 1], scale[row * n + n - 1]);
        let sinh_median: Vec<f64> =
            (0..blen).map(|i| (q[(row * blen + i) * nq + mid] as f64 - loc0 as f64) / scale0 as f64).collect();
        let (lj, sj) = advancing_block_frames(
            &context[row * t..(row + 1) * t], &mask[row * t..(row + 1) * t], ps, &sinh_median, n_pred, cfg.minimum_scale,
        );
        for i in 0..blen {
            let j = i / ps;
            let (l, s) = if j == 0 { (loc0, scale0) } else { (lj[j] as f32, sj[j] as f32) };
            let base = (row * blen + i) * nq;
            for k in 0..nq {
                out[base + k] = (q[base + k] - loc0) / scale0 * s + l;
            }
            clamp_nonfinite(&mut out[base..base + nq]);
        }
    }
    out
}
