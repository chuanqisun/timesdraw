//! The Timestral network: patch embedding, time/variate transformer over `[Enc ; Dec]` tokens,
//! and the monotone quantile head. Mirrors `timestral/{model,layers,masks}.py`.

use crate::forecast::patch_stats;
use crate::kernels::{axpy, dot, fast_exp, matmul, rms_norm, sigmoid_mul, silu, swiglu};
use crate::weights::Tensors;
use serde::Deserialize;

pub type Trace = Vec<(String, Vec<f32>)>;

fn d_true() -> bool {
    true
}
fn d_gate() -> String {
    "sdpa_sigmoid".into()
}

#[derive(Deserialize, Clone, Debug)]
pub struct Config {
    pub patch_size: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub mlp_hidden_dim: usize,
    pub block_schedule: Vec<String>,
    pub quantile_knots: Vec<f64>,
    pub block_size: usize,
    pub context_length: usize,
    #[serde(default = "d_theta")]
    pub rope_theta: f64,
    #[serde(default = "d_half")]
    pub rope_partial_factor: f64,
    #[serde(default = "d_eps")]
    pub norm_eps: f64,
    #[serde(default)]
    pub attn_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default = "d_gate")]
    pub time_attn_gate: String,
    #[serde(default = "d_gate")]
    pub variate_attn_gate: String,
    #[serde(default = "d_min_scale")]
    pub minimum_scale: f64,
    #[serde(default = "d_true")]
    pub use_asinh: bool,
    #[serde(default = "d_true")]
    pub attend_missing_patches: bool,
    #[serde(default = "d_two")]
    pub quantile_width_factor: f64,
}
fn d_theta() -> f64 {
    10000.0
}
fn d_half() -> f64 {
    0.5
}
fn d_eps() -> f64 {
    1e-6
}
fn d_min_scale() -> f64 {
    1e-6
}
fn d_two() -> f64 {
    2.0
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }
    pub fn num_quantiles(&self) -> usize {
        self.quantile_knots.len()
    }
    pub fn median_index(&self) -> usize {
        self.quantile_knots.iter().position(|&q| q == 0.5).unwrap()
    }
}

/// A linear layer stored transposed (`wt[k][n]`) for the matmul kernel.
pub struct Linear {
    k: usize,
    n: usize,
    wt: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl Linear {
    /// `weight` is PyTorch's `[out, in]`; output column `c` is taken from source row `rows(c)`.
    fn new(weight: (Vec<usize>, Vec<f32>), bias: Option<Vec<f32>>, rows: impl Fn(usize) -> usize) -> Result<Self, String> {
        let (shape, w) = weight;
        if shape.len() != 2 {
            return Err("linear weight must be 2-D".into());
        }
        let (n, k) = (shape[0], shape[1]);
        let mut wt = vec![0.0; k * n];
        for c in 0..n {
            let src = &w[rows(c) * k..(rows(c) + 1) * k];
            for (kk, v) in src.iter().enumerate() {
                wt[kk * n + c] = *v;
            }
        }
        Ok(Self { k, n, wt, bias })
    }

    fn apply(&self, x: &[f32], m: usize) -> Vec<f32> {
        let mut out = vec![0.0; m * self.n];
        matmul(x, m, self.k, &self.wt, self.n, self.bias.as_deref(), &mut out);
        out
    }
}

/// `output(silu(hidden(x))) + residual(x)`.
struct ResBlock {
    hidden: Linear,
    output: Linear,
    residual: Linear,
}

impl ResBlock {
    fn load(t: &Tensors, prefix: &str, bias: bool) -> Result<Self, String> {
        let lin = |name: &str| -> Result<Linear, String> {
            let b = if bias { Some(t.get(&format!("{prefix}.{name}.bias"))?.1) } else { None };
            Linear::new(t.get(&format!("{prefix}.{name}.weight"))?, b, |c| c)
        };
        Ok(Self { hidden: lin("hidden_layer")?, output: lin("output_layer")?, residual: lin("residual_layer")? })
    }

    fn apply(&self, x: &[f32], m: usize) -> Vec<f32> {
        let mut h = self.hidden.apply(x, m);
        for v in h.iter_mut() {
            *v = silu(*v);
        }
        let mut out = self.output.apply(&h, m);
        for (o, r) in out.iter_mut().zip(self.residual.apply(x, m)) {
            *o += r;
        }
        out
    }
}

struct Attn {
    is_time: bool,
    qkv: Linear,
    o: Linear,
    gate: Option<Linear>,
}

struct Block {
    norms: Vec<Vec<f32>>,
    attns: Vec<Attn>,
    mlp_norm: Vec<f32>,
    w12: Linear,
    w3: Linear,
}

pub struct Model {
    pub cfg: Config,
    patch_proj: ResBlock,
    forecast_query: Vec<f32>,
    blocks: Vec<Block>,
    out_norm: Vec<f32>,
    head: ResBlock,
    inv_freq: Vec<f64>,
}

impl Model {
    pub fn load(config_json: &str, safetensors: &[u8]) -> Result<Self, String> {
        let cfg: Config = serde_json::from_str(config_json).map_err(|e| format!("config.json: {e}"))?;
        if cfg.attn_bias || cfg.mlp_bias {
            return Err("biased attention/MLP layers are not supported".into());
        }
        let t = Tensors::parse(safetensors)?;
        let (d, nh, hd) = (cfg.d_model, cfg.num_heads, cfg.head_dim());
        let mut blocks = Vec::new();
        for (li, kind) in cfg.block_schedule.iter().enumerate() {
            let axes: &[bool] = match kind.as_str() {
                "time" => &[true],
                "variate" => &[false],
                "time_variate" => &[true, false],
                "variate_time" => &[false, true],
                other => return Err(format!("unknown layer kind {other}")),
            };
            let p = format!("transformer.layers.{li}");
            let (mut norms, mut attns) = (Vec::new(), Vec::new());
            for (ai, &is_time) in axes.iter().enumerate() {
                norms.push(t.get(&format!("{p}.attn_norms.{ai}.scale"))?.1);
                let a = format!("{p}.attns.{ai}");
                // The fused weight's rows are laid out (qkv, head_dim, n_heads); regroup to (qkv, n_heads, head_dim).
                let qkv = Linear::new(t.get(&format!("{a}.wQKV.weight"))?, None, |c| {
                    let (which, rest) = (c / d, c % d);
                    which * d + (rest % hd) * nh + rest / hd
                })?;
                let o = Linear::new(t.get(&format!("{a}.wO.weight"))?, None, |c| c)?;
                let gate_kind = if is_time { &cfg.time_attn_gate } else { &cfg.variate_attn_gate };
                let gate = if gate_kind == "sdpa_sigmoid" {
                    Some(Linear::new(t.get(&format!("{a}._gate.weight"))?, None, |c| c)?)
                } else {
                    None
                };
                attns.push(Attn { is_time, qkv, o, gate });
            }
            blocks.push(Block {
                norms,
                attns,
                mlp_norm: t.get(&format!("{p}.mlp_norm.scale"))?.1,
                w12: Linear::new(t.get(&format!("{p}.mlp.w12.weight"))?, None, |c| c)?,
                w3: Linear::new(t.get(&format!("{p}.mlp.w3.weight"))?, None, |c| c)?,
            });
        }
        let rope_dim = (hd as f64 * cfg.rope_partial_factor) as usize;
        let inv_freq = (0..rope_dim / 2).map(|j| 1.0 / cfg.rope_theta.powf(2.0 * j as f64 / rope_dim as f64)).collect();
        Ok(Self {
            patch_proj: ResBlock::load(&t, "representation.patch_proj", true)?,
            forecast_query: t.get("representation.forecast_query")?.1,
            out_norm: t.get("transformer.out_norm.scale")?.1,
            head: ResBlock::load(&t, "head.projection", false)?,
            blocks,
            inv_freq,
            cfg,
        })
    }

    /// Forecast `h_patches` patches after packed `(r, t)` context rows (`t` a multiple of the patch size).
    /// Rows with equal `groups` ids are variates of one series and attend to each other.
    /// Returns denormalized quantiles `(r, h_patches * patch_size, Q)`, anchored at the last context patch.
    pub fn predict_block(
        &self, inputs: &[f32], observed: &[bool], r: usize, t: usize, groups: &[usize], h_patches: usize,
        mut trace: Option<&mut Trace>,
    ) -> Vec<f32> {
        let cfg = &self.cfg;
        let (ps, d, nq) = (cfg.patch_size, cfg.d_model, cfg.num_quantiles());
        let (ll, h) = (t / ps, h_patches);
        let s = ll + h;
        let mut rec = |name: &str, v: &[f32]| {
            if let Some(tr) = trace.as_deref_mut() {
                tr.push((name.to_string(), v.to_vec()));
            }
        };

        // Causal per-patch normalization, then the patch embedding of [values, missing].
        let (loc, scale) = patch_stats(inputs, observed, r, t, ps, cfg.minimum_scale);
        let mut scaled = vec![0.0f32; r * t];
        let mut embed_in = vec![0.0f32; r * ll * 2 * ps];
        for row in 0..r {
            for p in 0..ll {
                let (lo, sc) = (loc[row * ll + p], scale[row * ll + p].max(f32::EPSILON));
                for i in 0..ps {
                    let idx = row * t + p * ps + i;
                    let v = if observed[idx] {
                        let z = (inputs[idx] - lo) / sc;
                        if cfg.use_asinh { z.asinh() } else { z }
                    } else {
                        0.0
                    };
                    scaled[idx] = v;
                    let base = (row * ll + p) * 2 * ps;
                    embed_in[base + i] = v;
                    embed_in[base + ps + i] = if observed[idx] { 0.0 } else { 1.0 };
                }
            }
        }
        rec("scaled", &scaled);
        let expand = |v: &[f32]| -> Vec<f32> { v.iter().flat_map(|&x| std::iter::repeat(x).take(ps)).collect() };
        rec("loc", &expand(&loc));
        rec("scale", &expand(&scale));
        let enc = self.patch_proj.apply(&embed_in, r * ll);
        rec("embed", &enc);

        let mut x = vec![0.0f32; r * s * d];
        for row in 0..r {
            x[row * s * d..(row * s + ll) * d].copy_from_slice(&enc[row * ll * d..(row + 1) * ll * d]);
            for j in 0..h {
                x[(row * s + ll + j) * d..(row * s + ll + j + 1) * d].copy_from_slice(&self.forecast_query);
            }
        }

        // Which patches are valid keys, and RoPE positions anchored at each series' first observed patch.
        let mut key_valid = vec![false; r * ll];
        for row in 0..r {
            let mut started = false;
            for p in 0..ll {
                let obs = observed[row * t + p * ps..row * t + (p + 1) * ps].iter().any(|&o| o);
                started |= obs;
                key_valid[row * ll + p] = if cfg.attend_missing_patches { started } else { obs };
            }
        }
        let first_obs: Vec<usize> = groups
            .iter()
            .map(|&g| {
                (0..ll)
                    .find(|&p| (0..r).any(|row| groups[row] == g && observed[row * t + p * ps..row * t + (p + 1) * ps].iter().any(|&o| o)))
                    .unwrap_or(0)
            })
            .collect();
        let positions: Vec<i64> = (0..r * s).map(|i| (i % s) as i64 - first_obs[i / s] as i64).collect();

        for (li, block) in self.blocks.iter().enumerate() {
            for (norm, attn) in block.norms.iter().zip(&block.attns) {
                let xn = rms_norm(&x, d, norm, cfg.norm_eps as f32);
                let mut out = if attn.is_time {
                    self.time_attn(attn, &xn, r, s, ll, &key_valid, &positions)
                } else {
                    self.variate_attn(attn, &xn, r, s, ll, &key_valid, groups)
                };
                if let Some(g) = &attn.gate {
                    sigmoid_mul(&mut out, &g.apply(&xn, r * s));
                }
                for (xv, ov) in x.iter_mut().zip(attn.o.apply(&out, r * s)) {
                    *xv += ov;
                }
            }
            let xn = rms_norm(&x, d, &block.mlp_norm, cfg.norm_eps as f32);
            let h12 = block.w12.apply(&xn, r * s);
            let hid = cfg.mlp_hidden_dim;
            let mut act = vec![0.0f32; r * s * hid];
            for (i, a) in act.chunks_exact_mut(hid).enumerate() {
                let row = &h12[i * 2 * hid..(i + 1) * 2 * hid];
                swiglu(&row[..hid], &row[hid..], a);
            }
            for (xv, ov) in x.iter_mut().zip(block.w3.apply(&act, r * s)) {
                *xv += ov;
            }
            rec(&format!("layer{li}"), &x);
        }
        let x = rms_norm(&x, d, &self.out_norm, cfg.norm_eps as f32);
        rec("transformer", &x);

        // Head on the decoder tokens.
        let mut dec = vec![0.0f32; r * h * d];
        for row in 0..r {
            dec[row * h * d..(row + 1) * h * d].copy_from_slice(&x[(row * s + ll) * d..(row * s + s) * d]);
        }
        let raw = self.head.apply(&dec, r * h);
        let mut q = vec![0.0f32; r * h * ps * nq];
        for (rv, qv) in raw.chunks_exact(nq).zip(q.chunks_exact_mut(nq)) {
            self.monotone(rv, qv);
        }
        rec("head", &q);

        // Back to raw space with the last context patch's frame: sinh(p) * scale + loc.
        for row in 0..r {
            let (lo, sc) = (loc[row * ll + ll - 1], scale[row * ll + ll - 1]);
            let block = &mut q[row * h * ps * nq..(row + 1) * h * ps * nq];
            for v in block.iter_mut() {
                let z = if cfg.use_asinh { v.sinh() } else { *v };
                *v = z * sc + lo;
            }
            for qv in block.chunks_exact_mut(nq) {
                crate::forecast::clamp_nonfinite(qv);
            }
        }
        rec("denorm", &q);
        q
    }

    /// Non-crossing quantiles: the median channel is free, the others fan out from it by cumulative
    /// `width_factor * dp * softplus(raw)`.
    fn monotone(&self, raw: &[f32], out: &mut [f32]) {
        let levels = &self.cfg.quantile_knots;
        let m = self.cfg.median_index();
        let wf = self.cfg.quantile_width_factor as f32;
        let softplus = |v: f32| if v > 20.0 { v } else { v.exp().ln_1p() };
        out[m] = raw[m];
        let mut acc = 0.0;
        for i in m + 1..levels.len() {
            acc += wf * (levels[i] - levels[i - 1]) as f32 * softplus(raw[i]);
            out[i] = raw[m] + acc;
        }
        acc = 0.0;
        for i in (0..m).rev() {
            acc += wf * (levels[i + 1] - levels[i]) as f32 * softplus(raw[i]);
            out[i] = raw[m] - acc;
        }
    }

    /// Time attention per row over patches, with RoPE. Encoder queries see earlier valid encoder keys
    /// (and themselves); decoder queries see all valid encoder keys and earlier decoder tokens.
    fn time_attn(&self, a: &Attn, xn: &[f32], r: usize, s: usize, ll: usize, key_valid: &[bool], pos: &[i64]) -> Vec<f32> {
        let cfg = &self.cfg;
        let (d, nh, hd) = (cfg.d_model, cfg.num_heads, cfg.head_dim());
        let half = hd / 2;
        let mut qkv = a.qkv.apply(xn, r * s);
        for tok in 0..r * s {
            for (j, f) in self.inv_freq.iter().enumerate() {
                let (sn, c) = (pos[tok] as f64 * f).sin_cos();
                let (sn, c) = (sn as f32, c as f32);
                for part in 0..2 {
                    for h in 0..nh {
                        let base = tok * 3 * d + part * d + h * hd;
                        let (u, v) = (qkv[base + j], qkv[base + j + half]);
                        qkv[base + j] = u * c - v * sn;
                        qkv[base + j + half] = v * c + u * sn;
                    }
                }
            }
        }
        let inv_sqrt = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0.0f32; r * s * d];
        let (mut keys, mut scores) = (Vec::with_capacity(s), Vec::with_capacity(s));
        for row in 0..r {
            for h in 0..nh {
                for qi in 0..s {
                    keys.clear();
                    if qi < ll {
                        keys.extend((0..=qi).filter(|&j| j == qi || key_valid[row * ll + j]));
                    } else {
                        keys.extend((0..ll).filter(|&j| key_valid[row * ll + j]));
                        keys.extend(ll..=qi);
                    }
                    let qv = &qkv[(row * s + qi) * 3 * d + h * hd..][..hd];
                    attend(&qkv, qv, keys.iter().map(|&j| row * s + j), d, h * hd, hd, inv_sqrt, &mut scores, &mut out[(row * s + qi) * d + h * hd..][..hd]);
                }
            }
        }
        out
    }

    /// Variate attention: at each token slot, rows of the same series attend to each other
    /// (no RoPE); a key row must hold a valid patch in that slot, and every row sees itself.
    fn variate_attn(&self, a: &Attn, xn: &[f32], r: usize, s: usize, ll: usize, key_valid: &[bool], groups: &[usize]) -> Vec<f32> {
        let cfg = &self.cfg;
        let (d, nh, hd) = (cfg.d_model, cfg.num_heads, cfg.head_dim());
        let qkv = a.qkv.apply(xn, r * s);
        let inv_sqrt = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0.0f32; r * s * d];
        let mut scores = Vec::with_capacity(r);
        for slot in 0..s {
            for row in 0..r {
                let keys: Vec<usize> = (0..r)
                    .filter(|&k| k == row || (groups[k] == groups[row] && (slot >= ll || key_valid[k * ll + slot])))
                    .collect();
                for h in 0..nh {
                    let qv = &qkv[(row * s + slot) * 3 * d + h * hd..][..hd];
                    attend(&qkv, qv, keys.iter().map(|&k| k * s + slot), d, h * hd, hd, inv_sqrt, &mut scores, &mut out[(row * s + slot) * d + h * hd..][..hd]);
                }
            }
        }
        out
    }
}

/// softmax(q·K^T / sqrt(hd)) · V over the given token indices, written to `out`.
#[allow(clippy::too_many_arguments)]
fn attend(
    qkv: &[f32], q: &[f32], toks: impl Iterator<Item = usize>, d: usize, head_off: usize, hd: usize, inv_sqrt: f32,
    scores: &mut Vec<f32>, out: &mut [f32],
) {
    let toks: Vec<usize> = toks.collect();
    scores.clear();
    scores.extend(toks.iter().map(|&k| dot(q, &qkv[k * 3 * d + d + head_off..][..hd]) * inv_sqrt));
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for sc in scores.iter_mut() {
        *sc = fast_exp(*sc - max);
        sum += *sc;
    }
    for (&k, &p) in toks.iter().zip(scores.iter()) {
        axpy(out, p / sum, &qkv[k * 3 * d + 2 * d + head_off..][..hd]);
    }
}
