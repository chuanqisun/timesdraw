//! Numeric kernels. Each has a plain scalar version and, when built with `+simd128` (the wasm
//! build), a SIMD128 version.

/// exp(x) from range reduction and a degree-7 Taylor polynomial; relative error ~1e-8.
#[inline]
pub fn fast_exp(x: f32) -> f32 {
    let x = x.clamp(-87.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E).round();
    let r = (x - n * 0.693_145_75) - n * 1.428_606_8e-6;
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (1.0 / 6.0
                    + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0 + r * (1.0 / 5040.0)))))));
    p * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + fast_exp(-x))
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// `out = X[m,k] · Wt[k,n] (+ bias)`, with `Wt` stored k-major (row `k` holds the `n` outputs).
pub fn matmul(x: &[f32], m: usize, k: usize, wt: &[f32], n: usize, bias: Option<&[f32]>, out: &mut [f32]) {
    debug_assert!(x.len() == m * k && wt.len() == k * n && out.len() == m * n);
    #[cfg(target_feature = "simd128")]
    simd::matmul(x, m, k, wt, n, bias, out);
    #[cfg(not(target_feature = "simd128"))]
    scalar_matmul(x, m, k, wt, n, bias, out);
}

#[cfg(not(target_feature = "simd128"))]
fn scalar_matmul(x: &[f32], m: usize, k: usize, wt: &[f32], n: usize, bias: Option<&[f32]>, out: &mut [f32]) {
    for i in 0..m {
        let o = &mut out[i * n..(i + 1) * n];
        match bias {
            Some(b) => o.copy_from_slice(b),
            None => o.fill(0.0),
        }
        for kk in 0..k {
            let xv = x[i * k + kk];
            for (ov, wv) in o.iter_mut().zip(&wt[kk * n..(kk + 1) * n]) {
                *ov += xv * wv;
            }
        }
    }
}

/// `sum(a * b)`.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_feature = "simd128")]
    return simd::dot(a, b);
    #[cfg(not(target_feature = "simd128"))]
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `out += a * v`.
pub fn axpy(out: &mut [f32], a: f32, v: &[f32]) {
    #[cfg(target_feature = "simd128")]
    simd::axpy(out, a, v);
    #[cfg(not(target_feature = "simd128"))]
    for (o, x) in out.iter_mut().zip(v) {
        *o += a * x;
    }
}

/// `x[i] *= sigmoid(g[i])`.
pub fn sigmoid_mul(x: &mut [f32], g: &[f32]) {
    #[cfg(target_feature = "simd128")]
    simd::sigmoid_mul(x, g);
    #[cfg(not(target_feature = "simd128"))]
    for (xv, gv) in x.iter_mut().zip(g) {
        *xv *= sigmoid(*gv);
    }
}

/// `out[i] = gate[i] * silu(up[i])`.
pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    #[cfg(target_feature = "simd128")]
    simd::swiglu(gate, up, out);
    #[cfg(not(target_feature = "simd128"))]
    for ((o, g), u) in out.iter_mut().zip(gate).zip(up) {
        *o = g * silu(*u);
    }
}

/// RMSNorm over each `d`-wide row of `x`, times `scale`.
pub fn rms_norm(x: &[f32], d: usize, scale: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for (xr, or) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let inv = 1.0 / (dot(xr, xr) / d as f32 + eps).sqrt();
        for ((o, v), s) in or.iter_mut().zip(xr).zip(scale) {
            *o = v * inv * s;
        }
    }
    out
}

#[cfg(target_feature = "simd128")]
mod simd {
    use core::arch::wasm32::*;

    /// 4 rows (or `MR`) by 8 columns of the output at a time, accumulated over all of `k`.
    #[inline(always)]
    unsafe fn kernel<const MR: usize>(
        x: &[f32], i: usize, k: usize, wt: &[f32], n: usize, j: usize, bias: Option<&[f32]>, out: &mut [f32],
    ) {
        let (b0, b1) = match bias {
            Some(b) => (v128_load(b.as_ptr().add(j) as *const v128), v128_load(b.as_ptr().add(j + 4) as *const v128)),
            None => (f32x4_splat(0.0), f32x4_splat(0.0)),
        };
        let mut acc = [[b0, b1]; MR];
        let mut wp = wt.as_ptr().add(j);
        for kk in 0..k {
            let w0 = v128_load(wp as *const v128);
            let w1 = v128_load(wp.add(4) as *const v128);
            for r in 0..MR {
                let xv = f32x4_splat(*x.get_unchecked((i + r) * k + kk));
                acc[r][0] = f32x4_add(acc[r][0], f32x4_mul(xv, w0));
                acc[r][1] = f32x4_add(acc[r][1], f32x4_mul(xv, w1));
            }
            wp = wp.add(n);
        }
        for r in 0..MR {
            let op = out.as_mut_ptr().add((i + r) * n + j);
            v128_store(op as *mut v128, acc[r][0]);
            v128_store(op.add(4) as *mut v128, acc[r][1]);
        }
    }

    pub fn matmul(x: &[f32], m: usize, k: usize, wt: &[f32], n: usize, bias: Option<&[f32]>, out: &mut [f32]) {
        let nv = n / 8 * 8;
        let mut i = 0;
        unsafe {
            while i + 4 <= m {
                for j in (0..nv).step_by(8) {
                    kernel::<4>(x, i, k, wt, n, j, bias, out);
                }
                i += 4;
            }
            while i < m {
                for j in (0..nv).step_by(8) {
                    kernel::<1>(x, i, k, wt, n, j, bias, out);
                }
                i += 1;
            }
        }
        if nv < n {
            for i in 0..m {
                for j in nv..n {
                    let mut s = bias.map_or(0.0, |b| b[j]);
                    for kk in 0..k {
                        s += x[i * k + kk] * wt[kk * n + j];
                    }
                    out[i * n + j] = s;
                }
            }
        }
    }

    #[inline(always)]
    fn hsum(v: v128) -> f32 {
        f32x4_extract_lane::<0>(v) + f32x4_extract_lane::<1>(v) + f32x4_extract_lane::<2>(v) + f32x4_extract_lane::<3>(v)
    }

    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let mut acc = [f32x4_splat(0.0); 4];
        let mut i = 0;
        unsafe {
            while i + 16 <= n {
                for c in 0..4 {
                    let av = v128_load(a.as_ptr().add(i + 4 * c) as *const v128);
                    let bv = v128_load(b.as_ptr().add(i + 4 * c) as *const v128);
                    acc[c] = f32x4_add(acc[c], f32x4_mul(av, bv));
                }
                i += 16;
            }
        }
        let mut s = hsum(f32x4_add(f32x4_add(acc[0], acc[1]), f32x4_add(acc[2], acc[3])));
        while i < n {
            s += a[i] * b[i];
            i += 1;
        }
        s
    }

    pub fn axpy(out: &mut [f32], a: f32, v: &[f32]) {
        let n = out.len();
        let av = f32x4_splat(a);
        let mut i = 0;
        unsafe {
            while i + 4 <= n {
                let o = out.as_mut_ptr().add(i) as *mut v128;
                v128_store(o, f32x4_add(v128_load(o), f32x4_mul(av, v128_load(v.as_ptr().add(i) as *const v128))));
                i += 4;
            }
        }
        while i < n {
            out[i] += a * v[i];
            i += 1;
        }
    }

    /// Same polynomial as `fast_exp`, four lanes at a time.
    #[inline(always)]
    fn exp4(x: v128) -> v128 {
        let x = f32x4_min(f32x4_max(x, f32x4_splat(-87.0)), f32x4_splat(88.0));
        let n = f32x4_nearest(f32x4_mul(x, f32x4_splat(std::f32::consts::LOG2_E)));
        let r = f32x4_sub(
            f32x4_sub(x, f32x4_mul(n, f32x4_splat(0.693_145_75))),
            f32x4_mul(n, f32x4_splat(1.428_606_8e-6)),
        );
        let mut p = f32x4_splat(1.0 / 5040.0);
        for c in [1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0] {
            p = f32x4_add(f32x4_mul(p, r), f32x4_splat(c));
        }
        let pow2 = i32x4_shl(i32x4_add(i32x4_trunc_sat_f32x4(n), i32x4_splat(127)), 23);
        f32x4_mul(p, pow2)
    }

    #[inline(always)]
    fn sigmoid4(x: v128) -> v128 {
        let one = f32x4_splat(1.0);
        f32x4_div(one, f32x4_add(one, exp4(f32x4_neg(x))))
    }

    pub fn sigmoid_mul(x: &mut [f32], g: &[f32]) {
        let n = x.len();
        let mut i = 0;
        unsafe {
            while i + 4 <= n {
                let xp = x.as_mut_ptr().add(i) as *mut v128;
                let gv = v128_load(g.as_ptr().add(i) as *const v128);
                v128_store(xp, f32x4_mul(v128_load(xp), sigmoid4(gv)));
                i += 4;
            }
        }
        while i < n {
            x[i] *= super::sigmoid(g[i]);
            i += 1;
        }
    }

    pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
        let n = out.len();
        let mut i = 0;
        unsafe {
            while i + 4 <= n {
                let u = v128_load(up.as_ptr().add(i) as *const v128);
                let g = v128_load(gate.as_ptr().add(i) as *const v128);
                v128_store(out.as_mut_ptr().add(i) as *mut v128, f32x4_mul(g, f32x4_mul(u, sigmoid4(u))));
                i += 4;
            }
        }
        while i < n {
            out[i] = gate[i] * super::silu(up[i]);
            i += 1;
        }
    }
}
