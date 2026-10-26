//! Compares the Rust port against PyTorch outputs written by `web/tools/make_golden.py`.

use std::path::{Path, PathBuf};
use timestral_wasm::forecast::Forecaster;
use timestral_wasm::model::Model;

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../tools/golden")
}

fn read_f32(name: &str) -> Vec<f32> {
    let bytes = std::fs::read(golden_dir().join(name)).unwrap_or_else(|_| panic!("missing {name}; run web/tools/make_golden.py"));
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn manifest() -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(golden_dir().join("manifest.json")).expect("run web/tools/make_golden.py")).unwrap()
}

fn load() -> Forecaster {
    let m = manifest();
    let dir = PathBuf::from(m["checkpoint"].as_str().unwrap());
    let model = Model::load(
        &std::fs::read_to_string(dir.join("config.json")).unwrap(),
        &std::fs::read(dir.join("model.safetensors")).unwrap(),
    )
    .unwrap();
    Forecaster::new(model)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert!(a.len() >= b.len(), "got {} values, expected {}", a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn stages_match_pytorch() {
    let mut fc = load();
    let ctx = read_f32("arc.in.f32");
    let (x, y) = ctx.split_at(120);
    fc.trace = Some(Vec::new());
    fc.forecast(&[x.to_vec(), y.to_vec()], 120).unwrap();
    let trace = fc.trace.take().unwrap();
    let mut worst = 0.0f32;
    for (name, got) in &trace {
        let want = read_f32(&format!("stage_{name}.f32"));
        // The Rust pass batches forward and mirrored series; the first rows are the forward pass.
        let d = max_diff(got, &want);
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
        println!("stage {name:12} max|diff| {d:.3e}  (max|ref| {scale:.3e})");
        worst = worst.max(d / scale);
    }
    assert!(worst < 1e-4, "relative stage error {worst}");
}

#[test]
fn forecasts_match_pytorch() {
    let m = manifest();
    let mut forecasters = std::collections::HashMap::new();
    for f in m["fixtures"].as_array().unwrap() {
        let name = f["name"].as_str().unwrap();
        let (t, h) = (f["t"].as_u64().unwrap() as usize, f["horizon"].as_u64().unwrap() as usize);
        let mc = f["max_context"].as_u64().map(|v| v as usize);
        let fc = forecasters.entry(mc).or_insert_with(|| {
            let mut fc = load();
            if let Some(mc) = mc {
                fc.max_context = mc;
            }
            fc
        });
        let ctx = read_f32(&format!("{name}.in.f32"));
        let want = read_f32(&format!("{name}.out.f32")); // (2, H, 21)
        let got = fc.forecast(&[ctx[..t].to_vec(), ctx[t..].to_vec()], h).unwrap();
        // Absolute error, relative to 1 + |reference| since far-horizon values can reach ~1e3.
        let (mut mid, mut tails) = (0.0f32, 0.0f32);
        for i in 0..got.len() {
            let k = i % 21;
            let d = (got[i] - want[i]).abs() / (1.0 + want[i].abs());
            if (1..=19).contains(&k) { mid = mid.max(d) } else { tails = tails.max(d) }
        }
        println!("{name:8} H={h:5} max|diff| 5-95%: {mid:.2e}  1%/99%: {tails:.2e}");
        assert!(mid <= 1e-4, "{name}: 5-95% quantiles differ by {mid}");
        assert!(tails <= 1e-2, "{name}: 1%/99% quantiles differ by {tails}");
    }
}
