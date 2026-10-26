//! Minimal safetensors reader: 8-byte little-endian header length, a JSON header, then raw data.

use std::collections::HashMap;

pub struct Tensors<'a> {
    data: &'a [u8],
    entries: HashMap<String, Entry>,
}

struct Entry {
    dtype: String,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

impl<'a> Tensors<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, String> {
        let header_len = u64::from_le_bytes(bytes.get(..8).ok_or("safetensors: file too short")?.try_into().unwrap()) as usize;
        let header = bytes.get(8..8 + header_len).ok_or("safetensors: truncated header")?;
        let json: serde_json::Value = serde_json::from_slice(header).map_err(|e| format!("safetensors header: {e}"))?;
        let data = &bytes[8 + header_len..];
        let mut entries = HashMap::new();
        for (name, v) in json.as_object().ok_or("safetensors: header is not an object")? {
            if name == "__metadata__" {
                continue;
            }
            let bad = || format!("safetensors: malformed entry {name}");
            let shape = v["shape"].as_array().ok_or_else(bad)?.iter().map(|d| d.as_u64().unwrap_or(0) as usize).collect();
            let off = v["data_offsets"].as_array().ok_or_else(bad)?;
            let (start, end) = (off[0].as_u64().ok_or_else(bad)? as usize, off[1].as_u64().ok_or_else(bad)? as usize);
            let dtype = v["dtype"].as_str().ok_or_else(bad)?.to_string();
            if end > data.len() || start > end {
                return Err(format!("safetensors: {name} out of range"));
            }
            entries.insert(name.clone(), Entry { dtype, shape, start, end });
        }
        Ok(Self { data, entries })
    }

    /// Tensor as `(shape, f32 values)`; F32, F16 and BF16 checkpoints are accepted.
    pub fn get(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        let e = self.entries.get(name).ok_or_else(|| format!("missing weight {name}"))?;
        let raw = &self.data[e.start..e.end];
        let vals: Vec<f32> = match e.dtype.as_str() {
            "F32" => raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            "BF16" => raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            "F16" => raw.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            other => return Err(format!("{name}: unsupported dtype {other}")),
        };
        if vals.len() != e.shape.iter().product::<usize>() {
            return Err(format!("{name}: size does not match shape"));
        }
        Ok((e.shape.clone(), vals))
    }

    pub fn has(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as i32, (h & 0x3ff) as u32);
    let sign = if s == 1 { -1.0 } else { 1.0 };
    match e {
        0 => sign * m as f32 * 2f32.powi(-24),
        31 => if m == 0 { sign * f32::INFINITY } else { f32::NAN },
        _ => sign * (1.0 + m as f32 / 1024.0) * 2f32.powi(e - 15),
    }
}
