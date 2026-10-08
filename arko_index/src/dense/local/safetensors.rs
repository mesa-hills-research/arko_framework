//! A reader for the safetensors format: an 8-byte little-endian header length, a JSON
//! header naming each tensor's dtype, shape and byte range, then the raw data. Tensors
//! are converted to `f32` on reading (F32, F16 and BF16 are accepted).
use crate::bench_v1::Error;
use std::collections::HashMap;

struct Entry {
    dtype: String,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

/// A parsed safetensors file, held in memory.
pub(crate) struct Tensors {
    bytes: Vec<u8>,
    data: usize,
    entries: HashMap<String, Entry>,
}

impl Tensors {
    pub(crate) fn parse(bytes: Vec<u8>) -> Result<Self, Error> {
        let bad = |what: &str| -> Error { format!("not a safetensors file: {what}").into() };
        let len: [u8; 8] = bytes
            .get(..8)
            .ok_or_else(|| bad("too short"))?
            .try_into()
            .unwrap();
        let len = usize::try_from(u64::from_le_bytes(len)).map_err(|_| bad("header length"))?;
        let data = 8usize
            .checked_add(len)
            .ok_or_else(|| bad("header length"))?;
        let header = bytes
            .get(8..data)
            .ok_or_else(|| bad("header past the end"))?;
        let header: HashMap<String, serde_json::Value> =
            serde_json::from_slice(header).map_err(|e| bad(&e.to_string()))?;
        let mut entries = HashMap::new();
        for (name, v) in header {
            if name == "__metadata__" {
                continue;
            }
            let dtype = v["dtype"].as_str().ok_or_else(|| bad("dtype"))?.to_owned();
            let shape = v["shape"]
                .as_array()
                .ok_or_else(|| bad("shape"))?
                .iter()
                .map(|d| d.as_u64().map(|d| d as usize).ok_or_else(|| bad("shape")))
                .collect::<Result<Vec<_>, _>>()?;
            let offsets = v["data_offsets"].as_array().ok_or_else(|| bad("offsets"))?;
            let offset = |i: usize| -> Result<usize, Error> {
                offsets
                    .get(i)
                    .and_then(|o| o.as_u64())
                    .map(|o| o as usize)
                    .ok_or_else(|| bad("offsets"))
            };
            let (start, end) = (offset(0)?, offset(1)?);
            if start > end || data + end > bytes.len() {
                return Err(bad(&format!("{name}: data past the end")));
            }
            entries.insert(
                name,
                Entry {
                    dtype,
                    shape,
                    start,
                    end,
                },
            );
        }
        Ok(Self {
            bytes,
            data,
            entries,
        })
    }

    pub(crate) fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// The first dimension of the tensor `name`.
    pub(crate) fn rows(&self, name: &str) -> Result<usize, Error> {
        self.entries
            .get(name)
            .and_then(|e| e.shape.first().copied())
            .ok_or_else(|| format!("the model has no tensor {name}").into())
    }

    /// The tensor `name` as `f32`s, checked to have `shape`.
    pub(crate) fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, Error> {
        let e = self
            .entries
            .get(name)
            .ok_or_else(|| format!("the model has no tensor {name}"))?;
        if e.shape != shape {
            return Err(format!("tensor {name} has shape {:?}, not {shape:?}", e.shape).into());
        }
        let raw = &self.bytes[self.data + e.start..self.data + e.end];
        let count: usize = shape.iter().product();
        let width = match e.dtype.as_str() {
            "F32" => 4,
            "F16" | "BF16" => 2,
            other => return Err(format!("tensor {name}: unsupported dtype {other}").into()),
        };
        if raw.len() != count * width {
            return Err(format!("tensor {name}: {} bytes for {count} values", raw.len()).into());
        }
        Ok(match e.dtype.as_str() {
            "F32" => raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            "F16" => raw
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            _ => raw
                .chunks_exact(2)
                .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
                .collect(),
        })
    }
}

/// IEEE half precision to single precision, exactly (subnormals, infinities, NaN).
pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal: shift the mantissa up to an implicit leading one.
            let (mut e, mut m) = (127 - 15 + 1, man);
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, _) => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A safetensors file of `(name, shape, values)` stored with `dtype` (`F32` or `F16`).
    pub(crate) fn write(tensors: &[(String, Vec<usize>, Vec<f32>)], dtype: &str) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, values) in tensors {
            let start = data.len();
            for v in values {
                match dtype {
                    "F32" => data.extend_from_slice(&v.to_le_bytes()),
                    _ => data.extend_from_slice(&f32_to_f16(*v).to_le_bytes()),
                }
            }
            header.insert(
                name.clone(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}),
            );
        }
        header.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
        let header = serde_json::to_vec(&header).unwrap();
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&header);
        out.extend_from_slice(&data);
        out
    }

    /// Round to nearest half (normal range only; enough for test weights).
    fn f32_to_f16(v: f32) -> u16 {
        if v == 0.0 {
            return 0;
        }
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let man = bits & 0x7f_ffff;
        assert!(
            (1..31).contains(&exp),
            "test value {v} outside the normal f16 range"
        );
        let mut h = sign as u32 | ((exp as u32) << 10) | (man >> 13);
        if man & 0x1000 != 0 {
            h += 1;
        }
        h as u16
    }

    #[test]
    fn half_precision_converts_exactly() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x0200), 2f32.powi(-15));
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn reads_f32_f16_and_rejects_bad_shapes() {
        let t = vec![("a".to_string(), vec![2, 2], vec![1.0, -2.0, 0.5, 3.0])];
        for dtype in ["F32", "F16"] {
            let tensors = Tensors::parse(write(&t, dtype)).unwrap();
            assert!(tensors.contains("a"));
            assert_eq!(
                tensors.get("a", &[2, 2]).unwrap(),
                vec![1.0, -2.0, 0.5, 3.0]
            );
            assert!(tensors.get("a", &[4]).is_err());
            assert!(tensors.get("b", &[2, 2]).is_err());
        }
        let mut bytes = write(&t, "F32");
        bytes.truncate(bytes.len() - 1);
        assert!(Tensors::parse(bytes).is_err());
        assert!(Tensors::parse(vec![1, 2, 3]).is_err());
    }
}
