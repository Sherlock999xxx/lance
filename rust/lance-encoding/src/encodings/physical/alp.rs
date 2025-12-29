// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Classic ALP (Adaptive Lossless floating-Point) miniblock encoding.
//!
//! # Buffer layout
//!
//! This is a **miniblock** physical encoding (see `encodings::logical::primitive::miniblock`).
//! The encoder emits **three value buffers per page**. For each chunk, `MiniBlockChunk.buffer_sizes`
//! contains three entries matching these buffers in order.
//!
//! - **Buffer 0 (payload):** per-chunk header + bitpacked integer deltas
//!   - `min`: i32 for f32 or i64 for f64 (little-endian)
//!   - `bit_width`: u8
//!   - `packed_deltas`: bitpacked `encoded[i] - min` for all values in the chunk
//! - **Buffer 1 (exceptions positions):** `u16` positions (little-endian), relative to the chunk
//! - **Buffer 2 (exceptions values):** original IEEE754 bit patterns for exceptions
//!   - f32: `u32` little-endian bits, one per exception
//!   - f64: `u64` little-endian bits, one per exception
//!
//! # Why this design
//!
//! - **Bitwise lossless:** exceptions store the original IEEE754 bit patterns. This preserves
//!   `-0.0` and NaN payloads, which cannot be guaranteed by float equality alone.
//! - **Miniblock-friendly random access:** decoding happens at chunk granularity. Positions use
//!   `u16` because chunk sizes are limited to 1024 (f32) / 512 (f64), keeping the exception index
//!   overhead small.
//! - **Plays well with `GeneralMiniBlockCompressor`:** it only compresses the *first* buffer. Placing
//!   the ALP main payload in buffer 0 maximizes the benefit while leaving the usually-small exception
//!   side buffers untouched.
//! - **Robustness:** if a page is not compressible (or a chunk cannot be encoded), the implementation
//!   falls back to the `ValueEncoder` (flat encoding) instead of forcing a larger representation.

use std::fmt::Debug;

use snafu::location;

use crate::buffer::LanceBuffer;
use crate::compression::MiniBlockDecompressor;
use crate::data::{BlockInfo, DataBlock, FixedWidthDataBlock};
use crate::encodings::logical::primitive::miniblock::{
    MiniBlockChunk, MiniBlockCompressed, MiniBlockCompressor, MAX_MINIBLOCK_BYTES,
};
use crate::encodings::physical::value::ValueEncoder;
use crate::format::pb21::CompressiveEncoding;
use crate::format::{pb21, ProtobufUtils21};

use lance_core::{Error, Result};

#[derive(Debug, Clone, Copy)]
struct Exponents {
    e: u8,
    f: u8,
}

#[inline]
fn bits_required_u64(v: u64) -> u8 {
    if v == 0 {
        0
    } else {
        (64 - v.leading_zeros()) as u8
    }
}

/// Bit-pack `u64` values into a dense byte stream using a fixed bit width.
///
/// This is a small, dependency-free packer that writes values LSB-first into a `u128` scratch
/// register and emits little-endian bytes. It is intentionally generic and used by this module
/// to store ALP integer deltas compactly without pulling in external bitpacking codecs.
fn pack_bits_u64(values: &[u64], width: u8) -> Vec<u8> {
    if width == 0 || values.is_empty() {
        return Vec::new();
    }
    let width = width as u32;
    debug_assert!(width <= 64);
    let mut out = Vec::with_capacity(((values.len() as u64 * width as u64 + 7) / 8) as usize);
    let mut buf: u128 = 0;
    let mut bits_in_buf: u32 = 0;
    let mask: u128 = (1u128 << width) - 1;
    for &v in values {
        buf |= ((v as u128) & mask) << bits_in_buf;
        bits_in_buf += width;
        while bits_in_buf >= 8 {
            out.push((buf & 0xFF) as u8);
            buf >>= 8;
            bits_in_buf -= 8;
        }
    }
    if bits_in_buf > 0 {
        out.push((buf & 0xFF) as u8);
    }
    out
}

/// Inverse of [`pack_bits_u64`].
///
/// This reads `num_values` packed integers from `bytes` using the fixed `width` and returns them
/// as `u64`. The implementation is intentionally simple (LSB-first with a `u128` scratch buffer)
/// because miniblock chunks are small and we want predictable behavior for correctness testing.
fn unpack_bits_u64(bytes: &[u8], num_values: usize, width: u8) -> Result<Vec<u64>> {
    if num_values == 0 || width == 0 {
        return Ok(vec![0u64; num_values]);
    }
    let width = width as u32;
    if width > 64 {
        return Err(Error::invalid_input("bit width out of range", location!()));
    }
    let mask: u128 = (1u128 << width) - 1;
    let mut out = Vec::with_capacity(num_values);
    let mut buf: u128 = 0;
    let mut bits_in_buf: u32 = 0;
    let mut idx = 0usize;
    for _ in 0..num_values {
        while bits_in_buf < width {
            if idx >= bytes.len() {
                return Err(Error::invalid_input("bitpacked input truncated", location!()));
            }
            buf |= (bytes[idx] as u128) << bits_in_buf;
            bits_in_buf += 8;
            idx += 1;
        }
        let v = (buf & mask) as u64;
        out.push(v);
        buf >>= width;
        bits_in_buf -= width;
    }
    Ok(out)
}

/// Pick sample indices for exponent search.
///
/// ALP exponent selection is relatively expensive. For large pages we use a small deterministic
/// sample (up to 32 values) to estimate the best `(e, f)` without scanning the whole input.
fn sample_positions(num_values: usize, sample_size: usize) -> Vec<usize> {
    if num_values <= sample_size {
        return (0..num_values).collect();
    }
    let step = num_values / sample_size;
    (0..sample_size).map(|i| i * step).collect()
}

/// Exhaustively search ALP exponents for `f32` using a small sample.
///
/// We try all `0 <= f < e <= 10` and pick the pair with the smallest estimated encoded size.
/// The estimate uses the classic ALP output (`encode_f32`) and models the downstream integer
/// packing as `min + bitpacked(range)` plus exception overhead. This keeps selection stable and
/// fast while tracking the same storage layout used by actual chunk encoding.
fn find_best_exponents_f32(values: &[f32]) -> Exponents {
    let positions = sample_positions(values.len(), 32);
    let sample = positions
        .into_iter()
        .map(|i| values[i])
        .collect::<Vec<_>>();

    let mut best = Exponents { e: 0, f: 0 };
    let mut best_bytes = usize::MAX;
    for e in (0u8..=10).rev() {
        for f in 0u8..e {
            let exp = Exponents { e, f };
            let (encoded, patch_count) = encode_f32(&sample, exp);
            let bytes = estimate_alp_size_i64(&encoded, patch_count, 4);
            if bytes < best_bytes || (bytes == best_bytes && e - f < best.e - best.f) {
                best = exp;
                best_bytes = bytes;
            }
        }
    }
    best
}

/// Exhaustively search ALP exponents for `f64` using a small sample.
///
/// Same strategy as [`find_best_exponents_f32`], but with the `f64` exponent range (`e <= 18`).
fn find_best_exponents_f64(values: &[f64]) -> Exponents {
    let positions = sample_positions(values.len(), 32);
    let sample = positions
        .into_iter()
        .map(|i| values[i])
        .collect::<Vec<_>>();

    let mut best = Exponents { e: 0, f: 0 };
    let mut best_bytes = usize::MAX;
    for e in (0u8..=18).rev() {
        for f in 0u8..e {
            let exp = Exponents { e, f };
            let (encoded, patch_count) = encode_f64(&sample, exp);
            let bytes = estimate_alp_size_i64(&encoded, patch_count, 8);
            if bytes < best_bytes || (bytes == best_bytes && e - f < best.e - best.f) {
                best = exp;
                best_bytes = bytes;
            }
        }
    }
    best
}

fn estimate_alp_size_i64(encoded: &[i64], patch_count: usize, value_bytes: usize) -> usize {
    if encoded.is_empty() {
        return 0;
    }
    let mut min = i64::MAX;
    let mut max = i64::MIN;
    for &v in encoded {
        min = min.min(v);
        max = max.max(v);
    }
    let range = max.wrapping_sub(min) as u64;
    let width = bits_required_u64(range) as usize;
    let packed = (encoded.len() * width + 7) / 8;
    let header = value_bytes + 1; // min + bit_width
    let exceptions = patch_count * (2 + value_bytes); // pos(u16) + value bits
    header + packed + exceptions
}

#[inline]
fn fast_round_f32(v: f32) -> f32 {
    const SWEET: f32 = (1u32 << 23) as f32 + (1u32 << 22) as f32;
    (v + SWEET) - SWEET
}

#[inline]
fn fast_round_f64(v: f64) -> f64 {
    const SWEET: f64 = (1u64 << 52) as f64 + (1u64 << 51) as f64;
    (v + SWEET) - SWEET
}

const F10_F32: [f32; 11] = [
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
    100000.0,
    1000000.0,
    10000000.0,
    100000000.0,
    1000000000.0,
    10000000000.0,
];
const IF10_F32: [f32; 11] = [
    1.0,
    0.1,
    0.01,
    0.001,
    0.0001,
    0.00001,
    0.000001,
    0.0000001,
    0.00000001,
    0.000000001,
    0.0000000001,
];

const F10_F64: [f64; 24] = [
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
    100000.0,
    1000000.0,
    10000000.0,
    100000000.0,
    1000000000.0,
    10000000000.0,
    100000000000.0,
    1000000000000.0,
    10000000000000.0,
    100000000000000.0,
    1000000000000000.0,
    10000000000000000.0,
    100000000000000000.0,
    1000000000000000000.0,
    10000000000000000000.0,
    100000000000000000000.0,
    1000000000000000000000.0,
    10000000000000000000000.0,
    100000000000000000000000.0,
];
const IF10_F64: [f64; 24] = [
    1.0,
    0.1,
    0.01,
    0.001,
    0.0001,
    0.00001,
    0.000001,
    0.0000001,
    0.00000001,
    0.000000001,
    0.0000000001,
    0.00000000001,
    0.000000000001,
    0.0000000000001,
    0.00000000000001,
    0.000000000000001,
    0.0000000000000001,
    0.00000000000000001,
    0.000000000000000001,
    0.0000000000000000001,
    0.00000000000000000001,
    0.000000000000000000001,
    0.0000000000000000000001,
    0.00000000000000000000001,
];

/// Classic ALP encode for `f32`, used for exponent selection only.
///
/// This function computes the transformed integer stream and counts how many values must be
/// stored as exceptions when requiring **bitwise** round-tripping (`to_bits()` equality).
///
/// For size estimation we replace exception slots with a "fill" value (the first encodable value).
/// This matches the behavior of the real chunk encoder: reducing the integer range improves
/// downstream delta+bitpacking even when exceptions exist.
fn encode_f32(values: &[f32], exp: Exponents) -> (Vec<i64>, usize) {
    let mut encoded = Vec::with_capacity(values.len());
    let mut patches = 0usize;
    let mut fill: Option<i32> = None;

    for &v in values {
        if !v.is_finite() {
            encoded.push(0);
            patches += 1;
            continue;
        }
        let scaled = v * F10_F32[exp.e as usize] * IF10_F32[exp.f as usize];
        if !scaled.is_finite() {
            encoded.push(0);
            patches += 1;
            continue;
        }
        let rounded = fast_round_f32(scaled);
        let enc = rounded as i32;
        let decoded = (enc as f32) * F10_F32[exp.f as usize] * IF10_F32[exp.e as usize];
        if decoded.to_bits() == v.to_bits() {
            if fill.is_none() {
                fill = Some(enc);
            }
            encoded.push(enc as i64);
        } else {
            encoded.push(0);
            patches += 1;
        }
    }

    if let Some(fill) = fill {
        for (i, &v) in values.iter().enumerate() {
            if !v.is_finite() {
                encoded[i] = fill as i64;
                continue;
            }
            let scaled = v * F10_F32[exp.e as usize] * IF10_F32[exp.f as usize];
            if !scaled.is_finite() {
                encoded[i] = fill as i64;
                continue;
            }
            let rounded = fast_round_f32(scaled);
            let enc = rounded as i32;
            let decoded = (enc as f32) * F10_F32[exp.f as usize] * IF10_F32[exp.e as usize];
            if decoded.to_bits() != v.to_bits() {
                encoded[i] = fill as i64;
            }
        }
    }

    (encoded, patches)
}

/// Classic ALP encode for `f64`, used for exponent selection only.
///
/// See [`encode_f32`] for the design. The core difference is the exponent range and integer type.
fn encode_f64(values: &[f64], exp: Exponents) -> (Vec<i64>, usize) {
    let mut encoded = Vec::with_capacity(values.len());
    let mut patches = 0usize;
    let mut fill: Option<i64> = None;

    for &v in values {
        if !v.is_finite() {
            encoded.push(0);
            patches += 1;
            continue;
        }
        let scaled = v * F10_F64[exp.e as usize] * IF10_F64[exp.f as usize];
        if !scaled.is_finite() {
            encoded.push(0);
            patches += 1;
            continue;
        }
        let rounded = fast_round_f64(scaled);
        let enc = rounded as i64;
        let decoded = (enc as f64) * F10_F64[exp.f as usize] * IF10_F64[exp.e as usize];
        if decoded.to_bits() == v.to_bits() {
            if fill.is_none() {
                fill = Some(enc);
            }
            encoded.push(enc);
        } else {
            encoded.push(0);
            patches += 1;
        }
    }

    if let Some(fill) = fill {
        for (i, &v) in values.iter().enumerate() {
            if !v.is_finite() {
                encoded[i] = fill;
                continue;
            }
            let scaled = v * F10_F64[exp.e as usize] * IF10_F64[exp.f as usize];
            if !scaled.is_finite() {
                encoded[i] = fill;
                continue;
            }
            let rounded = fast_round_f64(scaled);
            let enc = rounded as i64;
            let decoded = (enc as f64) * F10_F64[exp.f as usize] * IF10_F64[exp.e as usize];
            if decoded.to_bits() != v.to_bits() {
                encoded[i] = fill;
            }
        }
    }

    (encoded, patches)
}

/// Encodes f32/f64 values using the ALP miniblock layout described in the module docs.
///
/// This is an *opt-in* encoder (selected via compression metadata/config) and is expected to be
/// version-gated by the caller (Lance file version >= 2.2). The encoder may fall back to
/// `ValueEncoder` if ALP is ineffective for the given page.
#[derive(Debug, Clone)]
pub struct AlpMiniBlockEncoder {
    bits_per_value: u64,
}

impl AlpMiniBlockEncoder {
    pub fn new(bits_per_value: u64) -> Self {
        assert!(bits_per_value == 32 || bits_per_value == 64);
        Self { bits_per_value }
    }

    fn max_chunk_size(&self) -> usize {
        match self.bits_per_value {
            32 => 1024,
            64 => 512,
            _ => unreachable!(),
        }
    }
}

impl MiniBlockCompressor for AlpMiniBlockEncoder {
    fn compress(&self, page: DataBlock) -> Result<(MiniBlockCompressed, CompressiveEncoding)> {
        let DataBlock::FixedWidth(data) = page else {
            return Err(Error::invalid_input(
                "ALP encoding only supports FixedWidth data blocks",
                location!(),
            ));
        };

        if data.bits_per_value != self.bits_per_value {
            return Err(Error::invalid_input(
                "ALP bits_per_value mismatch",
                location!(),
            ));
        }

        if data.num_values == 0 {
            let encoding = ProtobufUtils21::alp(self.bits_per_value as u32, 1, 0);
            return Ok((
                MiniBlockCompressed {
                    data: vec![],
                    chunks: vec![],
                    num_values: 0,
                },
                encoding,
            ));
        }

        let max_chunk = self.max_chunk_size();
        let bytes_per_value = (self.bits_per_value / 8) as usize;
        let raw_size = data.num_values as usize * bytes_per_value;

        let exponents = match self.bits_per_value {
            32 => {
                let words = data.data.borrow_to_typed_slice::<u32>();
                let floats = words
                    .as_ref()
                    .iter()
                    .map(|b| f32::from_bits(*b))
                    .collect::<Vec<_>>();
                find_best_exponents_f32(&floats)
            }
            64 => {
                let words = data.data.borrow_to_typed_slice::<u64>();
                let floats = words
                    .as_ref()
                    .iter()
                    .map(|b| f64::from_bits(*b))
                    .collect::<Vec<_>>();
                find_best_exponents_f64(&floats)
            }
            _ => unreachable!(),
        };

        let mut buf0 = Vec::new();
        let mut buf1 = Vec::new();
        let mut buf2 = Vec::new();
        let mut chunks = Vec::new();

        let mut offset = 0usize;
        let bytes = data.data.as_ref();

        while offset < bytes.len() {
            let remaining_values = (bytes.len() - offset) / bytes_per_value;
            let chunk_values = remaining_values.min(max_chunk);
            let chunk_bytes_len = chunk_values * bytes_per_value;
            let chunk_bytes = &bytes[offset..offset + chunk_bytes_len];

            let chunk_encoded_res = match self.bits_per_value {
                32 => {
                    let words = bytemuck::try_cast_slice::<u8, u32>(chunk_bytes).map_err(|_| {
                        Error::invalid_input("invalid f32 buffer alignment", location!())
                    })?;
                    let floats = words.iter().map(|b| f32::from_bits(*b)).collect::<Vec<_>>();
                    encode_chunk_f32(&floats, exponents)
                }
                64 => {
                    let words = bytemuck::try_cast_slice::<u8, u64>(chunk_bytes).map_err(|_| {
                        Error::invalid_input("invalid f64 buffer alignment", location!())
                    })?;
                    let floats = words.iter().map(|b| f64::from_bits(*b)).collect::<Vec<_>>();
                    encode_chunk_f64(&floats, exponents)
                }
                _ => unreachable!(),
            };
            let chunk_encoded = match chunk_encoded_res {
                Ok(v) => v,
                Err(_) => return ValueEncoder::default().compress(DataBlock::FixedWidth(data)),
            };
            let sizes = chunk_encoded.append_to(&mut buf0, &mut buf1, &mut buf2)?;

            let total_value_bytes = sizes.0 as u64 + sizes.1 as u64 + sizes.2 as u64;
            if total_value_bytes > MAX_MINIBLOCK_BYTES {
                return ValueEncoder::default().compress(DataBlock::FixedWidth(data));
            }

            let log_num_values = if offset + chunk_bytes_len == bytes.len() {
                0
            } else {
                (chunk_values as u64).ilog2() as u8
            };
            chunks.push(MiniBlockChunk {
                buffer_sizes: vec![sizes.0, sizes.1, sizes.2],
                log_num_values,
            });

            offset += chunk_bytes_len;
        }

        let compressed_size = buf0.len() + buf1.len() + buf2.len();
        if compressed_size >= raw_size {
            return ValueEncoder::default().compress(DataBlock::FixedWidth(data));
        }

        let encoding =
            ProtobufUtils21::alp(self.bits_per_value as u32, exponents.e as u32, exponents.f as u32);
        Ok((
            MiniBlockCompressed {
                data: vec![
                    LanceBuffer::from(buf0),
                    LanceBuffer::from(buf1),
                    LanceBuffer::from(buf2),
                ],
                chunks,
                num_values: data.num_values,
            },
            encoding,
        ))
    }
}

struct ChunkEncoded {
    bits_per_value: u64,
    min: i64,
    bit_width: u8,
    packed_deltas: Vec<u8>,
    exception_positions: Vec<u16>,
    exception_bits: Vec<u8>,
}

impl ChunkEncoded {
    fn append_to(
        self,
        buf0: &mut Vec<u8>,
        buf1: &mut Vec<u8>,
        buf2: &mut Vec<u8>,
    ) -> Result<(u32, u32, u32)> {
        let start0 = buf0.len();
        let start1 = buf1.len();
        let start2 = buf2.len();

        match self.bits_per_value {
            32 => {
                buf0.extend_from_slice(&(self.min as i32).to_le_bytes());
            }
            64 => {
                buf0.extend_from_slice(&self.min.to_le_bytes());
            }
            _ => unreachable!(),
        }
        buf0.push(self.bit_width);
        buf0.extend_from_slice(&self.packed_deltas);

        for pos in self.exception_positions {
            buf1.extend_from_slice(&pos.to_le_bytes());
        }
        buf2.extend_from_slice(&self.exception_bits);

        let s0 = (buf0.len() - start0) as u32;
        let s1 = (buf1.len() - start1) as u32;
        let s2 = (buf2.len() - start2) as u32;
        Ok((s0, s1, s2))
    }
}

fn encode_chunk_f32(values: &[f32], exp: Exponents) -> Result<ChunkEncoded> {
    let mut encoded = Vec::with_capacity(values.len());
    let mut exception_positions = Vec::new();
    let mut exception_bits = Vec::new();
    let mut fill: Option<i32> = None;

    for (i, &v) in values.iter().enumerate() {
        let bits = v.to_bits();
        if !v.is_finite() || bits == 0x8000_0000 {
            exception_positions.push(i as u16);
            exception_bits.extend_from_slice(&bits.to_le_bytes());
            encoded.push(0i32);
            continue;
        }

        let scaled = v * F10_F32[exp.e as usize] * IF10_F32[exp.f as usize];
        if !scaled.is_finite() {
            exception_positions.push(i as u16);
            exception_bits.extend_from_slice(&bits.to_le_bytes());
            encoded.push(0i32);
            continue;
        }
        let rounded = fast_round_f32(scaled);
        let enc = rounded as i32;
        let decoded = (enc as f32) * F10_F32[exp.f as usize] * IF10_F32[exp.e as usize];
        if decoded.to_bits() == bits {
            if fill.is_none() {
                fill = Some(enc);
            }
            encoded.push(enc);
        } else {
            exception_positions.push(i as u16);
            exception_bits.extend_from_slice(&bits.to_le_bytes());
            encoded.push(0i32);
        }
    }

    let Some(fill) = fill else {
        return Err(Error::invalid_input(
            "ALP chunk has no encodable values",
            location!(),
        ));
    };
    for &pos in &exception_positions {
        encoded[pos as usize] = fill;
    }

    let mut min = i32::MAX;
    let mut max = i32::MIN;
    for &v in &encoded {
        min = min.min(v);
        max = max.max(v);
    }
    let range = (max as i64 - min as i64) as u64;
    let bit_width = bits_required_u64(range);

    let deltas = encoded
        .iter()
        .map(|&v| (v.wrapping_sub(min)) as u32 as u64)
        .collect::<Vec<_>>();
    let packed = pack_bits_u64(&deltas, bit_width);

    Ok(ChunkEncoded {
        bits_per_value: 32,
        min: min as i64,
        bit_width,
        packed_deltas: packed,
        exception_positions,
        exception_bits,
    })
}

fn encode_chunk_f64(values: &[f64], exp: Exponents) -> Result<ChunkEncoded> {
    let mut encoded = Vec::with_capacity(values.len());
    let mut exception_positions = Vec::new();
    let mut exception_bits = Vec::new();
    let mut fill: Option<i64> = None;

    for (i, &v) in values.iter().enumerate() {
        let bits = v.to_bits();
        if !v.is_finite() || bits == 0x8000_0000_0000_0000 {
            exception_positions.push(i as u16);
            exception_bits.extend_from_slice(&bits.to_le_bytes());
            encoded.push(0i64);
            continue;
        }
        let scaled = v * F10_F64[exp.e as usize] * IF10_F64[exp.f as usize];
        if !scaled.is_finite() {
            exception_positions.push(i as u16);
            exception_bits.extend_from_slice(&bits.to_le_bytes());
            encoded.push(0i64);
            continue;
        }
        let rounded = fast_round_f64(scaled);
        let enc = rounded as i64;
        let decoded = (enc as f64) * F10_F64[exp.f as usize] * IF10_F64[exp.e as usize];
        if decoded.to_bits() == bits {
            if fill.is_none() {
                fill = Some(enc);
            }
            encoded.push(enc);
        } else {
            exception_positions.push(i as u16);
            exception_bits.extend_from_slice(&bits.to_le_bytes());
            encoded.push(0i64);
        }
    }

    let Some(fill) = fill else {
        return Err(Error::invalid_input(
            "ALP chunk has no encodable values",
            location!(),
        ));
    };
    for &pos in &exception_positions {
        encoded[pos as usize] = fill;
    }

    let mut min = i64::MAX;
    let mut max = i64::MIN;
    for &v in &encoded {
        min = min.min(v);
        max = max.max(v);
    }
    let range = max.wrapping_sub(min) as u64;
    let bit_width = bits_required_u64(range);

    let deltas = encoded
        .iter()
        .map(|&v| v.wrapping_sub(min) as u64)
        .collect::<Vec<_>>();
    let packed = pack_bits_u64(&deltas, bit_width);

    Ok(ChunkEncoded {
        bits_per_value: 64,
        min,
        bit_width,
        packed_deltas: packed,
        exception_positions,
        exception_bits,
    })
}

/// Decodes values encoded by `AlpMiniBlockEncoder`.
///
/// See the module-level "Buffer layout" section for the expected buffer ordering and
/// per-chunk structure.
#[derive(Debug)]
pub struct AlpMiniBlockDecompressor {
    bits_per_value: u64,
    exp: Exponents,
}

impl AlpMiniBlockDecompressor {
    pub fn from_description(desc: &pb21::Alp) -> Result<Self> {
        let bits_per_value = desc.bits_per_value as u64;
        if bits_per_value != 32 && bits_per_value != 64 {
            return Err(Error::invalid_input("ALP bits_per_value must be 32 or 64", location!()));
        }
        let exp = Exponents {
            e: desc.exponent_e as u8,
            f: desc.exponent_f as u8,
        };

        if exp.f >= exp.e {
            return Err(Error::invalid_input("ALP requires exponent_f < exponent_e", location!()));
        }

        match bits_per_value {
            32 => {
                if exp.e as usize >= F10_F32.len() || exp.f as usize >= F10_F32.len() {
                    return Err(Error::invalid_input("ALP f32 exponents out of range", location!()));
                }
            }
            64 => {
                if exp.e as usize >= F10_F64.len() || exp.f as usize >= F10_F64.len() {
                    return Err(Error::invalid_input("ALP f64 exponents out of range", location!()));
                }
            }
            _ => unreachable!(),
        }

        Ok(Self {
            bits_per_value,
            exp,
        })
    }
}

impl MiniBlockDecompressor for AlpMiniBlockDecompressor {
    fn decompress(&self, data: Vec<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        if num_values == 0 {
            return Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
                data: LanceBuffer::empty(),
                bits_per_value: self.bits_per_value,
                num_values: 0,
                block_info: BlockInfo::new(),
            }));
        }

        if data.len() != 3 {
            return Err(Error::invalid_input(
                format!("ALP decompression expects 3 buffers, got {}", data.len()),
                location!(),
            ));
        }

        let n = usize::try_from(num_values).map_err(|_| {
            Error::invalid_input("ALP chunk too large for usize", location!())
        })?;

        let buf0 = data[0].as_ref();
        let buf1 = data[1].as_ref();
        let buf2 = data[2].as_ref();

        let (min, bit_width, packed) = match self.bits_per_value {
            32 => {
                if buf0.len() < 5 {
                    return Err(Error::invalid_input("ALP buffer0 too small", location!()));
                }
                let min = i32::from_le_bytes([buf0[0], buf0[1], buf0[2], buf0[3]]) as i64;
                let bit_width = buf0[4];
                (min, bit_width, &buf0[5..])
            }
            64 => {
                if buf0.len() < 9 {
                    return Err(Error::invalid_input("ALP buffer0 too small", location!()));
                }
                let min = i64::from_le_bytes([
                    buf0[0], buf0[1], buf0[2], buf0[3], buf0[4], buf0[5], buf0[6], buf0[7],
                ]);
                let bit_width = buf0[8];
                (min, bit_width, &buf0[9..])
            }
            _ => unreachable!(),
        };

        let deltas = unpack_bits_u64(packed, n, bit_width)?;

        let mut out_bytes = Vec::with_capacity(n * (self.bits_per_value as usize / 8));
        match self.bits_per_value {
            32 => {
                for d in deltas {
                    let enc = (min as i32).wrapping_add(d as u32 as i32);
                    let decoded =
                        (enc as f32) * F10_F32[self.exp.f as usize] * IF10_F32[self.exp.e as usize];
                    out_bytes.extend_from_slice(&decoded.to_bits().to_le_bytes());
                }
            }
            64 => {
                for d in deltas {
                    let enc = (min as i64).wrapping_add(d as i64);
                    let decoded =
                        (enc as f64) * F10_F64[self.exp.f as usize] * IF10_F64[self.exp.e as usize];
                    out_bytes.extend_from_slice(&decoded.to_bits().to_le_bytes());
                }
            }
            _ => unreachable!(),
        }

        if buf1.len() % 2 != 0 {
            return Err(Error::invalid_input("ALP exception positions not u16-aligned", location!()));
        }
        let exception_count = buf1.len() / 2;
        let value_bytes = (self.bits_per_value / 8) as usize;
        if buf2.len() != exception_count * value_bytes {
            return Err(Error::invalid_input(
                "ALP exception values length mismatch",
                location!(),
            ));
        }

        for i in 0..exception_count {
            let pos = u16::from_le_bytes([buf1[i * 2], buf1[i * 2 + 1]]) as usize;
            if pos >= n {
                return Err(Error::invalid_input("ALP exception position out of range", location!()));
            }
            let start = i * value_bytes;
            let end = start + value_bytes;
            out_bytes[pos * value_bytes..(pos + 1) * value_bytes].copy_from_slice(&buf2[start..end]);
        }

        Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::from(out_bytes),
            bits_per_value: self.bits_per_value,
            num_values,
            block_info: BlockInfo::new(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistics::ComputeStat;

    fn round_trip_f32(values: &[f32]) {
        let bytes = values.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect::<Vec<_>>();
        let mut block = FixedWidthDataBlock {
            data: LanceBuffer::from(bytes),
            bits_per_value: 32,
            num_values: values.len() as u64,
            block_info: BlockInfo::new(),
        };
        block.compute_stat();
        let encoder = AlpMiniBlockEncoder::new(32);
        let (compressed, encoding) = encoder.compress(DataBlock::FixedWidth(block)).unwrap();
        let pb21::compressive_encoding::Compression::Alp(desc) = encoding.compression.unwrap() else {
            panic!("expected ALP encoding")
        };
        assert_eq!(compressed.data.len(), 3);
        let decompressor = AlpMiniBlockDecompressor::from_description(&desc).unwrap();

        let mut vals_in_prev = 0u64;
        let mut buffer_offsets = vec![0usize; compressed.data.len()];
        let mut out = Vec::new();
        for chunk in &compressed.chunks {
            let chunk_vals = chunk.num_values(vals_in_prev, compressed.num_values);
            vals_in_prev += chunk_vals;
            let buffers = chunk
                .buffer_sizes
                .iter()
                .zip(compressed.data.iter().zip(buffer_offsets.iter_mut()))
                .map(|(sz, (buf, off))| {
                    let start = *off;
                    let end = start + *sz as usize;
                    *off = end;
                    buf.slice_with_length(start, *sz as usize)
                })
                .collect::<Vec<_>>();

            let decoded = decompressor.decompress(buffers, chunk_vals).unwrap();
            let DataBlock::FixedWidth(decoded) = decoded else { panic!("expected fixed width") };
            let words = decoded.data.borrow_to_typed_slice::<u32>();
            out.extend(words.as_ref().iter().map(|b| f32::from_bits(*b)));
        }

        assert_eq!(out.len(), values.len());
        for (a, b) in out.iter().zip(values.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    fn round_trip_f64(values: &[f64]) {
        let bytes = values.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect::<Vec<_>>();
        let mut block = FixedWidthDataBlock {
            data: LanceBuffer::from(bytes),
            bits_per_value: 64,
            num_values: values.len() as u64,
            block_info: BlockInfo::new(),
        };
        block.compute_stat();
        let encoder = AlpMiniBlockEncoder::new(64);
        let (compressed, encoding) = encoder.compress(DataBlock::FixedWidth(block)).unwrap();
        let pb21::compressive_encoding::Compression::Alp(desc) = encoding.compression.unwrap() else {
            panic!("expected ALP encoding")
        };
        assert_eq!(compressed.data.len(), 3);
        let decompressor = AlpMiniBlockDecompressor::from_description(&desc).unwrap();

        let mut vals_in_prev = 0u64;
        let mut buffer_offsets = vec![0usize; compressed.data.len()];
        let mut out = Vec::new();
        for chunk in &compressed.chunks {
            let chunk_vals = chunk.num_values(vals_in_prev, compressed.num_values);
            vals_in_prev += chunk_vals;
            let buffers = chunk
                .buffer_sizes
                .iter()
                .zip(compressed.data.iter().zip(buffer_offsets.iter_mut()))
                .map(|(sz, (buf, off))| {
                    let start = *off;
                    let end = start + *sz as usize;
                    *off = end;
                    buf.slice_with_length(start, *sz as usize)
                })
                .collect::<Vec<_>>();

            let decoded = decompressor.decompress(buffers, chunk_vals).unwrap();
            let DataBlock::FixedWidth(decoded) = decoded else { panic!("expected fixed width") };
            let words = decoded.data.borrow_to_typed_slice::<u64>();
            out.extend(words.as_ref().iter().map(|b| f64::from_bits(*b)));
        }

        assert_eq!(out.len(), values.len());
        for (a, b) in out.iter().zip(values.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn test_round_trip_with_exceptions_f32() {
        let mut values = (0..1024).map(|v| v as f32).collect::<Vec<_>>();
        values[3] = -0.0;
        values[7] = f32::from_bits(0x7FC0_0001);
        values[11] = f32::INFINITY;
        round_trip_f32(&values);
    }

    #[test]
    fn test_round_trip_with_exceptions_f64() {
        let mut values = (0..512).map(|v| v as f64).collect::<Vec<_>>();
        values[2] = -0.0;
        values[5] = f64::from_bits(0x7FF8_0000_0000_0001);
        values[9] = f64::NEG_INFINITY;
        round_trip_f64(&values);
    }

    #[test]
    fn test_fallback_when_not_beneficial() {
        let values = vec![-0.0f32; 1024];
        let bytes = values.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect::<Vec<_>>();
        let mut block = FixedWidthDataBlock {
            data: LanceBuffer::from(bytes),
            bits_per_value: 32,
            num_values: values.len() as u64,
            block_info: BlockInfo::new(),
        };
        block.compute_stat();
        let encoder = AlpMiniBlockEncoder::new(32);
        let (_compressed, encoding) = encoder.compress(DataBlock::FixedWidth(block)).unwrap();
        assert!(matches!(
            encoding.compression.unwrap(),
            pb21::compressive_encoding::Compression::Flat(_)
        ));
    }
}
