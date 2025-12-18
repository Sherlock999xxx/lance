// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::{Float16Type, Float32Type, Float64Type};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float16Array, Float32Array, Float64Array, UInt8Array,
};
use arrow_schema::{DataType, Field};
use bitvec::prelude::{BitVec, Lsb0};
use deepsize::DeepSizeOf;
use lance_arrow::{ArrowFloatType, FixedSizeListArrayExt, FloatArray, FloatType};
use lance_core::{Error, Result};
use ndarray::s;
use num_traits::{AsPrimitive, FromPrimitive};
use rand_distr::Distribution;
use snafu::location;

use crate::vector::bq::storage::{
    RabitQuantizationMetadata, RabitQuantizationStorage, RABIT_CODE_COLUMN, RABIT_METADATA_KEY,
};
use crate::vector::bq::transform::{ADD_FACTORS_FIELD, SCALE_FACTORS_FIELD};
use crate::vector::bq::RQBuildParams;
use crate::vector::quantizer::{Quantization, Quantizer, QuantizerBuildParams};

/// Build parameters for RabitQuantizer.
///
/// num_bits: the number of bits per dimension.
pub struct RabitBuildParams {
    pub num_bits: u8,
}

impl Default for RabitBuildParams {
    fn default() -> Self {
        Self { num_bits: 1 }
    }
}

impl QuantizerBuildParams for RabitBuildParams {
    fn sample_size(&self) -> usize {
        // RabitQ doesn't need to sample any data
        0
    }
}

#[derive(Debug, Clone, DeepSizeOf)]
pub struct RabitQuantizer {
    metadata: RabitQuantizationMetadata,
}

impl RabitQuantizer {
    pub fn new<T: ArrowFloatType>(num_bits: u8, dim: i32) -> Self {
        Self::try_new_with_rotate::<T>(num_bits, dim, true).expect("RabitQ init should succeed")
    }

    pub fn try_new_with_rotate<T: ArrowFloatType>(
        num_bits: u8,
        dim: i32,
        rotate: bool,
    ) -> Result<Self> {
        if !rotate && num_bits != 1 {
            // Without rotation, we only support 1 bit per dimension for now (code_dim == dim).
            // Supporting num_bits > 1 would require defining how to expand dimensions without a
            // projection.
            return Err(Error::invalid_input(
                "RabitQ rotate=false currently requires num_bits == 1",
                location!(),
            ));
        }

        // we don't need to calculate the inverse of P,
        // just take the generated matrix as P^{-1}
        let code_dim = dim * num_bits as i32;
        let rotate_mat = {
            let values: Vec<T::Native> = if rotate {
                let rotate_mat = random_orthogonal::<T>(code_dim as usize);
                let (rotate_mat, _) = rotate_mat.into_raw_vec_and_offset();
                rotate_mat
            } else {
                let zero = T::Native::from_f64(0.0).unwrap();
                let one = T::Native::from_f64(1.0).unwrap();
                let code_dim = code_dim as usize;
                let mut values = Vec::with_capacity(code_dim * code_dim);
                for i in 0..code_dim {
                    for j in 0..code_dim {
                        values.push(if i == j { one } else { zero });
                    }
                }
                values
            };

            match T::FLOAT_TYPE {
                FloatType::Float16 | FloatType::Float32 | FloatType::Float64 => {
                    let rotate_mat = T::ArrayType::from(values);
                    FixedSizeListArray::try_new_from_values(rotate_mat, code_dim).unwrap()
                }
                _ => unimplemented!("RabitQ does not support data type: {:?}", T::FLOAT_TYPE),
            }
        };

        let metadata = RabitQuantizationMetadata {
            rotate_mat: Some(rotate_mat),
            rotate_mat_position: 0,
            num_bits,
            rotate,
            packed: true,
        };
        Ok(Self { metadata })
    }

    pub fn num_bits(&self) -> u8 {
        self.metadata.num_bits
    }

    #[inline]
    fn rotate_mat_flat<T: ArrowFloatType>(&self) -> &[T::Native] {
        let rotate_mat = self
            .metadata
            .rotate_mat
            .as_ref()
            .expect("rotate_mat not loaded");
        rotate_mat
            .values()
            .as_any()
            .downcast_ref::<T::ArrayType>()
            .unwrap()
            .as_slice()
    }

    #[inline]
    fn rotate_mat<T: ArrowFloatType>(&'_ self) -> ndarray::ArrayView2<'_, T::Native> {
        let code_dim = self.code_dim();
        assert!(
            self.metadata.rotate,
            "rotate_mat requested but rotate=false"
        );
        ndarray::ArrayView2::from_shape((code_dim, code_dim), self.rotate_mat_flat::<T>()).unwrap()
    }

    pub fn dim(&self) -> usize {
        self.code_dim() / self.metadata.num_bits as usize
    }

    // compute the dot product of v_q * v_r
    pub fn codes_res_dot_dists<T: ArrowFloatType>(
        &self,
        residual_vectors: &FixedSizeListArray,
    ) -> Result<Vec<f32>>
    where
        T::Native: AsPrimitive<f32>,
    {
        let dim = self.dim();
        if residual_vectors.value_length() as usize != dim {
            return Err(Error::invalid_input(
                format!(
                    "Vector dimension mismatch: {} != {}",
                    residual_vectors.value_length(),
                    dim
                ),
                location!(),
            ));
        }

        let (_, ip_rq_res, _) = self.transform_with_rotated::<T>(residual_vectors, false)?;
        Ok(ip_rq_res)
    }

    pub(crate) fn quantize_with_rotated_and_ip(
        &self,
        vectors: &FixedSizeListArray,
        keep_rotated: bool,
    ) -> Result<(ArrayRef, Vec<f32>, Option<FixedSizeListArray>)> {
        match vectors.value_type() {
            DataType::Float16 => self.transform_with_rotated::<Float16Type>(vectors, keep_rotated),
            DataType::Float32 => self.transform_with_rotated::<Float32Type>(vectors, keep_rotated),
            DataType::Float64 => self.transform_with_rotated::<Float64Type>(vectors, keep_rotated),
            value_type => Err(Error::invalid_input(
                format!("Unsupported data type: {:?}", value_type),
                location!(),
            )),
        }
    }

    fn transform<T: ArrowFloatType>(
        &self,
        residual_vectors: &FixedSizeListArray,
    ) -> Result<ArrayRef>
    where
        T::Native: AsPrimitive<f32>,
    {
        let (codes, _, _) = self.transform_with_rotated::<T>(residual_vectors, false)?;
        Ok(codes)
    }

    fn transform_with_rotated<T: ArrowFloatType>(
        &self,
        residual_vectors: &FixedSizeListArray,
        keep_rotated: bool,
    ) -> Result<(ArrayRef, Vec<f32>, Option<FixedSizeListArray>)>
    where
        T::Native: AsPrimitive<f32>,
    {
        // we don't need to normalize the residual vectors,
        // because the sign of P^{-1} * v_r is the same as P^{-1} * v_r / ||v_r||
        let n = residual_vectors.len();
        let dim = self.dim();
        let code_dim = self.code_dim();
        debug_assert_eq!(residual_vectors.values().len(), n * dim);

        let vectors = ndarray::ArrayView2::from_shape(
            (n, dim),
            residual_vectors
                .values()
                .as_any()
                .downcast_ref::<T::ArrayType>()
                .unwrap()
                .as_slice(),
        )
        .map_err(|e| Error::invalid_input(e.to_string(), location!()))?;
        let vectors = vectors.t();

        let sqrt_dim = (dim as f32 * self.metadata.num_bits as f32).sqrt();

        let (ip_rq_res, bv, rotated_vectors) = if self.metadata.rotate {
            let rotate_mat = self.rotate_mat::<T>();
            let rotate_mat = rotate_mat.slice(s![.., 0..dim]);
            let rotated_vectors = rotate_mat.dot(&vectors);

            let mut ip_rq_res = Vec::with_capacity(n);
            for vec_idx in 0..n {
                let sum_abs = (0..code_dim)
                    .map(|j| rotated_vectors[(j, vec_idx)].as_().abs())
                    .sum::<f32>();
                ip_rq_res.push(sum_abs / sqrt_dim);
            }

            let bv: BitVec<u8, Lsb0> = BitVec::from_iter(
                rotated_vectors
                    .t()
                    .iter()
                    .map(|v| v.as_().is_sign_positive()),
            );
            (ip_rq_res, bv, Some(rotated_vectors))
        } else {
            // rotate=false: code_dim must equal dim (enforced at build time).
            debug_assert_eq!(code_dim, dim);

            let mut ip_rq_res = Vec::with_capacity(n);
            for vec_idx in 0..n {
                let sum_abs = (0..dim)
                    .map(|j| vectors[(j, vec_idx)].as_().abs())
                    .sum::<f32>();
                ip_rq_res.push(sum_abs / sqrt_dim);
            }

            let bv: BitVec<u8, Lsb0> =
                BitVec::from_iter(vectors.t().iter().map(|v| v.as_().is_sign_positive()));
            (ip_rq_res, bv, None)
        };
        let codes = UInt8Array::from(bv.into_vec());
        debug_assert_eq!(codes.len(), n * code_dim / u8::BITS as usize);
        let codes = Arc::new(FixedSizeListArray::try_new_from_values(
            codes,
            code_dim as i32 / u8::BITS as i32, // num_bits -> num_bytes
        )?);

        let rotated_vectors = if keep_rotated {
            if self.metadata.rotate {
                let rotated_vectors =
                    rotated_vectors.expect("rotate=true implies rotated_vectors is present");
                let mut rotated_values = Vec::with_capacity(n * code_dim);
                for vec_idx in 0..n {
                    for j in 0..code_dim {
                        rotated_values.push(rotated_vectors[(j, vec_idx)].as_());
                    }
                }
                let rotated_values = Float32Array::from(rotated_values);
                Some(FixedSizeListArray::try_new_from_values(
                    rotated_values,
                    code_dim as i32,
                )?)
            } else {
                // rotate=false: the "rotated" vectors are just the original residual vectors,
                // converted to Float32 if needed.
                let values = residual_vectors.values();
                let rotated_values: Vec<f32> = match values.data_type() {
                    DataType::Float32 => values
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .expect("float32")
                        .values()
                        .to_vec(),
                    DataType::Float16 => values
                        .as_any()
                        .downcast_ref::<Float16Array>()
                        .expect("float16")
                        .values()
                        .iter()
                        .map(|v| v.to_f32())
                        .collect(),
                    DataType::Float64 => values
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .expect("float64")
                        .values()
                        .iter()
                        .map(|v| *v as f32)
                        .collect(),
                    dt => {
                        return Err(Error::invalid_input(
                            format!("Unsupported data type: {:?}", dt),
                            location!(),
                        ))
                    }
                };
                Some(FixedSizeListArray::try_new_from_values(
                    Float32Array::from(rotated_values),
                    code_dim as i32,
                )?)
            }
        } else {
            None
        };

        Ok((codes, ip_rq_res, rotated_vectors))
    }
}

impl Quantization for RabitQuantizer {
    type BuildParams = RQBuildParams;
    type Metadata = RabitQuantizationMetadata;
    type Storage = RabitQuantizationStorage;

    fn build(
        data: &dyn Array,
        _: lance_linalg::distance::DistanceType,
        params: &Self::BuildParams,
    ) -> Result<Self> {
        let q = match data.as_fixed_size_list().value_type() {
            DataType::Float16 => Self::try_new_with_rotate::<Float16Type>(
                params.num_bits,
                data.as_fixed_size_list().value_length(),
                params.rotate,
            )?,
            DataType::Float32 => Self::try_new_with_rotate::<Float32Type>(
                params.num_bits,
                data.as_fixed_size_list().value_length(),
                params.rotate,
            )?,
            DataType::Float64 => Self::try_new_with_rotate::<Float64Type>(
                params.num_bits,
                data.as_fixed_size_list().value_length(),
                params.rotate,
            )?,
            dt => {
                return Err(Error::invalid_input(
                    format!("Unsupported data type: {:?}", dt),
                    location!(),
                ))
            }
        };
        Ok(q)
    }

    fn retrain(&mut self, _data: &dyn Array) -> Result<()> {
        Ok(())
    }

    fn code_dim(&self) -> usize {
        self.metadata
            .rotate_mat
            .as_ref()
            .expect("rotate_mat not loaded")
            .len()
    }

    fn column(&self) -> &'static str {
        RABIT_CODE_COLUMN
    }

    fn use_residual(_: lance_linalg::distance::DistanceType) -> bool {
        true
    }

    fn quantize(&self, vectors: &dyn Array) -> Result<arrow_array::ArrayRef> {
        let vectors = vectors.as_fixed_size_list();
        match vectors.value_type() {
            DataType::Float16 => self.transform::<Float16Type>(vectors),
            DataType::Float32 => self.transform::<Float32Type>(vectors),
            DataType::Float64 => self.transform::<Float64Type>(vectors),
            value_type => Err(Error::invalid_input(
                format!("Unsupported data type: {:?}", value_type),
                location!(),
            )),
        }
    }

    fn metadata_key() -> &'static str {
        RABIT_METADATA_KEY
    }

    fn quantization_type() -> crate::vector::quantizer::QuantizationType {
        crate::vector::quantizer::QuantizationType::Rabit
    }

    fn metadata(
        &self,
        args: Option<crate::vector::quantizer::QuantizationMetadata>,
    ) -> Self::Metadata {
        let mut metadata = self.metadata.clone();
        if let Some(args) = args {
            metadata.packed = args.transposed;
        }
        metadata
    }

    fn from_metadata(
        metadata: &Self::Metadata,
        _: lance_linalg::distance::DistanceType,
    ) -> Result<Quantizer> {
        Ok(Quantizer::Rabit(Self {
            metadata: metadata.clone(),
        }))
    }

    fn field(&self) -> Field {
        Field::new(
            RABIT_CODE_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt8, true)),
                self.code_dim() as i32 / u8::BITS as i32, // num_bits -> num_bytes
            ),
            true,
        )
    }

    fn extra_fields(&self) -> Vec<Field> {
        vec![ADD_FACTORS_FIELD.clone(), SCALE_FACTORS_FIELD.clone()]
    }
}

impl TryFrom<Quantizer> for RabitQuantizer {
    type Error = Error;

    fn try_from(quantizer: Quantizer) -> Result<Self> {
        match quantizer {
            Quantizer::Rabit(quantizer) => Ok(quantizer),
            _ => Err(Error::invalid_input(
                "Cannot convert non-RabitQuantizer to RabitQuantizer",
                location!(),
            )),
        }
    }
}

impl From<RabitQuantizer> for Quantizer {
    fn from(quantizer: RabitQuantizer) -> Self {
        Self::Rabit(quantizer)
    }
}

fn random_normal_matrix(n: usize) -> ndarray::Array2<f64> {
    let mut rng = rand::rng();
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    ndarray::Array2::from_shape_simple_fn((n, n), || normal.sample(&mut rng))
}

// implement the householder qr decomposition referenced from https://en.wikipedia.org/wiki/Householder_transformation#QR_decomposition
fn householder_qr(a: ndarray::Array2<f64>) -> (ndarray::Array2<f64>, ndarray::Array2<f64>) {
    let (m, n) = a.dim();
    let mut q = ndarray::Array2::eye(m);
    let mut r = a;

    for k in 0..n.min(m - 1) {
        let mut x = r.slice(s![k.., k]).to_owned();
        let x_norm = x.dot(&x).sqrt();

        if x_norm < f64::EPSILON {
            continue;
        }

        // Create Householder vector
        let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
        x[0] += sign * x_norm;
        let u = &x / x.dot(&x).sqrt();

        // Apply Householder transformation to R
        // Compute outer product manually
        let mut u_outer = ndarray::Array2::zeros((m - k, m - k));
        for i in 0..(m - k) {
            for j in 0..(m - k) {
                u_outer[[i, j]] = u[i] * u[j];
            }
        }
        let h = ndarray::Array2::eye(m - k) - 2.0 * u_outer;

        // Apply transformation to R
        let r_block = r.slice(s![k.., k..]).to_owned();
        let h_r = h.dot(&r_block);
        r.slice_mut(s![k.., k..]).assign(&h_r);

        // Apply transformation to Q
        let q_block = q.slice(s![.., k..]).to_owned();
        let q_h = q_block.dot(&h);
        q.slice_mut(s![.., k..]).assign(&q_h);
    }

    (q, r)
}

fn random_orthogonal<T: ArrowFloatType>(n: usize) -> ndarray::Array2<T::Native>
where
    T::Native: FromPrimitive,
{
    let a = random_normal_matrix(n);
    let (q, _) = householder_qr(a);

    // cast f64 matrix to T::Native matrix
    q.mapv(|v| T::Native::from_f64(v).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use arrow::datatypes::UInt8Type;
    use arrow_array::FixedSizeListArray;
    use rstest::rstest;

    #[rstest]
    #[case(8)]
    #[case(16)]
    #[case(32)]
    fn test_householder_qr(#[case] n: usize) {
        let a = random_normal_matrix(n);
        let (m, n) = a.dim();

        let (q, r) = householder_qr(a.clone());

        // Check Q is orthogonal: Q^T * Q should be identity
        let q_t_q = q.t().dot(&q);
        for i in 0..m {
            for j in 0..m {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(q_t_q[[i, j]], expected, epsilon = 1e-5);
            }
        }

        // Check QR decomposition: Q * R should equal original matrix
        let qr = q.dot(&r);
        for i in 0..m {
            for j in 0..n {
                assert_relative_eq!(qr[[i, j]], a[[i, j]], epsilon = 1e-5);
            }
        }

        // Check R is upper triangular
        for i in 1..n.min(m) {
            for j in 0..i {
                assert_relative_eq!(r[[i, j]], 0.0, epsilon = 1e-5);
            }
        }

        // Additional check: Q should have shape (m, m) and R should have shape (m, n)
        assert_eq!(q.dim(), (m, m));
        assert_eq!(r.dim(), (m, n));
    }

    #[test]
    fn test_rabitq_rotate_false_skips_rotation() {
        let dim = 8;
        let rq = RabitQuantizer::try_new_with_rotate::<Float32Type>(1, dim, false).unwrap();

        assert!(!rq.metadata(None).rotate);
        assert_eq!(rq.code_dim(), dim as usize);

        // One vector: alternating signs, expecting code byte 0b0101_0101 (0x55).
        let values = Float32Array::from(vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
        let fsl = FixedSizeListArray::try_new_from_values(values, dim).unwrap();

        let (codes, _ip_rq_res, rotated) = rq.quantize_with_rotated_and_ip(&fsl, true).unwrap();
        let rotated = rotated.unwrap();
        assert_eq!(
            rotated.values().as_primitive::<Float32Type>().values(),
            fsl.values().as_primitive::<Float32Type>().values()
        );

        let codes = codes.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
        assert_eq!(codes.value_length(), 1);
        let code_byte = codes.values().as_primitive::<UInt8Type>().value(0);
        assert_eq!(code_byte, 0x55);
    }
}
