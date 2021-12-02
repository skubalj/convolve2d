//! Definitions for various kernels that can be generated automatically.
//!
//! Many of the kernels defined in this module currently require the `"std"` feature, and use the
//! [`DynamicMatrix`] type. However, once "complex expressions" are implemented for const generics,
//! these will be able to be changed to [`StaticMatrix`]es, as their sizes can be checked at
//! compile time.

use crate::DynamicMatrix;
use std::vec;

/// Generate a Gaussian kernel with the specified standard deviation.
///
/// The current implementation requires the `"std"` feature flag. However, once complex expressions
/// for const generics is stabilized, the `size` parameter can be a compile time constant checked
/// by the type system.
#[cfg(feature = "std")]
pub fn gaussian(size: usize, std_dev: f64) -> DynamicMatrix<f64> {
    let stride = (size >> 1) as f64;
    let exp_coefficient = -0.5 / (std_dev * std_dev);
    let coefficient = 1.0 / std_dev;
    let allocation = size * size;

    // Set the values according to the gaussian function
    let mut data = vec![0.0; allocation];
    for (i, item) in data.iter_mut().enumerate() {
        let r = (i / size) as f64 - stride;
        let c = (i % size) as f64 - stride;
        let x_sq = r * r + c * c;
        *item = coefficient * f64::exp(x_sq * exp_coefficient);
    }

    // Normalize the values
    let sum = data.iter().sum::<f64>();
    if sum > 0.0 {
        data.iter_mut().for_each(|x| *x /= sum);
    }

    DynamicMatrix::new(size, size, data).unwrap()
}
