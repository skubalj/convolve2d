//! Definitions for various kernels that can be generated automatically.
//!
//! Many of the kernels defined in this module currently require the `"std"` feature, and use the
//! [`DynamicMatrix`] type. However, once "complex expressions" are implemented for const generics,
//! these will be able to be changed to [`StaticMatrix`]es, as their sizes can be checked at
//! compile time.

use crate::{DynamicMatrix, StaticMatrix};
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

/// A sobel filter that works in the X direction
#[rustfmt::skip]
pub fn sobel_x<T: From<i8>>() -> StaticMatrix<T, 9> {
    StaticMatrix::new(
        3,
        3,
        [
            T::from(-1),  T::from(0),  T::from(1),
            T::from(-2),  T::from(0),  T::from(2),
            T::from(-1),  T::from(0),  T::from(1),
        ],
    )
    .unwrap()
}

/// A sobel filter that works in the Y direction
#[rustfmt::skip]
pub fn sobel_y<T: From<i8>>() -> StaticMatrix<T, 9> {
    StaticMatrix::new(
        3,
        3,
        [
            T::from(1),  T::from(2),  T::from(1),
            T::from(0),  T::from(0),  T::from(0),
            T::from(-1),  T::from(-2),  T::from(-1),
        ],
    )
    .unwrap()
}

/// A laplacian filter that works in a cross 
#[rustfmt::skip]
pub fn laplacian_cross<T: From<i8>>() -> StaticMatrix<T, 9> {
    StaticMatrix::new(
        3,
        3,
        [
            T::from(0),  T::from(-1),  T::from(0),
            T::from(-1),  T::from(4),  T::from(-1),
            T::from(0),  T::from(-1),  T::from(0),
        ],
    )
    .unwrap()
}

/// A laplacian filter that takes pixel data from the diagonals as well.
#[rustfmt::skip]
pub fn laplacian_full<T: From<i8>>() -> StaticMatrix<T, 9> {
    StaticMatrix::new(
        3,
        3,
        [
            T::from(-1),  T::from(-1),  T::from(-1),
            T::from(-1),  T::from(8),  T::from(-1),
            T::from(-1),  T::from(-1),  T::from(-1),
        ],
    )
    .unwrap()
}
