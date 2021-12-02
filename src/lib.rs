// Disable the standard library
#![no_std]

#[cfg(feature = "std")]
extern crate std;
#[cfg(feature = "std")]
pub use crate::convolution::get_convolution;
#[cfg(feature = "std")]
pub use crate::matrix::DynamicMatrix;

mod convolution;
#[cfg(feature = "image")]
mod image_ext;
pub mod kernels;
mod matrix;

pub use crate::convolution::convolve2d;
pub use crate::matrix::{Matrix, MatrixMut, StaticMatrix};
