// Unconditionally disable the standard library
#![no_std]

// Re-enable the standard library
#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "image")]
mod image_ext;

mod convolution;
pub mod kernels;
mod matrix;

pub use crate::convolution::convolve2d;
pub use crate::matrix::std_dependent::OwnedMatrix;
pub use crate::matrix::{Matrix, MatrixMut};
