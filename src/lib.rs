// Disable the standard library
#![no_std]

// If the 'std' feature is enabled, re-enable the standard library
#[cfg(feature = "std")]
extern crate std;

mod convolution;
#[cfg(feature = "image")]
mod image_ext;
mod matrix;
mod subpixels;

// Library Public API
pub mod kernel;

pub use crate::{
    convolution::write_convolution,
    matrix::{Matrix, MatrixMut, StaticMatrix},
    subpixels::SubPixels,
};

#[cfg(feature = "std")]
pub use crate::{convolution::convolve2d, matrix::DynamicMatrix};
