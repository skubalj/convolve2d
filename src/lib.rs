//! This crate defines an *easy* and *extensible* way to conduct image convolutions, in a way that
//! is *free of system dependencies*, and works with `no_std`.
//!
//! The purpose of `convolve2d` is to provide a single package that provides everything you need to
//! conduct image convolutions suitable for computer vision or image manipulation. Here's a breif
//! overview of what's on offer:
//!
//! * **Two convolution functions**: allowing you to pass your own buffer if speed is important, or
//!   have a buffer allocated and returned for a more idiomatic interface.
//!
//! * **Traits**: Convolution is defined generically across the [`Matrix`] trait. If you have a
//!   custom image type, simply define an implementation of `Matrix`, and you're good to go!
//!
//! * **Built-in `image` Support**: We also offer support for the `image` library through a feature
//!   flag (disabled by default), allowing you to seamlessly use the types you're already used to!
//!
//! * **`rayon`**: Compute convolutions in parallel using the `rayon` flag. (Enabled by default)
//!
//! * **`no_std` Operation**: to suit the needs of specialty systems or WASM.
//!
//! * **Kernel Generators**: The [`kernel`] module provides generation functions for a number of
//!   kernels commonly used in image processing.
//!
//! While other convolution libraries may be more efficient, use a faster algorithm, or run
//! on the GPU, this library's main focus is providing a complete convolution experience that is
//! portable and easy to use.
//!
//! # Features:
//!
//! The following features are supported:
//!
//! | Feature | Default | Description |
//! | :------ | :------ | :---------- |
//! | `std`   | Yes     | Allow access to the standard library, enabling the `DynamicMatrix` type. |
//! | `rayon` | Yes     | Use rayon to compute convolutions in parallel.                           |
//! | `image` | No      | Add extensions for interoperation with the `image` crate.                |
//! | `full`  | No      | All features.                                                            |
//!
//! To use the library in `no_std` mode, simply disable all features:
//! ```toml
//! convolve2d = { version = "0.3.0", default-features = false }
//! ```
//!
//! # Notes on `image` Compatibility
//! Compatibility with the `image` library is provided using the `image` feature flag. This flag
//! provides the following features:
//!
//! * The various pixel formats (`Rgb`, `Luma`, etc...) can now be converted to and from the
//!   `SubPixels` type. This allows them to be scaled and added as required for convolutions.
//!
//! * `ImageBuffer` can be converted to and from `DynamicMatrix`es using `into` and `from`.
//!
//! * `ImageBuffer`s for which the pixel type is `Luma` can be used as `Matrix`es directly. This is
//!   because each element in the underlying data structure is one pixel. (Whereas in an RGB image,
//!   each element is one subpixel, meaning we need to group with `SubPixels`)

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
    convolution::{write_convolution, write_convolution_saturating},
    matrix::{Matrix, MatrixMut, StaticMatrix},
    subpixels::SubPixels,
};

#[cfg(feature = "std")]
pub use crate::{
    convolution::{convolve2d, convolve2d_saturating},
    matrix::DynamicMatrix,
};

/// A trait for types that can add without overflowing
pub trait SaturatingAdd<Rhs = Self> {
    /// The resulting type after applying addition
    type Output;

    /// Add `other` to `rhs`, without overflowing
    fn saturating_add(self, rhs: Rhs) -> Self::Output;
}

/// A trait for types that can be multiplied without overflowing
pub trait SaturatingMul<Rhs = Self> {
    /// The resulting type after applying multiplication
    type Output;

    /// Multiply `self` by `rhs`, without overflowing
    fn saturating_mul(self, rhs: Rhs) -> Self::Output;
}

macro_rules! saturating_impl {
    ($($t:ty),+) => {
        $(
            impl SaturatingAdd<$t> for $t {
                type Output = Self;

                #[inline]
                fn saturating_add(self, v: Self) -> Self {
                    <$t>::saturating_add(self, v)
                }
            }

            impl SaturatingMul<$t> for $t {
                type Output = Self;

                #[inline]
                fn saturating_mul(self, v: Self) -> Self {
                    <$t>::saturating_mul(self, v)
                }
            }
        )+
    };
}

saturating_impl!(u8, u16, u32, u64, u128, usize);
saturating_impl!(i8, i16, i32, i64, i128, isize);
