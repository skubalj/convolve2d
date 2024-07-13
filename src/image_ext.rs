//! This module contains extensions to the `image` library to allow for greater compatibility.
//!
//! Here, we define how `image` types are converted into our working types, and back out.

#[cfg(feature = "std")]
use crate::DynamicMatrix;
use crate::{Matrix, SubPixels};
use core::ops::Deref;
use image::{ImageBuffer, Luma, LumaA, Pixel, Primitive, Rgb, Rgba};
#[cfg(feature = "std")]
use std::prelude::v1::*;

/// Implementations to covnert to and from the SubPixels type for each color format supported by
/// the `image` library.
macro_rules! from_subpixels {
    {$($type:ident, $n:expr;)*} => {$(
        impl<T: Primitive> From<SubPixels<T, $n>> for $type<T> {
            fn from(sp: SubPixels<T, $n>) -> Self {
                $type(sp.0)
            }
        }

        impl<T: Primitive> From<$type<T>> for SubPixels<T, $n> {
            fn from(format: $type<T>) -> Self {
                format.0.into()
            }
        }
    )*}
}

from_subpixels! {Luma, 1; LumaA, 2; Rgb, 3; Rgba, 4;}

/// An implementation of [`Matrix`] for grayscale ImageBuffers. This works because each pixel is
/// one element in the underlying container.
impl<T, C> Matrix<T> for ImageBuffer<Luma<T>, C>
where
    T: Primitive + 'static,
    C: Deref<Target = [T]>,
{
    fn get_width(&self) -> usize {
        self.width() as usize
    }

    fn get_height(&self) -> usize {
        self.height() as usize
    }

    fn get_data(&self) -> &[T] {
        self.as_raw()
    }
}

#[cfg(feature = "std")]
impl<P, SP, const N: usize> From<ImageBuffer<P, Vec<SP>>> for DynamicMatrix<SubPixels<SP, N>>
where
    P: 'static + Pixel<Subpixel = SP> + Into<SubPixels<SP, N>>,
    SP: 'static + Primitive,
{
    fn from(buf: ImageBuffer<P, Vec<P::Subpixel>>) -> Self {
        let subpixel_data = buf.pixels().map(|&x| x.into()).collect();
        // `unwrap` is safe here because `ImageBuffer` ensures the size is correct.
        Self::new(buf.width() as usize, buf.height() as usize, subpixel_data).unwrap()
    }
}

#[cfg(feature = "std")]
impl<P, SP, const N: usize> From<DynamicMatrix<SubPixels<SP, N>>> for ImageBuffer<P, Vec<SP>>
where
    P: 'static + Pixel<Subpixel = SP> + From<SubPixels<SP, N>>,
    SP: 'static + Primitive,
{
    fn from(buf: DynamicMatrix<SubPixels<SP, N>>) -> Self {
        let (width, height, data) = buf.into_parts();
        let vec = data.into_iter().flat_map(|x| x.0).collect();
        // This works, even though buf has been partially moved because of disjoint capture
        // `unwrap` is safe here, because `DynamicMatrix`'s size is checked on creation
        ImageBuffer::from_vec(width as u32, height as u32, vec).unwrap()
    }
}
