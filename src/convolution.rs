//! Definitions of the two convolution functions provided by the library

use crate::matrix::{FlippedMatrix, Matrix, MatrixMut};
use crate::{SaturatingAdd, SaturatingMul};
use core::ops::{Add, Mul};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

// Re-import the standard library
#[cfg(feature = "std")]
use crate::DynamicMatrix;
#[cfg(feature = "std")]
use std::prelude::v1::*;
#[cfg(feature = "std")]
use std::vec;

/// Perform a 2D convolution on the specified image with the provided kernel.
///
/// This function is a convient interface for the [`write_convolution`] function, automatically
/// generating a new allocation in which to store the convolution. In most cases, this function
/// should be preferred, as it is the more idiomatic implemntation. However, in higher performance
/// scenarios, or in contexts in which greater control is needed, `write_convolution` may still be
/// useful
///
/// Naturally, as this function uses the `DynamicMatrix` type, it requires the `std` feature.
///
/// # Example
/// ```
/// use convolve2d::{convolve2d, DynamicMatrix};
/// let mat = DynamicMatrix::new(3, 3, vec![
///     0, 0, 0,
///     0, 1, 0,
///     0, 0, 0,
/// ]).unwrap();
///
/// let kernel = DynamicMatrix::new(3, 3, vec![
///     1, 2, 3,
///     4, 5, 6,
///     7, 8, 9,
/// ]).unwrap();
///
/// let output = convolve2d(&mat, &kernel);
/// assert_eq!(output, DynamicMatrix::new(3, 3, vec![9, 8, 7, 6, 5, 4, 3, 2, 1]).unwrap());
/// ```
///
/// # Panics
/// If the kernel's `get_value` method does not return `Some` for all row and column values in the
/// ranges `0..kenrel.get_height()` and `0..kernel.get_width()`.
///
/// Why should this panic? It would be easy to return a `Result` instead, but having `get_value`
/// fail to return a value when we expect it to be valid indicates a programming failure, rather
/// than a simple error in execution.
#[cfg(feature = "std")]
pub fn convolve2d<T, K, O>(image: &impl Matrix<T>, kernel: &impl Matrix<K>) -> DynamicMatrix<O>
where
    T: Mul<K, Output = O> + Clone + Send + Sync,
    K: Clone + Send + Sync,
    O: Add<Output = O> + Default + Clone + Send,
{
    let allocation = image.get_width() * image.get_height();
    let mut out = DynamicMatrix::new(
        image.get_width(),
        image.get_height(),
        vec![O::default(); allocation],
    )
    .unwrap();
    write_convolution(image, kernel, &mut out);
    out
}

/// Write the convolution of the provided image and kernel into the specified buffer.
///
/// The name of this function is meant to evoke memories of [`std::fmt::write`], which also takes
/// a sink as an output parameter.
///
/// While this function avoids allocations, and is therefore slightly faster, you may prefer the
/// [`convolve2d`] function for a more idiomatic approach.
///
/// # Example
/// ```
/// use convolve2d::{write_convolution, StaticMatrix};
/// let mat = StaticMatrix::new(3, 3, [
///     0, 0, 0,
///     0, 1, 0,
///     0, 0, 0,
/// ]).unwrap();
///
/// let kernel = StaticMatrix::new(3, 3, [
///     1, 2, 3,
///     4, 5, 6,
///     7, 8, 9,
/// ]).unwrap();
///
/// let mut output = StaticMatrix::new(3, 3, [0; 9]).unwrap();
/// write_convolution(&mat, &kernel, &mut output);
/// assert_eq!(output, StaticMatrix::new(3, 3, [9, 8, 7, 6, 5, 4, 3, 2, 1]).unwrap());
/// ```
///
/// # Panics
/// If the kernel's `get_value` method does not return `Some` for all row and column values in the
/// ranges `0..kenrel.get_height()` and `0..kernel.get_width()`.
///
/// Why should this panic? It would be easy to return a `Result` instead, but having `get_value`
/// fail to return a value when we expect it to be valid indicates a programming failure, rather
/// than a simple error in execution.
pub fn write_convolution<T, K, O>(
    image: &impl Matrix<T>,
    kernel: &impl Matrix<K>,
    out: &mut impl MatrixMut<O>,
) where
    T: Mul<K, Output = O> + Clone + Send + Sync,
    K: Clone + Send + Sync,
    O: Add<Output = O> + Clone + Send,
{
    // Flip the kernel, as is the custom for convolutions
    let kernel = FlippedMatrix(kernel);

    let kernel_stride_x = (kernel.get_width() >> 1) as isize;
    let kernel_stride_y = (kernel.get_height() >> 1) as isize;

    for row in 0..kernel.get_height() {
        // Calculate how many rows there are between the top of the image and the top of the kernel.
        let rows_off_center = row as isize - kernel_stride_y;

        for col in 0..kernel.get_width() {
            // Calculate how many columns there are between the left side of the image and the left
            // side of the kernel
            let cols_off_center = col as isize - kernel_stride_x;

            // Determine the number of elements that the image row needs to be shifted
            let alignment = rows_off_center * image.get_width() as isize + cols_off_center;

            // Apply this kernel value and add to the buffer
            update_buffer(
                image.get_data(),
                kernel.get_value(row, col).unwrap().clone(),
                alignment,
                out.get_data_mut(),
            );
        }
    }
}

/// Perform a 2D convolution on the specified image with the provided kernel, without integer overflow.
///
/// This function is a convient interface for the [`write_convolution`] function, automatically
/// generating a new allocation in which to store the convolution. In most cases, this function
/// should be preferred, as it is the more idiomatic implemntation. However, in higher performance
/// scenarios, or in contexts in which greater control is needed, `write_convolution` may still be
/// useful
///
/// Naturally, as this function uses the `DynamicMatrix` type, it requires the `std` feature.
///
/// # Example
/// ```
/// use convolve2d::{convolve2d, DynamicMatrix};
/// let mat = DynamicMatrix::new(3, 3, vec![
///     0, 0, 0,
///     0, 1, 0,
///     0, 0, 0,
/// ]).unwrap();
///
/// let kernel = DynamicMatrix::new(3, 3, vec![
///     1, 2, 3,
///     4, 5, 6,
///     7, 8, 9,
/// ]).unwrap();
///
/// let output = convolve2d(&mat, &kernel);
/// assert_eq!(output, DynamicMatrix::new(3, 3, vec![9, 8, 7, 6, 5, 4, 3, 2, 1]).unwrap());
/// ```
///
/// # Panics
/// If the kernel's `get_value` method does not return `Some` for all row and column values in the
/// ranges `0..kenrel.get_height()` and `0..kernel.get_width()`.
///
/// Why should this panic? It would be easy to return a `Result` instead, but having `get_value`
/// fail to return a value when we expect it to be valid indicates a programming failure, rather
/// than a simple error in execution.
#[cfg(feature = "std")]
pub fn convolve2d_saturating<T, K, O>(
    image: &impl Matrix<T>,
    kernel: &impl Matrix<K>,
) -> DynamicMatrix<O>
where
    T: SaturatingMul<K, Output = O> + Clone + Send + Sync,
    K: Clone + Send + Sync,
    O: SaturatingAdd<Output = O> + Default + Clone + Send,
{
    let allocation = image.get_width() * image.get_height();
    let mut out = DynamicMatrix::new(
        image.get_width(),
        image.get_height(),
        vec![O::default(); allocation],
    )
    .unwrap();
    write_convolution_saturating(image, kernel, &mut out);
    out
}

/// Write the convolution of the provided image and kernel into the specified buffer, without integer overflow
///
/// The name of this function is meant to evoke memories of [`std::fmt::write`], which also takes
/// a sink as an output parameter.
///
/// While this function avoids allocations, and is therefore slightly faster, you may prefer the
/// [`convolve2d`] function for a more idiomatic approach.
///
/// # Example
/// ```
/// use convolve2d::{write_convolution, StaticMatrix};
/// let mat = StaticMatrix::new(3, 3, [
///     0, 0, 0,
///     0, 1, 0,
///     0, 0, 0,
/// ]).unwrap();
///
/// let kernel = StaticMatrix::new(3, 3, [
///     1, 2, 3,
///     4, 5, 6,
///     7, 8, 9,
/// ]).unwrap();
///
/// let mut output = StaticMatrix::new(3, 3, [0; 9]).unwrap();
/// write_convolution(&mat, &kernel, &mut output);
/// assert_eq!(output, StaticMatrix::new(3, 3, [9, 8, 7, 6, 5, 4, 3, 2, 1]).unwrap());
/// ```
///
/// # Panics
/// If the kernel's `get_value` method does not return `Some` for all row and column values in the
/// ranges `0..kenrel.get_height()` and `0..kernel.get_width()`.
///
/// Why should this panic? It would be easy to return a `Result` instead, but having `get_value`
/// fail to return a value when we expect it to be valid indicates a programming failure, rather
/// than a simple error in execution.
pub fn write_convolution_saturating<T, K, O>(
    image: &impl Matrix<T>,
    kernel: &impl Matrix<K>,
    out: &mut impl MatrixMut<O>,
) where
    T: SaturatingMul<K, Output = O> + Clone + Send + Sync,
    K: Clone + Send + Sync,
    O: SaturatingAdd<Output = O> + Clone + Send,
{
    // Flip the kernel, as is the custom for convolutions
    let kernel = FlippedMatrix(kernel);

    let kernel_stride_x = (kernel.get_width() >> 1) as isize;
    let kernel_stride_y = (kernel.get_height() >> 1) as isize;

    for row in 0..kernel.get_height() {
        // Calculate how many rows there are between the top of the image and the top of the kernel.
        let rows_off_center = row as isize - kernel_stride_y;

        for col in 0..kernel.get_width() {
            // Calculate how many columns there are between the left side of the image and the left
            // side of the kernel
            let cols_off_center = col as isize - kernel_stride_x;

            // Determine the number of elements that the image row needs to be shifted
            let alignment = rows_off_center * image.get_width() as isize + cols_off_center;

            // Apply this kernel value and add to the buffer
            update_buffer_saturating(
                image.get_data(),
                kernel.get_value(row, col).unwrap().clone(),
                alignment,
                out.get_data_mut(),
            );
        }
    }
}

/// Convert the provided alignment to padding and choke values.
///
/// If the provided alignment is positive, that implies that we need to pad our output stream. If
/// the provided alignment is negative, that implies we need to choke up on our output stream,
/// throwing away the first `n` elements.
fn alignment_to_choke_padding(alignment: isize) -> (usize, usize) {
    // Use the alignment calculation to determine our choke and padding numbers
    let mut choke = 0;
    let mut padding = 0;
    if alignment < 0 {
        choke = alignment.unsigned_abs();
    } else {
        padding = alignment as usize;
    }
    (choke, padding)
}

/// Update the output buffer, multiplying the image by the kernel value and adding it to the
/// buffer at the specified alignment.
fn update_buffer<T, K, O>(image: &[T], kernel_value: K, alignment: isize, buf: &mut [O])
where
    T: Mul<K, Output = O> + Clone + Send + Sync,
    K: Clone + Send + Sync,
    O: Add<Output = O> + Clone + Send,
{
    let (choke, padding) = alignment_to_choke_padding(alignment);

    #[cfg(not(feature = "rayon"))]
    let image_iter = image.iter();
    #[cfg(feature = "rayon")]
    let image_iter = image.par_iter();

    #[cfg(not(feature = "rayon"))]
    let buf_iter = buf.iter_mut();
    #[cfg(feature = "rayon")]
    let buf_iter = buf.par_iter_mut();

    image_iter
        .map(|x| x.clone() * kernel_value.clone())
        .skip(choke)
        .zip(buf_iter.skip(padding))
        .for_each(|(n, a)| *a = a.clone() + n)
}

/// Update the output buffer, multiplying the image by the kernel value and adding it to the
/// buffer at the specified alignment.
fn update_buffer_saturating<T, K, O>(image: &[T], kernel_value: K, alignment: isize, buf: &mut [O])
where
    T: SaturatingMul<K, Output = O> + Clone + Send + Sync,
    K: Clone + Send + Sync,
    O: SaturatingAdd<Output = O> + Clone + Send,
{
    let (choke, padding) = alignment_to_choke_padding(alignment);

    #[cfg(not(feature = "rayon"))]
    let image_iter = image.iter();
    #[cfg(feature = "rayon")]
    let image_iter = image.par_iter();

    #[cfg(not(feature = "rayon"))]
    let buf_iter = buf.iter_mut();
    #[cfg(feature = "rayon")]
    let buf_iter = buf.par_iter_mut();

    image_iter
        .map(|x| x.clone().saturating_mul(kernel_value.clone()))
        .skip(choke)
        .zip(buf_iter.skip(padding))
        .for_each(|(n, a)| *a = a.clone().saturating_add(n))
}

#[cfg(test)]
mod tests {
    use super::update_buffer;
    use crate::{write_convolution, write_convolution_saturating, StaticMatrix};
    use test_case::test_case;

    #[test_case(-5, [12, 14, 16, 18, 0, 0, 0, 0, 0]; "alignment_n5")]
    #[test_case(-1, [4, 6, 8, 10, 12, 14, 16, 18, 0]; "alignment_n1")]
    #[test_case(0, [2, 4, 6, 8, 10, 12, 14, 16, 18]; "alignment_0")]
    #[test_case(1, [0, 2, 4, 6, 8, 10, 12, 14, 16]; "alignment_1")]
    #[test_case(5, [0, 0, 0, 0, 0, 2, 4, 6, 8]; "alignment_5")]
    fn update_buffer_t(alignment: isize, arr: [u32; 9]) {
        let image = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut output = [0; 9];
        update_buffer(&image, 2u32, alignment, &mut output);
        assert_eq!(output, arr);
    }

    #[cfg(feature = "std")]
    #[test]
    fn convolve2d_smoke_test() {
        let img = StaticMatrix::new(3, 3, [0, 0, 0, 0, 1, 0, 0, 0, 0]).unwrap();
        let kernel = StaticMatrix::new(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();

        let output = crate::convolve2d(&img, &kernel);

        let expected =
            crate::DynamicMatrix::new(3, 3, std::vec![9, 8, 7, 6, 5, 4, 3, 2, 1]).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn number_grid() {
        let img = StaticMatrix::new(3, 3, [0, 0, 0, 0, 1, 0, 0, 0, 0]).unwrap();
        let kernel = StaticMatrix::new(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let mut output = StaticMatrix::new(3, 3, [0; 9]).unwrap();

        write_convolution(&img, &kernel, &mut output);

        let expected = StaticMatrix::new(3, 3, [9, 8, 7, 6, 5, 4, 3, 2, 1]).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn test_saturating() {
        let img: StaticMatrix<u8, 9> =
            StaticMatrix::new(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let kernel = StaticMatrix::new(1, 1, [128]).unwrap();
        let mut output = StaticMatrix::new(3, 3, [0; 9]).unwrap();

        write_convolution_saturating(&img, &kernel, &mut output);

        let expected =
            StaticMatrix::new(3, 3, [128, 255, 255, 255, 255, 255, 255, 255, 255]).unwrap();
        assert_eq!(output, expected);
    }
}
