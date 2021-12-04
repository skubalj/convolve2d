use crate::matrix::{FlippedMatrix, Matrix, MatrixMut};
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
/// This function is a convient interface for the [`convolve2d`] function, which stores the
/// generated convolution in a new allocation.
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
/// Perform a 2D convolution on the specified image with the provided kernel, storing the result
/// in the provided buffer.
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

/// Convert the provided alignment to padding and choke values.
fn alignment_to_choke_padding(alignment: isize) -> (usize, usize) {
    // Use the alignment calculation to determine our choke and padding numbers
    let mut choke = 0;
    let mut padding = 0;
    if alignment < 0 {
        choke = alignment.abs() as usize;
    } else {
        padding = alignment as usize;
    }
    (choke, padding)
}

#[cfg(not(feature = "rayon"))]
fn update_buffer<T, K, O>(image: &[T], kernel_value: K, alignment: isize, buf: &mut [O])
where
    T: Mul<K, Output = O> + Clone,
    K: Clone,
    O: Add<Output = O> + Clone,
{
    let (choke, padding) = alignment_to_choke_padding(alignment);

    image
        .iter()
        .map(|x| x.clone() * kernel_value.clone())
        .skip(choke)
        .zip(buf.iter_mut().skip(padding))
        .for_each(|(n, a)| *a = a.clone() + n)
}

#[cfg(feature = "rayon")]
fn update_buffer<T, K, O>(image: &[T], kernel_value: K, alignment: isize, buf: &mut [O])
where
    T: Mul<K, Output = O> + Clone + Send + Sync,
    K: Clone + Send + Sync,
    O: Add<Output = O> + Clone + Send,
{
    let (choke, padding) = alignment_to_choke_padding(alignment);

    image
        .par_iter()
        .map(|x| x.clone() * kernel_value.clone())
        .skip(choke)
        .zip(buf.par_iter_mut().skip(padding))
        .for_each(|(n, a)| *a = a.clone() + n)
}

#[cfg(test)]
mod tests {
    use super::update_buffer;
    use crate::{write_convolution, StaticMatrix};
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

    #[test]
    fn number_grid() {
        let img = StaticMatrix::new(3, 3, [0, 0, 0, 0, 1, 0, 0, 0, 0]).unwrap();
        let kernel = StaticMatrix::new(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let mut output = StaticMatrix::new(3, 3, [0; 9]).unwrap();

        write_convolution(&img, &kernel, &mut output);

        let expected = StaticMatrix::new(3, 3, [9, 8, 7, 6, 5, 4, 3, 2, 1]).unwrap();
        assert_eq!(output, expected);
    }
}
