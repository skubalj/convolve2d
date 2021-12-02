use crate::matrix::{FlippedMatrix, Matrix, MatrixMut};
use core::ops::{Add, Mul};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

pub fn convolve2d<T, K, O>(
    image: &impl Matrix<T>,
    kernel: &impl Matrix<K>,
    out: &mut impl MatrixMut<O>,
) where
    T: Mul<K, Output = O> + Clone + Send + Sync,
    K: Clone + Send + Sync,
    O: Add<Output = O> + Default + Clone + Send,
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

            // Use the alignment calculation to determine our choke and padding numbers
            let mut choke = 0;
            let mut padding = 0;
            if alignment < 0 {
                choke = alignment.abs() as usize;
            } else {
                padding = alignment as usize;
            }

            // Apply this kernel value and add to the buffer
            update_buffer(
                image.get_data(),
                kernel.get_value(row, col).unwrap().clone(),
                choke,
                padding,
                out.get_data_mut(),
            );
        }
    }
}

#[cfg(not(feature = "rayon"))]
fn update_buffer<T, K, O>(image: &[T], kernel_value: K, choke: usize, padding: usize, buf: &mut [O])
where
    T: Mul<K, Output = O> + Clone,
    K: Clone,
    O: Add<Output = O> + Default + Clone,
{
    image
        .iter()
        .map(|x| x.clone() * kernel_value.clone())
        .skip(choke)
        .zip(buf.iter_mut().skip(padding))
        .for_each(|(n, a)| *a = a.clone() + n)
}

#[cfg(feature = "rayon")]
fn update_buffer<T, K, O>(image: &[T], kernel_value: K, choke: usize, padding: usize, buf: &mut [O])
where
    T: Mul<K, Output = O> + Clone + Send + Sync,
    K: Clone + Send + Sync,
    O: Add<Output = O> + Default + Clone + Send,
{
    image
        .par_iter()
        .map(|x| x.clone() * kernel_value.clone())
        .skip(choke)
        .zip(buf.par_iter_mut().skip(padding))
        .for_each(|(n, a)| *a = a.clone() + n)
}

#[cfg(test)]
mod tests {
    use crate::{convolve2d, OwnedMatrix};
    use std::vec;

    #[test]
    fn number_grid() {
        let img = OwnedMatrix::new(3, 3, vec![0, 0, 0, 0, 1, 0, 0, 0, 0]).unwrap();
        let kernel = OwnedMatrix::new(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let mut output = OwnedMatrix::new(3, 3, vec![0; 9]).unwrap();

        convolve2d(&img, &kernel, &mut output);

        let expected = OwnedMatrix::new(3, 3, vec![9, 8, 7, 6, 5, 4, 3, 2, 1]).unwrap();
        assert_eq!(output, expected);
    }
}
