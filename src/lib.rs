use crate::matrix::FlippedMatrix;
use std::ops::{Add, Mul};

#[cfg(feature = "image")]
mod image_ext;

pub mod kernels;
mod matrix;
pub use crate::matrix::{Matrix, OwnedMatrix};

pub fn convolve2d<T, K, O>(image: impl Matrix<T>, kernel: impl Matrix<K>) -> OwnedMatrix<O>
where
    T: Mul<K, Output = O> + Clone,
    K: Clone,
    O: Add<Output = O> + Default + Clone,
{
    // Flip the kernel, as is the custom for convolutions
    let kernel = FlippedMatrix(kernel);

    let kernel_stride_x = (kernel.get_width() >> 1) as isize;
    let kernel_stride_y = (kernel.get_height() >> 1) as isize;

    let mut buf = vec![O::default(); image.get_width() * image.get_height()];

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
            let value = kernel.get_value(row, col).unwrap().clone();
            image
                .get_data()
                .iter()
                .map(|x| x.clone() * value.clone())
                .skip(choke)
                .zip(buf.iter_mut().skip(padding))
                .for_each(|(n, a)| *a = a.clone() + n)
        }
    }

    OwnedMatrix::new(image.get_width(), image.get_height(), buf)
        .expect("Generated image is invalid size?")
}

#[cfg(test)]
mod tests {
    use crate::{convolve2d, OwnedMatrix};

    #[test]
    fn number_grid() {
        let img = OwnedMatrix::new(3, 3, vec![0, 0, 0, 0, 1, 0, 0, 0, 0]).unwrap();
        let kernel = OwnedMatrix::new(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let expected = OwnedMatrix::new(3, 3, vec![9, 8, 7, 6, 5, 4, 3, 2, 1]).unwrap();

        assert_eq!(convolve2d(img, kernel), expected);
    }
}
