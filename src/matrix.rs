//! Definition of the [`Matrix`] trait

/// The `Matrix` trait provides an easily extendable interface for data in the program
pub trait Matrix<T> {
    /// Get the width of the matrix
    fn get_width(&self) -> usize;

    /// Get the height of the matrix
    fn get_height(&self) -> usize;

    /// Retrieve the data stored in this matrix
    fn get_data(&self) -> &[T];

    /// Get the value stored at the given row and column of the matrix
    fn get_value(&self, row: usize, col: usize) -> Option<&T> {
        self.get_data().get(row * self.get_width() + col)
    }
}

/// A subtype of [`Matrix`] allowing mutable access to the underlying data
pub trait MatrixMut<T>: Matrix<T> {
    /// Get a mutable slice to the underlying matrix data
    fn get_data_mut(&mut self) -> &mut [T];
}

/// A wrapper around a matrix that flips the values in `get_value`
///
/// The primary use for this wrapper is to flip the kernel before conducting the convolution
#[repr(transparent)]
pub struct FlippedMatrix<'a, M>(pub &'a M);

impl<'a, M, T> Matrix<T> for FlippedMatrix<'a, M>
where
    M: Matrix<T>,
{
    #[inline]
    fn get_width(&self) -> usize {
        self.0.get_width()
    }

    #[inline]
    fn get_height(&self) -> usize {
        self.0.get_height()
    }

    #[inline]
    fn get_data(&self) -> &[T] {
        self.0.get_data()
    }

    fn get_value(&self, row: usize, col: usize) -> Option<&T> {
        let new_row = self.get_height() - row - 1;
        let new_col = self.get_width() - col - 1;
        self.0.get_value(new_row, new_col)
    }
}

#[cfg(feature = "std")]
pub mod std_dependent {
    use super::{Matrix, MatrixMut};
    use std::vec::Vec;

    #[derive(Debug, Clone, PartialEq)]
    pub struct OwnedMatrix<T> {
        width: usize,
        height: usize,
        data: Vec<T>,
    }

    impl<T> OwnedMatrix<T> {
        /// Create a new `OwnedMatrix` with the specified data
        ///
        /// Returns `None` if the length of the provided data is not `width * height`.
        ///
        /// # Example
        /// ```
        /// # use convolve2d::OwnedMatrix;
        /// # use std::vec;
        /// assert!(OwnedMatrix::new(2, 3, vec![0f64; 6]).is_some());
        /// assert!(OwnedMatrix::new(2, 3, vec![0f64; 5]).is_none());
        /// ```
        pub fn new(width: usize, height: usize, data: Vec<T>) -> Option<Self> {
            if width * height == data.len() {
                Some(Self {
                    width,
                    height,
                    data,
                })
            } else {
                None
            }
        }
    }

    impl<T> Matrix<T> for OwnedMatrix<T> {
        fn get_width(&self) -> usize {
            self.width
        }

        fn get_height(&self) -> usize {
            self.height
        }

        fn get_data(&self) -> &[T] {
            self.data.as_slice()
        }
    }

    impl<T> MatrixMut<T> for OwnedMatrix<T> {
        fn get_data_mut(&mut self) -> &mut [T] {
            self.data.as_mut_slice()
        }
    }
}
