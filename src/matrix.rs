//! Definition of the [`Matrix`] trait

use crate::SubPixels;
#[cfg(feature = "std")]
use std::prelude::v1::*;

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

/// A `Matrix` with a size known at compile time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticMatrix<T, const N: usize> {
    /// The number of columns in the matrix
    pub width: usize,
    /// The number of rows in the matrix
    pub height: usize,
    /// The set of all values in this matrix
    pub data: [T; N],
}

impl<T, const N: usize> StaticMatrix<T, N> {
    /// Create a new `StaticMatrix` with the specified dimensions.
    ///
    /// Returns `None` if the length of the provided data is not `width * height`
    ///
    /// # Example
    /// ```
    /// # use convolve2d::StaticMatrix;
    /// assert!(StaticMatrix::new(2, 2, [1, 2, 3, 4]).is_some());
    /// assert!(StaticMatrix::new(2, 3, [1, 2, 3, 4, 5]).is_none());
    /// ```
    pub fn new(width: usize, height: usize, data: [T; N]) -> Option<Self> {
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

    pub fn map<F, O>(self, operation: F) -> StaticMatrix<O, N>
    where
        F: Fn(T) -> O,
        O: Default + Copy,
    {
        let mut arr = [O::default(); N];
        for (i, x) in self.data.into_iter().enumerate() {
            arr[i] = operation(x);
        }
        StaticMatrix::new(self.width, self.height, arr).unwrap()
    }
}

impl<T: Copy, const M: usize, const N: usize> StaticMatrix<SubPixels<T, N>, M> {
    pub fn map_subpixels<F, O>(self, operation: F) -> StaticMatrix<SubPixels<O, N>, M>
    where
        F: Fn(T) -> O + Copy,
        O: Default + Copy,
    {
        let mut arr = [SubPixels::default(); M];
        for (i, x) in self.data.into_iter().enumerate() {
            arr[i] = x.map(operation);
        }
        StaticMatrix::new(self.width, self.height, arr).unwrap()
    }
}

impl<T, const N: usize> Matrix<T> for StaticMatrix<T, N> {
    fn get_width(&self) -> usize {
        self.width
    }

    fn get_height(&self) -> usize {
        self.height
    }

    fn get_data(&self) -> &[T] {
        &self.data
    }
}

impl<T, const N: usize> MatrixMut<T> for StaticMatrix<T, N> {
    fn get_data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

/// A concrete implementation of `Matrix` for which the size is not known at compile time.
///
/// Requires the `"std"` feature to be enabled.
#[cfg(feature = "std")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicMatrix<T> {
    /// The number of columns in the matrix
    pub width: usize,
    /// The number of rows in the matrix
    pub height: usize,
    /// The set of all values in this matrix
    pub data: Vec<T>,
}

#[cfg(feature = "std")]
impl<T> DynamicMatrix<T> {
    /// Create a new `OwnedMatrix` with the specified data
    ///
    /// Returns `None` if the length of the provided data is not `width * height`.
    ///
    /// # Example
    /// ```
    /// # use convolve2d::DynamicMatrix;
    /// # use std::vec;
    /// assert!(DynamicMatrix::new(2, 3, vec![0f64; 6]).is_some());
    /// assert!(DynamicMatrix::new(2, 3, vec![0f64; 5]).is_none());
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

    pub fn map<F: Fn(T) -> O, O>(self, operation: F) -> DynamicMatrix<O> {
        let arr = self.data.into_iter().map(operation).collect();
        DynamicMatrix::new(self.width, self.height, arr).unwrap()
    }
}

#[cfg(feature = "std")]
impl<T: Copy, const N: usize> DynamicMatrix<SubPixels<T, N>> {
    pub fn map_subpixels<F, O>(self, operation: F) -> DynamicMatrix<SubPixels<O, N>>
    where
        F: Fn(T) -> O + Copy,
        O: Default + Copy,
    {
        let arr = self.data.into_iter().map(|sp| sp.map(operation)).collect();
        DynamicMatrix::new(self.width, self.height, arr).unwrap()
    }
}

#[cfg(feature = "std")]
impl<T> Matrix<T> for DynamicMatrix<T> {
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

#[cfg(feature = "std")]
impl<T> MatrixMut<T> for DynamicMatrix<T> {
    fn get_data_mut(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::FlippedMatrix;
    use crate::{Matrix, StaticMatrix};

    #[test]
    fn flipped_matrix() {
        let mat = StaticMatrix::new(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let flipped = FlippedMatrix(&mat);
        assert_eq!(flipped.get_value(0, 0), Some(&9));
        assert_eq!(flipped.get_value(2, 1), Some(&2));
        assert_eq!(flipped.get_value(0, 2), Some(&7));
        assert_eq!(flipped.get_value(1, 1), Some(&5));
    }
}
