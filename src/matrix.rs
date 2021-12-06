//! Definitions for the [`Matrix`] trait, and the concrete implementations provided by the library.

use crate::SubPixels;
#[cfg(feature = "std")]
use std::prelude::v1::*;

/// An easily implementable interface for types that can be used in a convolution.
///
/// # Extensibility
/// To be able to feed your own types into the convolution functions, simply define an
/// implementation of `Matrix` for your types.
///
/// Note that while `Matrix` makes no demands of its generic type `T`, in practice `T` must conform
/// to the generic requirements of the convolution functions. ([`convolve2d`](crate::convolve2d)
/// and [`write_convolution`](crate::write_convolution)).
///
/// Note that it is expected that the slice returned from `get_data` has length `width * height`.
/// If this invariant is violated, you **are not** going to violate memory safety, but the result of
/// the convolution **will most likely** be garbage.
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

/// A subtype of [`Matrix`] allowing mutable access to the underlying data.
///
/// # Extensibility
/// Implement this trait (in addition to `Matrix`) if you want to use one of your own types as the
/// output buffer for [`write_convoution`](crate::write_convolution). This trait is not required
/// if you only want to use your types as inputs.
pub trait MatrixMut<T>: Matrix<T> {
    /// Get a mutable slice to the underlying matrix data
    fn get_data_mut(&mut self) -> &mut [T];
}

/// A wrapper around a matrix that flips the values in `get_value`
///
/// The primary use for this wrapper is to flip the kernel before conducting the convolution
#[repr(transparent)]
pub(crate) struct FlippedMatrix<'a, M>(pub &'a M);

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

/// A [`Matrix`] with a size known at compile time.
///
/// The `StaticMatrix` type provides a simple, heap free way to use this library. However, it means
/// that the size of the matrix has to be known at compile time. Additionally, as all matrix data
/// is stored on the stack, copying a `StaticMatrix` can be a time consuming task. (This is why I
/// chose to make `StaticMatrix` not implement `Clone`, even if `T` does.) The `std` feature
/// provides access to the [`DynamicMatrix`], which will generally be easier to use.
///
/// However, even with the `std` feature enabled, you may find `StaticMatrix` to be handy for
/// defining kernels which have a known value at compile time. Many of the kernels in the
/// [`kernel`](crate::kernel) module use `StaticMatrix` for their implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticMatrix<T, const N: usize> {
    /// The number of columns in the matrix
    width: usize,
    /// The number of rows in the matrix
    height: usize,
    /// The set of all values in this matrix
    data: [T; N],
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

    /// Perform a map operation on this matrix.
    ///
    /// Each element in the matrix body is given to the provided function, and the results are
    /// aggregated into a new `StaticMatrix`. A common use for this function is to convert between
    /// integers and floating point numbers.
    ///
    /// If your matrix's element type is `SubPixels`, then consider using
    /// [`map_subpixels`](StaticMatrix::map_subpixels) to apply the operation to each subpixel,
    /// rather than to each pixel group.
    ///
    /// # Example
    /// ```
    /// # use convolve2d::StaticMatrix;
    /// let mat: StaticMatrix<u32, 4> = StaticMatrix::new(2, 2, [1, 2, 3, 4]).unwrap();
    /// assert_eq!(
    ///     mat.map(|x| x as f64),
    ///     StaticMatrix::new(2, 2, [1.0, 2.0, 3.0, 4.0]).unwrap()
    /// );
    /// ```
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

    /// Consume `self`, and return the width, height, and matrix data. (in that order).
    ///
    /// # Extensibility
    /// This method is intended to be used to destructure a `StaticMatrix` into its fields so that
    /// it can be converted cleanly to a user defined type.
    pub fn into_parts(self) -> (usize, usize, [T; N]) {
        (self.width, self.height, self.data)
    }
}

impl<T: Copy, const M: usize, const N: usize> StaticMatrix<SubPixels<T, N>, M> {
    /// Perform a map operation on each of the individual subpixel elements in the matrix.
    ///
    /// This function is a shortcut for calling [`StaticMatrix::map`], then calling
    /// [`SubPixels::map`] with the provided function.
    ///
    /// # Example
    /// ```
    /// # use convolve2d::{StaticMatrix, SubPixels};
    /// let mat = StaticMatrix::new(2, 2, [
    ///     SubPixels([1, 2, 3]), SubPixels([4, 5, 6]),
    ///     SubPixels([7, 8, 9]), SubPixels([10, 11, 12])
    /// ]).unwrap();
    ///
    /// let expected = StaticMatrix::new(2, 2, [
    ///     SubPixels([2, 4, 6]), SubPixels([8, 10, 12]),
    ///     SubPixels([14, 16, 18]), SubPixels([20, 22, 24])
    /// ]).unwrap();
    ///
    /// assert_eq!(mat.map_subpixels(|x| x * 2), expected);
    /// ```
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

/// A concrete implementation of [`Matrix`] for which the size is not known at compile time.
///
/// Requires the `std` feature to be enabled. If you're working without the standard library,
/// see [`StaticMatrix`].
///
/// The `DynamicMatrix` type is the preferred concrete implemenatation of `Matrix` that we provide,
/// as it stores its data in a `Vec`. If you're building a matrix to do something, this should
/// probably be your first stop, especially if your matrix is large.
#[cfg(feature = "std")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicMatrix<T> {
    /// The number of columns in the matrix
    width: usize,
    /// The number of rows in the matrix
    height: usize,
    /// The set of all values in this matrix
    data: Vec<T>,
}

#[cfg(feature = "std")]
impl<T> DynamicMatrix<T> {
    /// Create a new `DynamicMatrix` with the specified data
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

    /// Perform a map operation on this matrix.
    ///
    /// Each element in the matrix body is given to the provided function, and the results are
    /// aggregated into a new `DynamicMatrix`. A common use for this function is to convert between
    /// integers and floating point numbers.
    ///
    /// If your matrix's element type is `SubPixels`, then consider using
    /// [`map_subpixels`](DynamicMatrix::map_subpixels) to apply the operation to each subpixel,
    /// rather than to each pixel group.
    ///
    /// # Example
    /// ```
    /// # use convolve2d::DynamicMatrix;
    /// let mat: DynamicMatrix<u32> = DynamicMatrix::new(2, 2, vec![1, 2, 3, 4]).unwrap();
    /// assert_eq!(
    ///     mat.map(|x| x as f64),
    ///     DynamicMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap()
    /// );
    /// ```
    pub fn map<F: Fn(T) -> O, O>(self, operation: F) -> DynamicMatrix<O> {
        let arr = self.data.into_iter().map(operation).collect();
        DynamicMatrix::new(self.width, self.height, arr).unwrap()
    }

    /// Consume `self`, and return the width, height, and matrix data. (in that order).
    ///
    /// # Extensibility
    /// This function is intended to be used to destructure the `DynamicMatrix` into something that
    /// can be converted to a user defined type.
    pub fn into_parts(self) -> (usize, usize, Vec<T>) {
        (self.width, self.height, self.data)
    }
}

#[cfg(feature = "std")]
impl<T: Copy, const N: usize> DynamicMatrix<SubPixels<T, N>> {
    /// Perform a map operation on each of the individual subpixel elements in the matrix.
    ///
    /// This function is a shortcut for calling [`DynamicMatrix::map`], then calling
    /// [`SubPixels::map`] with the provided function.
    ///
    /// # Example
    /// ```
    /// # use convolve2d::{DynamicMatrix, SubPixels};
    /// let mat = DynamicMatrix::new(2, 2, vec![
    ///     SubPixels([1, 2, 3]), SubPixels([4, 5, 6]),
    ///     SubPixels([7, 8, 9]), SubPixels([10, 11, 12])
    /// ]).unwrap();
    ///
    /// let expected = DynamicMatrix::new(2, 2, vec![
    ///     SubPixels([2, 4, 6]), SubPixels([8, 10, 12]),
    ///     SubPixels([14, 16, 18]), SubPixels([20, 22, 24])
    /// ]).unwrap();
    ///
    /// assert_eq!(mat.map_subpixels(|x| x * 2), expected);
    /// ```
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
