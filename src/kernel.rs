//! Definitions for various kernels that can be generated automatically.
//! 
//! The gaussian and box blur filters can be used to blur images while the sobel and laplacian 
//! filters are commonly used for edge detection. See the documentation on each function for more
//! detail.

/// Generate a Gaussian kernel with the specified standard deviation.
///
/// The current implementation requires the `"std"` feature flag. However, once complex expressions
/// for const generics is stabilized, the `size` parameter can be a compile time constant checked
/// by the type system.
///
/// The output matrix of this function is normalized so that the sum is 1.
///
/// # Example
/// ```
/// # use convolve2d::{DynamicMatrix, kernel};
/// let k1 = kernel::gaussian(5, 1.0);
/// let k2 = k1.map(|x| (x * 1000.0) as i32); // Convert into integers so we can use `==`
/// assert_eq!(k2, DynamicMatrix::new(5, 5, vec![
///      2, 13,  21, 13,  2,
///     13, 59,  98, 59, 13,
///     21, 98, 162, 98, 21,
///     13, 59,  98, 59, 13,
///      2, 13,  21, 13,  2
/// ]).unwrap())
/// ```
#[cfg(feature = "std")]
pub fn gaussian(size: usize, std_dev: f64) -> crate::DynamicMatrix<f64> {
    let stride = (size >> 1) as f64;
    let exp_coefficient = -0.5 / (std_dev * std_dev);
    let coefficient = 1.0 / std_dev;
    let allocation = size * size;

    // Set the values according to the gaussian function
    let mut data = std::vec![0.0; allocation];
    for (i, item) in data.iter_mut().enumerate() {
        let r = (i / size) as f64 - stride;
        let c = (i % size) as f64 - stride;
        let x_sq = r * r + c * c;
        *item = coefficient * f64::exp(x_sq * exp_coefficient);
    }

    // Normalize the values
    let sum = data.iter().sum::<f64>();
    if sum > 0.0 {
        data.iter_mut().for_each(|x| *x /= sum);
    }

    crate::DynamicMatrix::new(size, size, data).unwrap()
}

/// Generate a kernel used for box blur, normalized to 1.
///
/// The current implementation requires the `"std"` feature flag. However, once complex expressions
/// for const generics is stabilized, the `size` parameter can be a compile time constant checked
/// by the type system.
///
/// # Example
/// ```
/// # use convolve2d::{DynamicMatrix, kernel};
/// let k1 = kernel::box_blur(4);
/// assert_eq!(k1, DynamicMatrix::new(4, 4, vec![0.0625; 16]).unwrap())
/// ```
#[cfg(feature = "std")]
pub fn box_blur(size: usize) -> crate::DynamicMatrix<f64> {
    let value = 1.0 / (size * size) as f64;
    crate::DynamicMatrix::new(size, size, std::vec![value; size * size]).unwrap()
}

/// Sobel filters, commonly used for edge detection
pub mod sobel {
    use crate::StaticMatrix;

    /// A sobel filter that works in the X direction
    /// 
    /// This function is generic so that you can choose the data type that works best for you. 
    /// 
    /// # Example
    /// ```
    /// # use convolve2d::kernel;
    /// let mat = kernel::sobel::x::<i8>();
    /// let kernel = [
    ///     -1, 0, 1,
    ///     -2, 0, 2,
    ///     -1, 0, 1,
    /// ];
    /// 
    /// assert_eq!(mat.into_parts().2, kernel);
    /// ```
    #[rustfmt::skip]
    pub fn x<T: From<i8>>() -> StaticMatrix<T, 9> {
        StaticMatrix::new(
            3,
            3,
            [
                T::from(-1),  T::from(0),  T::from(1),
                T::from(-2),  T::from(0),  T::from(2),
                T::from(-1),  T::from(0),  T::from(1),
            ],
        )
        .unwrap()
    }

    /// A sobel filter that works in the Y direction
    /// 
    /// This function is generic so that you can choose the data type that works best for you. 
    /// 
    /// # Example
    /// ```
    /// # use convolve2d::kernel;
    /// let mat = kernel::sobel::y::<i8>();
    /// let kernel = [
    ///      1,  2,  1,
    ///      0,  0,  0,
    ///     -1, -2, -1,
    /// ];
    /// 
    /// assert_eq!(mat.into_parts().2, kernel);
    /// ```
    #[rustfmt::skip]
    pub fn y<T: From<i8>>() -> StaticMatrix<T, 9> {
        StaticMatrix::new(
            3,
            3,
            [
                T::from(1),  T::from(2),  T::from(1),
                T::from(0),  T::from(0),  T::from(0),
                T::from(-1),  T::from(-2),  T::from(-1),
            ],
        )
        .unwrap()
    }
}

/// Laplacian filters, used for edge detection
pub mod laplacian {
    use crate::StaticMatrix;

    /// A laplacian filter that works in a cross, ignoring data from the diagonals
    /// 
    /// This function is generic so that you can choose the data type that works best for you. 
    /// 
    /// # Example
    /// ```
    /// # use convolve2d::kernel;
    /// let mat = kernel::laplacian::cross::<i8>();
    /// let kernel = [
    ///      0, -1,  0,
    ///     -1,  4, -1,
    ///      0, -1,  0,
    /// ];
    /// 
    /// assert_eq!(mat.into_parts().2, kernel);
    /// ```
    #[rustfmt::skip]
    pub fn cross<T: From<i8>>() -> StaticMatrix<T, 9> {
        StaticMatrix::new(
            3,
            3,
            [
                T::from(0),  T::from(-1),  T::from(0),
                T::from(-1),  T::from(4),  T::from(-1),
                T::from(0),  T::from(-1),  T::from(0),
            ],
        )
        .unwrap()
    }

    /// A laplacian filter that works in all directions, taking pixel data from the diagonals as
    /// well as the xy cross.
    /// 
    /// This function is generic so that you can choose the data type that works best for you. 
    /// 
    /// # Example
    /// ```
    /// # use convolve2d::kernel;
    /// let mat = kernel::laplacian::full::<i8>();
    /// let kernel = [
    ///     -1, -1, -1,
    ///     -1,  8, -1,
    ///     -1, -1, -1,
    /// ];
    /// 
    /// assert_eq!(mat.into_parts().2, kernel);
    /// ```
    #[rustfmt::skip]
    pub fn full<T: From<i8>>() -> StaticMatrix<T, 9> {
        StaticMatrix::new(
            3,
            3,
            [
                T::from(-1),  T::from(-1),  T::from(-1),
                T::from(-1),  T::from(8),  T::from(-1),
                T::from(-1),  T::from(-1),  T::from(-1),
            ],
        )
        .unwrap()
    }
}
