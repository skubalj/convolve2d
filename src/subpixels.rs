use core::ops::{Add, Mul};

/// A collection of subpixels that should make working with multi-channeled images more convenient.
///
/// This struct implements both `Add` and `Mul`, so that it can be used as the data type for a
/// [`Matrix`](crate::Matrix)
///
/// Instead of needing to divide the Red, Green, and Blue channels out so that each has its own
/// image `Matrix`, using `SubPixels` gives you the ability to perform all three convolutions at
/// once.
///
/// # Example
/// ```
/// # use convolve2d::SubPixels;
/// let sp1 = SubPixels([1, 2, 3]);
/// let sp2 = SubPixels([4, 5, 6]);
///
/// assert_eq!(sp1 * 2, SubPixels([2, 4, 6]));
/// assert_eq!(sp1 + sp2, SubPixels([5, 7, 9]));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubPixels<T: Copy, const N: usize>(pub [T; N]);

impl<T: Copy, const N: usize> SubPixels<T, N> {
    /// Perform an infallible type conversion
    ///
    /// # Example
    /// ```
    /// # use convolve2d::SubPixels;
    /// let pixel: SubPixels<u8, 3> = SubPixels([1, 2, 3]);
    /// assert_eq!(pixel.convert::<f64>(), SubPixels([1.0, 2.0, 3.0]));
    /// ```
    pub fn convert<U: From<T> + Copy + Default>(self) -> SubPixels<U, N> {
        let mut arr = [U::default(); N];
        for (a, n) in arr.iter_mut().zip(self.0) {
            *a = U::from(n);
        }
        SubPixels(arr)
    }

    /// Perform a fallible type conversion
    ///
    /// # Example
    /// ```
    /// # use convolve2d::SubPixels;
    /// let pixel: SubPixels<u32, 3> = SubPixels([1, 2, 3]);
    /// assert_eq!(pixel.try_convert::<u8>(), Ok(SubPixels([1, 2, 3])));
    /// ```
    pub fn try_convert<U: TryFrom<T> + Copy + Default>(self) -> Result<SubPixels<U, N>, U::Error> {
        let mut arr = [U::default(); N];
        for (a, n) in arr.iter_mut().zip(self.0) {
            *a = U::try_from(n)?;
        }
        Ok(SubPixels(arr))
    }

    /// Perform a map operation, applying the provided function to each subpixel.
    ///
    /// # Example
    /// ```
    /// # use convolve2d::SubPixels;
    /// let pixel = SubPixels([1, 2, 3]);
    /// assert_eq!(pixel.map(|x| x + 1), SubPixels([2, 3, 4]));
    /// ```
    pub fn map<F, O>(self, operation: F) -> SubPixels<O, N>
    where
        F: Fn(T) -> O,
        O: Default + Copy,
    {
        let mut arr = [O::default(); N];
        for (i, x) in self.0.into_iter().enumerate() {
            arr[i] = operation(x);
        }
        SubPixels(arr)
    }
}

impl<T: Copy, const N: usize> From<[T; N]> for SubPixels<T, N> {
    fn from(other: [T; N]) -> Self {
        Self(other)
    }
}

impl<T: Add<Output = T> + Copy, const N: usize> Add for SubPixels<T, N> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        for (i, x) in rhs.0.into_iter().enumerate() {
            self.0[i] = self.0[i] + x;
        }
        self
    }
}

impl<T, C, O, const N: usize> Mul<C> for SubPixels<T, N>
where
    C: Copy,
    T: Mul<C, Output = O> + Copy,
    O: Default + Copy,
{
    type Output = SubPixels<O, N>;

    fn mul(self, rhs: C) -> Self::Output {
        let mut arr = [O::default(); N];
        self.0
            .into_iter()
            .map(|a| a * rhs)
            .enumerate()
            .for_each(|(i, v)| arr[i] = v);
        SubPixels(arr)
    }
}

impl<T: Copy + Default, const N: usize> Default for SubPixels<T, N> {
    fn default() -> Self {
        Self([T::default(); N])
    }
}
