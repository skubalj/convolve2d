`convolve2d`: Image Convolutions in Rust
========================================

This crate defines an *easy* and *extensible* way to conduct image convolutions, in a way that is 
*free of system dependencies*, and works with `no_std`. Sound cool? Read on!

The purpose of `convolve2d` is to provide a single package that provides everything you need to 
conduct image convolutions suitable for computer vision or image manipulation. Here's a breif 
overview of what's on offer:

* **Two convolution functions**: allowing you to pass your own buffer if speed is important, or 
  have a buffer allocated and returned for a more idiomatic interface.

* **Traits**: Convolution is defined generically across the `Matrix` trait. If you have a custom 
  image type, simply define an implementation of `Matrix`, and you're good to go!

* **Built-in `image` Support**: We also offer support for the `image` library, through a feature 
  flag (disabled by default), allowing you to seamlessly use the types you're already used to!

* **`rayon`**: Compute convolutions in parallel using the `rayon` flag. (Enabled by default)

* **`no_std` Operation**: to suit the needs of specialty systems or WASM.

* **Kernel Generators**: The `kernel` module provides generation functions for a number of kernels
  commonly used in image processing.

While other convolution libraries may be more efficient, using a faster algorithm, or running on the
GPU, this library's main focus is providing a complete convolution experience that is portable and 
easy to use.

## Example:
This example shows how easy it is to perform convolutions when using the extensions for the `image`
library. (See the `image` feature)

```rust
use image::RgbImage;
use convolve2d::*;

// Simply use `into` to convert from an `ImageBuffer` to a `DynamicMatrix`.
let image_buffer: RgbImage = ...;
let img: DynamicMatrix<SubPixels<u8, 3>> = image_buffer.into();

// Convert our color space to floating point, since our gaussian will be `f64`s
let img: DynamicMatrix<SubPixels<f64, 3>> = img.map_subpixels(|sp| sp as f64 / 255.0);

// Generate a 5x5 gaussian with standard deviation 2.0
let kernel = kernel::gaussian(5, 2.0);

// Perform the convolution, getting back a new `DynamicMatrix`
let convolution = convolve2d(&img, &kernel);

// Convert the color space back to 8-bit colors 
let convolution = convolution.map_subpixels(|sp| f64::round(sp * 255.0) as u8);

// Convert back into an `RgbImage` and save using the `image` library
RgbImage::from(convolution).save("output.png").expect("Unable to save image");
```

## Features:

The following features are supported:

| Feature | Default | Description |
| :------ | :------ | :---------- |
| `std`   | Yes     | Allow access to the standard library, enabling the `DynamicMatrix` type. |
| `rayon` | Yes     | Use rayon to compute convolutions in parallel.                           |
| `image` | No      | Add extensions for interoperation with the `image` crate.                |
| `full`  | No      | All features.                                                            |

To use the library in `no_std` mode, simply disable all features: 
```toml
convolve2d = { version = "0.1.0", default-features = false }
```

## Acknowledgment:
Thanks to the following packages!
| Crate                                             | Owner / Maintainer        | License           |
| :------------------------------------------------ | :------------------------ | :---------------- |
| [`image`](https://crates.io/crates/image)         | HeroicKatora, fintelia    | MIT               |
| [`rayon`](https://crates.io/crates/rayon)         | Josh Stone, Niko Matsakis | Apache 2.0 or MIT |
| [`test-case`](https://crates.io/crates/test-case) | Wojciech Polak, Luke Biel | MIT               |

And to the Rust community at large!

## Contributions:
Is something not clear? Do we need another kernel type? This library came about as a personal
project, but feel free to submit issues or PRs on GitLab!

## License:
This crate is released under the terms of the MIT License. 

Copyright (C) 2021 Joseph Skubal