`convolve2d`: Image Convolutions in Rust
========================================

This crate defines an *easy* and *extensible* way to conduct image convolutions, in a way that is 
*dependency free*, and works with `no_std`. Sound cool? Read on!

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

This library is still in the early phase of development, and there's more to come! Stay tuned!

## Acknowledgment:
Thanks to the following packages!
| Crate                                             | Owner / Maintainer        | License           |
| :------------------------------------------------ | :------------------------ | :---------------- |
| [`image`](https://crates.io/crates/image)         | HeroicKatora, fintelia    | MIT               |
| [`rayon`](https://crates.io/crates/rayon)         | Josh Stone, Niko Matsakis | Apache 2.0 or MIT |
| [`test-case`](https://crates.io/crates/test-case) | Wojciech Polak, Luke Biel | MIT               |

And to the Rust community at large!

## License:
This crate is released under the terms of the MIT License. 

Copyright (C) 2021 Joseph Skubal