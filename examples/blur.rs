//! A simple demo to test the gaussian generation function and benchmark convolutions with large
//! kernels

use convolve2d::*;
#[cfg(feature = "image")]
use image::{io::Reader as ImageReader, RgbImage};
use std::time::Instant;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "GaussianBlur")]
struct Args {
    #[structopt(name = "IMAGE")]
    image: String,

    #[structopt(long, default_value = "5")]
    size: usize,

    #[structopt(long, default_value = "1.0")]
    std_dev: f64,
}

#[cfg(not(feature = "image"))]
fn main() {
    println!("This demo requires the optional `image` flag. Rerun with `--all-features`");
}

#[cfg(feature = "image")]
fn main() {
    let opt = Args::from_args();

    let img: DynamicMatrix<SubPixels<u8, 3>> = ImageReader::open(opt.image)
        .expect("Unable to open image")
        .decode()
        .expect("Unable to decode image")
        .to_rgb8()
        .into();
    let img: DynamicMatrix<SubPixels<f64, 3>> = img.map_subpixels(|sp| sp as f64 / 255.0);

    let kg_start = Instant::now();
    let kernel = kernel::gaussian(opt.size, opt.std_dev);
    let kernel_time = Instant::now() - kg_start;

    let cv_start = Instant::now();
    let convolution = convolve2d(&img, &kernel);
    let convolution_time = Instant::now() - cv_start;

    let convolution = convolution.map_subpixels(|sp| f64::round(sp.abs() * 255.0) as u8);
    RgbImage::from(convolution)
        .save("output.png")
        .expect("Unable to save image");

    println!(
        "Kernel Generation Time: {:.3}ms",
        kernel_time.as_secs_f64() * 1e3
    );
    println!(
        "Convolution Time: {:.3}ms",
        convolution_time.as_secs_f64() * 1e3
    );
}
