//! A simple demo to test different edge detection filters

use clap::{Parser, ValueEnum};
use convolve2d::*;
use image::{GrayImage, ImageReader};
use std::fmt::Display;
use std::ops::Sub;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Kernel {
    SobelX,
    SobelY,
    LaplacianCross,
    LaplacianFull,
}

impl Display for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SobelX => write!(f, "sobel_x"),
            Self::SobelY => write!(f, "sobel_y"),
            Self::LaplacianCross => write!(f, "laplacian_cross"),
            Self::LaplacianFull => write!(f, "laplacian_full"),
        }
    }
}

#[derive(Parser, Debug)]
struct Args {
    image: String,
    kernel: Kernel,
}

fn main() {
    let args = Args::parse();

    let img: DynamicMatrix<SubPixels<u8, 1>> = ImageReader::open(args.image)
        .expect("Unable to open image")
        .decode()
        .expect("Unable to decode image")
        .into_luma8()
        .into();
    let img = img.map_subpixels(|sp| sp as i32);

    let kernel: StaticMatrix<i32, 9> = match args.kernel {
        Kernel::SobelX => kernel::sobel::x(),
        Kernel::SobelY => kernel::sobel::y(),
        Kernel::LaplacianCross => kernel::laplacian::cross(),
        Kernel::LaplacianFull => kernel::laplacian::full(),
    };

    let cv_start = Instant::now();
    let convolution = convolve2d(&img, &kernel);
    let cv_stop = Instant::now();

    let convolution = convolution.map_subpixels(|x| x.abs() as u8);
    GrayImage::from(convolution)
        .save("output.png")
        .expect("Unable to save image");

    println!(
        "Convolution Time: {:.3}ms",
        cv_stop.sub(cv_start).as_secs_f64() * 1e3
    );
}
