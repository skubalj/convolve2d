//! A simple demo to test different edge detection filters

use clap::Parser;
use convolve2d::*;
use image::{io::Reader as ImageReader, GrayImage};
use std::ops::Sub;
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    image: String,
    kernel: String,
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

    let kernel: StaticMatrix<i32, 9> = match args.kernel.as_str() {
        "sobel_x" => kernel::sobel::x(),
        "sobel_y" => kernel::sobel::y(),
        "laplacian_cross" => kernel::laplacian::cross(),
        "laplacian_full" => kernel::laplacian::full(),
        _ => return,
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
