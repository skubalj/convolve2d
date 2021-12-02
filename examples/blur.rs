use convolve2d::*;
use image::io::Reader as ImageReader;
use image::GrayImage;
use std::env::args;
use std::ops::Sub;
use std::time::Instant;

fn print_usage() {
    println!(
        "Usage: {} <filepath> <kernel_size> <std_dev>",
        args().nth(0).unwrap()
    );
}

fn main() {
    let filepath = match args().nth(1) {
        Some(x) => x,
        None => {
            print_usage();
            return;
        }
    };

    let size = match args().nth(2) {
        Some(x) => x.parse().unwrap(),
        None => {
            print_usage();
            return;
        }
    };

    let std_dev = match args().nth(3) {
        Some(x) => x.parse().unwrap(),
        None => {
            print_usage();
            return;
        }
    };

    let img = ImageReader::open(filepath)
        .expect("Unable to open image")
        .decode()
        .expect("Unable to decode image")
        .into_luma8();

    let floating = img.as_raw().iter().map(|&x| x as f64 / 255.0).collect();
    let img_mat =
        DynamicMatrix::new(img.width() as usize, img.height() as usize, floating).unwrap();

    let kg_start = Instant::now();
    let kernel = kernel::gaussian(size, std_dev);
    let kg_stop = Instant::now();
    let convolution = get_convolution(&img_mat, &kernel);
    let cv_stop = Instant::now();

    let out_vec = convolution
        .get_data()
        .iter()
        .map(|x| (x * 255.0) as u8)
        .collect();
    GrayImage::from_vec(img.width(), img.height(), out_vec)
        .unwrap()
        .save("output.png")
        .expect("Unable to save image");

    println!(
        "Kernel Generation Time: {:.3}ms",
        kg_stop.sub(kg_start).as_secs_f64() * 1e3
    );
    println!(
        "Convolution Time: {:.3}ms",
        cv_stop.sub(kg_stop).as_secs_f64() * 1e3
    );
}
