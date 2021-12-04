use convolve2d::*;
use image::io::Reader as ImageReader;
use image::RgbImage;
use std::ops::Sub;
use std::time::Instant;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
struct Args {
    #[structopt(name = "IMAGE")]
    image: String,

    #[structopt(name = "KERNEL")]
    kernel: String,
}

fn main() {
    let args = Args::from_args();

    let img: DynamicMatrix<SubPixels<u8, 3>> = ImageReader::open(args.image)
        .expect("Unable to open image")
        .decode()
        .expect("Unable to decode image")
        .into_rgb8()
        .into();
    let img = img.map_subpixels(|sp| sp as f64 / 255.0);

    let kernel: StaticMatrix<f64, 9> = match args.kernel.as_str() {
        "sobel_x" => kernel::sobel_x(),
        "sobel_y" => kernel::sobel_y(),
        "laplacian_cross" => kernel::laplacian_cross(),
        "laplacian_full" => kernel::laplacian_full(),
        _ => return,
    };

    let cv_start = Instant::now();
    let convolution = convolve2d(&img, &kernel);
    let cv_stop = Instant::now();

    let convolution = convolution.map_subpixels(|x| f64::round(x.abs() * 255.0) as u8);
    RgbImage::from(convolution)
        .save("output.png")
        .expect("Unable to save image");

    println!(
        "Convolution Time: {:.3}ms",
        cv_stop.sub(cv_start).as_secs_f64() * 1e3
    );
}
