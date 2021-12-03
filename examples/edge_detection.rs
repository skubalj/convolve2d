use convolve2d::*;
use image::io::Reader as ImageReader;
use image::GrayImage;
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

    let img = ImageReader::open(args.image)
        .expect("Unable to open image")
        .decode()
        .expect("Unable to decode image")
        .into_luma8();

    let floating = img.as_raw().iter().map(|&x| x as f64 / 255.0).collect();
    let img_mat =
        DynamicMatrix::new(img.width() as usize, img.height() as usize, floating).unwrap();

    let kernel: StaticMatrix<f64, 9> = match args.kernel.as_str() {
        "sobel_x" => kernel::sobel_x(),
        "sobel_y" => kernel::sobel_y(),
        "laplacian_cross" => kernel::laplacian_cross(),
        "laplacian_full" => kernel::laplacian_full(),
        _ => return,
    };

    let cv_start = Instant::now();
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
        "Convolution Time: {:.3}ms",
        cv_stop.sub(cv_start).as_secs_f64() * 1e3
    );
}
