// use cxx::CxxVector;

mod jet;

use jet::Jet;
use nalgebra::U1;

#[cxx::bridge(namespace = "org::ceres_example")]
mod ffi {

    // TODO(lucasw) Convert to and from the ceres::Jet, provide all Jet operations in native rust
    // also?
    // See ceres-solver/include/ceres/jet.h
    // struct RustJet {
    //   a: f64,
    //   TODO(lucasw) don't use a Vec, instead a fixed array based on ceres::Jet v size.
    //   v: Vec<f64>,
    // }

    extern "Rust" {
        // "extern function with generic parameters is not supported yet"
        // fn evaluate<T>(val: T) -> T;
        fn evaluate(val: f64) -> f64;
    }

    // C++ types and signatures exposed to Rust.
    unsafe extern "C++" {
        include!("ceres_cxx/include/ceres_example.h");

        type CeresExample;

        fn new_ceres_example() -> UniquePtr<CeresExample>;

        // fn run<T>(&self, vals: &Vec<T>);
        fn run(&self, vals: &Vec<f64>);
    }
}

pub fn evaluate(val: f64) -> f64 {
    let residual = 12.3 - val;
    residual
}

fn main() {
    let j0 = Jet::new(2.5, 0.1, U1);
    println!("jet {:?}", j0);

    let j1 = Jet::new(3.0, 0.01, U1);
    let v2 = j1.v + j0.v;
    println!("jet op {:?}.v + {:?}.v = {:?}", j1.v, j0.v, v2);
    let j2 = j1 + j0;
    println!("jet op {:?} + {:?} = {:?}", j1, j0, j2);

    let j3 = j1 * j0;
    println!("jet op {:?} * {:?} = {:?}", j1, j0, j3);

    let ceres_example = ffi::new_ceres_example();
    // is it possible to create a CxxVector on the rust side?  There isn't a CxxVector::new()
    // let vals = CxxVector;  // ::<f64>();
    let mut vals = Vec::new();
    vals.push(5.342);
    vals.push(8.0);
    ceres_example.run(&vals);
}
