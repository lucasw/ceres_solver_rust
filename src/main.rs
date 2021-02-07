// use cxx::CxxVector;

mod jet;

use jet::Jet;
use nalgebra::U1;

#[cxx::bridge(namespace = "org::ceres_example")]
mod ffi {
    extern "Rust" {
        // "extern function with generic parameters is not supported yet"
        // fn evaluate<T>(val: T) -> T;
        fn evaluate(val: f64) -> f64;
        fn evaluate_raw_jet(val: f64, val_v: &[f64; 1], residual: &mut f64, residual_v: &mut [f64; 1]);
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
    let residual = 15.31 - val;
    residual
}

pub fn evaluate_raw_jet(val: f64, val_v: &[f64; 1], residual: &mut f64, residual_v: &mut [f64; 1]) {
    *residual = 15.31 - val;
    *residual_v = *val_v;
    residual_v[0] *= 0.9;
    println!("rust x {}, {:?} -> residual {}, {:?}", val, val_v, residual, residual_v);
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
