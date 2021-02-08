// Lucas Walter
// 2020-2021

mod jet;
mod trajectory;

use cxx::UniquePtr;
use jet::Jet;
use nalgebra::U1;
use std::time::Instant;
use trajectory::WayPoint;

#[cxx::bridge(namespace = "org::ceres_example")]
mod ffi {
    // rust to expose to C++
    extern "Rust" {
        // "extern function with generic parameters is not supported yet"
        // fn evaluate<T>(val: T) -> T;
        fn evaluate(val: f64) -> f64;
        fn evaluate_raw_jet(
            val_a: f64,
            val_v: &[f64; 1],
            residual_a: &mut f64,
            residual_v: &mut [f64; 1],
        );

        /*
        fn evaluate_trajectory(
            start: &WayPoint,
            target: &WayPoint,
            j0_a: f64,
            j0_v: &[f64; 1],
            j1_a: f64,
            j1_v: &[f64; 1],
            j2_a: f64,
            j2_v: &[f64; 1],
            j3_a: f64,
            j3_v: &[f64; 1],
            residual0_a: &mut f64,
            residual0_v: &mut [f64; 1],
            residual1_a: &mut f64,
            residual1_v: &mut [f64; 1],
            residual2_a: &mut f64,
            residual2_v: &mut [f64; 1],
            residual3_a: &mut f64,
            residual3_v: &mut [f64; 1],
        );
        */

        type WayPoint;
    }

    // C++ types and signatures exposed to Rust.
    unsafe extern "C++" {
        include!("ceres_cxx/include/ceres_example.h");

        type CeresExample;

        fn new_ceres_example() -> UniquePtr<CeresExample>;

        // These can't be anything but &Vec because of cxx limitations
        #[allow(clippy::ptr_arg)]
        fn run_numeric(&self, vals: &Vec<f64>);
        #[allow(clippy::ptr_arg)]
        fn run_auto(&self, vals: &Vec<f64>);

        // TODO(lucasw) need to return the solution
        // fn find_trajectory(&self, start: &WayPoint, target: &WayPoint);
    }
}

pub fn evaluate(val: f64) -> f64 {
    let target = 15.31;
    target - val
}

pub fn evaluate_raw_jet(
    val_a: f64,
    val_v: &[f64; 1],
    residual_a: &mut f64,
    residual_v: &mut [f64; 1],
) {
    let target = Jet::new(15.31, 0.0, U1);
    let val = Jet::from_vec(val_a, val_v, U1);
    let residual = target - val;
    *residual_a = residual.a;
    *residual_v = residual.get_vec();
    // println!("rust x {:?} -> residual {:?}", val, residual);
}

fn simple_example(ceres_example: &cxx::UniquePtr<ffi::CeresExample>) {
    let j0 = Jet::new(2.5, 0.1, U1);
    println!("jet {:?}", j0);

    let j1 = Jet::new(3.0, 0.01, U1);
    let v2 = j1.v + j0.v;
    println!("jet op {:?}.v + {:?}.v = {:?}", j1.v, j0.v, v2);
    let j2 = j1 + j0;
    println!("jet op {:?} + {:?} = {:?}", j1, j0, j2);

    let j3 = j1 * j0;
    println!("jet op {:?} * {:?} = {:?}", j1, j0, j3);

    // is it possible to create a CxxVector on the rust side?  There isn't a CxxVector::new()
    // let vals = CxxVector;  // ::<f64>();
    let mut vals = Vec::new();
    vals.push(5.342);
    vals.push(8.0);

    // TODO(lucasw) these are too quick to measure accurately, also the first one to run goes
    // slower
    let start = Instant::now();
    ceres_example.run_auto(&vals);
    println!("elapsed {:?}", start.elapsed());

    let start = Instant::now();
    ceres_example.run_numeric(&vals);
    println!("elapsed {:?}", start.elapsed());
}

fn main() {
    /*
    let ceres_example = ffi::new_ceres_example();

    simple_example(&ceres_example);

    let start = trajectory::WayPoint {
        time: 0.0,
        position: 1.0,
        velocity: -1.0,
        acceleration: 0.5,
        jerk: 0.0,
    };

    let target = trajectory::WayPoint {
        time: 1.0,
        position: 4.0,
        velocity: 0.0,
        acceleration: 0.0,
        jerk: 0.0,
    };

    ceres_example.find_trajectory(&start, &target);
*/

    trajectory::test_generate();
}
