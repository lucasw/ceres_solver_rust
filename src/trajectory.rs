#[derive(Clone, Copy, Debug)]
pub struct WayPoint {
    pub time: f64,
    pub position: f64,
    pub velocity: f64,
    pub acceleration: f64,
    pub jerk: f64,
}

impl WayPoint {
    fn new(time: f64,
           position: f64,
           velocity: f64,
           acceleration: f64,
           jerk: f64,
    ) -> WayPoint {
        WayPoint {
            time,
            position,
            velocity,
            acceleration,
            jerk,
        }
    }
    fn csv_print(&self) {
        println!("{:.6}, {:+.6}, {:+.6}, {:+.6}, {:+.6}",
                 self.time, self.position, self.velocity, self.acceleration, self.jerk);
    }
}

pub fn generate(
    start: WayPoint,
    end_time: f64,
    jerks: &[f64],
    waypoints: &mut [WayPoint],
) {
    let steps = waypoints.len() - 1;
    let tdiff = end_time - start.time;
    let step_interval = tdiff / (steps as f64);
    // println!("# {} {} {}", tdiff, steps, step_interval);

    waypoints[0] = start;

    // TODO(lucasw) change to use itertools
    for i in 0..waypoints.len() - 1 {
        let prev = waypoints[i];
        if i == 0 {
            println!("time, position, velocity, acceleration, jerk");
            prev.csv_print();
        }
        let mut cur = prev.clone();

        if jerks.len() > 0 {
            let jerk_ind = i * jerks.len() / waypoints.len();
            cur.jerk = jerks[jerk_ind];
        } else {
            cur.jerk = start.jerk;
        }
        cur.acceleration += cur.jerk * step_interval;
        cur.velocity += prev.acceleration * step_interval;
        cur.position += prev.velocity * step_interval;
        cur.time += step_interval;
        waypoints[i + 1] = cur;
        cur.csv_print();
    }
}

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
) {

}

pub fn test_generate() {
    let start = WayPoint {
        time: 0.0,
        position: 1.0,
        velocity: -1.0,
        acceleration: 0.5,
        jerk: 0.0,
    };

    // make num waypoints +1 to account for starting waypoint
    let mut waypoints = vec![start; 41];
    let jerks = vec![1.0, 4.5, 2.0, 2.0, -7.8];
    generate(start, start.time + 1.0, &jerks, &mut waypoints);
}
