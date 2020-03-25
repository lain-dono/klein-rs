use approx::abs_diff_eq;
use core::f32::consts::FRAC_PI_2;
use klein::{Branch, Line, Motor, Rotor, Translator};

#[test]
fn rotor_exp_log() {
    let r = Rotor::new(FRAC_PI_2, 0.3, -3.0, 1.0);
    let b: Branch = r.log();
    let r2: Rotor = b.exp();

    abs_diff_eq!(r2.scalar(), r.scalar());
    abs_diff_eq!(r2.e12(), r.e12());
    abs_diff_eq!(r2.e13(), r.e13());
    abs_diff_eq!(r2.e23(), r.e23());
}

#[test]
fn rotor_sqrt() {
    let r1 = Rotor::new(FRAC_PI_2, 0.3, -3.0, 1.0);
    let r2: Rotor = r1.sqrt();
    let r3 = r2 * r2;
    abs_diff_eq!(r1.scalar(), r3.scalar());
    abs_diff_eq!(r1.e12(), r3.e12());
    abs_diff_eq!(r1.e13(), r3.e13());
    abs_diff_eq!(r1.e23(), r3.e23());
}

#[test]
fn motor_exp_log_sqrt() {
    // Construct a motor from a translator and rotor
    let r = Rotor::new(FRAC_PI_2, 0.3, -3.0, 1.0);
    let t = Translator::new(12.0, -2.0, 0.4, 1.0);
    let m1: Motor = r * t;
    let l: Line = m1.log();
    let m2: Motor = l.exp();

    abs_diff_eq!(m1.scalar(), m2.scalar());
    abs_diff_eq!(m1.e12(), m2.e12());
    abs_diff_eq!(m1.e31(), m2.e31());
    abs_diff_eq!(m1.e23(), m2.e23());
    abs_diff_eq!(m1.e01(), m2.e01());
    abs_diff_eq!(m1.e02(), m2.e02());
    abs_diff_eq!(m1.e03(), m2.e03());
    abs_diff_eq!(m1.e0123(), m2.e0123());

    let m3 = m1.sqrt() * m1.sqrt();
    abs_diff_eq!(m1.scalar(), m3.scalar());
    abs_diff_eq!(m1.e12(), m3.e12());
    abs_diff_eq!(m1.e31(), -m3.e31());
    abs_diff_eq!(m1.e23(), m3.e23());
    abs_diff_eq!(m1.e01(), m3.e01());
    abs_diff_eq!(m1.e02(), m3.e02());
    abs_diff_eq!(m1.e03(), m3.e03());
    abs_diff_eq!(m1.e0123(), m3.e0123());
}

#[test]
fn motor_slerp() {
    // Construct a motor from a translator and rotor
    let r = Rotor::new(FRAC_PI_2, 0.3, -3.0, 1.0);
    let t = Translator::new(12.0, -2.0, 0.4, 1.0);
    let m1: Motor = r * t;
    let l: Line = m1.log();
    // Divide the motor action into three equal steps
    let step: Line = l / 3.0;
    let step: Motor = step.exp();
    let m2: Motor = step * step * step;
    abs_diff_eq!(m1.scalar(), m2.scalar());
    abs_diff_eq!(m1.e12(), m2.e12());
    abs_diff_eq!(m1.e31(), m2.e31());
    abs_diff_eq!(m1.e23(), m2.e23());
    abs_diff_eq!(m1.e01(), m2.e01());
    abs_diff_eq!(m1.e02(), m2.e02());
    abs_diff_eq!(m1.e03(), m2.e03());
    abs_diff_eq!(m1.e0123(), m2.e0123());
}

#[test]
fn motor_blend() {
    let r1 = Rotor::new(FRAC_PI_2, 0.0, 0.0, 1.0);
    let t1 = Translator::new(1.0, 0.0, 0.0, 1.0);
    let m1: Motor = r1 * t1;

    let r2 = Rotor::new(FRAC_PI_2, 0.3, -3.0, 1.0);
    let t2 = Translator::new(12.0, -2.0, 0.4, 1.0);
    let m2: Motor = r2 * t2;

    let motion: Motor = m2 * m1.reversed(); // ~
    let step: Line = motion.log() / 4.0;
    let step: Motor = step.exp();

    // Applying motor_step 0 times to m1 is m1.
    // Applying motor_step 4 times to m1 is m2 * ~m1;
    let result: Motor = step * step * step * step * m1;
    abs_diff_eq!(result.scalar(), m2.scalar());
    abs_diff_eq!(result.e12(), m2.e12());
    abs_diff_eq!(result.e31(), m2.e31());
    abs_diff_eq!(result.e23(), m2.e23());
    abs_diff_eq!(result.e01(), m2.e01());
    abs_diff_eq!(result.e02(), m2.e02());
    abs_diff_eq!(result.e03(), m2.e03());
    abs_diff_eq!(result.e0123(), m2.e0123());
}
