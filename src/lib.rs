#![feature(stdarch)]
#![allow(deprecated, non_snake_case, unused_unsafe)]
#![warn(clippy::all)]

#[macro_use]
pub mod arch;

mod exp_log;
mod join;

mod multivector_ep;
mod multivector_gp;
mod multivector_ip;

mod direction; // done f32x4
mod dual; // done scalar
mod line; // done
mod matrix;
mod motor;
mod plane; // done
mod point; // done
mod rotor;
mod translator; // done

mod macros;

pub use self::{
    direction::Direction,
    dual::Dual,
    line::{Branch, IdealLine, Line},
    matrix::{Mat3x4, Mat4x4},
    motor::Motor,
    plane::Plane,
    point::{Origin, Point},
    rotor::Rotor,
    translator::Translator,
};

/*
pub fn direction(x: f32, y: f32, z: f32) -> Direction {
    Direction::new(x, y, z)
}

pub fn dual(p: f32, q: f32) -> Dual {
    Dual::new(p, q)
}

pub fn branch(a: f32, b: f32, c: f32) -> Branch {
    Branch::new(a, b, c)
}

pub fn ideal_line(a: f32, b: f32, c: f32) -> IdealLine {
    IdealLine::new(a, b, c)
}

pub fn line(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32) -> Line {
    Line::new(a, b, c, d, e, f)
}

pub fn motor(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32) -> Motor {
    Motor::new(a, b, c, d, e, f, g, h)
}

pub fn plane(a: f32, b: f32, c: f32, d: f32) -> Plane {
    Plane::new(a, b, c, d)
}

pub fn point(x: f32, y: f32, z: f32) -> Point {
    Point::new(x, y, z)
}

pub fn rotor(a: f32, b: f32, c: f32, d: f32) -> Rotor {
    Rotor::new(a, b, c, d)
}

pub fn translator(delta: f32, x: f32, y: f32, z: f32) -> Translator {
    Translator::new(delta, x, y, z)
}
*/
