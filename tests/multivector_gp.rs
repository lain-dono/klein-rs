use approx::abs_diff_eq;
use klein::{Branch, Line, Motor, Plane, Point, Rotor, Translator};

#[test]
fn plane_plane() {
    // d*e_0 + a*e_1 + b*e_2 + c*e_3
    let p1 = Plane::new(1.0, 2.0, 3.0, 4.0);
    let p2 = Plane::new(2.0, 3.0, -1.0, -2.0);
    let p12: Motor = p1 * p2;
    assert_eq!(p12.scalar(), 5.0);
    assert_eq!(p12.e12(), -1.0);
    assert_eq!(p12.e31(), 7.0);
    assert_eq!(p12.e23(), -11.0);
    assert_eq!(p12.e01(), 10.0);
    assert_eq!(p12.e02(), 16.0);
    assert_eq!(p12.e03(), 2.0);
    assert_eq!(p12.e0123(), 0.0);

    let p3: Plane = (p1 * p2).sqrt().conj_plane(p2);
    assert!(p3.approx_eq(p1, 0.001));

    let p1 = p1.normalized();

    let m: Motor = p1 * p1;
    abs_diff_eq!(m.scalar(), 1.0);

    let p1 = Plane::new(1.0, 2.0, 3.0, 4.0);
    let m: Motor = p1 / p1;
    abs_diff_eq!(m.scalar(), 1.0);
    assert_eq!(m.e12(), 0.0);
    assert_eq!(m.e31(), 0.0);
    assert_eq!(m.e23(), 0.0);
    assert_eq!(m.e01(), 0.0);
    assert_eq!(m.e02(), 0.0);
    assert_eq!(m.e03(), 0.0);
    assert_eq!(m.e0123(), 0.0);
}

// plane*point
#[test]
fn plane_mul_point() {
    // d*e_0 + a*e_1 + b*e_2 + c*e_3
    let p1 = Plane::new(1.0, 2.0, 3.0, 4.0);
    // x*e_032 + y*e_013 + z*e_021 + e_123
    let p2 = Point::new(-2.0, 1.0, 4.0);

    let p1p2: Motor = p1 * p2;
    assert_eq!(p1p2.scalar(), 0.0);
    assert_eq!(p1p2.e01(), -5.0);
    assert_eq!(p1p2.e02(), 10.0);
    assert_eq!(p1p2.e03(), -5.0);
    assert_eq!(p1p2.e12(), 3.0);
    assert_eq!(p1p2.e31(), 2.0);
    assert_eq!(p1p2.e23(), 1.0);
    assert_eq!(p1p2.e0123(), 16.0);
}

// line-normalization
#[test]
fn line_normalization() {
    let l = Line::new(1.0, 2.0, 3.0, 3.0, 2.0, 1.0);
    let l = l.normalized();
    let m: Motor = l * l.reversed();
    abs_diff_eq!(m.scalar(), 1.0);
    abs_diff_eq!(m.e23(), 0.0);
    abs_diff_eq!(m.e31(), 0.0);
    abs_diff_eq!(m.e12(), 0.0);
    abs_diff_eq!(m.e01(), 0.0);
    abs_diff_eq!(m.e02(), 0.0);
    abs_diff_eq!(m.e03(), 0.0);
    abs_diff_eq!(m.e0123(), 0.0);
}

#[test]
fn branch_branch() {
    let b1 = Branch::new(2.0, 1.0, 3.0);
    let b2 = Branch::new(1.0, -2.0, -3.0);
    let r: Rotor = b2 * b1;
    assert_eq!(r.scalar(), 9.0);
    assert_eq!(r.e23(), 3.0);
    assert_eq!(r.e13(), 9.0);
    assert_eq!(r.e12(), -5.0);

    let b1 = b1.normalized();
    let b2 = b2.normalized();
    let b3 = (b2 * b1).sqrt().reversed().conj_branch(b1);
    abs_diff_eq!(b3.x(), b2.x());
    abs_diff_eq!(b3.y(), b2.y());
    abs_diff_eq!(b3.z(), b2.z());

    let b = Branch::new(2.0, 1.0, 3.0);
    let r: Rotor = b / b;
    abs_diff_eq!(r.scalar(), 1.0);
    assert_eq!(r.e23(), 0.0);
    assert_eq!(r.e13(), 0.0);
    assert_eq!(r.e12(), 0.0);
}

// line*line
#[test]
fn line_mul_line() {
    // a*e01 + b*e02 + c*e03 + d*e23 + e*e31 + f*e12
    let l1 = Line::new(1.0, 0.0, 0.0, 3.0, 2.0, 1.0);
    let l2 = Line::new(0.0, 1.0, 0.0, 4.0, 1.0, -2.0);

    let l1l2: Motor = l1 * l2;
    assert_eq!(l1l2.scalar(), -12.0);
    assert_eq!(l1l2.e12(), 5.0);
    assert_eq!(l1l2.e31(), -10.0);
    assert_eq!(l1l2.e23(), 5.0);
    assert_eq!(l1l2.e01(), 1.0);
    assert_eq!(l1l2.e02(), -2.0);
    assert_eq!(l1l2.e03(), -4.0);
    assert_eq!(l1l2.e0123(), 6.0);

    let l1 = l1.normalized();
    let l2 = l2.normalized();
    let l3 = (l1 * l2).sqrt().conj_line(l2);
    assert!(l3.approx_eq(-l1, 0.001));
}

// line/line
#[test]
fn line_div_line() {
    let l = Line::new(1.0, -2.0, 2.0, -3.0, 3.0, -4.0);
    let m: Motor = l / l;
    abs_diff_eq!(m.scalar(), 1.0);
    assert_eq!(m.e12(), 0.0);
    assert_eq!(m.e31(), 0.0);
    assert_eq!(m.e23(), 0.0);
    abs_diff_eq!(m.e01(), 0.0);
    abs_diff_eq!(m.e02(), 0.0);
    abs_diff_eq!(m.e03(), 0.0);
    abs_diff_eq!(m.e0123(), 0.0);
}

// point*plane
#[test]
fn point_mul_plane() {
    // x*e_032 + y*e_013 + z*e_021 + e_123
    let p1 = Point::new(-2.0, 1.0, 4.0);
    // d*e_0 + a*e_1 + b*e_2 + c*e_3
    let p2 = Plane::new(1.0, 2.0, 3.0, 4.0);

    let p1p2: Motor = p1 * p2;
    assert_eq!(p1p2.scalar(), 0.0);
    assert_eq!(p1p2.e01(), -5.0);
    assert_eq!(p1p2.e02(), 10.0);
    assert_eq!(p1p2.e03(), -5.0);
    assert_eq!(p1p2.e12(), 3.0);
    assert_eq!(p1p2.e31(), 2.0);
    assert_eq!(p1p2.e23(), 1.0);
    assert_eq!(p1p2.e0123(), -16.0);
}

// point*point point/point
#[test]
fn point_point() {
    // x*e_032 + y*e_013 + z*e_021 + e_123
    let p1 = Point::new(1.0, 2.0, 3.0);
    let p2 = Point::new(-2.0, 1.0, 4.0);

    let p1p2: Translator = p1 * p2;
    abs_diff_eq!(p1p2.e01(), -3.0);
    abs_diff_eq!(p1p2.e02(), -1.0);
    abs_diff_eq!(p1p2.e03(), 1.0);

    let p3: Point = (p1p2).sqrt().conj_point(p2);
    abs_diff_eq!(p3.x(), 1.0);
    abs_diff_eq!(p3.y(), 2.0);
    abs_diff_eq!(p3.z(), 3.0);

    let p1 = Point::new(1.0, 2.0, 3.0);
    let t: Translator = p1 / p1;
    assert_eq!(t.e01(), 0.0);
    assert_eq!(t.e02(), 0.0);
    assert_eq!(t.e03(), 0.0);
}

// translator/translator
#[test]
fn translator_div_translator() {
    let t1 = Translator::new(3.0, 1.0, -2.0, 3.0);
    let t2: Translator = t1 / t1;
    assert_eq!(t2.e01(), 0.0);
    assert_eq!(t2.e02(), 0.0);
    assert_eq!(t2.e03(), 0.0);
}

// rotor*translator
#[test]
fn rotor_mul_translator() {
    let r = Rotor::raw(1.0, 0.0, 0.0, 1.0);
    let t = Translator::raw(1.0, 0.0, 0.0, 0.0);
    let m: Motor = r * t;
    assert_eq!(m.scalar(), 1.0);
    assert_eq!(m.e01(), 0.0);
    assert_eq!(m.e02(), 0.0);
    assert_eq!(m.e03(), 1.0);
    assert_eq!(m.e23(), 0.0);
    assert_eq!(m.e31(), 0.0);
    assert_eq!(m.e12(), 1.0);
    assert_eq!(m.e0123(), 1.0);
}

// translator*rotor
#[test]
fn translator_mul_rotor() {
    let r = Rotor::raw(1.0, 0.0, 0.0, 1.0);
    let t = Translator::raw(1.0, 0.0, 0.0, 0.0);
    let m: Motor = t * r;
    assert_eq!(m.scalar(), 1.0);
    assert_eq!(m.e01(), 0.0);
    assert_eq!(m.e02(), 0.0);
    assert_eq!(m.e03(), 1.0);
    assert_eq!(m.e23(), 0.0);
    assert_eq!(m.e31(), 0.0);
    assert_eq!(m.e12(), 1.0);
    assert_eq!(m.e0123(), 1.0);
}

// motor*rotor
#[test]
fn motor_mul_rotor() {
    let r1 = Rotor::raw(1.0, 2.0, 3.0, 4.0);
    let t = Translator::raw(3.0, -2.0, 1.0, -3.0);
    let r2 = Rotor::raw(-4.0, 2.0, -3.0, 1.0);
    let m1: Motor = (t * r1) * r2;
    let m2: Motor = t * (r1 * r2);
    assert_eq!(m1, m2);
}

// rotor*motor
#[test]
fn rotor_mul_rotor() {
    let r1 = Rotor::raw(1.0, 2.0, 3.0, 4.0);
    let t = Translator::raw(3.0, -2.0, 1.0, -3.0);
    let r2 = Rotor::raw(-4.0, 2.0, -3.0, 1.0);
    let m1: Motor = r2 * (r1 * t);
    let m2: Motor = (r2 * r1) * t;
    assert_eq!(m1, m2);
}

// motor*translator
#[test]
fn motor_mul_translator() {
    let r = Rotor::raw(1.0, 2.0, 3.0, 4.0);
    let t1 = Translator::raw(3.0, -2.0, 1.0, -3.0);
    let t2 = Translator::raw(-4.0, 2.0, -3.0, 1.0);
    let m1: Motor = (r * t1) * t2;
    let m2: Motor = r * (t1 * t2);
    assert_eq!(m1, m2);
}

// translator*motor
#[test]
fn translator_mul_motor() {
    let r = Rotor::raw(1.0, 2.0, 3.0, 4.0);
    let t1 = Translator::raw(3.0, -2.0, 1.0, -3.0);
    let t2 = Translator::raw(-4.0, 2.0, -3.0, 1.0);
    let m1: Motor = t2 * (r * t1);
    let m2: Motor = (t2 * r) * t1;
    assert_eq!(m1, m2);
}

#[test]
fn motor_motor() {
    let m1 = Motor::new(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    let m2 = Motor::new(6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0);

    let mul: Motor = m1 * m2;
    assert_eq!(mul.scalar(), -86.0);
    assert_eq!(mul.e23(), 36.0);
    assert_eq!(mul.e31(), 32.0);
    assert_eq!(mul.e12(), 52.0);
    assert_eq!(mul.e01(), -38.0);
    assert_eq!(mul.e02(), -76.0);
    assert_eq!(mul.e03(), -66.0);
    assert_eq!(mul.e0123(), 384.0);

    let div: Motor = m1 / m1;
    abs_diff_eq!(div.scalar(), 1.0);
    assert_eq!(div.e23(), 0.0);
    assert_eq!(div.e31(), 0.0);
    assert_eq!(div.e12(), 0.0);
    assert_eq!(div.e01(), 0.0);
    abs_diff_eq!(div.e02(), 0.0);
    abs_diff_eq!(div.e03(), 0.0);
    abs_diff_eq!(div.e0123(), 0.0);
}
