use klein::{IdealLine, Line, Plane, Point};

// plane|plane
#[test]
fn plane_plane() {
    // d*e_0 + a*e_1 + b*e_2 + c*e_3
    let p1 = Plane::new(1.0, 2.0, 3.0, 4.0);
    let p2 = Plane::new(2.0, 3.0, -1.0, -2.0);
    let p12 = p1 | p2;
    assert_eq!(p12, 5.0);
}

// plane|line
#[test]
fn plane_line() {
    // d*e_0 + a*e_1 + b*e_2 + c*e_3
    let p1 = Plane::new(1.0, 2.0, 3.0, 4.0);

    // a*e01 + b*e01 + c*e02 + d*e23 + e*e31 + f*e12
    let l1 = Line::new(0.0, 0.0, 1.0, 4.0, 1.0, -2.0);

    let p1l1: Plane = p1 | l1;
    assert_eq!(p1l1.e0(), -3.0);
    assert_eq!(p1l1.e1(), 7.0);
    assert_eq!(p1l1.e2(), -14.0);
    assert_eq!(p1l1.e3(), 7.0);
}

// plane|ideal-line
#[test]
fn plane_ideal_line() {
    // d*e_0 + a*e_1 + b*e_2 + c*e_3
    let p1 = Plane::new(1.0, 2.0, 3.0, 4.0);

    // a*e01 + b*e02 + c*e03
    let l1 = IdealLine::new(-2.0, 1.0, 4.0);

    let p1l1: Plane = p1 | l1;
    assert_eq!(p1l1.e0(), -12.0);
}

// plane|point
#[test]
fn plane_point() {
    // d*e_0 + a*e_1 + b*e_2 + c*e_3
    let p1 = Plane::new(1.0, 2.0, 3.0, 4.0);
    // x*e_032 + y*e_013 + z*e_021 + e_123
    let p2 = Point::new(-2.0, 1.0, 4.0);

    let p1p2: Line = p1 | p2;
    assert_eq!(p1p2.e01(), -5.0);
    assert_eq!(p1p2.e02(), 10.0);
    assert_eq!(p1p2.e03(), -5.0);
    assert_eq!(p1p2.e12(), 3.0);
    assert_eq!(p1p2.e31(), 2.0);
    assert_eq!(p1p2.e23(), 1.0);
}

// line|plane
#[test]
fn line_plane() {
    // d*e_0 + a*e_1 + b*e_2 + c*e_3
    let p1 = Plane::new(1.0, 2.0, 3.0, 4.0);

    // a*e01 + b*e01 + c*e02 + d*e23 + e*e31 + f*e12
    let l1 = Line::new(0.0, 0.0, 1.0, 4.0, 1.0, -2.0);

    let p1l1: Plane = l1 | p1;
    assert_eq!(p1l1.e0(), 3.0);
    assert_eq!(p1l1.e1(), -7.0);
    assert_eq!(p1l1.e2(), 14.0);
    assert_eq!(p1l1.e3(), -7.0);
}

// line|line
#[test]
fn line_line() {
    // a*e01 + b*e01 + c*e02 + d*e23 + e*e31 + f*e12
    let l1 = Line::new(1.0, 0.0, 0.0, 3.0, 2.0, 1.0);
    let l2 = Line::new(0.0, 1.0, 0.0, 4.0, 1.0, -2.0);

    let l1l2: f32 = l1 | l2;
    assert_eq!(l1l2, -12.0);
}

// line|point
#[test]
fn line_point() {
    // a*e01 + b*e01 + c*e02 + d*e23 + e*e31 + f*e12
    let l1 = Line::new(0.0, 0.0, 1.0, 3.0, 2.0, 1.0);
    // x*e_032 + y*e_013 + z*e_021 + e_123
    let p2 = Point::new(-2.0, 1.0, 4.0);

    let l1p2: Plane = l1 | p2;
    assert_eq!(l1p2.e0(), 0.0);
    assert_eq!(l1p2.e1(), -3.0);
    assert_eq!(l1p2.e2(), -2.0);
    assert_eq!(l1p2.e3(), -1.0);
}

// point|plane
#[test]
fn point_plane() {
    // x*e_032 + y*e_013 + z*e_021 + e_123
    let p1 = Point::new(-2.0, 1.0, 4.0);
    // d*e_0 + a*e_1 + b*e_2 + c*e_3
    let p2 = Plane::new(1.0, 2.0, 3.0, 4.0);

    let p1p2: Line = p1 | p2;
    assert_eq!(p1p2.e01(), -5.0);
    assert_eq!(p1p2.e02(), 10.0);
    assert_eq!(p1p2.e03(), -5.0);
    assert_eq!(p1p2.e12(), 3.0);
    assert_eq!(p1p2.e31(), 2.0);
    assert_eq!(p1p2.e23(), 1.0);
}

// point|line
#[test]
fn point_line() {
    // a*e01 + b*e01 + c*e02 + d*e23 + e*e31 + f*e12
    let l1 = Line::new(0.0, 0.0, 1.0, 3.0, 2.0, 1.0);
    // x*e_032 + y*e_013 + z*e_021 + e_123
    let p2 = Point::new(-2.0, 1.0, 4.0);

    let l1p2: Plane = p2 | l1;
    assert_eq!(l1p2.e0(), 0.0);
    assert_eq!(l1p2.e1(), -3.0);
    assert_eq!(l1p2.e2(), -2.0);
    assert_eq!(l1p2.e3(), -1.0);
}

// point|point
#[test]
fn point_point() {
    // x*e_032 + y*e_013 + z*e_021 + e_123
    let p1 = Point::new(1.0, 2.0, 3.0);
    let p2 = Point::new(-2.0, 1.0, 4.0);

    let p1p2 = p1 | p2;
    assert_eq!(p1p2, -1.0);
}

// project point to line
#[test]
fn project_point_to_line() {
    let p1 = Point::new(2.0, 2.0, 0.0);
    let p2 = Point::new(0.0, 0.0, 0.0);
    let p3 = Point::new(1.0, 0.0, 0.0);
    let l: Line = p2 & p3;
    let mut p4: Point = (l | p1) ^ l;
    p4.normalize();

    approx::abs_diff_eq!(p4.e123(), 1.0);
    approx::abs_diff_eq!(p4.x(), 2.0);
    approx::abs_diff_eq!(p4.y(), 0.0);
    approx::abs_diff_eq!(p4.z(), 0.0);
}
