use klein::{Dual, IdealLine, Line, Plane, Point};

// plane^plane
#[test]
fn plane_plane() {
    // d*e_0 + a*e_1 + b*e_2 + c*e_3
    let p1 = Plane::new(1.0, 2.0, 3.0, 4.0);
    let p2 = Plane::new(2.0, 3.0, -1.0, -2.0);
    let p12: Line = p1 ^ p2;
    assert_eq!(p12.e01(), 10.0);
    assert_eq!(p12.e02(), 16.0);
    assert_eq!(p12.e03(), 2.0);
    assert_eq!(p12.e12(), -1.0);
    assert_eq!(p12.e31(), 7.0);
    assert_eq!(p12.e23(), -11.0);
}

// plane^line
#[test]
fn plane_line() {
    // d*e_0 + a*e_1 + b*e_2 + c*e_3
    let p1 = Plane::new(1.0, 2.0, 3.0, 4.0);

    // a*e01 + b*e02 + c*e03 + d*e23 + e*e31 + f*e12
    let l1 = Line::new(0.0, 0.0, 1.0, 4.0, 1.0, -2.0);

    let p1l1: Point = p1 ^ l1;
    assert_eq!(p1l1.e021(), 8.0);
    assert_eq!(p1l1.e013(), -5.0);
    assert_eq!(p1l1.e032(), -14.0);
    assert_eq!(p1l1.e123(), 0.0);
}

// plane^ideal-line
#[test]
fn plane_ideal_line() {
    // d*e_0 + a*e_1 + b*e_2 + c*e_3
    let p1 = Plane::new(1.0, 2.0, 3.0, 4.0);

    // a*e01 + b*e02 + c*e03
    let l1 = IdealLine::new(-2.0, 1.0, 4.0);

    let p1l1: Point = p1 ^ l1;
    assert_eq!(p1l1.e021(), 5.0);
    assert_eq!(p1l1.e013(), -10.0);
    assert_eq!(p1l1.e032(), 5.0);
    assert_eq!(p1l1.e123(), 0.0);
}

// plane^point
#[test]
fn plane_point() {
    // d*e_0 + a*e_1 + b*e_2 + c*e_3
    let p1 = Plane::new(1.0, 2.0, 3.0, 4.0);
    // x*e_032 + y*e_013 + z*e_021 + e_123
    let p2 = Point::new(-2.0, 1.0, 4.0);

    let p1p2: Dual = p1 ^ p2;
    assert_eq!(p1p2.scalar(), 0.0);
    assert_eq!(p1p2.e0123(), 16.0);
}

// line^plane
#[test]
fn line_plane() {
    // d*e_0 + a*e_1 + b*e_2 + c*e_3
    let p1 = Plane::new(1.0, 2.0, 3.0, 4.0);

    // a*e01 + b*e01 + c*e02 + d*e23 + e*e31 + f*e12
    let l1 = Line::new(0.0, 0.0, 1.0, 4.0, 1.0, -2.0);

    let p1l1: Point = l1 ^ p1;
    assert_eq!(p1l1.e021(), 8.0);
    assert_eq!(p1l1.e013(), -5.0);
    assert_eq!(p1l1.e032(), -14.0);
    assert_eq!(p1l1.e123(), 0.0);
}

// line^line
#[test]
fn line_line() {
    // a*e01 + b*e01 + c*e02 + d*e23 + e*e31 + f*e12
    let l1 = Line::new(1.0, 0.0, 0.0, 3.0, 2.0, 1.0);
    let l2 = Line::new(0.0, 1.0, 0.0, 4.0, 1.0, -2.0);

    let l1l2: Dual = l1 ^ l2;
    assert_eq!(l1l2.e0123(), 6.0);
}

// line^ideal-line
#[test]
fn line_ideal_line() {
    // a*e01 + b*e01 + c*e02 + d*e23 + e*e31 + f*e12
    let l1 = Line::new(0.0, 0.0, 1.0, 3.0, 2.0, 1.0);
    // a*e01 + b*e02 + c*e03
    let l2 = IdealLine::new(-2.0, 1.0, 4.0);

    let l1l2: Dual = l1 ^ l2;
    assert_eq!(l1l2.e0123(), 0.0);
    assert_eq!(l1l2.scalar(), 0.0);
}

// ideal-line^plane
#[test]
fn ideal_line_plane() {
    // d*e_0 + a*e_1 + b*e_2 + c*e_3
    let p1 = Plane::new(1.0, 2.0, 3.0, 4.0);

    // a*e01 + b*e02 + c*e03
    let l1 = IdealLine::new(-2.0, 1.0, 4.0);

    let p1l1: Point = l1 ^ p1;
    assert_eq!(p1l1.e021(), 5.0);
    assert_eq!(p1l1.e013(), -10.0);
    assert_eq!(p1l1.e032(), 5.0);
    assert_eq!(p1l1.e123(), 0.0);
}

// point^plane
#[test]
fn point_plane() {
    // x*e_032 + y*e_013 + z*e_021 + e_123
    let p1 = Point::new(-2.0, 1.0, 4.0);
    // d*e_0 + a*e_1 + b*e_2 + c*e_3
    let p2 = Plane::new(1.0, 2.0, 3.0, 4.0);

    let p1p2: Dual = p1 ^ p2;
    assert_eq!(p1p2.e0123(), -16.0);
}
