use klein::{Line, Plane, Point};

// +z line
#[test]
fn positive_z_line() {
    let p1 = Point::new(0.0, 0.0, 0.0);
    let p2 = Point::new(0.0, 0.0, 1.0);
    let p12: Line = p1 & p2;
    assert_eq!(p12.e12(), 1.0);
}

// +y line
#[test]
fn positive_y_line() {
    let p1 = Point::new(0.0, -1.0, 0.0);
    let p2 = Point::new(0.0, 0.0, 0.0);
    let p12: Line = p1 & p2;
    assert_eq!(p12.e31(), 1.0);
}

// +x line
#[test]
fn positive_x_line() {
    let p1 = Point::new(-2.0, 0.0, 0.0);
    let p2 = Point::new(-1.0, 0.0, 0.0);
    let p12: Line = p1 & p2;
    assert_eq!(p12.e23(), 1.0);
}

// plane-construction
#[test]
fn plane_construction() {
    let p1 = Point::new(1.0, 3.0, 2.0);
    let p2 = Point::new(-1.0, 5.0, 2.0);
    let p3 = Point::new(2.0, -1.0, -4.0);

    let p123: Plane = p1 & p2 & p3;

    // Check that all 3 points lie on the plane
    let p1 = p123.e1() + p123.e2() * 3.0 + p123.e3() * 2.0 + p123.e0();
    let p2 = -p123.e1() + p123.e2() * 5.0 + p123.e3() * 2.0 + p123.e0();
    let p3 = p123.e1() * 2.0 - p123.e2() - p123.e3() * 4.0 + p123.e0();
    assert_eq!(p1, 0.0);
    assert_eq!(p2, 0.0);
    assert_eq!(p3, 0.0);
}
