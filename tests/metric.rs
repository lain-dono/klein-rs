use klein::{Line, Plane, Point};

#[test]
fn measure_point_to_point() {
    let p1 = Point::new(1.0, 0.0, 0.0);
    let p2 = Point::new(0.0, 1.0, 0.0);
    let l: Line = p1 & p2;
    // Produce the squared distance between p1 and p2
    assert_eq!(l.squared_norm(), 2.0);
}

#[test]
fn measure_point_to_plane() {
    //    Plane p2
    //    /
    //   / \ line perpendicular to
    //  /   \ p2 through p1
    // 0------x--------->
    //        p1

    // (2, 0, 0)
    let p1 = Point::new(2.0, 0.0, 0.0);
    // Plane x - y = 0
    let p2 = Plane::new(1.0, -1.0, 0.0, 0.0);
    let p2 = p2.normalized();
    // Distance from point p1 to plane p2
    let root_two = f32::sqrt(2.0);
    assert_eq!((p1 & p2).scalar().abs(), root_two);
    assert_eq!((p1 ^ p2).e0123().abs(), root_two);
}

#[test]
fn measure_point_to_line() {
    let l = Line::new(0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    let p = Point::new(0.0, 1.0, 2.0);
    let distance = (l & p).norm();
    assert_eq!(distance, f32::sqrt(2.0));
}
