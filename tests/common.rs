use klein::{point, plane, rotor, motor};

#[test]
fn multivector_sum_points() {
    let p1 = point(1.0, 2.0, 3.0);
    let p2 = point(2.0, 3.0, -1.0);

    let p3 = p1 + p2;
    assert_eq!(p3.x(), 1.0 + 2.0);
    assert_eq!(p3.y(), 2.0 + 3.0);
    assert_eq!(p3.z(), 3.0 + -1.0);

    let p4 = p1 - p2;
    assert_eq!(p4.x(), 1.0 - 2.0);
    assert_eq!(p4.y(), 2.0 - 3.0);
    assert_eq!(p4.z(), 3.0 - -1.0);
}

#[test]
fn multivector_sum_planes() {
    let p = plane(1.0, 3.0, 4.0, -5.0);
    let p_norm = p | p;
    assert_ne!(p_norm, 1.0);

    let p = p.normalized();
    let p_norm = p | p;
    approx::abs_diff_eq!(p_norm, 1.0);
}

#[test]
fn rotor_constrain() {
    let r1 = rotor(1.0, 2.0, 3.0, 4.0);
    let r2 = r1.constrained();
    assert_eq!(r1, r2);

    let r1 = -r1;
    let r2 = r1.constrained();
    assert_eq!(r1, -r2);
}

#[test]
fn motor_constrain() {
    let m1 = motor(
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    );
    let m2 = m1.constrained();
    assert_eq!(m1, m2);

    let m1 = -m1;
    let m2 = m1.constrained();
    assert_eq!(m1, -m2);
}
