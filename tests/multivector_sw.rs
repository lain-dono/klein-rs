use approx::abs_diff_eq;
use klein::{
    arch::{f32x4, sw02},
    Line, Motor, Plane, Point, Rotor, Translator,
};
use std::f32::consts::FRAC_PI_2;

#[test]
fn simd_sandwich() {
    let a = f32x4::new(4.0, 3.0, 2.0, 1.0);
    let b = f32x4::new(-1.0, -2.0, -3.0, -4.0);

    let ab = sw02(a, b).into_array();

    assert_eq!(ab[0], 9.0);
    assert_eq!(ab[1], 2.0);
    assert_eq!(ab[2], 3.0);
    assert_eq!(ab[3], 4.0);
}

#[test]
fn reflect_plane() {
    let p1 = Plane::new(3.0, 2.0, 1.0, -1.0);
    let p2 = Plane::new(1.0, 2.0, -1.0, -3.0);
    let p3: Plane = p1.reflect_plane(p2);

    assert_eq!(p3.e0(), 30.0);
    assert_eq!(p3.e1(), 22.0);
    assert_eq!(p3.e2(), -4.0);
    assert_eq!(p3.e3(), 26.0);
}

#[test]
fn reflect_line() {
    let p = Plane::new(3.0, 2.0, 1.0, -1.0);
    // a*e01 + b*e01 + c*e02 + d*e23 + e*e31 + f*e12
    let l1 = Line::new(1.0, -2.0, 3.0, 6.0, 5.0, -4.0);
    let l2 = p.reflect_line(l1);
    assert_eq!(l2.e01(), 28.0);
    assert_eq!(l2.e02(), -72.0);
    assert_eq!(l2.e03(), 32.0);
    assert_eq!(l2.e12(), 104.0);
    assert_eq!(l2.e31(), 26.0);
    assert_eq!(l2.e23(), 60.0);
}

#[test]
fn reflect_point() {
    let p1 = Plane::new(3.0, 2.0, 1.0, -1.0);
    let p2 = Point::new(4.0, -2.0, -1.0);
    let p3 = p1.reflect_point(p2);
    assert_eq!(p3.e021(), -26.0);
    assert_eq!(p3.e013(), -52.0);
    assert_eq!(p3.e032(), 20.0);
    assert_eq!(p3.e123(), 14.0);
}

#[test]
//#[ignore]
fn rotor_line() {
    // Make an unnormalized rotor to verify correctness
    let p1 = f32x4::from_array([1.0, 4.0, -3.0, 2.0]);
    let r = Rotor::from(p1);
    // a*e01 + b*e01 + c*e02 + d*e23 + e*e31 + f*e12
    let l1 = Line::new(-1.0, 2.0, -3.0, -6.0, 5.0, 4.0);
    let l2: Line = r.conj_line(l1);
    assert_eq!(l2.e01(), -110.0);
    assert_eq!(l2.e02(), 20.0);
    assert_eq!(l2.e03(), 10.0);
    assert_eq!(l2.e12(), -240.0);
    assert_eq!(l2.e31(), 102.0);
    assert_eq!(l2.e23(), -36.0);
}

#[test]
fn rotor_point() {
    let r = Rotor::new(FRAC_PI_2, 0.0, 0.0, 1.0);
    let p1 = Point::new(1.0, 0.0, 0.0);
    let p2: Point = r.conj_point(p1);
    assert_eq!(p2.x(), 0.0);
    abs_diff_eq!(p2.y(), 1.0);
    assert_eq!(p2.z(), 0.0);
}

#[test]
fn translator_point() {
    let t = Translator::new(1.0, 0.0, 0.0, 1.0);
    let p1 = Point::new(1.0, 0.0, 0.0);
    let p2: Point = t.conj_point(p1);
    assert_eq!(p2.x(), 1.0);
    assert_eq!(p2.y(), 0.0);
    assert_eq!(p2.z(), 1.0);
}

#[test]
fn translator_line() {
    let p2 = f32x4::from_array([0.0, -5.0, -2.0, 2.0]);
    let t = Translator::from(p2);

    // a*e01 + b*e01 + c*e02 + d*e23 + e*e31 + f*e12
    let l1 = Line::new(-1.0, 2.0, -3.0, -6.0, 5.0, 4.0);
    let l2: Line = t.conj_line(l1);
    assert_eq!(l2.e01(), 35.0);
    assert_eq!(l2.e02(), -14.0);
    assert_eq!(l2.e03(), 71.0);
    assert_eq!(l2.e12(), 4.0);
    assert_eq!(l2.e31(), 5.0);
    assert_eq!(l2.e23(), -6.0);
}

#[test]
fn construct_motor() {
    let  r = Rotor::new (FRAC_PI_2, 0.0, 0.0, 1.0);
    let t = Translator::new (1.0, 0.0, 0.0, 1.0);
    let m: Motor  = r * t;
    let p1 = Point::new (1.0, 0.0, 0.0);
    let p2 = m.conj_point(p1);
    assert_eq!(p2.x(), 0.0);
    abs_diff_eq!(p2.y(), 1.0);
    abs_diff_eq!(p2.z(), 1.0);

    // Rotation and translation about the same axis commutes
    let m  = t * r;
    let p2 = m.conj_point(p1);
    assert_eq!(p2.x(), 0.0);
    abs_diff_eq!(p2.y(), 1.0);
    abs_diff_eq!(p2.z(), 1.0);

    let l: Line  = m.log();
    assert_eq!(l.e23(), 0.0);
    abs_diff_eq!(l.e12(), -0.7854, epsilon = 0.001);
    assert_eq!(l.e31(), 0.0);
    assert_eq!(l.e01(), 0.0);
    assert_eq!(l.e02(), 0.0);
    abs_diff_eq!(l.e03(), -0.5);
}

#[test]
fn construct_motor_via_screw_axis() {
    let line = Line::new(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let m = Motor::from_line(FRAC_PI_2, 1.0, line);
    let p1 = Point::new(1.0, 0.0, 0.0);
    let p2 = m.conj_point(p1);
    abs_diff_eq!(p2.x(), 0.0);
    abs_diff_eq!(p2.y(), 1.0);
    abs_diff_eq!(p2.z(), 1.0);
}

#[test]
fn motor_plane() {
    let m = Motor::new(1.0, 4.0, 3.0, 2.0, 5.0, 6.0, 7.0, 8.0);
    let p1 = Plane::new(3.0, 2.0, 1.0, -1.0);
    let p2 = m.conj_plane(p1);
    assert_eq!(p2.x(), 78.0);
    assert_eq!(p2.y(), 60.0);
    assert_eq!(p2.z(), 54.0);
    assert_eq!(p2.d(), 358.0);
}

#[test]
#[ignore]
fn motor_plane_variadic() {
    /*
    motor m{1.0, 4.0, 3.0, 2.0, 5.0, 6.0, 7.0, 8.0};
    plane ps[2] = {{3.0, 2.0, 1.0, -1.0}, {3.0, 2.0, 1.0, -1.0}};
    plane ps2[2];
    m(ps, ps2, 2);

    for (size_t i = 0; i != 2; ++i)
    {
        assert_eq!(ps2[i].x(), 78.0);
        assert_eq!(ps2[i].y(), 60.0);
        assert_eq!(ps2[i].z(), 54.0);
        assert_eq!(ps2[i].d(), 358.0);
    }
    */
}

#[test]
fn motor_point() {
    let m = Motor::new(1.0, 4.0, 3.0, 2.0, 5.0, 6.0, 7.0, 8.0);
    let p1 = Point::new(-1.0, 1.0, 2.0);
    let p2 = m.conj_point(p1);
    assert_eq!(p2.x(), -12.0);
    assert_eq!(p2.y(), -86.0);
    assert_eq!(p2.z(), -86.0);
    assert_eq!(p2.w(), 30.0);
}

#[test]
#[ignore]
fn motor_point_variadic() {
    /*
    motor m{1.0, 4.0, 3.0, 2.0, 5.0, 6.0, 7.0, 8.0};
    point ps[2] = {{-1.0, 1.0, 2.0}, {-1.0, 1.0, 2.0}};
    point ps2[2];
    m(ps, ps2, 2);

    for (size_t i = 0; i != 2; ++i)
    {
        assert_eq!(ps2[i].x(), -12.0);
        assert_eq!(ps2[i].y(), -86.0);
        assert_eq!(ps2[i].z(), -86.0);
        assert_eq!(ps2[i].w(), 30.0);
    }
    */
}

#[test]
//#[ignore]
fn motor_line() {
    let m = Motor::new (2.0, 4.0, 3.0, -1.0, -5.0, -2.0, 2.0, -3.0);
    // a*e01 + b*e01 + c*e02 + d*e23 + e*e31 + f*e12
    let l1 = Line::new (-1.0, 2.0, -3.0, -6.0, 5.0, 4.0);
    let l2 = m.conj_line(l1);
    assert_eq!(l2.e01(), 6.0);
    assert_eq!(l2.e02(), 522.0);
    assert_eq!(l2.e03(), 96.0);
    assert_eq!(l2.e12(), -214.0);
    assert_eq!(l2.e31(), -148.0);
    assert_eq!(l2.e23(), -40.0);
}

#[test]
#[ignore]
fn motor_line_variadic() {
    /*
    motor m{2.0, 4.0, 3.0, -1.0, -5.0, -2.0, 2.0, -3.0};
    // a*e01 + b*e01 + c*e02 + d*e23 + e*e31 + f*e12
    line ls[2]
        = {{-1.0, 2.0, -3.0, -6.0, 5.0, 4.0}, {-1.0, 2.0, -3.0, -6.0, 5.0, 4.0}};
    line ls2[2];
    m(ls, ls2, 2);

    for (size_t i = 0; i != 2; ++i)
    {
        assert_eq!(ls2[i].e01(), 6.0);
        assert_eq!(ls2[i].e02(), 522.0);
        assert_eq!(ls2[i].e03(), 96.0);
        assert_eq!(ls2[i].e12(), -214.0);
        assert_eq!(ls2[i].e31(), -148.0);
        assert_eq!(ls2[i].e23(), -40.0);
    }
    */
}

#[test]
fn motor_origin() {
    let r = Rotor::new(FRAC_PI_2, 0.0, 0.0, 1.0);
    let t = Translator::new(1.0, 0.0, 0.0, 1.0);
    let m: Motor = r * t;
    let p: Point = m.conj_origin();
    assert_eq!(p.x(), 0.0);
    assert_eq!(p.y(), 0.0);
    abs_diff_eq!(p.z(), 1.0);
}

#[test]
#[ignore]
fn motor_to_matrix4x4() {
    /*
    motor m{1.0, 4.0, 3.0, 2.0, 5.0, 6.0, 7.0, 8.0};
    __m128 p1    = _mm_set_ps(1.0, 2.0, 1.0, -1.0);
    mat4x4 m_mat = m.as_mat4x4();
    __m128 p2    = m_mat(p1);
    float buf[4];
    _mm_storeu_ps(buf, p2);

    assert_eq!(buf[0], -12.0);
    assert_eq!(buf[1], -86.0);
    assert_eq!(buf[2], -86.0);
    assert_eq!(buf[3], 30.0);
    */
}

#[test]
#[ignore]
fn motor_to_matrix3x4() {
    /*
    motor m{1.0, 4.0, 3.0, 2.0, 5.0, 6.0, 7.0, 8.0};
    m.normalize();
    __m128 p1    = _mm_set_ps(1.0, 2.0, 1.0, -1.0);
    mat3x4 m_mat = m.as_mat3x4();
    __m128 p2    = m_mat(p1);
    float buf[4];
    _mm_storeu_ps(buf, p2);

    assert_eq!(buf[0], doctest::Approx(-12.0 / 30.0));
    assert_eq!(buf[1], doctest::Approx(-86.0 / 30.0));
    assert_eq!(buf[2], doctest::Approx(-86.0 / 30.0));
    assert_eq!(buf[3], 1.0);
    */
}

#[test]
fn normalize_motor() {
    let m = Motor::new(1.0, 4.0, 3.0, 2.0, 5.0, 6.0, 7.0, 8.0);
    let m = m.normalized();
    let norm = m * m.reversed();
    abs_diff_eq!(norm.scalar(), 1.0);
    abs_diff_eq!(norm.e0123(), 0.0);
}

#[test]
fn motor_sqrt() {
    let line = Line::new(3.0, 1.0, 2.0, 4.0, -2.0, 1.0).normalized();
    let m = Motor::from_line(FRAC_PI_2, 3.0, line);

    let m2 = m.sqrt();
    let m2 = m2 * m2;
    abs_diff_eq!(m.scalar(), m2.scalar());
    abs_diff_eq!(m.e01(), m2.e01());
    abs_diff_eq!(m.e02(), m2.e02());
    abs_diff_eq!(m.e03(), m2.e03());
    abs_diff_eq!(m.e23(), m2.e23());
    abs_diff_eq!(m.e31(), m2.e31());
    abs_diff_eq!(m.e12(), m2.e12());
    abs_diff_eq!(m.e0123(), m2.e0123());
}

#[test]
fn rotor_sqrt() {
    let r = Rotor::new(FRAC_PI_2, 1.0, 2.0, 3.0);

    let r2 = r.sqrt();
    let r2 = r2 * r2;
    abs_diff_eq!(r2.scalar(), r.scalar());
    abs_diff_eq!(r2.e23(), r.e23());
    abs_diff_eq!(r2.e13(), r.e13());
    abs_diff_eq!(r2.e12(), r.e12());
}

#[test]
fn normalize_rotor() {
    let p1 = f32x4::new(4.0, -3.0, 3.0, 28.0);
    let r = Rotor::from(p1).normalized();
    let norm: Rotor = r * r.reversed();
    abs_diff_eq!(norm.scalar(), 1.0);
    abs_diff_eq!(norm.e12(), 0.0);
    abs_diff_eq!(norm.e13(), 0.0);
    abs_diff_eq!(norm.e23(), 0.0);
}
