use klein::{Line, Point, Plane};

#[test]
fn simd_sandwich() {
    unsafe {
        use core::arch::x86_64::*;
        let a = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
        let b = _mm_set_ps(-1.0, -2.0, -3.0, -4.0);
        let mut ab = [0.0f32; 4];
        _mm_store_ps(ab.as_mut_ptr(), klein::arch::sw02(a, b));

        assert_eq!(ab[0], 9.0);
        assert_eq!(ab[1], 2.0);
        assert_eq!(ab[2], 3.0);
        assert_eq!(ab[3], 4.0);
    }
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

/*
TEST_CASE("rotor-line")
{
    // Make an unnormalized rotor to verify correctness
    float data[4] = {1.0, 4.0, -3.0, 2.0};
    rotor r;
    r.load_normalized(data);
    // a*e01 + b*e01 + c*e02 + d*e23 + e*e31 + f*e12
    line l1{-1.0, 2.0, -3.0, -6.0, 5.0, 4.0};
    line l2{r(l1)};
    assert_eq!(l2.e01(), -110.0);
    assert_eq!(l2.e02(), 20.0);
    assert_eq!(l2.e03(), 10.0);
    assert_eq!(l2.e12(), -240.0);
    assert_eq!(l2.e31(), 102.0);
    assert_eq!(l2.e23(), -36.0);
}

TEST_CASE("rotor-point")
{
    rotor r{M_PI * 0.5f, 0, 0, 1.0};
    point p1{1, 0, 0};
    point p2 = r(p1);
    assert_eq!(p2.x(), 0.0);
    assert_eq!(p2.y(), doctest::Approx(1.0));
    assert_eq!(p2.z(), 0.0);
}

TEST_CASE("translator-point")
{
    translator t{1.0, 0.0, 0.0, 1.0};
    point p1{1, 0, 0};
    point p2 = t(p1);
    assert_eq!(p2.x(), 1.0);
    assert_eq!(p2.y(), 0.0);
    assert_eq!(p2.z(), 1.0);
}

TEST_CASE("translator-line")
{
    float data[4] = {0.0, -5.0, -2.0, 2.0};
    translator t;
    t.load_normalized(data);
    // a*e01 + b*e01 + c*e02 + d*e23 + e*e31 + f*e12
    line l1{-1.0, 2.0, -3.0, -6.0, 5.0, 4.0};
    line l2{t(l1)};
    assert_eq!(l2.e01(), 35.0);
    assert_eq!(l2.e02(), -14.0);
    assert_eq!(l2.e03(), 71.0);
    assert_eq!(l2.e12(), 4.0);
    assert_eq!(l2.e31(), 5.0);
    assert_eq!(l2.e23(), -6.0);
}

TEST_CASE("construct-motor")
{
    rotor r{M_PI * 0.5f, 0, 0, 1.0};
    translator t{1.0, 0.0, 0.0, 1.0};
    motor m = r * t;
    point p1{1, 0, 0};
    point p2 = m(p1);
    assert_eq!(p2.x(), 0.0);
    assert_eq!(p2.y(), doctest::Approx(1.0));
    assert_eq!(p2.z(), doctest::Approx(1.0));

    // Rotation and translation about the same axis commutes
    m  = t * r;
    p2 = m(p1);
    assert_eq!(p2.x(), 0.0);
    assert_eq!(p2.y(), doctest::Approx(1.0));
    assert_eq!(p2.z(), doctest::Approx(1.0));

    line l = log(m);
    assert_eq!(l.e23(), 0.0);
    assert_eq!(l.e12(), doctest::Approx(-0.7854).epsilon(0.001));
    assert_eq!(l.e31(), 0.0);
    assert_eq!(l.e01(), 0.0);
    assert_eq!(l.e02(), 0.0);
    assert_eq!(l.e03(), doctest::Approx(-0.5));
}

TEST_CASE("construct-motor-via-screw-axis")
{
    motor m{M_PI * 0.5f, 1.0, line{0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    point p1{1, 0, 0};
    point p2 = m(p1);
    assert_eq!(p2.x(), doctest::Approx(0.0));
    assert_eq!(p2.y(), doctest::Approx(1.0));
    assert_eq!(p2.z(), doctest::Approx(1.0));
}

TEST_CASE("motor-plane")
{
    motor m{1.0, 4.0, 3.0, 2.0, 5.0, 6.0, 7.0, 8.0};
    plane p1{3.0, 2.0, 1.0, -1.0};
    plane p2 = m(p1);
    assert_eq!(p2.x(), 78.0);
    assert_eq!(p2.y(), 60.0);
    assert_eq!(p2.z(), 54.0);
    assert_eq!(p2.d(), 358.0);
}

TEST_CASE("motor-plane-variadic")
{
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
}

TEST_CASE("motor-point")
{
    motor m{1.0, 4.0, 3.0, 2.0, 5.0, 6.0, 7.0, 8.0};
    point p1{-1.0, 1.0, 2.0};
    point p2 = m(p1);
    assert_eq!(p2.x(), -12.0);
    assert_eq!(p2.y(), -86.0);
    assert_eq!(p2.z(), -86.0);
    assert_eq!(p2.w(), 30.0);
}

TEST_CASE("motor-point-variadic")
{
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
}

TEST_CASE("motor-line")
{
    motor m{2.0, 4.0, 3.0, -1.0, -5.0, -2.0, 2.0, -3.0};
    // a*e01 + b*e01 + c*e02 + d*e23 + e*e31 + f*e12
    line l1{-1.0, 2.0, -3.0, -6.0, 5.0, 4.0};
    line l2{m(l1)};
    assert_eq!(l2.e01(), 6.0);
    assert_eq!(l2.e02(), 522.0);
    assert_eq!(l2.e03(), 96.0);
    assert_eq!(l2.e12(), -214.0);
    assert_eq!(l2.e31(), -148.0);
    assert_eq!(l2.e23(), -40.0);
}

TEST_CASE("motor-line-variadic")
{
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
}

TEST_CASE("motor-origin")
{
    rotor r{M_PI * 0.5f, 0, 0, 1.0};
    translator t{1.0, 0.0, 0.0, 1.0};
    motor m = r * t;
    point p = m(origin{});
    assert_eq!(p.x(), 0.0);
    assert_eq!(p.y(), 0.0);
    assert_eq!(p.z(), doctest::Approx(1.0));
}

TEST_CASE("motor-to-matrix")
{
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
}

TEST_CASE("motor-to-matrix-3x4")
{
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
}

TEST_CASE("normalize-motor")
{
    motor m{1.0, 4.0, 3.0, 2.0, 5.0, 6.0, 7.0, 8.0};
    m.normalize();
    motor norm = m * ~m;
    assert_eq!(norm.scalar(), doctest::Approx(1.0));
    assert_eq!(norm.e0123(), doctest::Approx(0.0));
}

TEST_CASE("motor-sqrt")
{
    motor m{M_PI * 0.5f, 3.0, line{3.0, 1.0, 2.0, 4.0, -2.0, 1.0}.normalized()};

    motor m2 = sqrt(m);
    m2       = m2 * m2;
    assert_eq!(m.scalar(), doctest::Approx(m2.scalar()));
    assert_eq!(m.e01(), doctest::Approx(m2.e01()));
    assert_eq!(m.e02(), doctest::Approx(m2.e02()));
    assert_eq!(m.e03(), doctest::Approx(m2.e03()));
    assert_eq!(m.e23(), doctest::Approx(m2.e23()));
    assert_eq!(m.e31(), doctest::Approx(m2.e31()));
    assert_eq!(m.e12(), doctest::Approx(m2.e12()));
    assert_eq!(m.e0123(), doctest::Approx(m2.e0123()));
}

TEST_CASE("rotor-sqrt")
{
    rotor r{M_PI * 0.5f, 1, 2, 3};

    rotor r2 = sqrt(r);
    r2       = r2 * r2;
    assert_eq!(r2.scalar(), doctest::Approx(r.scalar()));
    assert_eq!(r2.e23(), doctest::Approx(r.e23()));
    assert_eq!(r2.e31(), doctest::Approx(r.e31()));
    assert_eq!(r2.e12(), doctest::Approx(r.e12()));
}

TEST_CASE("normalize-rotor")
{
    rotor r;
    r.p1_ = _mm_set_ps(4.0, -3.0, 3.0, 28.0);
    r.normalize();
    rotor norm = r * ~r;
    assert_eq!(norm.scalar(), doctest::Approx(1.0));
    assert_eq!(norm.e12(), doctest::Approx(0.0));
    assert_eq!(norm.e31(), doctest::Approx(0.0));
    assert_eq!(norm.e23(), doctest::Approx(0.0));
}
*/
