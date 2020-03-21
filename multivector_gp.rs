
SUBCASE("plane*plane")
{
    // d*e_0 + a*e_1 + b*e_2 + c*e_3
    plane p1{1.0, 2.0, 3.0, 4.0};
    plane p2{2.0, 3.0, -1.0, -2.0};
    motor p12 = p1 * p2;
    assert_eq!(p12.scalar(), 5.0);
    assert_eq!(p12.e12(), -1.0);
    assert_eq!(p12.e31(), 7.0);
    assert_eq!(p12.e23(), -11.0);
    assert_eq!(p12.e01(), 10.0);
    assert_eq!(p12.e02(), 16.0);
    assert_eq!(p12.e03(), 2.0);
    assert_eq!(p12.e0123(), 0.0);

    plane p3 = sqrt(p1 * p2)(p2);
    assert_eq!(p3.approx_eq(p1, 0.001f), true);

    p1.normalize();
    motor m = p1 * p1;
    assert_eq!(m.scalar(), doctest::Approx(1.0));
}

SUBCASE("plane/plane")
{
    plane p1{1.0, 2.0, 3.0, 4.0};
    motor m = p1 / p1;
    assert_eq!(m.scalar(), doctest::Approx(1.0));
    assert_eq!(m.e12(), 0.0);
    assert_eq!(m.e31(), 0.0);
    assert_eq!(m.e23(), 0.0);
    assert_eq!(m.e01(), 0.0);
    assert_eq!(m.e02(), 0.0);
    assert_eq!(m.e03(), 0.0);
    assert_eq!(m.e0123(), 0.0);
}

SUBCASE("plane*point")
{
    // d*e_0 + a*e_1 + b*e_2 + c*e_3
    plane p1{1.0, 2.0, 3.0, 4.0};
    // x*e_032 + y*e_013 + z*e_021 + e_123
    point p2{-2.0, 1.0, 4.0};

    motor p1p2 = p1 * p2;
    assert_eq!(p1p2.scalar(), 0.0);
    assert_eq!(p1p2.e01(), -5.0);
    assert_eq!(p1p2.e02(), 10.0);
    assert_eq!(p1p2.e03(), -5.0);
    assert_eq!(p1p2.e12(), 3.0);
    assert_eq!(p1p2.e31(), 2.0);
    assert_eq!(p1p2.e23(), 1.0);
    assert_eq!(p1p2.e0123(), 16.0);
}

SUBCASE("line-normalization")
{
    line l{1.0, 2.0, 3.0, 3.0, 2.0, 1.0};
    l.normalize();
    motor m = l * ~l;
    assert_eq!(m.scalar(), doctest::Approx(1.0));
    assert_eq!(m.e23(), doctest::Approx(0.0));
    assert_eq!(m.e31(), doctest::Approx(0.0));
    assert_eq!(m.e12(), doctest::Approx(0.0));
    assert_eq!(m.e01(), doctest::Approx(0.0));
    assert_eq!(m.e02(), doctest::Approx(0.0));
    assert_eq!(m.e03(), doctest::Approx(0.0));
    assert_eq!(m.e0123(), doctest::Approx(0.0));
}

SUBCASE("branch*branch")
{
    branch b1{2.0, 1.0, 3.0};
    branch b2{1.0, -2.0, -3.0};
    rotor r = b2 * b1;
    assert_eq!(r.scalar(), 9.0);
    assert_eq!(r.e23(), 3.0);
    assert_eq!(r.e31(), 9.0);
    assert_eq!(r.e12(), -5.0);

    b1.normalize();
    b2.normalize();
    branch b3 = ~sqrt(b2 * b1)(b1);
    assert_eq!(b3.x(), doctest::Approx(b2.x()));
    assert_eq!(b3.y(), doctest::Approx(b2.y()));
    assert_eq!(b3.z(), doctest::Approx(b2.z()));
}

SUBCASE("branch/branch")
{
    branch b{2.0, 1.0, 3.0};
    rotor r = b / b;
    assert_eq!(r.scalar(), doctest::Approx(1.0));
    assert_eq!(r.e23(), 0.0);
    assert_eq!(r.e31(), 0.0);
    assert_eq!(r.e12(), 0.0);
}

SUBCASE("line*line")
{
    // a*e01 + b*e02 + c*e03 + d*e23 + e*e31 + f*e12
    line l1{1.0, 0.0, 0.0, 3.0, 2.0, 1.0};
    line l2{0.0, 1.0, 0.0, 4.0, 1.0, -2.0};

    motor l1l2 = l1 * l2;
    assert_eq!(l1l2.scalar(), -12.0);
    assert_eq!(l1l2.e12(), 5.0);
    assert_eq!(l1l2.e31(), -10.0);
    assert_eq!(l1l2.e23(), 5.0);
    assert_eq!(l1l2.e01(), 1.0);
    assert_eq!(l1l2.e02(), -2.0);
    assert_eq!(l1l2.e03(), -4.0);
    assert_eq!(l1l2.e0123(), 6.0);

    l1.normalize();
    l2.normalize();
    line l3 = sqrt(l1 * l2)(l2);
    assert_eq!(l3.approx_eq(-l1, 0.001f), true);
}

SUBCASE("line/line")
{
    line l{1.0, -2.0, 2.0, -3.0, 3.0, -4.0};
    motor m = l / l;
    assert_eq!(m.scalar(), doctest::Approx(1.0));
    assert_eq!(m.e12(), 0.0);
    assert_eq!(m.e31(), 0.0);
    assert_eq!(m.e23(), 0.0);
    assert_eq!(m.e01(), 0.0);
    assert_eq!(m.e02(), 0.0);
    assert_eq!(m.e03(), doctest::Approx(0.0));
    assert_eq!(m.e0123(), doctest::Approx(0.0));
}

SUBCASE("point*plane")
{
    // x*e_032 + y*e_013 + z*e_021 + e_123
    point p1{-2.0, 1.0, 4.0};
    // d*e_0 + a*e_1 + b*e_2 + c*e_3
    plane p2{1.0, 2.0, 3.0, 4.0};

    motor p1p2 = p1 * p2;
    assert_eq!(p1p2.scalar(), 0.0);
    assert_eq!(p1p2.e01(), -5.0);
    assert_eq!(p1p2.e02(), 10.0);
    assert_eq!(p1p2.e03(), -5.0);
    assert_eq!(p1p2.e12(), 3.0);
    assert_eq!(p1p2.e31(), 2.0);
    assert_eq!(p1p2.e23(), 1.0);
    assert_eq!(p1p2.e0123(), -16.0);
}

SUBCASE("point*point")
{
    // x*e_032 + y*e_013 + z*e_021 + e_123
    point p1{1.0, 2.0, 3.0};
    point p2{-2.0, 1.0, 4.0};

    translator p1p2 = p1 * p2;
    assert_eq!(p1p2.e01(), doctest::Approx(-3.0));
    assert_eq!(p1p2.e02(), doctest::Approx(-1.0));
    assert_eq!(p1p2.e03(), doctest::Approx(1.0));

    point p3 = sqrt(p1p2)(p2);
    assert_eq!(p3.x(), doctest::Approx(1.0));
    assert_eq!(p3.y(), doctest::Approx(2.0));
    assert_eq!(p3.z(), doctest::Approx(3.0));
}

SUBCASE("point/point")
{
    point p1{1.0, 2.0, 3.0};
    translator t = p1 / p1;
    assert_eq!(t.e01(), 0.0);
    assert_eq!(t.e02(), 0.0);
    assert_eq!(t.e03(), 0.0);
}

SUBCASE("translator/translator")
{
    translator t1{3.0, 1.0, -2.0, 3.0};
    translator t2 = t1 / t1;
    assert_eq!(t2.e01(), 0.0);
    assert_eq!(t2.e02(), 0.0);
    assert_eq!(t2.e03(), 0.0);
}

SUBCASE("rotor*translator")
{
    rotor r;
    r.p1_ = _mm_set_ps(1.0, 0, 0, 1.0);
    translator t;
    t.p2_   = _mm_set_ps(1.0, 0, 0, 0.0);
    motor m = r * t;
    assert_eq!(m.scalar(), 1.0);
    assert_eq!(m.e01(), 0.0);
    assert_eq!(m.e02(), 0.0);
    assert_eq!(m.e03(), 1.0);
    assert_eq!(m.e23(), 0.0);
    assert_eq!(m.e31(), 0.0);
    assert_eq!(m.e12(), 1.0);
    assert_eq!(m.e0123(), 1.0);
}

SUBCASE("translator*rotor")
{
    rotor r;
    r.p1_ = _mm_set_ps(1.0, 0, 0, 1.0);
    translator t;
    t.p2_   = _mm_set_ps(1.0, 0, 0, 0.0);
    motor m = t * r;
    assert_eq!(m.scalar(), 1.0);
    assert_eq!(m.e01(), 0.0);
    assert_eq!(m.e02(), 0.0);
    assert_eq!(m.e03(), 1.0);
    assert_eq!(m.e23(), 0.0);
    assert_eq!(m.e31(), 0.0);
    assert_eq!(m.e12(), 1.0);
    assert_eq!(m.e0123(), 1.0);
}

SUBCASE("motor*rotor")
{
    rotor r1;
    r1.p1_ = _mm_set_ps(1.0, 2.0, 3.0, 4.0);
    translator t;
    t.p2_ = _mm_set_ps(3.0, -2.0, 1.0, -3.0);
    rotor r2;
    r2.p1_   = _mm_set_ps(-4.0, 2.0, -3.0, 1.0);
    motor m1 = (t * r1) * r2;
    motor m2 = t * (r1 * r2);
    assert_eq!(m1, m2);
}

SUBCASE("rotor*motor")
{
    rotor r1;
    r1.p1_ = _mm_set_ps(1.0, 2.0, 3.0, 4.0);
    translator t;
    t.p2_ = _mm_set_ps(3.0, -2.0, 1.0, -3.0);
    rotor r2;
    r2.p1_   = _mm_set_ps(-4.0, 2.0, -3.0, 1.0);
    motor m1 = r2 * (r1 * t);
    motor m2 = (r2 * r1) * t;
    assert_eq!(m1, m2);
}

SUBCASE("motor*translator")
{
    rotor r;
    r.p1_ = _mm_set_ps(1.0, 2.0, 3.0, 4.0);
    translator t1;
    t1.p2_ = _mm_set_ps(3.0, -2.0, 1.0, -3.0);
    translator t2;
    t2.p2_   = _mm_set_ps(-4.0, 2.0, -3.0, 1.0);
    motor m1 = (r * t1) * t2;
    motor m2 = r * (t1 * t2);
    assert_eq!(m1, m2);
}

SUBCASE("translator*motor")
{
    rotor r;
    r.p1_ = _mm_set_ps(1.0, 2.0, 3.0, 4.0);
    translator t1;
    t1.p2_ = _mm_set_ps(3.0, -2.0, 1.0, -3.0);
    translator t2;
    t2.p2_   = _mm_set_ps(-4.0, 2.0, -3.0, 1.0);
    motor m1 = t2 * (r * t1);
    motor m2 = (t2 * r) * t1;
    assert_eq!(m1, m2);
}

SUBCASE("motor*motor")
{
    motor m1{2, 3, 4, 5, 6, 7, 8, 9};
    motor m2{6, 7, 8, 9, 10, 11, 12, 13};
    motor m3 = m1 * m2;
    assert_eq!(m3.scalar(), -86.0);
    assert_eq!(m3.e23(), 36.0);
    assert_eq!(m3.e31(), 32.0);
    assert_eq!(m3.e12(), 52.0);
    assert_eq!(m3.e01(), -38.0);
    assert_eq!(m3.e02(), -76.0);
    assert_eq!(m3.e03(), -66.0);
    assert_eq!(m3.e0123(), 384.0);
}

SUBCASE("motor/motor")
{
    motor m1{2, 3, 4, 5, 6, 7, 8, 9};
    motor m2 = m1 / m1;
    assert_eq!(m2.scalar(), doctest::Approx(1.0));
    assert_eq!(m2.e23(), 0.0);
    assert_eq!(m2.e31(), 0.0);
    assert_eq!(m2.e12(), 0.0);
    assert_eq!(m2.e01(), 0.0);
    assert_eq!(m2.e02(), doctest::Approx(0.0));
    assert_eq!(m2.e03(), doctest::Approx(0.0));
    assert_eq!(m2.e0123(), doctest::Approx(0.0));
}