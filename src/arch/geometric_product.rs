use super::sse::*;
use core::arch::x86_64::*;

// Define functions of the form gpAB where A and B are partition indices.
// Each function so-defined computes the geometric product using vector intrinsics.
// The partition index determines which basis elements are present
// in each XMM component of the operand.

// A number of the computations in this file are performed symbolically in
// scripts/validation.klein

// Partition memory layouts
//     LSB --> MSB
// p0: (e0, e1, e2, e3)
// p1: (1, e12, e31, e23)
// p2: (e0123, e01, e02, e03)
// p3: (e123, e032, e013, e021)

// p0: (e0, e1, e2, e3)
// p1: (1, e23, e31, e12)
// p2: (e0123, e01, e02, e03)
pub unsafe fn gp00(a: __m128, b: __m128) -> (__m128, __m128) {
    // (a1 b1 + a2 b2 + a3 b3) +
    //
    // (a2 b3 - a3 b2) e23 +
    // (a3 b1 - a1 b3) e31 +
    // (a1 b2 - a2 b1) e12 +
    //
    // (a0 b1 - a1 b0) e01 +
    // (a0 b2 - a2 b0) e02 +
    // (a0 b3 - a3 b0) e03

    let p1 = _mm_mul_ps(swizzle!(a, 1, 3, 2, 1), swizzle!(b, 2, 1, 3, 1));

    let p1 = _mm_sub_ps(
        p1,
        _mm_xor_ps(
            _mm_set_ss(-0.0),
            _mm_mul_ps(swizzle!(a, 2, 1, 3, 2), swizzle!(b, 1, 3, 2, 2)),
        ),
    );

    // Add a3 b3 to the lowest component
    let p1 = _mm_add_ss(
        p1,
        _mm_mul_ps(swizzle!(a, 0, 0, 0, 3), swizzle!(b, 0, 0, 0, 3)),
    );

    // (a0 b0, a0 b1, a0 b2, a0 b3)
    let p2 = _mm_mul_ps(swizzle!(a, 0, 0, 0, 0), b);

    // Sub (a0 b0, a1 b0, a2 b0, a3 b0)
    // Note that the lowest component cancels
    let p2 = _mm_sub_ps(p2, _mm_mul_ps(a, swizzle!(b, 0, 0, 0, 0)));

    (p1, p2)
}

// p0: (e0, e1, e2, e3)
// p3: (e123, e032, e013, e021)
// p1: (1, e12, e31, e23)
// p2: (e0123, e01, e02, e03)
pub unsafe fn gp03_true(a: __m128, b: __m128) -> (__m128, __m128) {
    // a1 b0 e23 +
    // a2 b0 e31 +
    // a3 b0 e12 +
    // (a0 b0 + a1 b1 + a2 b2 + a3 b3) e0123 +
    // (a3 b2 - a2 b3) e01 +
    // (a1 b3 - a3 b1) e02 +
    // (a2 b1 - a1 b2) e03
    //
    // With flip:
    //
    // a1 b0 e23 +
    // a2 b0 e31 +
    // a3 b0 e12 +
    // -(a0 b0 + a1 b1 + a2 b2 + a3 b3) e0123 +
    // (a3 b2 - a2 b3) e01 +
    // (a1 b3 - a3 b1) e02 +
    // (a2 b1 - a1 b2) e03

    let p1 = _mm_mul_ps(a, swizzle!(b, 0, 0, 0, 0));
    let p1 = if cfg!(target_feature = "sse4.1") {
        _mm_blend_ps(p1, _mm_setzero_ps(), 1)
    } else {
        _mm_and_ps(p1, _mm_castsi128_ps(_mm_set_epi32(-1, -1, -1, 0)))
    };

    // (_, a3 b2, a1 b3, a2 b1)
    let p2 = _mm_mul_ps(swizzle!(a, 2, 1, 3, 0), swizzle!(b, 1, 3, 2, 0));
    let p2 = _mm_sub_ps(
        p2,
        _mm_mul_ps(swizzle!(a, 1, 3, 2, 0), swizzle!(b, 2, 1, 3, 0)),
    );

    // Compute a0 b0 + a1 b1 + a2 b2 + a3 b3 and store it in the low
    // component
    let tmp = dp(a, b);

    let tmp = _mm_xor_ps(tmp, _mm_set_ss(-0.0));

    let p2 = _mm_add_ps(p2, tmp);

    (p1, p2)
}

pub unsafe fn gp03_false(a: __m128, b: __m128) -> (__m128, __m128) {
    let p1 = _mm_mul_ps(a, swizzle!(b, 0, 0, 0, 0));
    let p1 = if cfg!(target_feature = "sse4.1") {
        _mm_blend_ps(p1, _mm_setzero_ps(), 1)
    } else {
        _mm_and_ps(p1, _mm_castsi128_ps(_mm_set_epi32(-1, -1, -1, 0)))
    };

    // (_, a3 b2, a1 b3, a2 b1)
    let p2 = _mm_mul_ps(swizzle!(a, 2, 1, 3, 0), swizzle!(b, 1, 3, 2, 0));
    let p2 = _mm_sub_ps(
        p2,
        _mm_mul_ps(swizzle!(a, 1, 3, 2, 0), swizzle!(b, 2, 1, 3, 0)),
    );

    // Compute a0 b0 + a1 b1 + a2 b2 + a3 b3 and store it in the low
    // component
    let tmp = dp(a, b);
    let p2 = _mm_add_ps(p2, tmp);

    (p1, p2)
}

// p1: (1, e23, e31, e12)
pub unsafe fn gp11(a: __m128, b: __m128) -> __m128 {
    // (a0 b0 - a1 b1 - a2 b2 - a3 b3) +
    // (a0 b1 - a2 b3 + a1 b0 + a3 b2)*e23
    // (a0 b2 - a3 b1 + a2 b0 + a1 b3)*e31
    // (a0 b3 - a1 b2 + a3 b0 + a2 b1)*e12

    // We use abcd to refer to the slots to avoid conflating bivector/scalar
    // coefficients with cartesian coordinates

    // In general, we can get rid of at most one swizzle
    let p1 = _mm_mul_ps(swizzle!(a, 0, 0, 0, 0), b);

    let p1 = _mm_sub_ps(
        p1,
        _mm_mul_ps(swizzle!(a, 1, 3, 2, 1), swizzle!(b, 2, 1, 3, 1)),
    );

    // In a separate register, accumulate the later components so we can
    // negate the lower single-precision element with a single instruction
    let tmp1 = _mm_mul_ps(swizzle!(a, 3, 2, 1, 2), swizzle!(b, 0, 0, 0, 2));
    let tmp2 = _mm_mul_ps(swizzle!(a, 2, 1, 3, 3), swizzle!(b, 1, 3, 2, 3));
    let tmp = _mm_xor_ps(_mm_add_ps(tmp1, tmp2), _mm_set_ss(-0.0));

    let p1 = _mm_add_ps(p1, tmp);
    p1
}

// p3: (e123, e021, e013, e032)
// p2: (e0123, e01, e02, e03)
pub unsafe fn gp33(a: __m128, b: __m128) -> __m128 {
    // (-a0 b0) +
    // (-a0 b1 + a1 b0) e01 +
    // (-a0 b2 + a2 b0) e02 +
    // (-a0 b3 + a3 b0) e03
    //
    // Produce a translator by dividing all terms by a0 b0

    let tmp = _mm_mul_ps(swizzle!(a, 0, 0, 0, 0), b);
    let tmp = _mm_mul_ps(tmp, _mm_set_ps(-1.0, -1.0, -1.0, -2.0));
    let tmp = _mm_add_ps(tmp, _mm_mul_ps(a, swizzle!(b, 0, 0, 0, 0)));

    // (0, 1, 2, 3) -> (0, 0, 2, 2)
    let ss = _mm_moveldup_ps(tmp);
    let ss = _mm_movelh_ps(ss, ss);
    let tmp = _mm_mul_ps(tmp, rcp_nr1(ss.into()).0);

    if cfg!(target_feature = "sse4.1") {
        _mm_blend_ps(tmp, _mm_setzero_ps(), 1)
    } else {
        _mm_and_ps(tmp, _mm_castsi128_ps(_mm_set_epi32(-1, -1, -1, 0)))
    }
}

pub unsafe fn gp_dl(u: f32, v: f32, b: __m128, c: __m128) -> (__m128, __m128) {
    // b1 u e23 +
    // b2 u e31 +
    // b3 u e12 +
    // (-b1 v + c1 u) e01 +
    // (-b2 v + c2 u) e02 +
    // (-b3 v + c3 u) e03
    let u_vec = _mm_set1_ps(u);
    let v_vec = _mm_set1_ps(v);
    let p1 = _mm_mul_ps(u_vec, b);
    let p2 = _mm_mul_ps(c, u_vec);
    let p2 = _mm_sub_ps(p2, _mm_mul_ps(b, v_vec));
    (p1, p2)
}

pub unsafe fn gp_rt_true(a: __m128, b: __m128) -> __m128 {
    // (a1 b1 + a2 b2 + a3 b3) e0123 +
    // (a0 b1 + a2 b3 - a3 b2) e01 +
    // (a0 b2 + a3 b1 - a1 b3) e02 +
    // (a0 b3 + a1 b2 - a2 b1) e03

    let p2 = _mm_mul_ps(swizzle!(a, 0, 0, 0, 1), swizzle!(b, 3, 2, 1, 1));
    let p2 = _mm_add_ps(
        p2,
        _mm_mul_ps(swizzle!(a, 1, 3, 2, 2), swizzle!(b, 2, 1, 3, 2)),
    );
    let p2 = _mm_sub_ps(
        p2,
        _mm_xor_ps(
            _mm_set_ss(-0.0),
            _mm_mul_ps(swizzle!(a, 2, 1, 3, 3), swizzle!(b, 1, 3, 2, 3)),
        ),
    );

    p2
}

pub unsafe fn gp_rt_false(a: __m128, b: __m128) -> __m128 {
    // (a1 b1 + a2 b2 + a3 b3) e0123 +
    // (a0 b1 + a3 b2 - a2 b3) e01 +
    // (a0 b2 + a1 b3 - a3 b1) e02 +
    // (a0 b3 + a2 b1 - a1 b2) e03

    let p2 = _mm_mul_ps(swizzle!(a, 0, 0, 0, 1), swizzle!(b, 3, 2, 1, 1));
    let p2 = _mm_add_ps(
        p2,
        _mm_mul_ps(swizzle!(a, 2, 1, 3, 2), swizzle!(b, 1, 3, 2, 2)),
    );
    _mm_sub_ps(
        p2,
        _mm_xor_ps(
            _mm_set_ss(-0.0),
            _mm_mul_ps(swizzle!(a, 1, 3, 2, 3), swizzle!(b, 2, 1, 3, 3)),
        ),
    )
}

pub unsafe fn gp12_true(a: __m128, b: __m128) -> __m128 {
    let p2 = gp_rt_true(a, b);
    _mm_sub_ps(
        p2,
        _mm_xor_ps(_mm_set_ss(-0.0), _mm_mul_ps(a, swizzle!(b, 0, 0, 0, 0))),
    )
}

pub unsafe fn gp12_false(a: __m128, b: __m128) -> __m128 {
    let p2 = gp_rt_false(a, b);
    _mm_sub_ps(
        p2,
        _mm_xor_ps(_mm_set_ss(-0.0), _mm_mul_ps(a, swizzle!(b, 0, 0, 0, 0))),
    )
}

// Optimized line * line operation
pub unsafe fn gp_ll(l1: &[__m128; 2], l2: &[__m128; 2], out: &mut [__m128; 2]) {
    // (-a1 b1 - a3 b3 - a2 b2) +
    // (a2 b1 - a1 b2) e12 +
    // (a1 b3 - a3 b1) e31 +
    // (a3 b2 - a2 b3) e23 +
    // (a1 c1 + a3 c3 + a2 c2 + b1 d1 + b3 d3 + b2 d2) e0123
    // (a3 c2 - a2 c3         + b2 d3 - b3 d2) e01 +
    // (a1 c3 - a3 c1         + b3 d1 - b1 d3) e02 +
    // (a2 c1 - a1 c2         + b1 d2 - b2 d1) e03 +
    let a = &l1[0];
    let d = &l1[1];
    let b = &l2[0];
    let c = &l2[1];

    let flip = _mm_set_ss(-0.0);

    let p1 = &mut *out.as_mut_ptr();
    let p2 = &mut *out.as_mut_ptr().add(1);

    *p1 = _mm_mul_ps(swizzle!(*a, 3, 1, 2, 1), swizzle!(*b, 2, 3, 1, 1));
    *p1 = _mm_xor_ps(*p1, flip);
    *p1 = _mm_sub_ps(
        *p1,
        _mm_mul_ps(swizzle!(*a, 2, 3, 1, 3), swizzle!(*b, 3, 1, 2, 3)),
    );
    let a2 = _mm_unpackhi_ps(*a, *a);
    let b2 = _mm_unpackhi_ps(*b, *b);
    *p1 = _mm_sub_ss(*p1, _mm_mul_ss(a2, b2));

    *p2 = _mm_mul_ps(swizzle!(*a, 2, 1, 3, 1), swizzle!(*c, 1, 3, 2, 1));
    *p2 = _mm_sub_ps(
        *p2,
        _mm_xor_ps(
            flip,
            _mm_mul_ps(swizzle!(*a, 1, 3, 2, 3), swizzle!(*c, 2, 1, 3, 3)),
        ),
    );
    *p2 = _mm_add_ps(
        *p2,
        _mm_mul_ps(swizzle!(*b, 1, 3, 2, 1), swizzle!(*d, 2, 1, 3, 1)),
    );
    *p2 = _mm_sub_ps(
        *p2,
        _mm_xor_ps(
            flip,
            _mm_mul_ps(swizzle!(*b, 2, 1, 3, 3), swizzle!(*d, 1, 3, 2, 3)),
        ),
    );
    let c2 = _mm_unpackhi_ps(*c, *c);
    let d2 = _mm_unpackhi_ps(*d, *d);
    *p2 = _mm_add_ss(*p2, _mm_mul_ss(a2, c2));
    *p2 = _mm_add_ss(*p2, _mm_mul_ss(b2, d2));
}

// Optimized motor * motor operation
pub unsafe fn gpMM(m1: &[__m128; 2], m2: &[__m128; 2], out: &mut [__m128; 2]) {
    // (a0 c0 - a1 c1 - a2 c2 - a3 c3) +
    // (a0 c1 + a3 c2 + a1 c0 - a2 c3) e23 +
    // (a0 c2 + a1 c3 + a2 c0 - a3 c1) e31 +
    // (a0 c3 + a2 c1 + a3 c0 - a1 c2) e12 +
    //
    // (a0 d0 + b0 c0 + a1 d1 + b1 c1 + a2 d2 + a3 d3 + b2 c2 + b3 c3)
    //  e0123 +
    // (a0 d1 + b1 c0 + a3 d2 + b3 c2 - a1 d0 - a2 d3 - b0 c1 - b2 c3)
    //  e01 +
    // (a0 d2 + b2 c0 + a1 d3 + b1 c3 - a2 d0 - a3 d1 - b0 c2 - b3 c1)
    //  e02 +
    // (a0 d3 + b3 c0 + a2 d1 + b2 c1 - a3 d0 - a1 d2 - b0 c3 - b1 c2)
    //  e03

    let a = &m1[0];
    let d = &m1[1];
    let b = &m2[0];
    let c = &m2[1];

    let e = &mut *out.as_mut_ptr();
    let f = &mut *out.as_mut_ptr().add(1);

    let a_xxxx = swizzle!(*a, 0, 0, 0, 0);
    let a_zyzw = swizzle!(*a, 3, 2, 1, 2);
    let a_ywyz = swizzle!(*a, 2, 1, 3, 1);
    let a_wzwy = swizzle!(*a, 1, 3, 2, 3);
    let c_wwyz = swizzle!(*c, 2, 1, 3, 3);
    let c_yzwy = swizzle!(*c, 1, 3, 2, 1);
    let s_flip = _mm_set_ss(-0.0);

    *e = _mm_mul_ps(a_xxxx, *c);
    let t = _mm_mul_ps(a_ywyz, c_yzwy);
    let t = _mm_add_ps(t, _mm_mul_ps(a_zyzw, swizzle!(*c, 0, 0, 0, 2)));
    let t = _mm_xor_ps(t, s_flip);
    *e = _mm_add_ps(*e, t);
    *e = _mm_sub_ps(*e, _mm_mul_ps(a_wzwy, c_wwyz));

    *f = _mm_mul_ps(a_xxxx, *d);
    *f = _mm_add_ps(*f, _mm_mul_ps(*b, swizzle!(*c, 0, 0, 0, 0)));
    *f = _mm_add_ps(*f, _mm_mul_ps(a_ywyz, swizzle!(*d, 1, 3, 2, 1)));
    *f = _mm_add_ps(*f, _mm_mul_ps(swizzle!(*b, 2, 1, 3, 1), c_yzwy));
    let t = _mm_mul_ps(a_zyzw, swizzle!(*d, 0, 0, 0, 2));
    let t = _mm_add_ps(t, _mm_mul_ps(a_wzwy, swizzle!(*d, 2, 1, 3, 3)));
    let t = _mm_add_ps(
        t,
        _mm_mul_ps(swizzle!(*b, 0, 0, 0, 2), swizzle!(*c, 3, 2, 1, 2)),
    );
    let t = _mm_add_ps(t, _mm_mul_ps(swizzle!(*b, 1, 3, 2, 3), c_wwyz));
    let t = _mm_xor_ps(t, s_flip);
    *f = _mm_sub_ps(*f, t);
}
