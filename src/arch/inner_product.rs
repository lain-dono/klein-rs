use super::sse::*;
use core::arch::x86_64::*;

// Partition memory layouts
//     LSB --> MSB
// p0: (e0, e1, e2, e3)
// p1: (1, e23, e31, e12)
// p2: (e0123, e01, e02, e03)
// p3: (e123, e032, e013, e021)

#[inline(always)]
pub unsafe fn dot00(a: __m128, b: __m128) -> f32 {
    // a1 b1 + a2 b2 + a3 b3
    let p1 = hi_dp(a, b);
    let mut out = 0.0;
    _mm_store_ss(&mut out, p1);
    out
}

// The symmetric inner product on these two partitions commutes
#[inline(always)]
pub unsafe fn dot03(a: __m128, b: __m128) -> (__m128, __m128) {
    // (a2 b1 - a1 b2) e03 +
    // (a3 b2 - a2 b3) e01 +
    // (a1 b3 - a3 b1) e02 +
    // a1 b0 e23 +
    // a2 b0 e31 +
    // a3 b0 e12

    let p1 = _mm_mul_ps(a, swizzle!(b, 0, 0, 0, 0));
    let p1 = if cfg!(target_feature = "sse4.1") {
        _mm_blend_ps(p1, _mm_setzero_ps(), 1)
    } else {
        _mm_and_ps(p1, _mm_castsi128_ps(_mm_set_epi32(-1, -1, -1, 0)))
    };

    let p2 = swizzle!(
        _mm_sub_ps(
            _mm_mul_ps(swizzle!(a, 1, 3, 2, 0), b),
            _mm_mul_ps(a, swizzle!(b, 1, 3, 2, 0))
        ),
        1,
        3,
        2,
        0
    );

    (p1, p2)
}

#[inline(always)]
pub unsafe fn dot11(a: __m128, b: __m128) -> f32 {
    let p1 = _mm_xor_ps(_mm_set_ss(-0.0), hi_dp_ss(a, b));
    let mut out = 0.0;
    _mm_store_ss(&mut out, p1);
    out
}

#[inline(always)]
pub unsafe fn dot33(a: __m128, b: __m128) -> f32 {
    // -a0 b0
    let p1 = _mm_mul_ps(_mm_set_ss(-1.0), _mm_mul_ss(a, b));
    let mut out = 0.0;
    _mm_store_ss(&mut out, p1);
    out
}

// Point | Line
#[inline(always)]
pub unsafe fn dot_ptl(a: __m128, b: __m128) -> __m128 {
    // (a1 b1 + a2 b2 + a3 b3) e0 +
    // -a0 b1 e1 +
    // -a0 b2 e2 +
    // -a0 b3 e3

    let dp = hi_dp_ss(a, b);
    let p0 = _mm_mul_ps(swizzle!(a, 0, 0, 0, 0), b);
    let p0 = _mm_xor_ps(p0, _mm_set_ps(-0.0, -0.0, -0.0, 0.0));

    let p0 = if cfg!(target_feature = "sse4.1") {
        _mm_blend_ps(p0, dp, 1)
    } else {
        _mm_add_ss(p0, dp)
    };

    p0
}

// Plane | Ideal Line
#[inline(always)]
pub unsafe fn dot_pil_true(a: __m128, c: __m128) -> __m128 {
    let p0 = hi_dp(a, c);

    p0
}

#[inline(always)]
pub unsafe fn dot_pil_false(a: __m128, c: __m128) -> __m128 {
    let p0 = dot_pil_true(a, c);
    let p0 = _mm_xor_ps(p0, _mm_set_ss(-0.0));

    p0
}

// Plane | Line
#[inline(always)]
pub unsafe fn dot_pl_false(a: __m128 , b: __m128 , c: __m128) -> __m128 {
    // -(a1 c1 + a2 c2 + a3 c3) e0 +
    // (a2 b1 - a1 b2) e3
    // (a3 b2 - a2 b3) e1 +
    // (a1 b3 - a3 b1) e2 +

    let p0 = _mm_mul_ps(swizzle!(a, 1, 3, 2, 0), b);
    let p0 = _mm_sub_ps(p0, _mm_mul_ps(a, swizzle!(b, 1, 3, 2, 0)));
    let p0 = _mm_sub_ss(swizzle!(p0, 1, 3, 2, 0), hi_dp_ss(a, c));

    p0
}

#[inline(always)]
pub unsafe fn dot_pl_true(a: __m128 , b: __m128 , c: __m128) -> __m128 {
    // (a1 c1 + a2 c2 + a3 c3) e0 +
    // (a1 b2 - a2 b1) e3
    // (a2 b3 - a3 b2) e1 +
    // (a3 b1 - a1 b3) e2 +

    let p0 = _mm_mul_ps(a, swizzle!(b, 1, 3, 2, 0));
    let p0 = _mm_sub_ps(p0, _mm_mul_ps(swizzle!(a, 1, 3, 2, 0), b));
    let p0 = _mm_add_ss(swizzle!(p0, 1, 3, 2, 0), hi_dp_ss(a, c));

    p0
}
