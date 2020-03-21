use super::{f32x4, sse::*};
use core::arch::x86_64::*;

// Partition memory layouts
//     LSB --> MSB
// p0: (e0, e1, e2, e3)
// p1: (1, e23, e31, e12)
// p2: (e0123, e01, e02, e03)
// p3: (e123, e032, e013, e021)

#[inline(always)]
pub unsafe fn ext00(a: __m128, b: __m128) -> (__m128, __m128) {
    // (a1 b2 - a2 b1) e12 +
    // (a2 b3 - a3 b2) e23 +
    // (a3 b1 - a1 b3) e31 +

    // (a0 b1 - a1 b0) e01 +
    // (a0 b2 - a2 b0) e02 +
    // (a0 b3 - a3 b0) e03

    let p1 = _mm_mul_ps(a, swizzle!(b, 1, 3, 2, 0));
    let p1 = swizzle!(
        _mm_sub_ps(p1, _mm_mul_ps(swizzle!(a, 1, 3, 2, 0), b)),
        1,
        3,
        2,
        0
    );

    let p2 = _mm_mul_ps(swizzle!(a, 0, 0, 0, 0), b);
    let p2 = _mm_sub_ps(p2, _mm_mul_ps(a, swizzle!(b, 0, 0, 0, 0)));

    // For both outputs above, we don't zero the lowest component because
    // we've arranged a cancelation

    (p1, p2)
}

// Plane ^ Branch (branch is a line through the origin)
#[inline(always)]
pub unsafe fn ext_pb(a: __m128, b: __m128) -> __m128 {
    // (a1 b1 + a2 b2 + a3 b3) e123 +
    // (-a0 b1) e032 +
    // (-a0 b2) e013 +
    // (-a0 b3) e021
    let p3 = _mm_mul_ps(
        _mm_mul_ps(swizzle!(a, 0, 0, 0, 1), b),
        _mm_set_ps(-1.0, -1.0, -1.0, 0.0),
    );

    _mm_add_ss(p3, hi_dp(a, b))
}

// p0 ^ p2 = p2 ^ p0
#[inline(always)]
pub unsafe fn ext02(a: __m128, b: __m128) -> __m128 {
    // (a1 b2 - a2 b1) e021
    // (a2 b3 - a3 b2) e032 +
    // (a3 b1 - a1 b3) e013 +

    let p3 = _mm_mul_ps(a, swizzle!(b, 1, 3, 2, 0));
    swizzle!(
        _mm_sub_ps(p3, _mm_mul_ps(swizzle!(a, 1, 3, 2, 0), b)),
        1,
        3,
        2,
        0
    )
}

#[inline(always)]
pub unsafe fn ext03_false(a: __m128, b: __m128) -> f32 {
    // (a0 b0 + a1 b1 + a2 b2 + a3 b3) e0123
    let p2 = dp(a, b);

    let mut q = 0.0;
    _mm_store_ss(&mut q, p2);
    q
}

// p0 ^ p3 = -p3 ^ p0
#[inline(always)]
pub unsafe fn ext03_true(a: __m128, b: __m128) -> f32 {
    let p2 = _mm_xor_ps(dp(a, b), _mm_set_ss(-0.0));

    let mut q = 0.0;
    _mm_store_ss(&mut q, p2);
    q
}
// The exterior products p2 ^ p2, p2 ^ p3, p3 ^ p2, and p3 ^ p3 all vanish
