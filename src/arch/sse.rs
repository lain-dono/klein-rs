// Provide convenience macros and utilities for invoking x86 SSE

use core::arch::x86_64::*;

// Little-endian XMM register swizzle
//
// KLN_SWIZZLE(reg, 3, 2, 1, 0) is the identity.
//
// This is undef-ed at the bottom of klein.hpp so as not to
// pollute the macro namespace
macro_rules! swizzle {
    ($reg:expr, $x:expr, $y:expr, $z:expr, $w:expr) => {
        core::arch::x86_64::_mm_shuffle_ps(
            $reg,
            $reg,
            core::arch::x86_64::_MM_SHUFFLE($x, $y, $z, $w),
        )
    };
}

// DP high components and caller ignores returned high components
#[inline(always)]
pub unsafe fn hi_dp_ss(a: __m128, b: __m128) -> __m128 {
    // 0 1 2 3 -> 1 + 2 + 3, 0, 0, 0
    let out = _mm_mul_ps(a, b);

    // 0 1 2 3 -> 1 1 3 3
    let hi = _mm_movehdup_ps(out);

    // 0 1 2 3 + 1 1 3 3 -> (0 + 1, 1 + 1, 2 + 3, 3 + 3)
    let sum = _mm_add_ps(hi, out);

    // unpacklo: 0 0 1 1
    let out = _mm_add_ps(sum, _mm_unpacklo_ps(out, out));

    // (1 + 2 + 3, _, _, _)
    _mm_movehl_ps(out, out)
}

// Reciprocal with an additional single Newton-Raphson refinement
#[inline(always)]
pub unsafe fn rcp_nr1(a: __m128) -> __m128 {
    // f(x) = 1/x - a
    // f'(x) = -1/x^2
    // x_{n+1} = x_n - f(x)/f'(x)
    //         = 2x_n - a x_n^2 = x_n (2 - a x_n)

    // ~2.7x baseline with ~22 bits of accuracy
    let xn = _mm_rcp_ps(a);
    let axn = _mm_mul_ps(a, xn);
    _mm_mul_ps(xn, _mm_sub_ps(_mm_set1_ps(2.0), axn))
}

// Reciprocal sqrt with an additional single Newton-Raphson refinement.
#[inline(always)]
pub unsafe fn rsqrt_nr1(a: __m128) -> __m128 {
    // f(x) = 1/x^2 - a
    // f'(x) = -1/(2x^(3/2))
    // Let x_n be the estimate, and x_{n+1} be the refinement
    // x_{n+1} = x_n - f(x)/f'(x)
    //         = 0.5 * x_n * (3 - a x_n^2)

    // From Intel optimization manual: expected performance is ~5.2x
    // baseline (sqrtps + divps) with ~22 bits of accuracy

    let xn = _mm_rsqrt_ps(a);
    let axn2 = _mm_mul_ps(xn, xn);
    let axn2 = _mm_mul_ps(a, axn2);
    let xn3 = _mm_sub_ps(_mm_set1_ps(3.0), axn2);
    _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(0.5), xn), xn3)
}

// Sqrt Newton-Raphson is evaluated in terms of rsqrt_nr1
#[inline(always)]
pub unsafe fn sqrt_nr1(a: __m128) -> __m128 {
    _mm_mul_ps(a, rsqrt_nr1(a))
}

#[cfg(target_feature = "sse4.1")]
#[inline(always)]
pub unsafe fn hi_dp(a: __m128, b: __m128) -> __m128 {
    _mm_dp_ps(a, b, 0b11100001)
}

// Equivalent to _mm_dp_ps(a, b, 0b11100001);
#[cfg(not(target_feature = "sse4.1"))]
#[inline(always)]
pub unsafe fn hi_dp(a: __m128, b: __m128) -> __m128 {
    // 0 1 2 3 -> 1 + 2 + 3, 0, 0, 0

    let out = _mm_mul_ps(a, b);

    // 0 1 2 3 -> 1 1 3 3
    let hi = _mm_movehdup_ps(out);

    // 0 1 2 3 + 1 1 3 3 -> (0 + 1, 1 + 1, 2 + 3, 3 + 3)
    let sum = _mm_add_ps(hi, out);

    // unpacklo: 0 0 1 1
    let out = _mm_add_ps(sum, _mm_unpacklo_ps(out, out));

    // (1 + 2 + 3, _, _, _)
    let out = _mm_movehl_ps(out, out);

    _mm_and_ps(out, _mm_castsi128_ps(_mm_set_epi32(0, 0, 0, -1)))
}

#[cfg(target_feature = "sse4.1")]
#[inline(always)]
pub unsafe fn hi_dp_bc(a: __m128, b: __m128) -> __m128 {
    return _mm_dp_ps(a, b, 0b11101111);
}
#[cfg(not(target_feature = "sse4.1"))]
#[inline(always)]
pub unsafe fn hi_dp_bc(a: __m128, b: __m128) -> __m128 {
    // Multiply across and mask low component
    let out = _mm_mul_ps(a, b);

    // 0 1 2 3 -> 1 1 3 3
    let hi = _mm_movehdup_ps(out);

    // 0 1 2 3 + 1 1 3 3 -> (0 + 1, 1 + 1, 2 + 3, 3 + 3)
    let sum = _mm_add_ps(hi, out);

    // unpacklo: 0 0 1 1
    let out = _mm_add_ps(sum, _mm_unpacklo_ps(out, out));

    swizzle!(out, 2, 2, 2, 2)
}

#[cfg(target_feature = "sse4.1")]
#[inline(always)]
pub unsafe fn dp(a: __m128, b: __m128) -> __m128 {
    _mm_dp_ps(a, b, 0b11110001)
}
#[cfg(not(target_feature = "sse4.1"))]
#[inline(always)]
pub unsafe fn dp(a: __m128, b: __m128) -> __m128 {
    // Multiply across and shift right (shifting in zeros)
    let out = _mm_mul_ps(a, b);
    let hi = _mm_movehdup_ps(out);

    // (a1 b1, a2 b2, a3 b3, 0) + (a2 b2, a2 b2, 0, 0)
    // = (a1 b1 + a2 b2, _, a3 b3, 0)
    let out = _mm_add_ps(hi, out);
    let out = _mm_add_ss(out, _mm_movehl_ps(hi, out));

    _mm_and_ps(out, _mm_castsi128_ps(_mm_set_epi32(0, 0, 0, -1)))
}

#[cfg(target_feature = "sse4.1")]
#[inline(always)]
pub unsafe fn dp_bc(a: __m128, b: __m128) -> __m128 {
    _mm_dp_ps(a, b, 0xff)
}

#[cfg(not(target_feature = "sse4.1"))]
#[inline(always)]
pub unsafe fn dp_bc(a: __m128, b: __m128) -> __m128 {
    // Multiply across and shift right (shifting in zeros)
    let out = _mm_mul_ps(a, b);
    let hi = _mm_movehdup_ps(out);

    // (a1 b1, a2 b2, a3 b3, 0) + (a2 b2, a2 b2, 0, 0)
    // = (a1 b1 + a2 b2, _, a3 b3, 0)
    let out = _mm_add_ps(hi, out);
    let out = _mm_add_ss(out, _mm_movehl_ps(hi, out));

    swizzle!(out, 0, 0, 0, 0)
}
