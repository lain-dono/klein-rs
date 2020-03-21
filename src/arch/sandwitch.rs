// Define functions of the form swAB where A and B are partition indices.
// Each function so-defined computes the sandwich operator using vector
// intrinsics. The partition index determines which basis elements are present
// in each XMM component of the operand.
//
// Notes:
// 1. The first argument is always the TARGET which is the multivector to apply
//    the sandwich operator to.
// 2. The second operator MAY be a bivector or motor (sandwiching with
//    a point or vector isn't supported at this time).
// 3. For efficiency, the sandwich operator is NOT implemented in terms of two
//    geometric products and a reversion. The result is nevertheless equivalent.

use super::{f32x4, sse::*};
use core::arch::x86_64::*;

// Partition memory layouts
//     LSB --> MSB
// p0: (e0, e1, e2, e3)
// p1: (1, e23, e31, e12)
// p2: (e0123, e01, e02, e03)
// p3: (e123, e032, e013, e021)

// Reflect a plane through another plane
// b * a * b
#[inline(always)]
pub unsafe fn sw00(a: __m128, b: __m128, p0_out: &mut __m128) {
    // (2a0(a2 b2 + a3 b3 + a1 b1) - b0(a1^2 + a2^2 + a3^2)) e0 +
    // (2a1(a2 b2 + a3 b3)         + b1(a1^2 - a2^2 - a3^2)) e1 +
    // (2a2(a3 b3 + a1 b1)         + b2(a2^2 - a3^2 - a1^2)) e2 +
    // (2a3(a1 b1 + a2 b2)         + b3(a3^2 - a1^2 - a2^2)) e3

    let a_zzwy = swizzle!(a, 1, 3, 2, 2);
    let a_wwyz = swizzle!(a, 2, 1, 3, 3);

    // Left block
    let tmp = _mm_mul_ps(a_zzwy, swizzle!(b, 1, 3, 2, 2));
    let tmp = _mm_add_ps(tmp, _mm_mul_ps(a_wwyz, swizzle!(b, 2, 1, 3, 3)));

    let a1 = _mm_movehdup_ps(a);
    let b1 = _mm_movehdup_ps(b);
    let tmp = _mm_add_ss(tmp, _mm_mul_ss(a1, b1));
    let tmp = _mm_mul_ps(tmp, _mm_add_ps(a, a));

    // Right block
    let a_yyzw = swizzle!(a, 3, 2, 1, 1);
    let tmp2 = _mm_xor_ps(_mm_mul_ps(a_yyzw, a_yyzw), _mm_set_ss(-0.0));
    let tmp2 = _mm_sub_ps(tmp2, _mm_mul_ps(a_zzwy, a_zzwy));
    let tmp2 = _mm_sub_ps(tmp2, _mm_mul_ps(a_wwyz, a_wwyz));
    let tmp2 = _mm_mul_ps(tmp2, b);

    *p0_out = _mm_add_ps(tmp, tmp2);
}

#[inline(always)]
pub unsafe fn sw10(a: __m128, b: __m128, p1: &mut __m128, p2: &mut __m128) {
    //                       b0(a1^2 + a2^2 + a3^2) +
    // (2a3(a1 b1 + a2 b2) + b3(a3^2 - a1^2 - a2^2)) e12 +
    // (2a1(a2 b2 + a3 b3) + b1(a1^2 - a2^2 - a3^2)) e23 +
    // (2a2(a3 b3 + a1 b1) + b2(a2^2 - a3^2 - a1^2)) e31 +
    //
    // 2a0(a1 b2 - a2 b1) e03
    // 2a0(a2 b3 - a3 b2) e01 +
    // 2a0(a3 b1 - a1 b3) e02 +

    let a_zyzw = swizzle!(a, 3, 2, 1, 2);
    let a_ywyz = swizzle!(a, 2, 1, 3, 1);
    let a_wzwy = swizzle!(a, 1, 3, 2, 3);

    let b_xzwy = swizzle!(b, 1, 3, 2, 0);

    let two_zero = _mm_set_ps(2.0, 2.0, 2.0, 0.0);
    *p1 = _mm_mul_ps(a, b);
    *p1 = _mm_add_ps(*p1, _mm_mul_ps(a_wzwy, b_xzwy));
    *p1 = _mm_mul_ps(*p1, _mm_mul_ps(a_ywyz, two_zero));

    let tmp = _mm_mul_ps(a_zyzw, a_zyzw);
    let tmp = _mm_add_ps(tmp, _mm_mul_ps(a_wzwy, a_wzwy));
    let tmp = _mm_xor_ps(tmp, _mm_set_ss(-0.0));
    let tmp = _mm_sub_ps(_mm_mul_ps(a_ywyz, a_ywyz), tmp);
    let tmp = _mm_mul_ps(swizzle!(b, 2, 1, 3, 0), tmp);

    *p1 = swizzle!(_mm_add_ps(*p1, tmp), 1, 3, 2, 0);

    *p2 = _mm_mul_ps(a_zyzw, b_xzwy);
    *p2 = _mm_sub_ps(*p2, _mm_mul_ps(a_wzwy, b));
    *p2 = _mm_mul_ps(*p2, _mm_mul_ps(swizzle!(a, 0, 0, 0, 0), two_zero));
    *p2 = swizzle!(*p2, 1, 3, 2, 0);
}

#[inline(always)]
pub unsafe fn sw20(a: __m128, b: __m128, p2: &mut __m128) {
    //                       -b0(a1^2 + a2^2 + a3^2) e0123 +
    // (-2a3(a1 b1 + a2 b2) + b3(a1^2 + a2^2 - a3^2)) e03
    // (-2a1(a2 b2 + a3 b3) + b1(a2^2 + a3^2 - a1^2)) e01 +
    // (-2a2(a3 b3 + a1 b1) + b2(a3^2 + a1^2 - a2^2)) e02 +

    let a_zzwy = swizzle!(a, 1, 3, 2, 2);
    let a_wwyz = swizzle!(a, 2, 1, 3, 3);

    *p2 = _mm_mul_ps(a, b);
    *p2 = _mm_add_ps(*p2, _mm_mul_ps(a_zzwy, swizzle!(b, 1, 3, 2, 0)));
    *p2 = _mm_mul_ps(*p2, _mm_mul_ps(a_wwyz, _mm_set_ps(-2.0, -2.0, -2.0, 0.0)));

    let a_yyzw = swizzle!(a, 3, 2, 1, 1);
    let tmp = _mm_mul_ps(a_yyzw, a_yyzw);
    let tmp = _mm_xor_ps(
        _mm_set_ss(-0.0),
        _mm_add_ps(tmp, _mm_mul_ps(a_zzwy, a_zzwy)),
    );
    let tmp = _mm_sub_ps(tmp, _mm_mul_ps(a_wwyz, a_wwyz));
    *p2 = _mm_add_ps(*p2, _mm_mul_ps(tmp, swizzle!(b, 2, 1, 3, 0)));
    *p2 = swizzle!(*p2, 1, 3, 2, 0);
}

#[inline(always)]
pub unsafe fn sw30(a: __m128, b: __m128, p3_out: &mut __m128) {
    //                                b0(a1^2 + a2^2 + a3^2)  e123 +
    // (-2a1(a0 b0 + a3 b3 + a2 b2) + b1(a2^2 + a3^2 - a1^2)) e032 +
    // (-2a2(a0 b0 + a1 b1 + a3 b3) + b2(a3^2 + a1^2 - a2^2)) e013 +
    // (-2a3(a0 b0 + a2 b2 + a1 b1) + b3(a1^2 + a2^2 - a3^2)) e021

    let a_zwyz = swizzle!(a, 2, 1, 3, 2);
    let a_yzwy = swizzle!(a, 1, 3, 2, 1);

    *p3_out = _mm_mul_ps(swizzle!(a, 0, 0, 0, 0), swizzle!(b, 0, 0, 0, 0));
    *p3_out = _mm_add_ps(*p3_out, _mm_mul_ps(a_zwyz, swizzle!(b, 2, 1, 3, 0)));
    *p3_out = _mm_add_ps(*p3_out, _mm_mul_ps(a_yzwy, swizzle!(b, 1, 3, 2, 0)));
    *p3_out = _mm_mul_ps(*p3_out, _mm_mul_ps(a, _mm_set_ps(-2.0, -2.0, -2.0, 0.0)));

    let tmp = _mm_mul_ps(a_yzwy, a_yzwy);
    let tmp = _mm_add_ps(tmp, _mm_mul_ps(a_zwyz, a_zwyz));
    let a_wyzw = swizzle!(a, 3, 2, 1, 3);
    let tmp = _mm_sub_ps(
        tmp,
        _mm_xor_ps(_mm_mul_ps(a_wyzw, a_wyzw), _mm_set_ss(-0.0)),
    );

    *p3_out = _mm_add_ps(*p3_out, _mm_mul_ps(b, tmp));
}

// Apply a translator to a plane.
// Assumes e0123 component of p2 is exactly 0
// p0: (e0, e1, e2, e3)
// p2: (e0123, e01, e02, e03)
// b * a * ~b
// The low component of p2 is expected to be the scalar component instead
#[inline(always)]
pub unsafe fn sw02(a: __m128, b: __m128) -> __m128 {
    // (a0 b0^2 + 2a1 b0 b1 + 2a2 b0 b2 + 2a3 b0 b3) e0 +
    // (a1 b0^2) e1 +
    // (a2 b0^2) e2 +
    // (a3 b0^2) e3
    //
    // Because the plane is projectively equivalent on multiplication by a
    // scalar, we can divide the result through by b0^2
    //
    // (a0 + 2a1 b1 / b0 + 2a2 b2 / b0 + 2a3 b3 / b0) e0 +
    // a1 e1 +
    // a2 e2 +
    // a3 e3
    //
    // The additive term clearly contains a dot product between the plane's
    // normal and the translation axis, demonstrating that the plane
    // "doesn't care" about translations along its span. More precisely, the
    // plane translates by the projection of the translator on the plane's
    // normal.

    // a1*b1 + a2*b2 + a3*b3 stored in the low component of tmp
    let tmp = hi_dp(a, b);

    let inv_b = rcp_nr1(b);
    // 2 / b0
    let inv_b = _mm_add_ss(inv_b, inv_b);
    let inv_b = _mm_and_ps(inv_b, _mm_castsi128_ps(_mm_set_epi32(0, 0, 0, -1)));
    let tmp = _mm_mul_ss(tmp, inv_b);

    // Add to the plane
    _mm_add_ps(a, tmp)
}

// Apply a translator to a line
// a := p1 input
// d := p2 input
// c := p2 translator
// out points to the start address of a line (p1, p2)
#[inline(always)]
pub unsafe fn sw_l2(a: __m128, d: __m128, c: __m128) -> (__m128, __m128) {
    // a0 +
    // a1 e23 +
    // a2 e31 +
    // a3 e12 +
    //
    // (2a0 c0 + d0) e0123 +
    // (2(a2 c3 - a3 c2 - a1 c0) + d1) e01 +
    // (2(a3 c1 - a1 c3 - a2 c0) + d2) e02 +
    // (2(a1 c2 - a2 c1 - a3 c0) + d3) e03

    let p1 = a;

    let p2 = _mm_mul_ps(swizzle!(a, 1, 3, 2, 0), swizzle!(c, 2, 1, 3, 0));

    // Add and subtract the same quantity in the low component to produce a
    // cancellation
    let p2 = _mm_sub_ps(
        p2,
        _mm_mul_ps(swizzle!(a, 2, 1, 3, 0), swizzle!(c, 1, 3, 2, 0)),
    );
    let p2 = _mm_sub_ps(
        p2,
        _mm_xor_ps(_mm_mul_ps(a, swizzle!(c, 0, 0, 0, 0)), _mm_set_ss(-0.0)),
    );
    let p2 = _mm_add_ps(p2, p2);
    let p2 = _mm_add_ps(p2, d);
    (p1, p2)
}

// Apply a translator to a point.
// Assumes e0123 component of p2 is exactly 0
// p2: (e0123, e01, e02, e03)
// p3: (e123, e032, e013, e021)
// b * a * ~b
#[inline(always)]
pub unsafe fn sw32(a: __m128, b: __m128) -> __m128 {
    // a0 e123 +
    // (a1 - 2 a0 b1) e032 +
    // (a2 - 2 a0 b2) e013 +
    // (a3 - 2 a0 b3) e021

    let tmp = _mm_mul_ps(swizzle!(a, 0, 0, 0, 0), b);
    let tmp = _mm_mul_ps(_mm_set_ps(-2.0, -2.0, -2.0, 0.0), tmp);
    _mm_add_ps(a, tmp)
}

// Apply a motor to a motor (works on lines as well)
// in points to the start of an array of motor inputs (alternating p1 and p2)
// out points to the start of an array of motor outputs (alternating p1 and p2)
//
// Note: in and out are permitted to alias iff a == out.

/*
//template <bool Variadic, bool Translate, bool InputP2>
pub unsafe fn sw_mm<'a>(
    input: impl Iterator<Item=&'a __m128>,
    b: __m128,
    c: Option<&'a __m128>,
    out: impl Iterator<Item=&'a mut __m128>,
) {
    // p1 block
    // a0(b0^2 + b1^2 + b2^2 + b3^2) +
    // (a1(b1^2 + b0^2 - b3^2 - b2^2) +
    //     2a2(b0 b3 + b1 b2) + 2a3(b1 b3 - b0 b2)) e23 +
    // (a2(b2^2 + b0^2 - b1^2 - b3^2) +
    //     2a3(b0 b1 + b2 b3) + 2a1(b2 b1 - b0 b3)) e31
    // (a3(b3^2 + b0^2 - b2^2 - b1^2) +
    //     2a1(b0 b2 + b3 b1) + 2a2(b3 b2 - b0 b1)) e12 +

    let b_xwyz   = swizzle!(b, 2, 1, 3, 0);
    let b_xzwy   = swizzle!(b, 1, 3, 2, 0);
    let b_yxxx   = swizzle!(b, 0, 0, 0, 1);
    let b_yxxx_2 = _mm_mul_ps(b_yxxx, b_yxxx);

    let tmp   = _mm_mul_ps(b, b);
    let tmp          = _mm_add_ps(tmp, b_yxxx_2);
    let b_tmp = swizzle!(b, 2, 1, 3, 2);
    let tmp2  = _mm_mul_ps(b_tmp, b_tmp);
    let b_tmp        = swizzle!(b, 1, 3, 2, 3);
    let tmp2         = _mm_add_ps(tmp2, _mm_mul_ps(b_tmp, b_tmp));
    let tmp          = _mm_sub_ps(tmp, _mm_xor_ps(tmp2, _mm_set_ss(-0.0)));
    // tmp needs to be scaled by a and set to p1_out

    let b_xxxx = swizzle!(b, 0, 0, 0, 0);
    let scale  = _mm_set_ps(2.0, 2.0, 2.0, 0.0);
    let tmp2          = _mm_mul_ps(b_xxxx, b_xwyz);
    let tmp2          = _mm_add_ps(tmp2, _mm_mul_ps(b, b_xzwy));
    let tmp2          = _mm_mul_ps(tmp2, scale);
    // tmp2 needs to be scaled by (a0, a2, a3, a1) and added to p1_out

    let tmp3 = _mm_mul_ps(b, b_xwyz);
    let tmp3        = _mm_sub_ps(tmp3, _mm_mul_ps(b_xxxx, b_xzwy));
    let tmp3        = _mm_mul_ps(tmp3, scale);
    // tmp3 needs to be scaled by (a0, a3, a1, a2) and added to p1_out

    // p2 block
    // (d coefficients are the components of the input line p2)
    // (2a0(b0 c0 - b1 c1 - b2 c2 - b3 c3) +
    //  d0(b1^2 + b0^2 + b2^2 + b3^2)) e0123 +
    //
    // (2a1(b1 c1 - b0 c0 - b3 c3 - b2 c2) +
    //  2a3(b1 c3 + b2 c0 + b3 c1 - b0 c2) +
    //  2a2(b1 c2 + b0 c3 + b2 c1 - b3 c0) +
    //  2d2(b0 b3 + b2 b1) +
    //  2d3(b1 b3 - b0 b2) +
    //  d1(b0^2 + b1^2 - b3^2 - b2^2)) e01 +
    //
    // (2a2(b2 c2 - b0 c0 - b3 c3 - b1 c1) +
    //  2a1(b2 c1 + b3 c0 + b1 c2 - b0 c3) +
    //  2a3(b2 c3 + b0 c1 + b3 c2 - b1 c0) +
    //  2d3(b0 b1 + b3 b2) +
    //  2d1(b2 b1 - b0 b3) +
    //  d2(b0^2 + b2^2 - b1^2 - b3^2)) e02 +
    //
    // (2a3(b3 c3 - b0 c0 - b1 c1 - b2 c2) +
    //  2a2(b3 c2 + b1 c0 + b2 c3 - b0 c1) +
    //  2a1(b3 c1 + b0 c2 + b1 c3 - b2 c0) +
    //  2d1(b0 b2 + b1 b3) +
    //  2d2(b3 b2 - b0 b1) +
    //  d3(b0^2 + b3^2 - b2^2 - b1^2)) e03

    // Rotation

    // tmp scaled by d and added to p2
    // tmp2 scaled by (d0, d2, d3, d1) and added to p2
    // tmp3 scaled by (d0, d3, d1, d2) and added to p2

    // Translation
    [[maybe_unused]] __m128 tmp4; // scaled by a and added to p2
    [[maybe_unused]] __m128 tmp5; // scaled by (a0, a3, a1, a2), added to p2
    [[maybe_unused]] __m128 tmp6; // scaled by (a0, a2, a3, a1), added to p2

    let translate = c.is_some();

    let (tmp4, tmp5, tmp6) = if let Some(c) = c{
        __m128 czero  = swizzle!(*c, 0, 0, 0, 0);
        __m128 c_xzwy = swizzle!(*c, 1, 3, 2, 0);
        __m128 c_xwyz = swizzle!(*c, 2, 1, 3, 0);

        tmp4 = _mm_mul_ps(b, *c);
        tmp4 = _mm_sub_ps(
            tmp4, _mm_mul_ps(b_yxxx, swizzle!(*c, 0, 0, 0, 1)));
        tmp4 = _mm_sub_ps(tmp4,
                            _mm_mul_ps(swizzle!(b, 1, 3, 3, 2),
                                        swizzle!(*c, 1, 3, 3, 2)));
        tmp4 = _mm_sub_ps(tmp4,
                            _mm_mul_ps(swizzle!(b, 2, 1, 2, 3),
                                        swizzle!(*c, 2, 1, 2, 3)));
        tmp4 = _mm_add_ps(tmp4, tmp4);

        tmp5 = _mm_mul_ps(b, c_xwyz);
        tmp5 = _mm_add_ps(tmp5, _mm_mul_ps(b_xzwy, czero));
        tmp5 = _mm_add_ps(tmp5, _mm_mul_ps(b_xwyz, *c));
        tmp5 = _mm_sub_ps(tmp5, _mm_mul_ps(b_xxxx, c_xzwy));
        tmp5 = _mm_mul_ps(tmp5, scale);

        tmp6 = _mm_mul_ps(b, c_xzwy);
        tmp6 = _mm_add_ps(tmp6, _mm_mul_ps(b_xxxx, c_xwyz));
        tmp6 = _mm_add_ps(tmp6, _mm_mul_ps(b_xzwy, *c));
        tmp6 = _mm_sub_ps(tmp6, _mm_mul_ps(b_xwyz, czero));
        tmp6 = _mm_mul_ps(tmp6, scale);
    } else {
        core::mem::uninitialized()
    };

    /*
    size_t limit            = Variadic ? count : 1;
    constexpr size_t stride = InputP2 ? 2 : 1;
    for (size_t i = 0; i != limit; ++i)
    {
        __m128 const& p1_in = in[stride * i]; // a
        __m128 p1_in_xzwy   = swizzle!(p1_in, 1, 3, 2, 0);
        __m128 p1_in_xwyz   = swizzle!(p1_in, 2, 1, 3, 0);

        __m128& p1_out = out[stride * i];

        p1_out = _mm_mul_ps(tmp, p1_in);
        p1_out = _mm_add_ps(p1_out, _mm_mul_ps(tmp2, p1_in_xzwy));
        p1_out = _mm_add_ps(p1_out, _mm_mul_ps(tmp3, p1_in_xwyz));

        if input_p2 {
            let const& p2_in = in[2 * i + 1]; // d
            __m128& p2_out      = out[2 * i + 1];
            p2_out              = _mm_mul_ps(tmp, p2_in);
            p2_out              = _mm_add_ps(
                p2_out, _mm_mul_ps(tmp2, swizzle!(p2_in, 1, 3, 2, 0)));
            p2_out = _mm_add_ps(
                p2_out, _mm_mul_ps(tmp3, swizzle!(p2_in, 2, 1, 3, 0)));
        }

        // If what is being applied is a rotor, the non-directional
        // components of the line are left untouched
        if translate {
            __m128& p2_out = out[2 * i + 1];
            p2_out         = _mm_add_ps(p2_out, _mm_mul_ps(tmp4, p1_in));
            p2_out = _mm_add_ps(p2_out, _mm_mul_ps(tmp5, p1_in_xwyz));
            p2_out = _mm_add_ps(p2_out, _mm_mul_ps(tmp6, p1_in_xzwy));
        }
    }
*/
}
*/

// Apply a motor to a plane
// a := p0
// b := p1
// c := p2
// If Translate is false, c is ignored (rotor application).
// If Variadic is true, a and out must point to a contiguous block of memory
// equivalent to __m128[count]
//template <bool Variadic = false, bool Translate = true>
#[inline(always)]
pub unsafe fn sw012<'a>(
    a: impl Iterator<Item = &'a __m128>,
    b: __m128,
    c: Option<&__m128>,
    out: impl Iterator<Item = &'a mut __m128>,
) {
    // LSB
    //
    // (2a3(b0 c3 + b1 c2 + b3 c0 - b2 c1) +
    //  2a2(b0 c2 + b3 c1 + b2 c0 - b1 c3) +
    //  2a1(b0 c1 + b2 c3 + b1 c0 - b3 c2) +
    //  a0 (b2^2 + b1^2 + b0^2 + b3^2)) e0 +
    //
    // (2a2(b0 b3 + b2 b1) +
    //  2a3(b1 b3 - b0 b2) +
    //  a1 (b0^2 + b1^2 - b3^2 - b2^2)) e1 +
    //
    // (2a3(b0 b1 + b3 b2) +
    //  2a1(b2 b1 - b0 b3) +
    //  a2 (b0^2 + b2^2 - b1^2 - b3^2)) e2 +
    //
    // (2a1(b0 b2 + b1 b3) +
    //  2a2(b3 b2 - b0 b1) +
    //  a3 (b0^2 + b3^2 - b2^2 - b1^2)) e3
    //
    // MSB
    //
    // Note the similarity between the results here and the rotor and
    // translator applied to the plane. The e1, e2, and e3 components do not
    // participate in the translation and are identical to the result after
    // the rotor was applied to the plane. The e0 component is displaced
    // similarly to the manner in which it is displaced after application of
    // a translator.

    // Double-cover scale
    let dc_scale = _mm_set_ps(2.0, 2.0, 2.0, 1.0);
    let b_xwyz = swizzle!(b, 2, 1, 3, 0);
    let b_xzwy = swizzle!(b, 1, 3, 2, 0);
    let b_xxxx = swizzle!(b, 0, 0, 0, 0);

    let tmp1 = _mm_mul_ps(swizzle!(b, 0, 0, 0, 2), swizzle!(b, 2, 1, 3, 2));
    let tmp1 = _mm_add_ps(
        tmp1,
        _mm_mul_ps(swizzle!(b, 1, 3, 2, 1), swizzle!(b, 3, 2, 1, 1)),
    );
    // Scale later with (a0, a2, a3, a1)
    let tmp1 = _mm_mul_ps(tmp1, dc_scale);

    let tmp2 = _mm_mul_ps(b, b_xwyz);

    let tmp2 = _mm_sub_ps(
        tmp2,
        _mm_xor_ps(
            _mm_set_ss(-0.0),
            _mm_mul_ps(swizzle!(b, 0, 0, 0, 3), swizzle!(b, 1, 3, 2, 3)),
        ),
    );
    // Scale later with (a0, a3, a1, a2)
    let tmp2 = _mm_mul_ps(tmp2, dc_scale);

    // Alternately add and subtract to improve low component stability
    let tmp3 = _mm_mul_ps(b, b);
    let tmp3 = _mm_sub_ps(tmp3, _mm_mul_ps(b_xwyz, b_xwyz));
    let tmp3 = _mm_add_ps(tmp3, _mm_mul_ps(b_xxxx, b_xxxx));
    let tmp3 = _mm_sub_ps(tmp3, _mm_mul_ps(b_xzwy, b_xzwy));
    // Scale later with a

    // Compute
    // 0 * _ +
    // 2a1(b0 c1 + b2 c3 + b1 c0 - b3 c2) +
    // 2a2(b0 c2 + b3 c1 + b2 c0 - b1 c3) +
    // 2a3(b0 c3 + b1 c2 + b3 c0 - b2 c1)
    // by decomposing into four vectors, factoring out the a components

    let translate = c.is_some();
    let tmp4 = if let Some(c) = c {
        let tmp4 = _mm_mul_ps(b_xxxx, *c);
        let tmp4 = _mm_add_ps(tmp4, _mm_mul_ps(b_xzwy, swizzle!(*c, 2, 1, 3, 0)));
        let tmp4 = _mm_add_ps(tmp4, _mm_mul_ps(b, swizzle!(*c, 0, 0, 0, 0)));

        // NOTE: The high component of tmp4 is meaningless here
        let tmp4 = _mm_sub_ps(tmp4, _mm_mul_ps(b_xwyz, swizzle!(*c, 1, 3, 2, 0)));
        _mm_mul_ps(tmp4, dc_scale)
    } else {
        core::mem::uninitialized()
    };

    // The temporaries (tmp1, tmp2, tmp3, tmp4)
    // strictly only have a dependence on b and c.

    for (a, p) in a.zip(out) {
        // Compute the lower block for components e1, e2, and e3
        *p = _mm_mul_ps(tmp1, swizzle!(*a, 1, 3, 2, 0));
        *p = _mm_add_ps(*p, _mm_mul_ps(tmp2, swizzle!(*a, 2, 1, 3, 0)));
        *p = _mm_add_ps(*p, _mm_mul_ps(tmp3, *a));

        if translate {
            let tmp5 = hi_dp(tmp4, *a);
            *p = _mm_add_ps(*p, tmp5);
        }
    }
}

// Apply a motor to a point
//template <bool Variadic, bool Translate>
pub unsafe fn sw312<'a>(
    a: impl Iterator<Item = &'a __m128>,
    b: __m128,
    c: Option<&'a __m128>,
    out: impl Iterator<Item = &'a mut __m128>,
) {
    // LSB
    // a0(b1^2 + b0^2 + b2^2 + b3^2) e123 +
    //
    // (2a0(b2 c3 - b0 c1 - b3 c2 - b1 c0) +
    //  2a3(b1 b3 - b0 b2) +
    //  2a2(b0 b3 +  b2 b1) +
    //  a1(b0^2 + b1^2 - b3^2 - b2^2)) e032
    //
    // (2a0(b3 c1 - b0 c2 - b1 c3 - b2 c0) +
    //  2a1(b2 b1 - b0 b3) +
    //  2a3(b0 b1 + b3 b2) +
    //  a2(b0^2 + b2^2 - b1^2 - b3^2)) e013 +
    //
    // (2a0(b1 c2 - b0 c3 - b2 c1 - b3 c0) +
    //  2a2(b3 b2 - b0 b1) +
    //  2a1(b0 b2 + b1 b3) +
    //  a3(b0^2 + b3^2 - b2^2 - b1^2)) e021 +
    // MSB
    //
    // Sanity check: For c1 = c2 = c3 = 0, the computation becomes
    // indistinguishable from a rotor application and the homogeneous
    // coordinate a0 does not participate. As an additional sanity check,
    // note that for a normalized rotor and homogenous point, the e123
    // component will remain unity.

    let two = _mm_set_ps(2.0, 2.0, 2.0, 0.0);
    let b_xxxx = swizzle!(b, 0, 0, 0, 0);
    let b_xwyz = swizzle!(b, 2, 1, 3, 0);
    let b_xzwy = swizzle!(b, 1, 3, 2, 0);

    let tmp1 = _mm_mul_ps(b, b_xwyz);
    let tmp1 = _mm_sub_ps(tmp1, _mm_mul_ps(b_xxxx, b_xzwy));
    let tmp1 = _mm_mul_ps(tmp1, two);
    // tmp1 needs to be scaled by (_, a3, a1, a2)

    let tmp2 = _mm_mul_ps(b_xxxx, b_xwyz);
    let tmp2 = _mm_add_ps(tmp2, _mm_mul_ps(b_xzwy, b));
    let tmp2 = _mm_mul_ps(tmp2, two);
    // tmp2 needs to be scaled by (_, a2, a3, a1)

    let tmp3 = _mm_mul_ps(b, b);
    let b_tmp = swizzle!(b, 0, 0, 0, 1);
    let tmp3 = _mm_add_ps(tmp3, _mm_mul_ps(b_tmp, b_tmp));
    let b_tmp = swizzle!(b, 2, 1, 3, 2);
    let tmp4 = _mm_mul_ps(b_tmp, b_tmp);
    let b_tmp = swizzle!(b, 1, 3, 2, 3);
    let tmp4 = _mm_add_ps(tmp4, _mm_mul_ps(b_tmp, b_tmp));
    let tmp3 = _mm_sub_ps(tmp3, _mm_xor_ps(tmp4, _mm_set_ss(-0.0)));
    // tmp3 needs to be scaled by (a0, a1, a2, a3)

    let translate = c.is_some();
    let tmp4 = if let Some(c) = c {
        let tmp4 = _mm_mul_ps(b_xzwy, swizzle!(*c, 2, 1, 3, 0));
        let tmp4 = _mm_sub_ps(tmp4, _mm_mul_ps(b_xxxx, *c));
        let tmp4 = _mm_sub_ps(tmp4, _mm_mul_ps(b_xwyz, swizzle!(*c, 1, 3, 2, 0)));
        let tmp4 = _mm_sub_ps(tmp4, _mm_mul_ps(b, swizzle!(*c, 0, 0, 0, 0)));

        // Mask low component and scale other components by 2
        let tmp4 = _mm_mul_ps(tmp4, two);
        // tmp4 needs to be scaled by (_, a0, a0, a0)
        tmp4
    } else {
        core::mem::uninitialized()
    };

    for (a, p) in a.zip(out) {
        *p = _mm_mul_ps(tmp1, swizzle!(*a, 2, 1, 3, 0));
        *p = _mm_add_ps(*p, _mm_mul_ps(tmp2, swizzle!(*a, 1, 3, 2, 0)));
        *p = _mm_add_ps(*p, _mm_mul_ps(tmp3, *a));

        if translate {
            *p = _mm_add_ps(*p, _mm_mul_ps(tmp4, swizzle!(*a, 0, 0, 0, 0)));
        }
    }
}

// Conjugate origin with motor. Unlike other operations the motor MUST be
// normalized prior to usage b is the rotor component (p1) c is the
// translator component (p2)
pub unsafe fn swo12(b: __m128, c: __m128) -> __m128 {
    //  (b0^2 + b1^2 + b2^2 + b3^2) e123 +
    // 2(b2 c3 - b1 c0 - b0 c1 - b3 c2) e032 +
    // 2(b3 c1 - b2 c0 - b0 c2 - b1 c3) e013 +
    // 2(b1 c2 - b3 c0 - b0 c3 - b2 c1) e021

    let tmp = _mm_mul_ps(b, swizzle!(c, 0, 0, 0, 0));
    let tmp = _mm_add_ps(tmp, _mm_mul_ps(swizzle!(b, 0, 0, 0, 0), c));
    let tmp = _mm_add_ps(
        tmp,
        _mm_mul_ps(swizzle!(b, 2, 1, 3, 0), swizzle!(c, 1, 3, 2, 0)),
    );
    let tmp = _mm_sub_ps(
        _mm_mul_ps(swizzle!(b, 1, 3, 2, 0), swizzle!(c, 2, 1, 3, 0)),
        tmp,
    );
    let tmp = _mm_mul_ps(tmp, _mm_set_ps(2.0, 2.0, 2.0, 0.0));

    // b0^2 + b1^2 + b2^2 + b3^2 assumed to equal 1
    // Set the low component to unity
    _mm_add_ps(tmp, _mm_set_ss(1.0))
}
