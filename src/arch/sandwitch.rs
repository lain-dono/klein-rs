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
pub fn sw00(a: f32x4, b: f32x4) -> f32x4 {
    // (2a0(a2 b2 + a3 b3 + a1 b1) - b0(a1^2 + a2^2 + a3^2)) e0 +
    // (2a1(a2 b2 + a3 b3)         + b1(a1^2 - a2^2 - a3^2)) e1 +
    // (2a2(a3 b3 + a1 b1)         + b2(a2^2 - a3^2 - a1^2)) e2 +
    // (2a3(a1 b1 + a2 b2)         + b3(a3^2 - a1^2 - a2^2)) e3

    let a_zzwy = shuffle!(a, [1, 3, 2, 2]);
    let a_wwyz = shuffle!(a, [2, 1, 3, 3]);
    let a_yyzw = shuffle!(a, [3, 2, 1, 1]);

    // Left block
    let left = a_zzwy * shuffle!(b, [1, 3, 2, 2]) + a_wwyz * shuffle!(b, [2, 1, 3, 3]);
    let left = left.add0(a.movehdup().mul0(b.movehdup())) * (a + a);

    // Right block
    let right = (a_yyzw * a_yyzw) ^ f32x4::set0(-0.0);
    let right = right - a_zzwy * a_zzwy - a_wwyz * a_wwyz;

    left + right * b
}

#[inline(always)]
pub fn sw10(a: f32x4, b: f32x4) -> (f32x4, f32x4) {
    //                       b0(a1^2 + a2^2 + a3^2) +
    // (2a3(a1 b1 + a2 b2) + b3(a3^2 - a1^2 - a2^2)) e12 +
    // (2a1(a2 b2 + a3 b3) + b1(a1^2 - a2^2 - a3^2)) e23 +
    // (2a2(a3 b3 + a1 b1) + b2(a2^2 - a3^2 - a1^2)) e31 +
    //
    // 2a0(a1 b2 - a2 b1) e03
    // 2a0(a2 b3 - a3 b2) e01 +
    // 2a0(a3 b1 - a1 b3) e02 +

    let a_zyzw = shuffle!(a, [3, 2, 1, 2]);
    let a_ywyz = shuffle!(a, [2, 1, 3, 1]);
    let a_wzwy = shuffle!(a, [1, 3, 2, 3]);

    let b_xzwy = shuffle!(b, [1, 3, 2, 0]);

    let two_zero = f32x4::new(2.0, 2.0, 2.0, 0.0);

    let p1 = (a_zyzw * a_zyzw + a_wzwy * a_wzwy) ^ f32x4::set0(-0.0);
    let p1 = (a_ywyz * a_ywyz - p1) * shuffle!(b, [2, 1, 3, 0]);
    let p1 = (a * b + a_wzwy * b_xzwy) * a_ywyz * two_zero + p1;
    let p1 = shuffle!(p1, [1, 3, 2, 0]);

    let p2 = (a_zyzw * b_xzwy - a_wzwy * b) * shuffle!(a, [0, 0, 0, 0]) * two_zero;
    let p2 = shuffle!(p2, [1, 3, 2, 0]);

    (p1, p2)
}

#[inline(always)]
pub fn sw20(a: f32x4, b: f32x4) -> f32x4 {
    //                       -b0(a1^2 + a2^2 + a3^2) e0123 +
    // (-2a3(a1 b1 + a2 b2) + b3(a1^2 + a2^2 - a3^2)) e03
    // (-2a1(a2 b2 + a3 b3) + b1(a2^2 + a3^2 - a1^2)) e01 +
    // (-2a2(a3 b3 + a1 b1) + b2(a3^2 + a1^2 - a2^2)) e02 +

    let a_zzwy = shuffle!(a, [1, 3, 2, 2]);
    let a_wwyz = shuffle!(a, [2, 1, 3, 3]);

    let p2 = a * b;
    let p2 = p2 + a_zzwy * shuffle!(b, [1, 3, 2, 0]);
    let p2 = p2 * a_wwyz * f32x4::new(-2.0, -2.0, -2.0, 0.0);

    let a_yyzw = shuffle!(a, [3, 2, 1, 1]);
    let tmp = a_yyzw * a_yyzw;
    let tmp = f32x4::set0(-0.0) ^ (tmp + a_zzwy * a_zzwy);
    let tmp = tmp - a_wwyz * a_wwyz;
    let p2 = p2 + tmp * shuffle!(b, [2, 1, 3, 0]);
    shuffle!(p2, [1, 3, 2, 0])
}

#[inline(always)]
pub fn sw30(a: f32x4, b: f32x4) -> f32x4 {
    //                                b0(a1^2 + a2^2 + a3^2)  e123 +
    // (-2a1(a0 b0 + a3 b3 + a2 b2) + b1(a2^2 + a3^2 - a1^2)) e032 +
    // (-2a2(a0 b0 + a1 b1 + a3 b3) + b2(a3^2 + a1^2 - a2^2)) e013 +
    // (-2a3(a0 b0 + a2 b2 + a1 b1) + b3(a1^2 + a2^2 - a3^2)) e021

    let a_zwyz = shuffle!(a, [2, 1, 3, 2]);
    let a_yzwy = shuffle!(a, [1, 3, 2, 1]);
    let a_wyzw = shuffle!(a, [3, 2, 1, 3]);

    let p3 = shuffle!(a, [0, 0, 0, 0]) * shuffle!(b, [0, 0, 0, 0]);

    let p3 = p3 + a_zwyz * shuffle!(b, [2, 1, 3, 0]) + a_yzwy * shuffle!(b, [1, 3, 2, 0]);
    let p3 = p3 * a * f32x4::new(-2.0, -2.0, -2.0, 0.0);

    p3 + b * (a_yzwy * a_yzwy + a_zwyz * a_zwyz - ((a_wyzw * a_wyzw) ^ f32x4::set0(-0.0)))
}

// Apply a translator to a plane.
// Assumes e0123 component of p2 is exactly 0
// p0: (e0, e1, e2, e3)
// p2: (e0123, e01, e02, e03)
// b * a * ~b
// The low component of p2 is expected to be the scalar component instead
#[inline(always)]
pub fn sw02(a: f32x4, b: f32x4) -> f32x4 {
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
    // 2 / b0
    // Add to the plane

    let inv_b = b.rcp_nr1();
    let inv_b = inv_b.add0(inv_b) & f32x4::cast_i32(0, 0, 0, -1);
    a + f32x4::hi_dp(a, b).mul0(inv_b)
}

// Apply a translator to a line
// a := p1 input
// d := p2 input
// c := p2 translator
// out points to the start address of a line (p1, p2)
#[inline(always)]
pub fn sw_l2(a: f32x4, d: f32x4, c: f32x4) -> (f32x4, f32x4) {
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
    let p2 = shuffle!(a, [1, 3, 2, 0]) * shuffle!(c, [2, 1, 3, 0]);

    // Add and subtract the same quantity in the low component to produce a
    // cancellation
    let p2 = p2 - shuffle!(a, [2, 1, 3, 0]) * shuffle!(c, [1, 3, 2, 0]);
    let p2 = p2 - ((a * shuffle!(c, [0, 0, 0, 0])) ^ f32x4::set0(-0.0));
    let p2 = p2 + p2 + d;

    (p1, p2)
}

// Apply a translator to a point.
// Assumes e0123 component of p2 is exactly 0
// p2: (e0123, e01, e02, e03)
// p3: (e123, e032, e013, e021)
// b * a * ~b
#[inline(always)]
pub fn sw32(a: f32x4, b: f32x4) -> f32x4 {
    // a0 e123 +
    // (a1 - 2 a0 b1) e032 +
    // (a2 - 2 a0 b2) e013 +
    // (a3 - 2 a0 b3) e021

    a + f32x4::new(-2.0, -2.0, -2.0, 0.0) * shuffle!(a, [0, 0, 0, 0]) * b
}

// Apply a motor to a motor (works on lines as well)
// in points to the start of an array of motor inputs (alternating p1 and p2)
// out points to the start of an array of motor outputs (alternating p1 and p2)
//
// Note: in and out are permitted to alias iff a == out.

//template <bool Variadic, bool Translate, bool InputP2>
pub fn sw_mm11(input: impl Iterator<Item = f32x4>, b: f32x4) -> impl Iterator<Item = f32x4> {
    // p1 block
    // a0(b0^2 + b1^2 + b2^2 + b3^2) +
    // (a1(b1^2 + b0^2 - b3^2 - b2^2) +
    //     2a2(b0 b3 + b1 b2) + 2a3(b1 b3 - b0 b2)) e23 +
    // (a2(b2^2 + b0^2 - b1^2 - b3^2) +
    //     2a3(b0 b1 + b2 b3) + 2a1(b2 b1 - b0 b3)) e31
    // (a3(b3^2 + b0^2 - b2^2 - b1^2) +
    //     2a1(b0 b2 + b3 b1) + 2a2(b3 b2 - b0 b1)) e12 +

    let b_xwyz = shuffle!(b, [2, 1, 3, 0]);
    let b_xzwy = shuffle!(b, [1, 3, 2, 0]);
    let b_yxxx = shuffle!(b, [0, 0, 0, 1]);
    let b_yxxx_2 = b_yxxx * b_yxxx;

    let b_tmp = shuffle!(b, [2, 1, 3, 2]);
    let b_tmp2 = b_tmp * b_tmp;
    let b_tmp = shuffle!(b, [1, 3, 2, 3]);
    let b_tmp2 = b_tmp2 + b_tmp * b_tmp;
    let tmp = b * b + b_yxxx_2 - (b_tmp2 ^ f32x4::set0(-0.0));
    // tmp needs to be scaled by a and set to p1_out

    let b_xxxx = shuffle!(b, [0, 0, 0, 0]);
    let scale = f32x4::new(2.0, 2.0, 2.0, 0.0);
    let tmp2 = (b_xxxx * b_xwyz + b * b_xzwy) * scale;
    // tmp2 needs to be scaled by (a0, a2, a3, a1) and added to p1_out

    let tmp3 = (b * b_xwyz - b_xxxx * b_xzwy) * scale;
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
    input.map(move |p1| {
        let p1_xzwy = shuffle!(p1, [1, 3, 2, 0]);
        let p1_xwyz = shuffle!(p1, [2, 1, 3, 0]);
        tmp * p1 + tmp2 * p1_xzwy + tmp3 * p1_xwyz
    })
}

pub fn sw_mm22<'a>(
    input: impl Iterator<Item = (&'a f32x4, &'a f32x4)>,
    b: f32x4,
    c: Option<&'a f32x4>,
    output: impl Iterator<Item = (&'a mut f32x4, &'a mut f32x4)>,
) {
    // p1 block
    // a0(b0^2 + b1^2 + b2^2 + b3^2) +
    // (a1(b1^2 + b0^2 - b3^2 - b2^2) +
    //     2a2(b0 b3 + b1 b2) + 2a3(b1 b3 - b0 b2)) e23 +
    // (a2(b2^2 + b0^2 - b1^2 - b3^2) +
    //     2a3(b0 b1 + b2 b3) + 2a1(b2 b1 - b0 b3)) e31
    // (a3(b3^2 + b0^2 - b2^2 - b1^2) +
    //     2a1(b0 b2 + b3 b1) + 2a2(b3 b2 - b0 b1)) e12 +

    let b_xwyz = shuffle!(b, [2, 1, 3, 0]);
    let b_xzwy = shuffle!(b, [1, 3, 2, 0]);
    let b_yxxx = shuffle!(b, [0, 0, 0, 1]);
    let b_yxxx_2 = b_yxxx * b_yxxx;

    let b_tmp = shuffle!(b, [2, 1, 3, 2]);
    let b_tmp2 = b_tmp * b_tmp;
    let b_tmp = shuffle!(b, [1, 3, 2, 3]);
    let b_tmp2 = b_tmp2 + b_tmp * b_tmp;
    let tmp = b * b + b_yxxx_2 - (b_tmp2 ^ f32x4::set0(-0.0));
    // tmp needs to be scaled by a and set to p1_out

    let b_xxxx = shuffle!(b, [0, 0, 0, 0]);
    let scale = f32x4::new(2.0, 2.0, 2.0, 0.0);
    let tmp2 = (b_xxxx * b_xwyz + b * b_xzwy) * scale;
    // tmp2 needs to be scaled by (a0, a2, a3, a1) and added to p1_out

    let tmp3 = (b * b_xwyz - b_xxxx * b_xzwy) * scale;
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

    let translate = c.is_some();

    // tmp4; // scaled by a and added to p2
    // tmp5; // scaled by (a0, a3, a1, a2), added to p2
    // tmp6; // scaled by (a0, a2, a3, a1), added to p2
    let (tmp4, tmp5, tmp6) = if let Some(&c) = c {
        let czero = shuffle!(c, [0, 0, 0, 0]);
        let c_xzwy = shuffle!(c, [1, 3, 2, 0]);
        let c_xwyz = shuffle!(c, [2, 1, 3, 0]);

        let tmp4 = b * c;
        let tmp4 = tmp4 - b_yxxx * shuffle!(c, [0, 0, 0, 1]);
        let tmp4 = tmp4 - shuffle!(b, [1, 3, 3, 2]) * shuffle!(c, [1, 3, 3, 2]);
        let tmp4 = tmp4 - shuffle!(b, [2, 1, 2, 3]) * shuffle!(c, [2, 1, 2, 3]);
        let tmp4 = tmp4 + tmp4;

        let tmp5 = (b * c_xwyz + b_xzwy * czero + b_xwyz * c - b_xxxx * c_xzwy) * scale;
        let tmp6 = (b * c_xzwy + b_xxxx * c_xwyz + b_xzwy * c - b_xwyz * czero) * scale;

        (tmp4, tmp5, tmp6)
    } else {
        unsafe { core::mem::uninitialized() }
    };

    for ((&p1_in, &p2_in), output) in input.zip(output) {
        let (p1_out, p2_out) = (output.0, output.1);

        let p1_in_xzwy = shuffle!(p1_in, [1, 3, 2, 0]);
        let p1_in_xwyz = shuffle!(p1_in, [2, 1, 3, 0]);

        let p2_in_xzwy = shuffle!(p2_in, [1, 3, 2, 0]);
        let p2_in_xwyz = shuffle!(p2_in, [2, 1, 3, 0]);

        *p1_out = tmp * p1_in + tmp2 * p1_in_xzwy + tmp3 * p1_in_xwyz;
        *p2_out = tmp * p2_in + tmp2 * p2_in_xzwy + tmp3 * p2_in_xwyz;

        // If what is being applied is a rotor, the non-directional
        // components of the line are left untouched
        if translate {
            *p2_out = *p2_out + tmp4 * p1_in;
            *p2_out = *p2_out + tmp5 * p1_in_xwyz;
            *p2_out = *p2_out + tmp6 * p1_in_xzwy;
        }
    }
}

// Apply a motor to a plane
// a := p0
// b := p1
// c := p2
// If Translate is false, c is ignored (rotor application).
// If Variadic is true, a and out must point to a contiguous block of memory
// equivalent to __m128[count]
//template <bool Variadic = false, bool Translate = true>
#[inline(always)]
pub fn sw012<'a>(
    a: impl Iterator<Item = &'a f32x4>,
    b: f32x4,
    c: Option<&f32x4>,
    out: impl Iterator<Item = &'a mut f32x4>,
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

    unsafe {
        // Double-cover scale
        let dc_scale = _mm_set_ps(2.0, 2.0, 2.0, 1.0);
        let b_xwyz = swizzle!(b.0, 2, 1, 3, 0);
        let b_xzwy = swizzle!(b.0, 1, 3, 2, 0);
        let b_xxxx = swizzle!(b.0, 0, 0, 0, 0);

        let tmp1 = _mm_mul_ps(swizzle!(b.0, 0, 0, 0, 2), swizzle!(b.0, 2, 1, 3, 2));
        let tmp1 = _mm_add_ps(
            tmp1,
            _mm_mul_ps(swizzle!(b.0, 1, 3, 2, 1), swizzle!(b.0, 3, 2, 1, 1)),
        );
        // Scale later with (a0, a2, a3, a1)
        let tmp1 = _mm_mul_ps(tmp1, dc_scale);

        let tmp2 = _mm_mul_ps(b.0, b_xwyz);

        let tmp2 = _mm_sub_ps(
            tmp2,
            _mm_xor_ps(
                _mm_set_ss(-0.0),
                _mm_mul_ps(swizzle!(b.0, 0, 0, 0, 3), swizzle!(b.0, 1, 3, 2, 3)),
            ),
        );
        // Scale later with (a0, a3, a1, a2)
        let tmp2 = _mm_mul_ps(tmp2, dc_scale);

        // Alternately add and subtract to improve low component stability
        let tmp3 = _mm_mul_ps(b.0, b.0);
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
            let tmp4 = _mm_mul_ps(b_xxxx, c.0);
            let tmp4 = _mm_add_ps(tmp4, _mm_mul_ps(b_xzwy, swizzle!(c.0, 2, 1, 3, 0)));
            let tmp4 = _mm_add_ps(tmp4, _mm_mul_ps(b.0, swizzle!(c.0, 0, 0, 0, 0)));

            // NOTE: The high component of tmp4 is meaningless here
            let tmp4 = _mm_sub_ps(tmp4, _mm_mul_ps(b_xwyz, swizzle!(c.0, 1, 3, 2, 0)));
            _mm_mul_ps(tmp4, dc_scale)
        } else {
            core::mem::uninitialized()
        };

        // The temporaries (tmp1, tmp2, tmp3, tmp4)
        // strictly only have a dependence on b and c.

        for (a, p) in a.zip(out) {
            // Compute the lower block for components e1, e2, and e3
            p.0 = _mm_mul_ps(tmp1, swizzle!(a.0, 1, 3, 2, 0));
            p.0 = _mm_add_ps(p.0, _mm_mul_ps(tmp2, swizzle!(a.0, 2, 1, 3, 0)));
            p.0 = _mm_add_ps(p.0, _mm_mul_ps(tmp3, a.0));

            if translate {
                let tmp5 = hi_dp(tmp4.into(), *a).0;
                p.0 = _mm_add_ps(p.0, tmp5);
            }
        }
    }
}

// Apply a motor to a point
//template <bool Variadic, bool Translate>
pub fn sw312<'a>(
    a: impl Iterator<Item = &'a f32x4>,
    b: f32x4,
    c: Option<&'a f32x4>,
    out: impl Iterator<Item = &'a mut f32x4>,
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

    unsafe {
        let two = _mm_set_ps(2.0, 2.0, 2.0, 0.0);
        let b_xxxx = swizzle!(b.0, 0, 0, 0, 0);
        let b_xwyz = swizzle!(b.0, 2, 1, 3, 0);
        let b_xzwy = swizzle!(b.0, 1, 3, 2, 0);

        let tmp1 = _mm_mul_ps(b.0, b_xwyz);
        let tmp1 = _mm_sub_ps(tmp1, _mm_mul_ps(b_xxxx, b_xzwy));
        let tmp1 = _mm_mul_ps(tmp1, two);
        // tmp1 needs to be scaled by (_, a3, a1, a2)

        let tmp2 = _mm_mul_ps(b_xxxx, b_xwyz);
        let tmp2 = _mm_add_ps(tmp2, _mm_mul_ps(b_xzwy, b.0));
        let tmp2 = _mm_mul_ps(tmp2, two);
        // tmp2 needs to be scaled by (_, a2, a3, a1)

        let tmp3 = _mm_mul_ps(b.0, b.0);
        let b_tmp = swizzle!(b.0, 0, 0, 0, 1);
        let tmp3 = _mm_add_ps(tmp3, _mm_mul_ps(b_tmp, b_tmp));
        let b_tmp = swizzle!(b.0, 2, 1, 3, 2);
        let tmp4 = _mm_mul_ps(b_tmp, b_tmp);
        let b_tmp = swizzle!(b.0, 1, 3, 2, 3);
        let tmp4 = _mm_add_ps(tmp4, _mm_mul_ps(b_tmp, b_tmp));
        let tmp3 = _mm_sub_ps(tmp3, _mm_xor_ps(tmp4, _mm_set_ss(-0.0)));
        // tmp3 needs to be scaled by (a0, a1, a2, a3)

        let translate = c.is_some();
        let tmp4 = if let Some(c) = c {
            let tmp4 = _mm_mul_ps(b_xzwy, swizzle!(c.0, 2, 1, 3, 0));
            let tmp4 = _mm_sub_ps(tmp4, _mm_mul_ps(b_xxxx, c.0));
            let tmp4 = _mm_sub_ps(tmp4, _mm_mul_ps(b_xwyz, swizzle!(c.0, 1, 3, 2, 0)));
            let tmp4 = _mm_sub_ps(tmp4, _mm_mul_ps(b.0, swizzle!(c.0, 0, 0, 0, 0)));

            // Mask low component and scale other components by 2
            // tmp4 needs to be scaled by (_, a0, a0, a0)
            _mm_mul_ps(tmp4, two)
        } else {
            core::mem::uninitialized()
        };

        for (a, p) in a.zip(out) {
            p.0 = _mm_mul_ps(tmp1, swizzle!(a.0, 2, 1, 3, 0));
            p.0 = _mm_add_ps(p.0, _mm_mul_ps(tmp2, swizzle!(a.0, 1, 3, 2, 0)));
            p.0 = _mm_add_ps(p.0, _mm_mul_ps(tmp3, a.0));

            if translate {
                p.0 = _mm_add_ps(p.0, _mm_mul_ps(tmp4, swizzle!(a.0, 0, 0, 0, 0)));
            }
        }
    }
}

// Conjugate origin with motor. Unlike other operations the motor MUST be
// normalized prior to usage b is the rotor component (p1) c is the
// translator component (p2)
pub fn swo12(b: f32x4, c: f32x4) -> f32x4 {
    //  (b0^2 + b1^2 + b2^2 + b3^2) e123 +
    // 2(b2 c3 - b1 c0 - b0 c1 - b3 c2) e032 +
    // 2(b3 c1 - b2 c0 - b0 c2 - b1 c3) e013 +
    // 2(b1 c2 - b3 c0 - b0 c3 - b2 c1) e021

    let tmp = b * shuffle!(c, [0, 0, 0, 0]);
    let tmp = tmp - shuffle!(b, [0, 0, 0, 0]) * c;
    let tmp = tmp + shuffle!(b, [2, 1, 3, 0]) * shuffle!(c, [1, 3, 2, 0]);
    let tmp = shuffle!(b, [1, 3, 2, 0]) * shuffle!(c, [2, 1, 3, 0]) - tmp;
    let tmp = tmp * f32x4::new(2.0, 2.0, 2.0, 0.0);

    // b0^2 + b1^2 + b2^2 + b3^2 assumed to equal 1
    // Set the low component to unity
    tmp + f32x4::set0(1.0)
}
