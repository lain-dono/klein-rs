use crate::{arch::f32x4, Branch, Dual, Line, Motor, Plane, Point, Rotor, Translator};

macro_rules! impl_gp {
    (|$a:ident: $a_ty:ty, $b:ident: $b_ty:ty| -> $output:ty $body:block) => {
        impl std::ops::Mul<$b_ty> for $a_ty {
            type Output = $output;

            #[inline]
            #[allow(unused_unsafe)]
            fn mul(self, other: $b_ty) -> Self::Output {
                unsafe {
                    let $a = self;
                    let $b = other;
                    $body
                }
            }
        }

        impl std::ops::Div<$b_ty> for $a_ty {
            type Output = $output;

            #[inline]
            #[allow(clippy::suspicious_arithmetic_impl)]
            fn div(self, other: $b_ty) -> Self::Output {
                self * other.inverse()
            }
        }
    };
}

impl_gp!(|a: Plane, b: Plane| -> Motor { Motor::from(gp00(a.p0.into(), b.p0.into())) });
impl_gp!(|a: Plane, b: Point| -> Motor { Motor::from(gp03_false(a.p0.into(), b.p3.into())) });
impl_gp!(|b: Point, a: Plane| -> Motor { Motor::from(gp03_true(a.p0.into(), b.p3.into())) });

/// Generate a rotor `r` such that `\widetilde{\sqrt{r}}` takes branch `b` to branch `a`.
impl_gp!(|a: Branch, b: Branch| -> Rotor { Rotor::from(gp11(a.p1.into(), b.p1.into())) });

/// Generates a motor $m$ that produces a screw motion about the common normal
/// to lines $a$ and $b$. The motor given by $\sqrt{m}$ takes $b$ to $a$
/// provided that $a$ and $b$ are both normalized.
impl_gp!(|a: Line, b: Line| -> Motor {
    // Optimized line * line operation
    // (-a1 b1 - a3 b3 - a2 b2) +
    // (a2 b1 - a1 b2) e12 +
    // (a1 b3 - a3 b1) e31 +
    // (a3 b2 - a2 b3) e23 +
    // (a1 c1 + a3 c3 + a2 c2 + b1 d1 + b3 d3 + b2 d2) e0123
    // (a3 c2 - a2 c3         + b2 d3 - b3 d2) e01 +
    // (a1 c3 - a3 c1         + b3 d1 - b1 d3) e02 +
    // (a2 c1 - a1 c2         + b1 d2 - b2 d1) e03 +

    let (a, d) = (a.p1, a.p2);
    let (b, c) = (b.p1, b.p2);

    let a2 = a.unpackhi();
    let b2 = b.unpackhi();
    let c2 = c.unpackhi();
    let d2 = d.unpackhi();

    let flip = f32x4::set_scalar(-0.0);

    let p1 = shuffle!(a, [3, 1, 2, 1]) * shuffle!(b, [2, 3, 1, 1]);
    let p1 = (p1 ^ flip) - shuffle!(a, [2, 3, 1, 3]) * shuffle!(b, [3, 1, 2, 3]);
    let p1 = p1.sub_scalar(a2.mul_scalar(b2));

    let p2 = shuffle!(a, [2, 1, 3, 1]) * shuffle!(c, [1, 3, 2, 1]);
    let p2 = p2 - (flip ^ (shuffle!(a, [1, 3, 2, 3]) * shuffle!(c, [2, 1, 3, 3])));
    let p2 = p2 + shuffle!(b, [1, 3, 2, 1]) * shuffle!(d, [2, 1, 3, 1]);
    let p2 = p2 - (flip ^ (shuffle!(b, [2, 1, 3, 3]) * shuffle!(d, [1, 3, 2, 3])));
    let p2 = p2.add_scalar(a2.mul_scalar(c2));
    let p2 = p2.add_scalar(b2.mul_scalar(d2));

    Motor::from((p1, p2))
});

/// Generates a translator $t$ that produces a displacement along the line
/// between points $a$ and $b$. The translator given by $\sqrt{t}$ takes $b$ to `a`.
impl_gp!(|a: Point, b: Point| -> Translator { Translator::from(gp33(a.p3.into(), b.p3.into())) });

/// Composes two rotational actions such that the produced rotor has the same
/// effect as applying rotor $b$, then rotor $a$.
impl_gp!(|a: Rotor, b: Rotor| -> Rotor { Rotor::from(gp11(a.p1.into(), b.p1.into())) });

/// The product of a dual number and a line effectively weights the line with a
/// rotational and translational quantity. Subsequent exponentiation will
/// produce a motor along the screw axis of line $b$ with rotation and
/// translation given by half the scalar and pseudoscalar parts of the dual
/// number `a` respectively.
impl_gp!(|a: Dual, b: Line| -> Line { Line::from(gp_dl(a.p, a.q, b.p1.into(), b.p2.into())) });

impl_gp!(|b: Line, a: Dual| -> Line { a * b });

/// Compose the action of a translator and rotor (`b` will be applied, then `a`)
impl_gp!(|a: Rotor, b: Translator| -> Motor {
    Motor {
        p1: a.p1,
        p2: gp_rt_false(a.p1, b.p2),
    }
});

/// Compose the action of a rotor and translator (`a` will be applied, then `b`)
impl_gp!(|b: Translator, a: Rotor| -> Motor {
    Motor {
        p1: a.p1,
        p2: gp_rt_true(a.p1, b.p2),
    }
});

/// Compose the action of two translators (this operation is commutative for
/// these operands).
impl_gp!(|a: Translator, b: Translator| -> Translator { a + b });

/// Compose the action of a rotor and motor (`b` will be applied, then `a`)
impl_gp!(|a: Rotor, b: Motor| -> Motor {
    Motor {
        p1: gp11(a.p1.into(), b.p1.into()).into(),
        p2: gp12_false(a.p1, b.p2),
    }
});

/// Compose the action of a rotor and motor (`a` will be applied, then `b`)
impl_gp!(|b: Motor, a: Rotor| -> Motor {
    Motor {
        p1: gp11(b.p1.into(), a.p1.into()).into(),
        p2: gp12_true(a.p1, b.p2),
    }
});

/// Compose the action of a translator and motor (`b` will be applied, then `a`)
impl_gp!(|a: Translator, b: Motor| -> Motor {
    Motor {
        p1: b.p1,
        p2: gp_rt_true(b.p1, a.p2) + b.p2,
    }
});

/// Compose the action of a translator and motor (`a` will be applied, then `b`)
impl_gp!(|b: Motor, a: Translator| -> Motor {
    Motor {
        p1: b.p1,
        p2: gp_rt_false(b.p1, a.p2) + b.p2,
    }
});

/// Compose the action of two motors (`b` will be applied, then `a`)
impl_gp!(|a: Motor, b: Motor| -> Motor {
    Motor::from(gp_mm(a.p1, a.p2, b.p1, b.p2))
});

use crate::arch::sse::*;
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

    // Compute a0 b0 + a1 b1 + a2 b2 + a3 b3 and store it in the low component
    let tmp = dp(a.into(), b.into()).0;
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
    let tmp = dp(a.into(), b.into()).0;
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

    _mm_add_ps(p1, tmp)
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
    let tmp = _mm_mul_ps(tmp, f32x4::rcp_nr1(ss.into()).0);

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

pub fn gp_rt_true(a: f32x4, b: f32x4) -> f32x4 {
    // (a1 b1 + a2 b2 + a3 b3) e0123 +
    // (a0 b1 + a2 b3 - a3 b2) e01 +
    // (a0 b2 + a3 b1 - a1 b3) e02 +
    // (a0 b3 + a1 b2 - a2 b1) e03

    let p2 = shuffle!(a, [0, 0, 0, 1]) * shuffle!(b, [3, 2, 1, 1]);
    let p2 = p2 + shuffle!(a, [1, 3, 2, 2]) * shuffle!(b, [2, 1, 3, 2]);

    p2 - (f32x4::set_scalar(-0.0) ^ (shuffle!(a, [2, 1, 3, 3]) * shuffle!(b, [1, 3, 2, 3])))
}

pub fn gp_rt_false(a: f32x4, b: f32x4) -> f32x4 {
    // (a1 b1 + a2 b2 + a3 b3) e0123 +
    // (a0 b1 + a3 b2 - a2 b3) e01 +
    // (a0 b2 + a1 b3 - a3 b1) e02 +
    // (a0 b3 + a2 b1 - a1 b2) e03

    let p2 = shuffle!(a, [0, 0, 0, 1]) * shuffle!(b, [3, 2, 1, 1]);
    let p2 = p2 + shuffle!(a, [2, 1, 3, 2]) * shuffle!(b, [1, 3, 2, 2]);
    p2 - (f32x4::set_scalar(-0.0) ^ (shuffle!(a, [1, 3, 2, 3]) * shuffle!(b, [2, 1, 3, 3])))
}

pub fn gp12_true(a: f32x4, b: f32x4) -> f32x4 {
    let p2 = gp_rt_true(a, b);
    p2 - (f32x4::set_scalar(-0.0) ^ (a * shuffle!(b, [0, 0, 0, 0])))
}

pub fn gp12_false(a: f32x4, b: f32x4) -> f32x4 {
    let p2 = gp_rt_false(a, b);
    p2 - (f32x4::set_scalar(-0.0) ^ (a * shuffle!(b, [0, 0, 0, 0])))
}

// Optimized motor * motor operation
pub fn gp_mm(a: f32x4, b: f32x4, c: f32x4, d: f32x4) -> (f32x4, f32x4) {
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

    let a_xxxx = shuffle!(a, [0, 0, 0, 0]);
    let a_zyzw = shuffle!(a, [3, 2, 1, 2]);
    let a_ywyz = shuffle!(a, [2, 1, 3, 1]);
    let a_wzwy = shuffle!(a, [1, 3, 2, 3]);
    let c_wwyz = shuffle!(c, [2, 1, 3, 3]);
    let c_yzwy = shuffle!(c, [1, 3, 2, 1]);
    let s_flip = f32x4::set_scalar(-0.0);

    let tmp = (a_ywyz * c_yzwy + a_zyzw * shuffle!(c, [0, 0, 0, 2])) ^ s_flip;
    let p1 = a_xxxx * c + tmp - a_wzwy * c_wwyz;

    let p2 = a_xxxx * d + b * shuffle!(c, [0, 0, 0, 0]);
    let p2 = p2 + a_ywyz * shuffle!(d, [1, 3, 2, 1]);
    let p2 = p2 + shuffle!(b, [2, 1, 3, 1]) * c_yzwy;
    let tmp = a_zyzw * shuffle!(d, [0, 0, 0, 2]);
    let tmp = tmp + a_wzwy * shuffle!(d, [2, 1, 3, 3]);
    let tmp = tmp + shuffle!(b, [0, 0, 0, 2]) * shuffle!(c, [3, 2, 1, 2]);
    let tmp = tmp + shuffle!(b, [1, 3, 2, 3]) * c_wwyz;
    let p2 = p2 - (tmp ^ s_flip);

    (p1, p2) // e f
}
