//! # Symmetric Inner Product (dot)
//!
//! The symmetric inner product takes two arguments and contracts the lower
//! graded element to the greater graded element. If lower graded element
//! spans an index that is not contained in the higher graded element, the
//! result is annihilated. Otherwise, the result is the part of the higher
//! graded element "most unlike" the lower graded element. Thus, the
//! symmetric inner product can be thought of as a bidirectional contraction
//! operator.
//!
//! There is some merit in providing both a left and right contraction
//! operator for explicitness. However, when using Klein, it's generally
//! clear what the interpretation of the symmetric inner product is with
//! respect to the projection on various entities.
//!
//! # Example "Angle between planes"
//!
//! ```cpp
//!     kln::plane a{x1, y1, z1, d1};
//!     kln::plane b{x2, y2, z2, d2};
//!
//!     // Compute the cos of the angle between two planes
//!     float cos_ang = a | b;
//! ```
//!
//! # Example "Line to plane through point"
//!
//! ```cpp
//!     kln::point a{x1, y1, z1};
//!     kln::plane b{x2, y2, z2, d2};
//!
//!     // The line l contains a and the shortest path from a to plane b.
//!     line l = a | b;
//! ```

use crate::arch::sse::*;
use crate::{arch::f32x4, IdealLine, Line, Plane, Point};
use core::arch::x86_64::*;

macro_rules! impl_dot {
    (|$a:ident: $a_ty:ty, $b:ident: $b_ty:ty| -> $output:ty $body:block) => {
        impl std::ops::BitOr<$b_ty> for $a_ty {
            type Output = $output;

            #[inline]
            #[allow(unused_unsafe)]
            fn bitor(self, other: $b_ty) -> Self::Output {
                unsafe {
                    let $a = self;
                    let $b = other;
                    $body
                }
            }
        }
    };
}

impl_dot!(|a: Plane, b: Plane| -> f32 { f32x4::hi_dp(a.p0.into(), b.p0.into()).first() });
impl_dot!(|a: Line, b: Line| -> f32 {
    (f32x4::all(-0.0) ^ f32x4::hi_dp_ss(a.p1.into(), b.p1.into())).first()
});
impl_dot!(|a: Point, b: Point| -> f32 {
    // -a0 b0
    let a = f32x4::from(a.p3);
    let b = f32x4::from(b.p3);

    (f32x4::all(-1.0) * (a * b)).first()
});

impl_dot!(|a: Plane, b: Line| -> Plane {
    #[inline(always)]
    pub unsafe fn dot_pl_false(a: __m128, b: __m128, c: __m128) -> __m128 {
        // -(a1 c1 + a2 c2 + a3 c3) e0 +
        // (a2 b1 - a1 b2) e3
        // (a3 b2 - a2 b3) e1 +
        // (a1 b3 - a3 b1) e2 +

        let p0 = _mm_mul_ps(swizzle!(a, 1, 3, 2, 0), b);
        let p0 = _mm_sub_ps(p0, _mm_mul_ps(a, swizzle!(b, 1, 3, 2, 0)));
        let p0 = _mm_sub_ss(swizzle!(p0, 1, 3, 2, 0), hi_dp_ss(a, c));

        p0
    }

    pub fn dot_pl_false_(a: f32x4, b: f32x4, c: f32x4) -> f32x4 {
        // -(a1 c1 + a2 c2 + a3 c3) e0 +
        // (a2 b1 - a1 b2) e3
        // (a3 b2 - a2 b3) e1 +
        // (a1 b3 - a3 b1) e2 +

        let p0 = shuffle!(a, [1, 3, 2, 0]) * b;
        let p0 = p0 - a * shuffle!(b, [1, 3, 2, 0]);
        let p0 = shuffle!(p0, [1, 3, 2, 0]) - f32x4::hi_dp_ss(a, c);
        p0
    }

    Plane::from(dot_pl_false(a.p0, b.p1, b.p2))
    //Plane::from(dot_pl_false_(a.p0.into(), b.p1.into(), b.p2.into()))
});
impl_dot!(|b: Line, a: Plane| -> Plane {
    #[inline(always)]
    pub unsafe fn dot_pl_true(a: __m128, b: __m128, c: __m128) -> __m128 {
        // (a1 c1 + a2 c2 + a3 c3) e0 +
        // (a1 b2 - a2 b1) e3
        // (a2 b3 - a3 b2) e1 +
        // (a3 b1 - a1 b3) e2 +

        let p0 = _mm_mul_ps(a, swizzle!(b, 1, 3, 2, 0));
        let p0 = _mm_sub_ps(p0, _mm_mul_ps(swizzle!(a, 1, 3, 2, 0), b));
        let p0 = _mm_add_ss(swizzle!(p0, 1, 3, 2, 0), hi_dp_ss(a, c));

        p0
    }
    Plane::from(dot_pl_true(a.p0, b.p1, b.p2))
});

impl_dot!(|a: Plane, b: IdealLine| -> Plane {
    let p0 = a.p0.into();
    let p2 = b.p2.into();
    (f32x4::hi_dp(p0, p2) ^ f32x4::all(-0.0)).into()
});
impl_dot!(|b: IdealLine, a: Plane| -> Plane {
    let p0 = a.p0.into();
    let p2 = b.p2.into();
    Plane::from(f32x4::hi_dp(p0, p2))
});

impl_dot!(|a: Plane, b: Point| -> Line {
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
    Line::from(dot03(a.p0, b.p3))
});
impl_dot!(|a: Point, b: Plane| -> Line { b | a });

impl_dot!(|a: Point, b: Line| -> Plane {
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
    Plane::from(dot_ptl(a.p3, b.p1))
});
impl_dot!(|a: Line, b: Point| -> Plane { b | a });

// Partition memory layouts
//     LSB --> MSB
// p0: (e0, e1, e2, e3)
// p1: (1, e23, e31, e12)
// p2: (e0123, e01, e02, e03)
// p3: (e123, e032, e013, e021)
