//! # Exterior Product (ext/meet)
//!
//! The exterior product between two basis elements extinguishes if the two
//! operands share any common index. Otherwise, the element produced is
//! equivalent to the union of the subspaces. A sign flip is introduced if
//! the concatenation of the element indices is an odd permutation of the
//! cyclic basis representation. The exterior product extends to general
//! multivectors by linearity.
//!
//! # example "Meeting two planes"
//!
//! ```cpp
//!     kln::plane p1{x1, y1, z1, d1};
//!     kln::plane p2{x2, y2, z2, d2};
//!
//!     // l lies at the intersection of p1 and p2.
//!     kln::line l = p1 ^ p2;
//! ```
//!
//! # example "Meeting a line and a plane"
//!
//! ```cpp
//!     kln::plane p1{x, y, z, d};
//!     kln::line l2{mx, my, mz, dx, dy, dz};
//!
//!     // p2 lies at the intersection of p1 and l2.
//!     kln::point p2 = p1 ^ l2;
//! ```

use crate::{arch::f32x4, Branch, Dual, IdealLine, Line, Plane, Point};

macro_rules! impl_meet {
    (|$a:ident: $a_ty:ty, $b:ident: $b_ty:ty| -> $output:ty $body:block) => {
        impl std::ops::BitXor<$b_ty> for $a_ty {
            type Output = $output;

            #[inline]
            fn bitxor(self, other: $b_ty) -> Self::Output {
                let $a = self;
                let $b = other;
                $body
            }
        }
    };
}

impl_meet!(|a: Plane, b: Plane| -> Line {
    let a = f32x4::from(a.p0);
    let b = f32x4::from(b.p0);

    // (a1 b2 - a2 b1) e12 +
    // (a2 b3 - a3 b2) e23 +
    // (a3 b1 - a1 b3) e31 +

    // (a0 b1 - a1 b0) e01 +
    // (a0 b2 - a2 b0) e02 +
    // (a0 b3 - a3 b0) e03

    let p1 = a * f32x4_swizzle!(b, 1, 3, 2, 0);
    let p1 = f32x4_swizzle!(p1 - f32x4_swizzle!(a, 1, 3, 2, 0) * b, 1, 3, 2, 0);

    let p2 = f32x4_swizzle!(a, 0, 0, 0, 0) * b;
    let p2 = p2 - a * f32x4_swizzle!(b, 0, 0, 0, 0);

    // For both outputs above, we don't zero the lowest component because
    // we've arranged a cancelation

    Line::from((p1, p2))
});

impl_meet!(|a: Plane, b: Branch| -> Point { Point::from(ext_pb(a.p0.into(), b.p1.into())) });
impl_meet!(|b: Branch, a: Plane| -> Point { a ^ b });

impl_meet!(|a: Plane, b: IdealLine| -> Point { Point::from(ext02(a.p0.into(), b.p2.into())) });
impl_meet!(|b: IdealLine, a: Plane| -> Point { a ^ b });

impl_meet!(|a: Plane, b: Line| -> Point {
    let p0 = f32x4::from(a.p0);
    let p1 = f32x4::from(b.p1);
    let p2 = f32x4::from(b.p2);
    let p3 = (ext02(p0, p2) + ext_pb(p0, p1)).into();
    Point { p3 }
});
impl_meet!(|b: Line, a: Plane| -> Point { a ^ b });

impl_meet!(|a: Plane, b: Point| -> Dual {
    // (a0 b0 + a1 b1 + a2 b2 + a3 b3) e0123
    let p0 = f32x4::from(a.p0);
    let p3 = f32x4::from(b.p3);
    Dual {
        p: 0.0,
        q: f32x4::dp(p0, p3).first(),
    }
});
impl_meet!(|b: Point, a: Plane| -> Dual {
    // p0 ^ p3 = -p3 ^ p0
    let p0 = f32x4::from(a.p0);
    let p3 = f32x4::from(b.p3);
    Dual {
        p: 0.0,
        q: (f32x4::dp(p0, p3) ^ f32x4::all(-0.0)).first(),
    }
});

impl_meet!(|a: Branch, b: IdealLine| -> Dual {
    Dual {
        p: 0.0,
        q: f32x4::hi_dp_ss(a.p1.into(), b.p2.into()).first(),
    }
});
impl_meet!(|b: IdealLine, a: Branch| -> Dual { a ^ b });

impl_meet!(|a: Line, b: Line| -> Dual {
    let x = f32x4::hi_dp_ss(a.p1.into(), b.p2.into()).first();
    let y = f32x4::hi_dp_ss(b.p1.into(), a.p2.into()).first();
    Dual { p: 0.0, q: x + y }
});

impl_meet!(|a: Line, b: IdealLine| -> Dual { Branch { p1: a.p1 } ^ b });
impl_meet!(|b: IdealLine, a: Line| -> Dual { a ^ b });

impl_meet!(|a: Line, b: Branch| -> Dual { IdealLine { p2: a.p2 } ^ b });
impl_meet!(|b: Branch, a: Line| -> Dual { a ^ b });

// Partition memory layouts
//     LSB --> MSB
// p0: (e0, e1, e2, e3)
// p1: (1, e23, e31, e12)
// p2: (e0123, e01, e02, e03)
// p3: (e123, e032, e013, e021)

// The exterior products p2 ^ p2, p2 ^ p3, p3 ^ p2, and p3 ^ p3 all vanish

// Plane ^ Branch (branch is a line through the origin)
#[inline(always)]
pub fn ext_pb(a: f32x4, b: f32x4) -> f32x4 {
    // (a1 b1 + a2 b2 + a3 b3) e123 +
    // (-a0 b1) e032 +
    // (-a0 b2) e013 +
    // (-a0 b3) e021

    f32x4_swizzle!(a, 0, 0, 0, 1) * b * f32x4::new(-1.0, -1.0, -1.0, 0.0) + f32x4::hi_dp(a, b)
}

// p0 ^ p2 = p2 ^ p0
#[inline(always)]
pub fn ext02(a: f32x4, b: f32x4) -> f32x4 {
    // (a1 b2 - a2 b1) e021
    // (a2 b3 - a3 b2) e032 +
    // (a3 b1 - a1 b3) e013 +

    let p3 = a * f32x4_swizzle!(b, 1, 3, 2, 0);
    let p3 = p3 - f32x4_swizzle!(a, 1, 3, 2, 0) * b;
    f32x4_swizzle!(p3, 1, 3, 2, 0)
}
