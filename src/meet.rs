//! # Exterior Product (ext)
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

use crate::{
    arch::{ext00, ext02, ext03_false, ext03_true, ext_pb, hi_dp_ss, f32x4},
    Branch, Dual, IdealLine, Line, Plane, Point,
};
use core::arch::x86_64::*;

macro_rules! impl_meet {
    (|$a:ident: $a_ty:ty, $b:ident: $b_ty:ty| -> $output:ty $body:block) => {
        impl std::ops::BitXor<$b_ty> for $a_ty {
            type Output = $output;

            #[inline]
            #[allow(unused_unsafe)]
            fn bitxor(self, other: $b_ty) -> Self::Output {
                unsafe {
                    let $a = self;
                    let $b = other;
                    $body
                }
            }
        }
    };
}

impl_meet!(|a: Plane, b: Plane| -> Line { Line::from(ext00(a.p0, b.p0)) });

impl_meet!(|a: Plane, b: Branch| -> Point { Point::from(ext_pb(a.p0, b.p1)) });
impl_meet!(|b: Branch, a: Plane| -> Point { a ^ b });

impl_meet!(|a: Plane, b: IdealLine| -> Point { Point::from(ext02(a.p0, b.p2)) });
impl_meet!(|b: IdealLine, a: Plane| -> Point { a ^ b });

impl_meet!(|a: Plane, b: Line| -> Point {
    (_mm_add_ps(ext02(a.p0, b.p2), ext_pb(a.p0, b.p1))).into()
});
impl_meet!(|b: Line, a: Plane| -> Point { a ^ b });

impl_meet!(|a: Plane, b: Point| -> Dual { Dual::new(0.0, ext03_false(a.p0, b.p3)) });
impl_meet!(|b: Point, a: Plane| -> Dual { Dual::new(0.0, ext03_true(a.p0, b.p3)) });

impl_meet!(|a: Branch, b: IdealLine| -> Dual {
    let dp = hi_dp_ss(a.p1, b.p2);
    let mut out = 0.0;
    _mm_store_ss(&mut out, dp);
    Dual { p: 0.0, q: out }
});
impl_meet!(|b: IdealLine, a: Branch| -> Dual { a ^ b });

impl_meet!(|a: Line, b: Line| -> Dual {
    let dp = hi_dp_ss(a.p1, b.p2);
    let mut out = [2.0; 2];
    _mm_store_ss(&mut out[0], dp);
    let dp = hi_dp_ss(b.p1, a.p2);
    _mm_store_ss(&mut out[1], dp);
    Dual {
        p: 0.0,
        q: out[0] + out[1],
    }
});

impl_meet!(|a: Line, b: IdealLine| -> Dual { Branch { p1: a.p1 } ^ b });
impl_meet!(|b: IdealLine, a: Line| -> Dual { a ^ b });

impl_meet!(|a: Line, b: Branch| -> Dual { IdealLine { p2: a.p2 } ^ b });
impl_meet!(|b: Branch, a: Line| -> Dual { a ^ b });
