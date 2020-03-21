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

use crate::{
    arch::{
        dot00, dot03, dot11, dot33, dot_pil_false, dot_pil_true, dot_pl_false, dot_pl_true, dot_ptl,
    },
    IdealLine, Line, Plane, Point,
};

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

impl_dot!(|a: Plane, b: Plane| -> f32 { dot00(a.p0, b.p0) });
impl_dot!(|a: Line, b: Line| -> f32 { dot11(a.p1, b.p1) });
impl_dot!(|a: Point, b: Point| -> f32 { dot33(a.p3, b.p3) });

impl_dot!(|a: Plane, b: Line| -> Plane { Plane::from(dot_pl_false(a.p0, b.p1, b.p2)) });
impl_dot!(|b: Line, a: Plane| -> Plane { Plane::from(dot_pl_true(a.p0, b.p1, b.p2)) });

impl_dot!(|a: Plane, b: IdealLine| -> Plane { Plane::from(dot_pil_false(a.p0, b.p2)) });
impl_dot!(|b: IdealLine, a: Plane| -> Plane { Plane::from(dot_pil_true(a.p0, b.p2)) });

impl_dot!(|a: Plane, b: Point| -> Line { Line::from(dot03(a.p0, b.p3)) });
impl_dot!(|a: Point, b: Plane| -> Line { b | a });

impl_dot!(|a: Point, b: Line| -> Plane { Plane::from(dot_ptl(a.p3, b.p1)) });
impl_dot!(|a: Line, b: Point| -> Plane { b | a });
