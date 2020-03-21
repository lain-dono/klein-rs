//! # Poincaré Dual (dual)
//!
//! The Poincaré Dual of an element is the "subspace complement" of the
//! argument with respect to the pseudoscalar in the exterior algebra. In
//! practice, it is a relabeling of the coordinates to their
//! dual-coordinates and is used most often to implement a "join" operation
//! in terms of the exterior product of the duals of each operand.
//!
//! Ex: The dual of the point $\mathbf{e}_{123} + 3\mathbf{e}_{013} -
//! 2\mathbf{e}_{021}$ (the point at
//! $(0, 3, -2)$) is the plane
//! $\mathbf{e}_0 + 3\mathbf{e}_2 - 2\mathbf{e}_3$.
//!
//! # Regressive Product (reg)
//!
//! The regressive product is implemented in terms of the exterior product.
//! Given multivectors $\mathbf{a}$ and $\mathbf{b}$, the regressive product
//! $\mathbf{a}\vee\mathbf{b}$ is equivalent to
//! $J(J(\mathbf{a})\wedge J(\mathbf{b}))$. Thus, both meets and joins
//! reside in the same algebraic structure.
//!
//! # example "Joining two points"
//!
//! ```cpp
//!     kln::point p1{x1, y1, z1};
//!     kln::point p2{x2, y2, z2};
//!
//!     // l contains both p1 and p2.
//!     kln::line l = p1 & p2;
//! ```
//!
//! # example "Joining a line and a point"
//!
//! ```cpp
//!     kln::point p1{x, y, z};
//!     kln::line l2{mx, my, mz, dx, dy, dz};
//!
//!     // p2 contains both p1 and l2.
//!     kln::plane p2 = p1 & l2;
//! ```

use crate::{Branch, Dual, IdealLine, Line, Plane, Point};

macro_rules! impl_dual {
    (|$a:ident: $a_ty:ty| -> $output:ty $body:block) => {
        impl std::ops::Not for $a_ty {
            type Output = $output;

            #[inline]
            fn not(self) -> Self::Output {
                let $a = self;
                $body
            }
        }
    };
}

impl_dual!(|a: Plane| -> Point { Point { p3: a.p0 } });
impl_dual!(|a: Point| -> Plane { Plane { p0: a.p3 } });
impl_dual!(|a: Line| -> Line { Line { p1: a.p2, p2: a.p1 } });
impl_dual!(|a: Branch| -> IdealLine { IdealLine { p2: a.p1 } });
impl_dual!(|a: IdealLine| -> Branch { Branch { p1: a.p2 } });
impl_dual!(|a: Dual| -> Dual { Dual { p: a.q, q: a.p } });

macro_rules! impl_reg {
    (|$a:ident: $a_ty:ty, $b:ident: $b_ty:ty| -> $output:ty $body:block) => {
        impl std::ops::BitAnd<$b_ty> for $a_ty {
            type Output = $output;

            #[inline]
            fn bitand(self, other: $b_ty) -> Self::Output {
                let $a = self;
                let $b = other;
                $body
            }
        }
    };
}

impl_reg!(|a: Point, b: Point| -> Line { !(!a ^ !b) });
impl_reg!(|a: Point, b: Line| -> Plane { !(!a ^ !b) });
impl_reg!(|b: Line, a: Point| -> Plane { a & b });
impl_reg!(|a: Point, b: Branch| -> Plane { !(!a ^ !b) });
impl_reg!(|b: Branch, a: Point| -> Plane { a & b });
impl_reg!(|a: Point, b: IdealLine| -> Plane { !(!a ^ !b) });
impl_reg!(|b: IdealLine, a: Point| -> Plane { a & b });
impl_reg!(|a: Plane, b: Point| -> Dual { !(!a ^ !b) });
impl_reg!(|b: Point, a: Plane| -> Dual { !(!a ^ !b) });
