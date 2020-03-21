//! Geometric Product (gp)
//!
//! The geometric product extends the exterior product with a notion of a
//! metric. When the subspace intersection of the operands of two basis
//! elements is non-zero, instead of the product extinguishing, the grade
//! collapses and a scalar weight is included in the final result according
//! to the metric. The geometric product can be used to build rotations, and
//! by extension, rotations and translations in projective space.
//!
//! # example "Rotor composition"
//!
//! ```cpp
//!     kln::rotor r1{ang1, x1, y1, z1};
//!     kln::rotor r2{ang2, x2, y2, z2};
//!
//!     // Compose rotors with the geometric product
//!     kln::rotor r3 = r1 * r2;; // r3 combines r2 and r1 in that order
//! ```
//!
//! # example "Two reflections"
//!
//! ```cpp
//!     kln::plane p1{x1, y1, z1, d1};
//!     kln::plane p2{x2, y2, z2, d2};
//!
//!     // The geometric product of two planes combines their reflections
//!     kln::motor m3 = p1 * p2; // m3 combines p2 and p1 in that order
//!     // If p1 and p2 were parallel, m3 would be a translation. Otherwise,
//!     // m3 would be a rotation.
//! ```
//!
//! Another common usage of the geometric product is to create a transformation
//! that takes one entity to another. Suppose we have two entities $a$ and $b$
//! and suppose that both entities are normalized such that $a^2 = b^2 = 1$.
//! Then, the action created by $\sqrt{ab}$ is the action that maps $b$ to $a$.
//!
//! # example "Motor between two lines"
//!
//! ```cpp
//!     kln::line l1{mx1, my1, mz1, dx1, dy1, dz1};
//!     kln::line l2{mx2, my2, mz2, dx2, dy2, dz2};
//!     // Ensure lines are normalized if they aren't already
//!     l1.normalize();
//!     l2.normalize();
//!     kln::motor m = kln::sqrt(l1 * l2);
//!
//!     kln::line l3 = m(l2);
//!     // l3 will be projectively equivalent to l1.
//! ```
//!
//! Also provided are division operators that multiply the first argument by the
//! inverse of the second argument.

/// Construct a motor $m$ such that $\sqrt{m}$ takes plane $b$ to plane $a$.
///
/// # example
///
/// ```cpp
///     kln::plane p1{x1, y1, z1, d1};
///     kln::plane p2{x2, y2, z2, d2};
///     kln::motor m = sqrt(p1 * p2);
///     plane p3 = m(p2);
///     // p3 will be approximately equal to p1
/// ```

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
            fn div(self, other: $b_ty) -> Self::Output {
                self * other.inverse()
            }
        }
    };
}

use crate::{
    arch::{gp00, gp03_false, gp03_true, gp11, gp_ll, gp33},
    Motor, Plane, Point, Rotor, Branch, Translator,
};

impl_gp!(|a: Plane, b: Plane| -> Motor { Motor::from(gp00(a.p0, b.p0)) });
impl_gp!(|a: Plane, b: Point| -> Motor { Motor::from(gp03_false(a.p0, b.p3)) });
impl_gp!(|b: Point, a: Plane| -> Motor { Motor::from(gp03_true(a.p0, b.p3)) });

/// Generate a rotor `r` such that `\widetilde{\sqrt{r}}` takes branch `b` to branch `a`.
impl_gp!(|a: Branch, b: Branch| -> Rotor { Rotor::from(gp11(a.p1, b.p1)) });


/*
/// Generates a motor $m$ that produces a screw motion about the common normal
/// to lines $a$ and $b$. The motor given by $\sqrt{m}$ takes $b$ to $a$
/// provided that $a$ and $b$ are both normalized.
impl_gp!(|a: Line, b: Line| -> Motor { Motor::from(gp_ll(a.p1, b.p1)) });
*/

/// Generates a translator $t$ that produces a displacement along the line
/// between points $a$ and $b$. The translator given by $\sqrt{t}$ takes $b$ to `a`.
impl_gp!(|a: Point, b: Point| -> Translator { Translator::from(gp33(a.p3, b.p3)) });

/// Composes two rotational actions such that the produced rotor has the same
/// effect as applying rotor $b$, then rotor $a$.
impl_gp!(|a: Rotor, b: Rotor| -> Rotor { Rotor::from(gp11(a.p1, b.p1)) });

/*
/// The product of a dual number and a line effectively weights the line with a
/// rotational and translational quantity. Subsequent exponentiation will
/// produce a motor along the screw axis of line $b$ with rotation and
/// translation given by half the scalar and pseudoscalar parts of the dual
/// number $a$ respectively.
line operator*(dual a, line b) noexcept
{
    line out;
    detail::gpDL(a.p, a.q, b.p1_, b.p2_, out.p1_, out.p2_);
    return out;
}

line operator*(line b, dual a) noexcept
{
    return a * b;
}

/// Compose the action of a translator and rotor (`b` will be applied, then `a`)
motor operator*(rotor a, translator b) noexcept
{
    motor out;
    out.p1_ = a.p1_;
    detail::gpRT<false>(a.p1_, b.p2_, out.p2_);
    return out;
}

/// Compose the action of a rotor and translator (`a` will be applied, then `b`)
motor operator*(translator b, rotor a) noexcept
{
    motor out;
    out.p1_ = a.p1_;
    detail::gpRT<true>(a.p1_, b.p2_, out.p2_);
    return out;
}

/// Compose the action of two translators (this operation is commutative for
/// these operands).
translator operator*(translator a,
                                                       translator b) noexcept
{
    return a + b;
}

/// Compose the action of a rotor and motor (`b` will be applied, then `a`)
motor operator*(rotor a, motor b) noexcept
{
    motor out;
    detail::gp11(a.p1_, b.p1_, out.p1_);
    detail::gp12<false>(a.p1_, b.p2_, out.p2_);
    return out;
}

/// Compose the action of a rotor and motor (`a` will be applied, then `b`)
motor operator*(motor b, rotor a) noexcept
{
    motor out;
    detail::gp11(b.p1_, a.p1_, out.p1_);
    detail::gp12<true>(a.p1_, b.p2_, out.p2_);
    return out;
}

/// Compose the action of a translator and motor (`b` will be applied, then `a`)
motor operator*(translator a, motor b) noexcept
{
    motor out;
    out.p1_ = b.p1_;
    detail::gpRT<true>(b.p1_, a.p2_, out.p2_);
    out.p2_ = _mm_add_ps(out.p2_, b.p2_);
    return out;
}

/// Compose the action of a translator and motor (`a` will be applied, then `b`)
motor operator*(motor b, translator a) noexcept
{
    motor out;
    out.p1_ = b.p1_;
    detail::gpRT<false>(b.p1_, a.p2_, out.p2_);
    out.p2_ = _mm_add_ps(out.p2_, b.p2_);
    return out;
}

/// Compose the action of two motors (`b` will be applied, then `a`)
motor operator*(motor a, motor b) noexcept
{
    motor out;
    detail::gpMM(a.p1_, b.p1_, &out.p1_);
    return out;
}

// Division operators

motor operator/(plane a, plane b) noexcept
{
    a * b.inverse()
}

translator operator/(point a, point b) noexcept
{
    a * b.inverse()
}

rotor operator/(branch a, branch b) noexcept
{
    a * b.inverse()
}

rotor operator/(rotor a, rotor b) noexcept
{
    a * b.inverse()
}

translator operator/(translator a, translator b) noexcept
{
    a * b.inverse()
}

motor operator/(line a, line b) noexcept
{
    a * b.inverse()
}

motor operator/(motor a, rotor b) noexcept
{
    a * b.inverse()
}

motor operator/(motor a, translator b) noexcept
{
    a * b.inverse()
}

motor operator/(motor a, motor b) noexcept
{
    a * b.inverse()
}
*/
