use crate::{Branch, Dual, IdealLine, Line, Plane, Point};

macro_rules! impl_dual {
    (|$a:ident: $a_ty:ty| -> $output:ty $body:block) => {
        impl core::ops::Not for $a_ty {
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
        impl core::ops::BitAnd<$b_ty> for $a_ty {
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
