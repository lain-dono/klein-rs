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
    // (a1 b2 - a2 b1) e12 +
    // (a2 b3 - a3 b2) e23 +
    // (a3 b1 - a1 b3) e31 +

    // (a0 b1 - a1 b0) e01 +
    // (a0 b2 - a2 b0) e02 +
    // (a0 b3 - a3 b0) e03

    let p1 = a.p0 * shuffle!(b.p0, [1, 3, 2, 0]);
    let p1 = shuffle!(p1 - shuffle!(a.p0, [1, 3, 2, 0]) * b.p0, [1, 3, 2, 0]);

    let p2 = shuffle!(a.p0, [0, 0, 0, 0]) * b.p0;
    let p2 = p2 - a.p0 * shuffle!(b.p0, [0, 0, 0, 0]);

    // For both outputs above, we don't zero the lowest component because
    // we've arranged a cancelation

    Line::from((p1, p2))
});

impl_meet!(|a: Plane, b: Branch| -> Point { Point::from(ext_pb(a.p0, b.p1)) });
impl_meet!(|b: Branch, a: Plane| -> Point { a ^ b });

impl_meet!(|a: Plane, b: IdealLine| -> Point { Point::from(ext02(a.p0, b.p2)) });
impl_meet!(|b: IdealLine, a: Plane| -> Point { a ^ b });

impl_meet!(|a: Plane, b: Line| -> Point { Point::from(ext02(a.p0, b.p2) + ext_pb(a.p0, b.p1)) });
impl_meet!(|b: Line, a: Plane| -> Point { a ^ b });

impl_meet!(|a: Plane, b: Point| -> Dual {
    // (a0 b0 + a1 b1 + a2 b2 + a3 b3) e0123
    Dual {
        p: 0.0,
        q: f32x4::dp(a.p0, b.p3).first(),
    }
});
impl_meet!(|b: Point, a: Plane| -> Dual {
    // p0 ^ p3 = -p3 ^ p0
    Dual {
        p: 0.0,
        q: (f32x4::dp(a.p0, b.p3) ^ f32x4::all(-0.0)).first(),
    }
});

impl_meet!(|a: Branch, b: IdealLine| -> Dual {
    Dual {
        p: 0.0,
        q: f32x4::hi_dp_ss(a.p1, b.p2).first(),
    }
});
impl_meet!(|b: IdealLine, a: Branch| -> Dual { a ^ b });

impl_meet!(|a: Line, b: Line| -> Dual {
    let x = f32x4::hi_dp_ss(a.p1, b.p2).first();
    let y = f32x4::hi_dp_ss(b.p1, a.p2).first();
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

    shuffle!(a, [0, 0, 0, 1]) * b * f32x4::new(-1.0, -1.0, -1.0, 0.0) + f32x4::hi_dp(a, b)
}

// p0 ^ p2 = p2 ^ p0
#[inline(always)]
pub fn ext02(a: f32x4, b: f32x4) -> f32x4 {
    // (a1 b2 - a2 b1) e021
    // (a2 b3 - a3 b2) e032 +
    // (a3 b1 - a1 b3) e013 +

    let p3 = a * shuffle!(b, [1, 3, 2, 0]);
    let p3 = p3 - shuffle!(a, [1, 3, 2, 0]) * b;
    shuffle!(p3, [1, 3, 2, 0])
}
