use crate::{arch::f32x4, IdealLine, Line, Plane, Point};

macro_rules! impl_dot {
    (|$a:ident: $a_ty:ty, $b:ident: $b_ty:ty| -> $output:ty $body:block) => {
        impl std::ops::BitOr<$b_ty> for $a_ty {
            type Output = $output;

            #[inline]
            fn bitor(self, other: $b_ty) -> Self::Output {
                let $a = self;
                let $b = other;
                $body
            }
        }
    };
}

impl_dot!(|a: Plane, b: Plane| -> f32 { f32x4::hi_dp(a.p0, b.p0).extract0() });
impl_dot!(|a: Line, b: Line| -> f32 { (f32x4::all(-0.0) ^ f32x4::hi_dp_ss(a.p1, b.p1)).extract0() });
impl_dot!(|a: Point, b: Point| -> f32 {
    // -a0 b0
    (f32x4::all(-1.0) * (a.p3 * b.p3)).extract0()
});

impl_dot!(|a: Plane, b: Line| -> Plane {
    // -(a1 c1 + a2 c2 + a3 c3) e0 +
    // (a2 b1 - a1 b2) e3
    // (a3 b2 - a2 b3) e1 +
    // (a1 b3 - a3 b1) e2 +

    let p0 = shuffle!(a.p0, [1, 3, 2, 0]) * b.p1;
    let p0 = p0 - a.p0 * shuffle!(b.p1, [1, 3, 2, 0]);
    let p0 = shuffle!(p0, [1, 3, 2, 0]).sub0(f32x4::hi_dp_ss(a.p0, b.p2));

    Plane::from(p0)
});
impl_dot!(|b: Line, a: Plane| -> Plane {

    // (a1 c1 + a2 c2 + a3 c3) e0 +
    // (a1 b2 - a2 b1) e3
    // (a2 b3 - a3 b2) e1 +
    // (a3 b1 - a1 b3) e2 +

    let p0 = a.p0 * shuffle!(b.p1, [1, 3, 2, 0]);
    let p0 = p0 - b.p1 * shuffle!(a.p0, [1, 3, 2, 0]);
    let p0 = shuffle!(p0, [1, 3, 2, 0]).add0(f32x4::hi_dp_ss(a.p0, b.p2));

    Plane::from(p0)
});

impl_dot!(|a: Plane, b: IdealLine| -> Plane {
    Plane::from(f32x4::hi_dp(a.p0, b.p2) ^ f32x4::all(-0.0))
});
impl_dot!(|b: IdealLine, a: Plane| -> Plane { Plane::from(f32x4::hi_dp(a.p0, b.p2)) });

impl_dot!(|a: Plane, b: Point| -> Line {
    // The symmetric inner product on these two partitions commutes

    // (a2 b1 - a1 b2) e03 +
    // (a3 b2 - a2 b3) e01 +
    // (a1 b3 - a3 b1) e02 +
    // a1 b0 e23 +
    // a2 b0 e31 +
    // a3 b0 e12

    let p1 = (a.p0 * shuffle!(b.p3, [0, 0, 0, 0])).blend_and();
    let sa = shuffle!(a.p0, [1, 3, 2, 0]);
    let sb = shuffle!(b.p3, [1, 3, 2, 0]);
    let p2 = shuffle!(b.p3 * sa - a.p0 * sb, [1, 3, 2, 0]);

    Line::from((p1, p2))
});
impl_dot!(|a: Point, b: Plane| -> Line { b | a });

impl_dot!(|a: Point, b: Line| -> Plane {
    // (a1 b1 + a2 b2 + a3 b3) e0 +
    // -a0 b1 e1 +
    // -a0 b2 e2 +
    // -a0 b3 e3

    let (a, b) = (a.p3, b.p1);

    let p0 = (b * shuffle!(a, [0, 0, 0, 0])) ^ f32x4::new(-0.0, -0.0, -0.0, 0.0);

    Plane::from(p0.blend1(f32x4::hi_dp_ss(a, b)))
});
impl_dot!(|a: Line, b: Point| -> Plane { b | a });

// Partition memory layouts
//     LSB --> MSB
// p0: (e0, e1, e2, e3)
// p1: (1, e23, e31, e12)
// p2: (e0123, e01, e02, e03)
// p3: (e123, e032, e013, e021)
