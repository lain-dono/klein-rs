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
    let a = f32x4::from(a.p0);
    let (b, c) = (f32x4::from(b.p1), f32x4::from(b.p2));

    // -(a1 c1 + a2 c2 + a3 c3) e0 +
    // (a2 b1 - a1 b2) e3
    // (a3 b2 - a2 b3) e1 +
    // (a1 b3 - a3 b1) e2 +

    let p0 = shuffle!(a, [1, 3, 2, 0]) * b;
    let p0 = p0 - a * shuffle!(b, [1, 3, 2, 0]);
    let p0 = shuffle!(p0, [1, 3, 2, 0]).sub_scalar(f32x4::hi_dp_ss(a, c));

    Plane::from(p0)
});
impl_dot!(|b: Line, a: Plane| -> Plane {
    let a = f32x4::from(a.p0);
    let (b, c) = (f32x4::from(b.p1), f32x4::from(b.p2));

    // (a1 c1 + a2 c2 + a3 c3) e0 +
    // (a1 b2 - a2 b1) e3
    // (a2 b3 - a3 b2) e1 +
    // (a3 b1 - a1 b3) e2 +

    let p0 = a * shuffle!(b, [1, 3, 2, 0]);
    let p0 = p0 - b * shuffle!(a, [1, 3, 2, 0]);
    let p0 = shuffle!(p0, [1, 3, 2, 0]).add_scalar(f32x4::hi_dp_ss(a, c));

    Plane::from(p0)
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
    let a = f32x4::from(a.p0);
    let b = f32x4::from(b.p3);
    // The symmetric inner product on these two partitions commutes

    // (a2 b1 - a1 b2) e03 +
    // (a3 b2 - a2 b3) e01 +
    // (a1 b3 - a3 b1) e02 +
    // a1 b0 e23 +
    // a2 b0 e31 +
    // a3 b0 e12

    let p1 = (a * shuffle!(b, [0, 0, 0, 0])).blend_and();
    let sa = shuffle!(a, [1, 3, 2, 0]);
    let sb = shuffle!(b, [1, 3, 2, 0]);
    let p2 = shuffle!(b * sa - a * sb, [1, 3, 2, 0]);

    Line::from((p1, p2))
});
impl_dot!(|a: Point, b: Plane| -> Line { b | a });

impl_dot!(|a: Point, b: Line| -> Plane {
    // (a1 b1 + a2 b2 + a3 b3) e0 +
    // -a0 b1 e1 +
    // -a0 b2 e2 +
    // -a0 b3 e3

    let a = f32x4::from(a.p3);
    let b = f32x4::from(b.p1);

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
