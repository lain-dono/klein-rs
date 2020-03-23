//! # Planes
//!
//! In projective geometry, planes are the fundamental element through which all
//! other entities are constructed. Lines are the meet of two planes, and points
//! are the meet of three planes (equivalently, a line and a plane).
//!
//! The plane multivector in PGA looks like
//! $d\mathbf{e}_0 + a\mathbf{e}_1 + b\mathbf{e}_2 + c\mathbf{e}_3$. Points
//! that reside on the plane satisfy the familiar equation
//! $d + ax + by + cz = 0$.

use super::{arch::f32x4, Line, Point};

#[derive(Clone, Copy)]
pub struct Plane {
    pub(crate) p0: f32x4,
}

impl Plane {
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        Self::from(f32x4::new(c, b, a, d))
    }

    /// Unaligned load of data. The `data` argument should point to 4 floats
    /// corresponding to the
    /// `(d, a, b, c)` components of the plane multivector where `d` occupies
    /// the lowest address in memory.
    ///
    /// # tip
    ///
    /// This is a faster mechanism for setting data compared to setting
    /// components one at a time.
    pub fn load(&mut self, data: [f32; 4]) {
        self.p0 = f32x4::from_array(data);
    }

    /// Normalize this plane `p` such that $p \cdot p = 1$.
    ///
    /// In order to compute the cosine of the angle between planes via the
    /// inner product operator `|`, the planes must be normalized. Producing a
    /// normalized rotor between two planes with the geometric product `*` also
    /// requires that the planes are normalized.
    pub fn normalize(&mut self) {
        let inv_norm = f32x4::hi_dp_bc(self.p0, self.p0)
            .rsqrt_nr1()
            .blend1(f32x4::set_scalar(1.0));

        self.p0 = inv_norm * self.p0;
    }

    /// Return a normalized copy of this plane.
    pub fn normalized(mut self) -> Self {
        self.normalize();
        self
    }

    /// Compute the plane norm, which is often used to compute distances
    /// between points and lines.
    ///
    /// Given a normalized point $P$ and normalized line $\ell$, the plane
    /// $P\vee\ell$ containing both $\ell$ and $P$ will have a norm equivalent
    /// to the distance between $P$ and $\ell$.
    pub fn norm(self) -> f32 {
        f32x4::hi_dp(self.p0, self.p0).sqrt_nr1().first()
    }

    pub fn invert(&mut self) {
        let inv_norm = f32x4::hi_dp_bc(self.p0, self.p0).rsqrt_nr1();
        self.p0 = self.p0 * inv_norm * inv_norm;
    }

    pub fn inverse(mut self) -> Self {
        self.invert();
        self
    }

    pub fn plane_eq(self, other: Self) -> bool {
        f32x4::bit_eq(self.into(), other.into())
    }

    pub fn approx_eq(self, other: Self, epsilon: f32) -> bool {
        f32x4::approx_eq(self.into(), other.into(), epsilon)
    }

    /// Reflect another plane $p_2$ through this plane $p_1$. The operation
    /// performed via this call operator is an optimized routine equivalent to
    /// the expression $p_1 p_2 p_1$.
    pub fn reflect_plane(self, p: Self) -> Self {
        Plane::from(crate::arch::sw00(self.p0, p.p0))
    }

    /// Reflect line $\ell$ through this plane $p$. The operation
    /// performed via this call operator is an optimized routine equivalent to
    /// the expression $p \ell p$.
    pub fn reflect_line(self, line: Line) -> Line {
        let (p1, p2) = crate::arch::sw10(self.p0, line.p1);
        let p2 = p2 + crate::arch::sw20(self.p0, line.p2);

        Line::from((p1, p2))
    }

    /// Reflect the point $P$ through this plane $p$. The operation
    /// performed via this call operator is an optimized routine equivalent to
    /// the expression $p P p$.
    pub fn reflect_point(self, p: Point) -> Point {
        Point::from(crate::arch::sw30(self.p0, p.p3))
    }
}
