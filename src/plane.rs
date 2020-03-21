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

use super::{Line, Point};
use core::arch::x86_64::*;

#[derive(Clone, Copy)]
pub struct Plane {
    pub(crate) p0: __m128,
}

impl Plane {
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        Self::from(unsafe { _mm_set_ps(c, b, a, d) })
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
        unsafe { self.p0 = _mm_loadu_ps(data.as_ptr()) }
    }

    /// Normalize this plane `p` such that $p \cdot p = 1$.
    ///
    /// In order to compute the cosine of the angle between planes via the
    /// inner product operator `|`, the planes must be normalized. Producing a
    /// normalized rotor between two planes with the geometric product `*` also
    /// requires that the planes are normalized.
    pub fn normalize(&mut self) {
        unsafe {
            use crate::arch::{rsqrt_nr1, hi_dp_bc};
            let inv_norm = rsqrt_nr1(hi_dp_bc(self.p0, self.p0));
            let inv_norm = if cfg!(target_feature = "sse4.1") {
                _mm_blend_ps(inv_norm, _mm_set_ss(1.0), 1)
            } else {
                _mm_add_ps(inv_norm, _mm_set_ss(1.0))
            };
            self.p0 = _mm_mul_ps(inv_norm, self.p0);
        }
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
        unsafe {
            let mut out = core::mem::uninitialized();
            _mm_store_ss(
                &mut out,
                crate::arch::sqrt_nr1(crate::arch::hi_dp(self.p0, self.p0)),
            );
            out
        }
    }

    pub fn invert(&mut self) {
        unsafe {
            let inv_norm = crate::arch::rsqrt_nr1(crate::arch::hi_dp_bc(self.p0, self.p0));
            self.p0 = _mm_mul_ps(inv_norm, self.p0);
            self.p0 = _mm_mul_ps(inv_norm, self.p0);
        }
    }

    pub fn inverse(mut self) -> Self {
        self.invert();
        self
    }

    pub fn plane_eq(self, other: Self) -> bool {
        unsafe { _mm_movemask_ps(_mm_cmpeq_ps(self.p0, other.p0)) == 0b1111 }
    }

    pub fn plane_approx_eq(self, other: Self, epsilon: f32) -> bool {
        unsafe {
            let eps = _mm_set1_ps(epsilon);
            let cmp = _mm_cmplt_ps(
                _mm_andnot_ps(_mm_set1_ps(-0.0), _mm_sub_ps(self.p0, other.p0)),
                eps,
            );
            _mm_movemask_ps(cmp) != 0b1111
        }
    }

    /// Reflect another plane $p_2$ through this plane $p_1$. The operation
    /// performed via this call operator is an optimized routine equivalent to
    /// the expression $p_1 p_2 p_1$.
    pub fn plane_reflect_plane(self, p: &Self) -> Self {
        unsafe {
            let mut out: Self = core::mem::uninitialized();
            crate::arch::sw00(self.p0, p.p0, &mut out.p0);
            out
        }
    }

    /// Reflect line $\ell$ through this plane $p$. The operation
    /// performed via this call operator is an optimized routine equivalent to
    /// the expression $p \ell p$.
    pub fn plane_reflect_line(self, line: &Line) -> Line {
        unsafe {
            let mut out: Line = core::mem::uninitialized();
            crate::arch::sw10(self.p0, line.p1, &mut out.p1, &mut out.p2);
            let mut p2_tmp = core::mem::uninitialized();
            crate::arch::sw20(self.p0, line.p2, &mut p2_tmp);
            out.p2 = _mm_add_ps(out.p2, p2_tmp);
            out
        }
    }

    /// Reflect the point $P$ through this plane $p$. The operation
    /// performed via this call operator is an optimized routine equivalent to
    /// the expression $p P p$.
    pub fn plane_reflect_point(self, p: &Point) -> Point {
        unsafe {
            let mut out: Point = core::mem::uninitialized();
            crate::arch::sw30(self.p0, p.p3, &mut out.p3);
            out
        }
    }
}
