//! # Translators
//!
//! A translator represents a rigid-body displacement along a normalized axis.
//! To apply the translator to a supported entity, the call operator is
//! available.
//!
//! ## Example
//!
//! ```c++
//!     // Initialize a point at (1, 3, 2)
//!     kln::point p{1.0, 3.0, 2.0};
//!
//!     // Create a normalized translator representing a 4-unit
//!     // displacement along the xz-axis.
//!     kln::translator r{4.0, 1.0, 0.0, 1.0};
//!
//!     // Displace our point using the created translator
//!     kln::point translated = r(p);
//! ```
//! We can translate lines and planes as well using the translator's call
//! operator.
//!
//! Translators can be multiplied to one another with the `*` operator to create
//! a new translator equivalent to the application of each factor.
//!
//! ## Example
//!
//! ```c++
//!     // Suppose we have 3 translators t1, t2, and t3
//!
//!     // The translator t created here represents the combined action of
//!     // t1, t2, and t3.
//!     kln::translator t = t3 * t2 * t1;
//! ```
//!
//! The same `*` operator can be used to compose the translator's action with
//! other rotors and motors.

use crate::{Line, Plane, Point, arch::f32x4};
use core::arch::x86_64::*;

#[derive(Clone, Copy)]
pub struct Translator {
    pub(crate) p2: __m128,
}

impl Translator {
    pub fn new(delta: f32, x: f32, y: f32, z: f32) -> Self {
        unsafe {
            let norm = (x * x + y * y + z * z).sqrt();
            let inv_norm = norm.recip();

            let half_d = -0.5 * delta;
            let p2 = _mm_mul_ps(_mm_set1_ps(half_d), _mm_set_ps(z, y, x, 0.0));
            let p2 = _mm_mul_ps(p2, _mm_set_ps(inv_norm, inv_norm, inv_norm, 0.0));
            Self { p2 }
        }
    }

    #[doc(hidden)]
    pub fn raw(a: f32, b: f32, c: f32, d: f32) -> Self {
        Self::from(f32x4::new(a, b, c, d).0)
    }

    /// Fast load operation for packed data that is already normalized. The
    /// argument `data` should point to a set of 4 float values with layout
    /// `(0.0, a, b, c)` corresponding to the multivector $a\mathbf{e}_{01} +
    /// b\mathbf{e}_{02} + c\mathbf{e}_{03}$.
    ///
    /// # Danger
    ///
    /// The translator data loaded this way *must* be normalized. That is,
    /// the quantity $-\sqrt{a^2 + b^2 + c^2}$ must be half the desired
    /// displacement.
    pub fn load_normalized(&mut self, data: [f32; 4]) {
        self.p2 = unsafe { _mm_loadu_ps(data.as_ptr()) };
    }

    pub fn invert(&mut self) {
        self.p2 = unsafe { _mm_xor_ps(_mm_set_ps(-0.0, -0.0, -0.0, 0.0), self.p2) };
    }

    pub fn inverse(mut self) -> Self {
        self.invert();
        self
    }

    /// Conjugates a plane $p$ with this translator and returns the result
    /// $tp\widetilde{t}$.
    pub fn conj_plane(&self, p: Plane) -> Plane {
        unsafe {
            let blend = if cfg!(target_feature = "sse4.1") {
                _mm_blend_ps(self.p2, _mm_set_ss(1.0), 1)
            } else {
                _mm_add_ps(self.p2, _mm_set_ss(1.0))
            };
            Plane::from(crate::arch::sw02(p.p0, blend))
        }
    }

    /// Conjugates a line $\ell$ with this translator and returns the result
    /// $t\ell\widetilde{t}$.
    pub fn conj_line(&self, l: Line) -> Line {
        unsafe { Line::from(crate::arch::sw_l2(l.p1, l.p2, self.p2)) }
    }

    /// Conjugates a point $p$ with this translator and returns the result
    /// $tp\widetilde{t}$.
    pub fn conj_point(&self, p: Point) -> Point {
        unsafe { Point::from(crate::arch::sw32(p.p3, self.p2)) }
    }
}
