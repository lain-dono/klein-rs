//! # Lines
//!
//! Klein provides three line classes: `line`, `branch`, and `ideal_line`. The
//! line class represents a full six-coordinate bivector. The branch contains
//! three non-degenerate components (aka, a line through the origin). The ideal
//! line represents the line at infinity. When the line is created as a meet
//! of two planes or join of two points (or carefully selected Plücker
//! coordinates), it will be a Euclidean line (factorizable as the meet of two vectors).

use core::arch::x86_64::*;

/// An ideal line represents a line at infinity and corresponds to the
/// multivector:
/// _a_**e**&#x2080;&#x2081; + _b_**e**&#x2080;&#x2082; + _c_**e**&#x2080;&#x2083;
#[derive(Clone, Copy)]
pub struct IdealLine {
    pub(crate) p2: __m128,
}

impl IdealLine {
    pub fn new(a: f32, b: f32, c: f32) -> Self {
        Self::from(unsafe { _mm_set_ps(c, b, a, 0.0) })
    }

    pub fn ideal_norm(self) -> f32 {
        self.squared_ideal_norm().sqrt()
    }

    pub fn squared_ideal_norm(self) -> f32 {
        unsafe {
            let mut out = core::mem::uninitialized();
            let dp = crate::arch::hi_dp(self.p2, self.p2);
            _mm_store_ss(&mut out, dp);
            out
        }
    }

    /// Reversion operator
    pub fn reverse(&mut self) {
        self.p2 = unsafe { _mm_xor_ps(self.p2, _mm_set_ps(-0.0, -0.0, -0.0, 0.0)) };
    }

    pub fn reversed(mut self) -> Self {
        self.reverse();
        self
    }
}

/// The `branch` both a line through the origin and also the principal branch of
/// the logarithm of a rotor.
///
/// The rotor branch will be most commonly constructed by taking the
/// logarithm of a normalized rotor. The branch may then be linearily scaled
/// to adjust the "strength" of the rotor, and subsequently re-exponentiated
/// to create the adjusted rotor.
///
/// # Example
///
///  Suppose we have a rotor $r$ and we wish to produce a rotor
///  `$\sqrt[4]{r}$` which performs a quarter of the rotation produced by
///  $r$. We can construct it like so:
///
///  ```c++
///      kln::branch b = r.log();
///      kln::rotor r_4 = (0.25f * b).exp();
///  ```
///
/// # Note
///
///  The branch of a rotor is technically a `line`, but because there are
///  no translational components, the branch is given its own type for
///  efficiency.
#[derive(Clone, Copy)]
pub struct Branch {
    pub(crate) p1: __m128,
}

impl Branch {
    /// Construct the branch as the following multivector:
    ///
    /// $$a \mathbf{e}_{23} + b\mathbf{e}_{31} + c\mathbf{e}_{23}$$
    ///
    /// To convince yourself this is a line through the origin, remember that
    /// such a line can be generated using the geometric product of two planes
    /// through the origin.
    pub fn new(a: f32, b: f32, c: f32) -> Self {
        Self::from(unsafe { _mm_set_ps(c, b, a, 0.0) })
    }

    /// Returns the square root of the quantity produced by `squared_norm`.
    pub fn norm(self) -> f32 {
        self.squared_norm().sqrt()
    }

    /// If a line is constructed as the regressive product (join) of
    /// two points, the squared norm provided here is the squared
    /// distance between the two points (provided the points are
    /// normalized).
    ///
    /// Returns `d^2 + e^2 + f^2`.
    pub fn squared_norm(self) -> f32 {
        unsafe {
            let mut out = core::mem::uninitialized();
            let dp = crate::arch::hi_dp(self.p1, self.p1);
            _mm_store_ss(&mut out, dp);
            out
        }
    }

    pub fn normalize(&mut self) {
        unsafe {
            let inv_norm = crate::arch::rsqrt_nr1(crate::arch::hi_dp_bc(self.p1, self.p1));
            self.p1 = _mm_mul_ps(self.p1, inv_norm);
        }
    }

    pub fn normalized(mut self) -> Self {
        self.normalize();
        self
    }

    pub fn invert(&mut self) {
        unsafe {
            let inv_norm = crate::arch::rsqrt_nr1(crate::arch::hi_dp_bc(self.p1, self.p1));
            self.p1 = _mm_mul_ps(self.p1, inv_norm);
            self.p1 = _mm_mul_ps(self.p1, inv_norm);
            self.p1 = _mm_xor_ps(_mm_set_ps(-0.0, -0.0, -0.0, 0.0), self.p1);
        }
    }

    pub fn inverse(mut self) -> Self {
        self.invert();
        self
    }

    pub fn reverse(&mut self) {
        self.p1 = unsafe { _mm_xor_ps(self.p1, _mm_set_ps(-0.0, -0.0, -0.0, 0.0)) };
    }

    pub fn reversed(mut self) -> Self {
        self.reverse();
        self
    }
}

/// A general line in `PGA` is given as a 6-coordinate bivector with a direct
/// correspondence to Plücker coordinates. All lines can be exponentiated using
/// the `exp` method to generate a motor.
#[derive(Clone, Copy)]
pub struct Line {
    // p1: (1, e12, e31, e23)
    pub(crate) p1: __m128,
    // p2: (e0123, e01, e02, e03)
    pub(crate) p2: __m128,
}

impl Line {
    /// A line is specifed by 6 coordinates which correspond to the line's
    /// [Plücker coordinates](https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates).
    /// The coordinates specified in this way correspond to the following
    /// multivector:
    ///
    /// $$a\mathbf{e}_{01} + b\mathbf{e}_{02} + c\mathbf{e}_{03} +\
    /// d\mathbf{e}_{23} + e\mathbf{e}_{31} + f\mathbf{e}_{12}$$
    pub fn new(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32) -> Self {
        Self {
            p1: unsafe { _mm_set_ps(f, e, d, 0.0) },
            p2: unsafe { _mm_set_ps(c, b, a, 0.0) },
        }
    }

    pub fn from_ideal_line(ideal_line: IdealLine) -> Self {
        Self {
            p1: unsafe { _mm_setzero_ps() },
            p2: ideal_line.p2,
        }
    }

    pub fn from_branch(branch: Branch) -> Self {
        Self {
            p1: branch.p1,
            p2: unsafe { _mm_setzero_ps() },
        }
    }

    pub fn store1(self) -> [f32; 4] {
        unsafe {
            let mut out = [0.0; 4];
            _mm_store_ps(out.as_mut_ptr(), self.p1);
            out
        }
    }

    pub fn store2(self) -> [f32; 4] {
        unsafe {
            let mut out = [0.0; 4];
            _mm_store_ps(out.as_mut_ptr(), self.p2);
            out
        }
    }

    /// Returns the square root of the quantity produced by
    /// `squared_norm`.
    pub fn norm(self) -> f32 {
        self.squared_norm().sqrt()
    }

    /// If a line is constructed as the regressive product (join) of
    /// two points, the squared norm provided here is the squared
    /// distance between the two points (provided the points are
    /// normalized). Returns $d^2 + e^2 + f^2$.
    pub fn squared_norm(self) -> f32 {
        unsafe {
            let mut out = core::mem::uninitialized();
            let dp = crate::arch::hi_dp(self.p1, self.p1);
            _mm_store_ss(&mut out, dp);
            out
        }
    }

    /// Normalize a line such that $\ell^2 = -1$.
    pub fn normalize(&mut self) {
        unsafe {
            // l = b + c where b is p1 and c is p2
            // l * ~l = |b|^2 - 2(b1 c1 + b2 c2 + b3 c3)e0123
            //
            // sqrt(l*~l) = |b| - (b1 c1 + b2 c2 + b3 c3)/|b| e0123
            //
            // 1/sqrt(l*~l) = 1/|b| + (b1 c1 + b2 c2 + b3 c3)/|b|^3 e0123
            //              = s + t e0123
            let b2 = crate::arch::hi_dp_bc(self.p1, self.p1);
            let s = crate::arch::rsqrt_nr1(b2);
            let bc = crate::arch::hi_dp_bc(self.p1, self.p2);
            let t = _mm_mul_ps(_mm_mul_ps(bc, crate::arch::rcp_nr1(b2.into()).0), s);

            // p1 * (s + t e0123) = s * p1 - t p1_perp
            let tmp = _mm_mul_ps(self.p2, s);
            self.p2 = _mm_sub_ps(tmp, _mm_mul_ps(self.p1, t));
            self.p1 = _mm_mul_ps(self.p1, s);
        }
    }

    /// Return a normalized copy of this line
    pub fn normalized(mut self) -> Self {
        self.normalize();
        self
    }

    pub fn invert(&mut self) {
        unsafe {
            // s, t computed as in the normalization
            let b2 = crate::arch::hi_dp_bc(self.p1, self.p1);
            let s = crate::arch::rsqrt_nr1(b2);
            let bc = crate::arch::hi_dp_bc(self.p1, self.p2);
            let b2_inv = crate::arch::rcp_nr1(b2.into()).0;
            let t = _mm_mul_ps(_mm_mul_ps(bc, b2_inv), s);
            let neg = _mm_set_ps(-0.0, -0.0, -0.0, 0.0);

            // p1 * (s + t e0123)^2 = (s * p1 - t p1_perp) * (s + t e0123)
            // = s^2 p1 - s t p1_perp - s t p1_perp
            // = s^2 p1 - 2 s t p1_perp
            // p2 * (s + t e0123)^2 = s^2 p2
            // NOTE: s^2 = b2_inv
            let st = _mm_mul_ps(s, t);
            let st = _mm_mul_ps(self.p1, st);
            self.p2 = _mm_sub_ps(_mm_mul_ps(self.p2, b2_inv), _mm_add_ps(st, st));
            self.p2 = _mm_xor_ps(self.p2, neg);

            self.p1 = _mm_xor_ps(_mm_mul_ps(self.p1, b2_inv), neg);
        }
    }

    pub fn inverse(mut self) -> Self {
        self.invert();
        self
    }

    pub fn reverse(&mut self) {
        unsafe {
            let flip = _mm_set_ps(-0.0, -0.0, -0.0, 0.0);
            self.p1 = _mm_xor_ps(self.p1, flip);
            self.p2 = _mm_xor_ps(self.p2, flip);
        }
    }

    pub fn reversed(mut self) -> Self {
        self.reverse();
        self
    }

    pub fn line_eq(self, other: Self) -> bool {
        unsafe {
            let p1_eq = _mm_cmpeq_ps(self.p1, other.p1);
            let p2_eq = _mm_cmpeq_ps(self.p2, other.p2);
            let eq = _mm_and_ps(p1_eq, p2_eq);
            _mm_movemask_ps(eq) == 0xf
        }
    }

    pub fn line_approx_eq(self, other: Self, epsilon: f32) -> bool {
        unsafe {
            let eps = _mm_set1_ps(epsilon);
            let neg = _mm_set1_ps(-0.0);
            let cmp1 = _mm_cmplt_ps(_mm_andnot_ps(neg, _mm_sub_ps(self.p1, other.p1)), eps);
            let cmp2 = _mm_cmplt_ps(_mm_andnot_ps(neg, _mm_sub_ps(self.p2, other.p2)), eps);
            let cmp = _mm_and_ps(cmp1, cmp2);
            _mm_movemask_ps(cmp) == 0xf
        }
    }
}
