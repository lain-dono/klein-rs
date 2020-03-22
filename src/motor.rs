//! # Motors
//!
//! A `motor` represents a kinematic motion in our algebra. From
//! [Chasles'
//! theorem](https://en.wikipedia.org/wiki/Chasles%27_theorem_(kinematics)), we
//! know that any rigid body displacement can be produced by a translation along
//! a line, followed or preceded by a rotation about an axis parallel to that
//! line. The motor algebra is isomorphic to the dual quaternions but exists
//! here in the same algebra as all the other geometric entities and actions at
//! our disposal. Operations such as composing a motor with a rotor or
//! translator are possible for example. The primary benefit to using a motor
//! over its corresponding matrix operation is twofold. First, you get the
//! benefit of numerical stability when composing multiple actions via the
//! geometric product (`*`). Second, because the motors constitute a continuous
//! group, they are amenable to smooth interpolation and differentiation.
//!
//! # example
//!
//! ```c++
//!     // Create a rotor representing a pi/2 rotation about the z-axis
//!     // Normalization is done automatically
//!     rotor r{M_PI * 0.5f, 0.0, 0.0, 1.0};
//!
//!     // Create a translator that represents a translation of 1 unit
//!     // in the yz-direction. Normalization is done automatically.
//!     translator t{1.0, 0.0, 1.0, 1.0};
//!
//!     // Create a motor that combines the action of the rotation and
//!     // translation above.
//!     motor m = r * t;
//!
//!     // Initialize a point at (1, 3, 2)
//!     kln::point p1{1.0, 3.0, 2.0};
//!
//!     // Translate p1 and rotate it to create a new point p2
//!     kln::point p2 = m(p1);
//! ```
//!
//! Motors can be multiplied to one another with the `*` operator to create
//! a new motor equivalent to the application of each factor.
//!
//! # example
//!
//! ```c++
//!     // Suppose we have 3 motors m1, m2, and m3
//!
//!     // The motor m created here represents the combined action of m1,
//!     // m2, and m3.
//!     kln::motor m = m3 * m2 * m1;
//! ```
//!
//! The same `*` operator can be used to compose the motor's action with other
//! translators and rotors.
//!
//! A demonstration of using the exponential and logarithmic map to blend
//! between two motors is provided in a test case
//! [here](https://github.com/jeremyong/Klein/blob/master/test/test_exp_log.cpp#L48).

use crate::{arch::f32x4, Plane, Point, Rotor, Translator};
use core::arch::x86_64::*;

#[derive(Clone, Copy)]
pub struct Motor {
    pub(crate) p1: __m128,
    pub(crate) p2: __m128,
}

impl Motor {
    /// Direct initialization from components. A more common way of creating a
    /// motor is to take a product between a rotor and a translator.
    /// The arguments coorespond to the multivector
    /// $a + b\mathbf{e}_{23} + c\mathbf{e}_{31} + d\mathbf{e}_{12} +\
    /// e\mathbf{e}_{01} + f\mathbf{e}_{02} + g\mathbf{e}_{03} +\
    /// h\mathbf{e}_{0123}$.
    pub fn new(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32) -> Self {
        Self {
            p1: unsafe { _mm_set_ps(d, c, b, a) },
            p2: unsafe { _mm_set_ps(g, f, e, h) },
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

    /*
    /// Produce a screw motion rotating and translating by given amounts along a
    /// provided Euclidean axis.
    motor(float ang_rad, float d, line l) noexcept
    {
        line log_m;
        detail::gpDL(
            -ang_rad * 0.5f, d * 0.5f, l.p1_, l.p2_, log_m.p1_, log_m.p2_);
        detail::exp(log_m.p1_, log_m.p2_, p1_, p2_);
    }

    motor(__m128 p1, __m128 p2) noexcept
        : p1_{p1}
        , p2_{p2}
    {}
    */

    #[inline]
    pub fn from_rotor(r: Rotor) -> Self {
        Self {
            p1: r.p1,
            p2: f32x4::zero().into(),
        }
    }

    pub fn from_translator(t: Translator) -> Self {
        Self {
            p1: f32x4::set_scalar(1.0).into(),
            p2: t.p2,
        }
    }

    /*
    /// Load motor data using two unaligned loads. This routine does *not*
    /// assume the data passed in this way is normalized.
    void load(float* in) noexcept
    {
        // Aligned and unaligned loads incur the same amount of latency and have
        // identical throughput on most modern processors
        p1_ = _mm_loadu_ps(in);
        p2_ = _mm_loadu_ps(in + 4);
    }
    */

    /// Normalizes this motor $m$ such that $m\widetilde{m} = 1$.
    pub fn normalize(&mut self) {
        unsafe {
            use crate::arch::{dp_bc, rcp_nr1, rsqrt_nr1};

            // m = b + c where b is p1 and c is p2
            //
            // m * ~m = |b|^2 + 2(b0 c0 - b1 c1 - b2 c2 - b3 c3)e0123
            //
            // The square root is given as:
            // |b| + (b0 c0 - b1 c1 - b2 c2 - b3 c3)/|b| e0123
            //
            // The inverse of this is given by:
            // 1/|b| + (-b0 c0 + b1 c1 + b2 c2 + b3 c3)/|b|^3 e0123 = s + t e0123
            //
            // Multiplying our original motor by this inverse will give us a
            // normalized motor.
            let b2 = dp_bc(self.p1, self.p1);
            let s = rsqrt_nr1(b2);
            let bc = dp_bc(_mm_xor_ps(self.p1, _mm_set_ss(-0.0)), self.p2);
            let t = _mm_mul_ps(_mm_mul_ps(bc, rcp_nr1(b2.into()).into()), s);

            // (s + t e0123) * motor =
            //
            // s b0 +
            // s b1 e23 +
            // s b2 e31 +
            // s b3 e12 +
            // (s c0 + t b0) e0123 +
            // (s c1 - t b1) e01 +
            // (s c2 - t b2) e02 +
            // (s c3 - t b3) e03

            let tmp = _mm_mul_ps(self.p2, s);
            self.p2 = _mm_sub_ps(tmp, _mm_xor_ps(_mm_mul_ps(self.p1, t), _mm_set_ss(-0.0)));
            self.p1 = _mm_mul_ps(self.p1, s);
        }
    }

    /// Return a normalized copy of this motor.
    pub fn normalized(mut self) -> Self {
        self.normalize();
        self
    }

    pub fn invert(&mut self) {
        unsafe {
            // s, t computed as in the normalization
            let b2 = crate::arch::dp_bc(self.p1, self.p1);
            let s = crate::arch::rsqrt_nr1(b2);
            let bc = crate::arch::dp_bc(_mm_xor_ps(self.p1, _mm_set_ss(-0.0)), self.p2);
            let b2_inv = crate::arch::rcp_nr1(b2.into()).0;
            let t = _mm_mul_ps(_mm_mul_ps(bc, b2_inv), s);
            let neg = _mm_set_ps(-0.0, -0.0, -0.0, 0.0);

            // p1 * (s + t e0123)^2 = (s * p1 - t p1_perp) * (s + t e0123)
            // = s^2 p1 - s t p1_perp - s t p1_perp
            // = s^2 p1 - 2 s t p1_perp
            // (the scalar component above needs to be negated)
            // p2 * (s + t e0123)^2 = s^2 p2 NOTE: s^2 = b2_inv
            let st = _mm_mul_ps(s, t);
            let st = _mm_mul_ps(self.p1, st);
            self.p2 = _mm_sub_ps(
                _mm_mul_ps(self.p2, b2_inv),
                _mm_xor_ps(_mm_add_ps(st, st), _mm_set_ss(-0.0)),
            );
            self.p2 = _mm_xor_ps(self.p2, neg);

            self.p1 = _mm_xor_ps(_mm_mul_ps(self.p1, b2_inv), neg);
        }
    }

    pub fn inverse(mut self) -> Self {
        self.invert();
        self
    }

    /// Constrains the motor to traverse the shortest arc
    pub fn constrain(&mut self) {
        unsafe {
            let mask = swizzle!(_mm_and_ps(self.p1, _mm_set_ss(-0.0)), 0, 0, 0, 0);
            self.p1 = _mm_xor_ps(mask, self.p1);
            self.p2 = _mm_xor_ps(mask, self.p2);
        }
    }

    pub fn constrained(mut self) -> Self {
        self.constrain();
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

    /// Bitwise comparison
    pub fn motor_eq(&self, other: &Self) -> bool {
        unsafe {
            let p1_eq = _mm_cmpeq_ps(self.p1, other.p1);
            let p2_eq = _mm_cmpeq_ps(self.p2, other.p2);
            let eq = _mm_and_ps(p1_eq, p2_eq);
            _mm_movemask_ps(eq) == 0xf
        }
    }

    pub fn approx_eq(&self, other: &Self, epsilon: f32) -> bool {
        unsafe {
            let eps = _mm_set1_ps(epsilon);
            let neg = _mm_set1_ps(-0.0);
            let cmp1 = _mm_cmplt_ps(_mm_andnot_ps(neg, _mm_sub_ps(self.p1, other.p1)), eps);
            let cmp2 = _mm_cmplt_ps(_mm_andnot_ps(neg, _mm_sub_ps(self.p2, other.p2)), eps);
            let cmp = _mm_and_ps(cmp1, cmp2);
            _mm_movemask_ps(cmp) == 0xf
        }
    }

    /*
    /// Convert this motor to a 3x4 column-major matrix representing this
    /// motor's action as a linear transformation. The motor must be normalized
    /// for this conversion to produce well-defined results, but is more
    /// efficient than a 4x4 matrix conversion.
    [[nodiscard]] mat3x4 as_mat3x4() const noexcept
    {
        mat3x4 out;
        mat4x4_12<true, true>(p1_, &p2_, out.cols);
        return out;
    }

    /// Convert this motor to a 4x4 column-major matrix representing this
    /// motor's action as a linear transformation.
    [[nodiscard]] mat4x4 as_mat4x4() const noexcept
    {
        mat4x4 out;
        mat4x4_12<true, false>(p1_, &p2_, out.cols);
        return out;
    }
    */

    /// Conjugates a plane $p$ with this motor and returns the result
    /// $mp\widetilde{m}$.
    pub fn conj_plane(self, p: Plane) -> Plane {
        unsafe {
            use core::iter::once;
            let mut out: Plane = core::mem::uninitialized();
            crate::arch::sw012(once(&p.p0), self.p1, Some(&self.p2), once(&mut out.p0));
            out
        }
    }

    /*
    /// Conjugates an array of planes with this motor in the input array and
    /// stores the result in the output array. Aliasing is only permitted when
    /// `in == out` (in place motor application).
    ///
    /// !!! tip
    ///
    ///     When applying a motor to a list of tightly packed planes, this
    ///     routine will be *significantly faster* than applying the motor to
    ///     each plane individually.
    void KLN_VEC_CALL operator()(plane* in, plane* out, size_t count) const
        noexcept
    {
        detail::sw012<true, true>(&in->p0_, p1_, &p2_, &out->p0_, count);
    }
    */

    /*
    /// Conjugates a line $\ell$ with this motor and returns the result
    /// $m\ell \widetilde{m}$.
    [[nodiscard]] line KLN_VEC_CALL operator()(line const& l) const noexcept
    {
        line out;
        detail::swMM<false, true, true>(&l.p1_, p1_, &p2_, &out.p1_);
        return out;
    }
    */

    /*
        /// Conjugates an array of lines with this motor in the input array and
        /// stores the result in the output array. Aliasing is only permitted when
        /// `in == out` (in place motor application).
        ///
        /// !!! tip
        ///
        ///     When applying a motor to a list of tightly packed lines, this
        ///     routine will be *significantly faster* than applying the motor to
        ///     each line individually.
        void KLN_VEC_CALL operator()(line* in, line* out, size_t count) const noexcept
        {
            detail::swMM<true, true, true>(&in->p1_, p1_, &p2_, &out->p1_, count);
        }

        /// Conjugates a point $p$ with this motor and returns the result
        /// $mp\widetilde{m}$.
        [[nodiscard]] point KLN_VEC_CALL operator()(point const& p) const noexcept
        {
            point out;
            detail::sw312<false, true>(&p.p3_, p1_, &p2_, &out.p3_);
            return out;
        }

        /// Conjugates an array of points with this motor in the input array and
        /// stores the result in the output array. Aliasing is only permitted when
        /// `in == out` (in place motor application).
        ///
        /// !!! tip
        ///
        ///     When applying a motor to a list of tightly packed points, this
        ///     routine will be *significantly faster* than applying the motor to
        ///     each point individually.
        void KLN_VEC_CALL operator()(point* in, point* out, size_t count) const
            noexcept
        {
            detail::sw312<true, true>(&in->p3_, p1_, &p2_, &out->p3_, count);
        }

        /// Conjugates the origin $O$ with this motor and returns the result
        /// $mO\widetilde{m}$.
        [[nodiscard]] point KLN_VEC_CALL operator()(origin) const noexcept
        {
            point out;
            out.p3_ = detail::swo12(p1_, p2_);
            return out;
        }

        /// Conjugates a direction $d$ with this motor and returns the result
        /// $md\widetilde{m}$.
        ///
        /// The cost of this operation is the same as the application of a rotor due
        /// to the translational invariance of directions (points at infinity).
        [[nodiscard]] direction KLN_VEC_CALL operator()(direction const& d) const
            noexcept
        {
            direction out;
            detail::sw312<false, false>(&d.p3_, p1_, nullptr, &out.p3_);
            return out;
        }

        /// Conjugates an array of directions with this motor in the input array and
        /// stores the result in the output array. Aliasing is only permitted when
        /// `in == out` (in place motor application).
        ///
        /// The cost of this operation is the same as the application of a rotor due
        /// to the translational invariance of directions (points at infinity).
        ///
        /// !!! tip
        ///
        ///     When applying a motor to a list of tightly packed directions, this
        ///     routine will be *significantly faster* than applying the motor to
        ///     each direction individually.
        void KLN_VEC_CALL operator()(direction* in, direction* out, size_t count) const
            noexcept
        {
            detail::sw312<true, false>(&in->p3_, p1_, nullptr, &out->p3_, count);
        }

    };
    */
}
