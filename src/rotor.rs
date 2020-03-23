use crate::{arch::f32x4, Direction, Plane, Point};
use core::arch::x86_64::*;

#[derive(Clone, Copy)]
pub struct Rotor {
    pub(crate) p1: __m128,
}

impl Rotor {
    /// Convenience constructor.
    ///
    /// Computes transcendentals and normalizes rotation axis.
    pub fn new(ang_rad: f32, x: f32, y: f32, z: f32) -> Self {
        let norm = (x * x + y * y + z * z).sqrt();
        let inv_norm = -1.0 / norm;

        let half = 0.5 * ang_rad;
        // Rely on compiler to coalesce these two assignments into a single
        // sincos call at instruction selection time
        let (sin, cos) = half.sin_cos();

        let scale = sin * inv_norm;

        unsafe {
            let p1 = _mm_set_ps(z, y, x, cos);
            let p1 = _mm_mul_ps(p1, _mm_set_ps(scale, scale, scale, 1.0));

            Self { p1 }
        }
    }

    #[doc(hidden)]
    pub fn raw(a: f32, b: f32, c: f32, d: f32) -> Self {
        Self::from(f32x4::new(a, b, c, d).0)
    }

    /// Fast load operation for packed data that is already normalized. The
    /// argument `data` should point to a set of 4 float values with layout `(a,
    /// b, c, d)` corresponding to the multivector
    /// $a + b\mathbf{e}_{23} + c\mathbf{e}_{31} + d\mathbf{e}_{12}$.
    ///
    /// # danger
    ///
    /// The rotor data loaded this way *must* be normalized. That is, the
    /// rotor $r$ must satisfy $r\widetilde{r} = 1$.
    pub fn load_normalized(data: [f32; 4]) -> Self {
        Self {
            p1: unsafe { _mm_loadu_ps(data.as_ptr()) },
        }
    }

    /// Normalize a rotor such that $\mathbf{r}\widetilde{\mathbf{r}} = 1$.
    pub fn normalize(&mut self) {
        unsafe {
            // A rotor is normalized if r * ~r is unity.
            use crate::arch::{dp_bc, rsqrt_nr1};
            let inv_norm = rsqrt_nr1(dp_bc(self.p1, self.p1));
            self.p1 = _mm_mul_ps(self.p1, inv_norm);
        }
    }

    /// Return a normalized copy of this rotor
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

    /// Constrains the rotor to traverse the shortest arc
    pub fn constrain(&mut self) {
        unsafe {
            let mask = swizzle!(_mm_and_ps(self.p1, _mm_set_ss(-0.0)), 0, 0, 0, 0);
            self.p1 = _mm_xor_ps(mask, self.p1);
        }
    }

    pub fn constrained(mut self) -> Self {
        self.constrain();
        self
    }

    pub fn reverse(&mut self) {
        self.p1 = unsafe { _mm_xor_ps(self.p1, _mm_set_ps(-0.0, -0.0, -0.0, 0.0)) };
    }

    pub fn reversed(mut self) -> Self {
        self.reverse();
        self
    }

    pub fn rotor_eq(&self, other: &Self) -> bool {
        unsafe { _mm_movemask_ps(_mm_cmpeq_ps(self.p1, other.p1)) == 0b1111 }
    }

    pub fn rotor_approx_eq(&self, other: &Self, epsilon: f32) -> bool {
        unsafe {
            let eps = _mm_set1_ps(epsilon);
            let cmp = _mm_cmplt_ps(
                _mm_andnot_ps(_mm_set1_ps(-0.0), _mm_sub_ps(self.p1, other.p1)),
                eps,
            );
            _mm_movemask_ps(cmp) != 0b1111
        }
    }

    /*
    /// Converts the rotor to a 3x4 column-major matrix. The results of this
    /// conversion are only defined if the rotor is normalized, and this
    /// conversion is preferable if so.
    [[nodiscard]] mat3x4 as_mat3x4() const noexcept
    {
        mat3x4 out;
        mat4x4_12<false, true>(p1_, nullptr, out.cols);
        return out;
    }

    /// Converts the rotor to a 4x4 column-major matrix.
    [[nodiscard]] mat4x4 as_mat4x4() const noexcept
    {
        mat4x4 out;
        mat4x4_12<false, false>(p1_, nullptr, out.cols);
        return out;
    }
    */

    /// Conjugates a plane $p$ with this rotor and returns the result
    /// $rp\widetilde{r}$.
    pub fn conj_plane(&self, p: &Plane) -> Plane {
        // NOTE: Conjugation of a plane and point with a rotor is identical
        unsafe {
            use core::iter::once;
            let mut out: Plane = core::mem::uninitialized();
            crate::arch::sw012(once(&p.p0), self.p1, None, once(&mut out.p0));
            out
        }
    }

    /// Conjugates an array of planes with this rotor in the input array and
    /// stores the result in the output array. Aliasing is only permitted when
    /// `in == out` (in place motor application).
    ///
    /// # tip
    ///
    /// When applying a rotor to a list of tightly packed planes, this
    /// routine will be *significantly faster* than applying the rotor to
    /// each plane individually.
    pub fn conj_plane_slice(&self, input: &[Plane], out: &mut [Plane]) {
        // NOTE: Conjugation of a plane and point with a rotor is identical
        unsafe {
            crate::arch::sw012(
                input.iter().map(|d| &d.p0),
                self.p1,
                None,
                out.iter_mut().map(|d| &mut d.p0),
            )
        }
    }

    /*
    [[nodiscard]] branch KLN_VEC_CALL operator()(branch const& b) const noexcept
    {
        branch out;
        detail::swMM<false, false, false>(&b.p1_, p1_, nullptr, &out.p1_);
        return out;
    }

    /// Conjugates a line $\ell$ with this rotor and returns the result
    /// $r\ell \widetilde{r}$.
    [[nodiscard]] line KLN_VEC_CALL operator()(line const& l) const noexcept
    {
        line out;
        detail::swMM<false, false, true>(&l.p1_, p1_, nullptr, &out.p1_);
        return out;
    }

    /// Conjugates an array of lines with this rotor in the input array and
    /// stores the result in the output array. Aliasing is only permitted when
    /// `in == out` (in place rotor application).
    ///
    /// !!! tip
    ///
    /// When applying a rotor to a list of tightly packed lines, this
    /// routine will be *significantly faster* than applying the rotor to
    /// each line individually.
    void KLN_VEC_CALL operator()(line* in, line* out, size_t count) const noexcept
    {
        detail::swMM<true, false, true>(&in->p1_, p1_, nullptr, &out->p1_, count);
    }
    */

    /// Conjugates a point `p` with this rotor and returns the result
    /// $rp\widetilde{r}$.
    pub fn conj_point(&self, p: Point) -> Point {
        // NOTE: Conjugation of a plane and point with a rotor is identical
        unsafe {
            use core::iter::once;
            let mut out: Point = core::mem::uninitialized();
            crate::arch::sw012(once(&p.p3), self.p1, None, once(&mut out.p3));
            out
        }
    }

    /// Conjugates an array of points with this rotor in the input array and
    /// stores the result in the output array. Aliasing is only permitted when
    /// `in == out` (in place rotor application).
    ///
    /// # tip
    ///
    /// When applying a rotor to a list of tightly packed points, this
    /// routine will be *significantly faster* than applying the rotor to
    /// each point individually.
    pub fn conj_point_slice(&self, input: &[Point], out: &mut [Point]) {
        // NOTE: Conjugation of a plane and point with a rotor is identical
        unsafe {
            crate::arch::sw012(
                input.iter().map(|d| &d.p3),
                self.p1,
                None,
                out.iter_mut().map(|d| &mut d.p3),
            );
        }
    }

    /// Conjugates a direction `d` with this rotor and returns the result
    /// $rd\widetilde{r}$.
    pub fn conj_dir(&self, d: &Direction) -> Direction {
        unsafe {
            use core::iter::once;
            let mut out: Direction = core::mem::uninitialized();
            // NOTE: Conjugation of a plane and point with a rotor is identical
            crate::arch::sw012(once(&d.p3.0), self.p1, None, once(&mut out.p3.0));
            out
        }
    }

    /// Conjugates an array of directions with this rotor in the input array and
    /// stores the result in the output array. Aliasing is only permitted when
    /// `in == out` (in place rotor application).
    ///
    /// # tip
    ///
    /// When applying a rotor to a list of tightly packed directions, this
    /// routine will be *significantly faster* than applying the rotor to
    /// each direction individually.
    pub fn conj_dir_slice(&self, input: &[Direction], out: &mut [Direction]) {
        // NOTE: Conjugation of a plane and point with a rotor is identical
        unsafe {
            crate::arch::sw012(
                input.iter().map(|d| &d.p3.0),
                self.p1,
                None,
                out.iter_mut().map(|d| &mut d.p3.0),
            )
        }
    }
}
