use crate::{arch::f32x4, Branch, Direction, Line, Plane, Point};

#[derive(Clone, Copy)]
pub struct Rotor {
    pub(crate) p1: f32x4,
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
        let p1 = f32x4::new(z, y, x, cos) * f32x4::new(scale, scale, scale, 1.0);
        Self { p1 }
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
        Self::from(f32x4::from_array(data))
    }

    /// Normalize a rotor such that $\mathbf{r}\widetilde{\mathbf{r}} = 1$.
    pub fn normalize(&mut self) {
        // A rotor is normalized if r * r.reverse() is unity.
        let inv_norm = f32x4::dp_bc(self.p1, self.p1).rsqrt_nr1();
        self.p1 = self.p1 * inv_norm;
    }

    /// Return a normalized copy of this rotor
    pub fn normalized(mut self) -> Self {
        self.normalize();
        self
    }

    pub fn invert(&mut self) {
        let inv_norm = f32x4::hi_dp_bc(self.p1, self.p1).rsqrt_nr1();
        self.p1 = self.p1 * inv_norm;
        self.p1 = self.p1 * inv_norm;
        self.p1 = f32x4::new(-0.0, -0.0, -0.0, 0.0) ^ self.p1;
    }

    pub fn inverse(mut self) -> Self {
        self.invert();
        self
    }

    /// Constrains the rotor to traverse the shortest arc
    pub fn constrain(&mut self) {
        let mask = shuffle!(self.p1 & f32x4::set0(-0.0), [0, 0, 0, 0]);
        self.p1 = mask ^ self.p1;
    }

    pub fn constrained(mut self) -> Self {
        self.constrain();
        self
    }

    pub fn reverse(&mut self) {
        self.p1 = self.p1 ^ f32x4::new(-0.0, -0.0, -0.0, 0.0);
    }

    pub fn reversed(mut self) -> Self {
        self.reverse();
        self
    }

    pub fn rotor_eq(self, other: Self) -> bool {
        f32x4::bit_eq(self.into(), other.into())
    }

    pub fn approx_eq(self, other: Self, epsilon: f32) -> bool {
        f32x4::approx_eq(self.into(), other.into(), epsilon)
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
        use core::iter::once;
        let mut out: Plane = unsafe { core::mem::uninitialized() };
        crate::arch::sw012(once(&p.p0), self.p1, None, once(&mut out.p0));
        out
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
        crate::arch::sw012(
            input.iter().map(|d| &d.p0),
            self.p1,
            None,
            out.iter_mut().map(|d| &mut d.p0),
        )
    }

    pub fn conj_branch(&self, b: Branch) -> Branch {
        use core::iter::once;
        let p1 = crate::arch::sw_mm11(once(b.p1), self.p1).next().unwrap();
        Branch { p1 }
    }

    /// Conjugates a line $\ell$ with this rotor and returns the result
    /// $`r\ell \widetilde{r}`$.
    pub fn conj_line(&self, l: Line) -> Line {
        use core::iter::once;
        unsafe {
            let mut out: Line = unsafe { core::mem::uninitialized() };
            crate::arch::sw_mm22(
                once((&l.p1, &l.p2)),
                self.p1,
                None,
                once((&mut out.p1, &mut out.p2)),
            );
            out
        }
    }

    /*
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
        use core::iter::once;
        let mut out: Point = unsafe { core::mem::uninitialized() };
        crate::arch::sw012(once(&p.p3), self.p1, None, once(&mut out.p3));
        out
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
        crate::arch::sw012(
            input.iter().map(|d| &d.p3),
            self.p1,
            None,
            out.iter_mut().map(|d| &mut d.p3),
        );
    }

    /// Conjugates a direction `d` with this rotor and returns the result
    /// $rd\widetilde{r}$.
    pub fn conj_dir(&self, d: &Direction) -> Direction {
        use core::iter::once;
        let mut out: Direction = unsafe { core::mem::uninitialized() };
        // NOTE: Conjugation of a plane and point with a rotor is identical
        crate::arch::sw012(once(&d.p3), self.p1, None, once(&mut out.p3));
        out
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
        crate::arch::sw012(
            input.iter().map(|d| &d.p3),
            self.p1,
            None,
            out.iter_mut().map(|d| &mut d.p3),
        )
    }
}
