use crate::{arch::f32x4, Line, Plane, Point};

#[derive(Clone, Copy)]
pub struct Translator {
    pub(crate) p2: f32x4,
}

impl Translator {
    pub fn new(delta: f32, x: f32, y: f32, z: f32) -> Self {
        let inv_norm = (x * x + y * y + z * z).sqrt().recip();

        let half_d = -0.5 * delta;
        let p2 = f32x4::all(half_d) * f32x4::new(z, y, x, 0.0);
        let p2 = p2 * f32x4::new(inv_norm, inv_norm, inv_norm, 0.0);
        Self::from(p2)
    }

    #[doc(hidden)]
    pub fn raw(a: f32, b: f32, c: f32, d: f32) -> Self {
        Self::from(f32x4::new(a, b, c, d))
    }

    /// Fast load operation for packed data that is already normalized. The
    /// argument `data` should point to a set of 4 float values with layout
    /// `(0.0, a, b, c)` corresponding to the multivector
    /// $`a\mathbf{e}_{01} + b\mathbf{e}_{02} + c\mathbf{e}_{03}`$.
    ///
    /// # Danger
    ///
    /// The translator data loaded this way *must* be normalized. That is,
    /// the quantity $`-\sqrt{a^2 + b^2 + c^2}`$ must be half the desired
    /// displacement.
    pub fn load_normalized(&mut self, data: [f32; 4]) {
        self.p2 = f32x4::from_array(data);
    }

    pub fn invert(&mut self) {
        self.p2 = f32x4::new(-0.0, -0.0, -0.0, 0.0) ^ self.p2;
    }

    pub fn inverse(mut self) -> Self {
        self.invert();
        self
    }

    /// Conjugates a plane $p$ with this translator and returns the result
    /// $tp\widetilde{t}$.
    pub fn conj_plane(&self, p: Plane) -> Plane {
        Plane::from(crate::arch::sw02(
            p.p0,
            self.p2.blend1(f32x4::set_scalar(1.0)),
        ))
    }

    /// Conjugates a line $`\ell`$ with this translator and returns the result
    /// $`t\ell\widetilde{t}`$.
    pub fn conj_line(&self, l: Line) -> Line {
        Line::from(crate::arch::sw_l2(l.p1, l.p2, self.p2))
    }

    /// Conjugates a point $p$ with this translator and returns the result
    /// $`tp\widetilde{t}`$.
    pub fn conj_point(&self, p: Point) -> Point {
        Point::from(crate::arch::sw32(p.p3, self.p2).0)
    }
}
