use crate::arch::f32x4;

/// The origin is a convenience type that occupies no memory but is castable to
/// a point entity. Several operations like conjugation of the origin by a motor
/// is optimized.
#[derive(Clone, Copy)]
pub struct Origin(pub(crate) Point);

impl Origin {
    /// On its own, the origin occupies no memory, but it can be casted as an
    /// entity at any point, at which point it is represented as
    /// $`\mathbf{e}_{123}`$.
    #[inline]
    pub fn into_point() -> Point {
        Point::from(f32x4::set0(1.0))
    }
}

#[derive(Clone, Copy)]
pub struct Point {
    pub(crate) p3: f32x4,
}

impl Point {
    /// Component-wise constructor (homogeneous coordinate is automatically
    /// initialized to 1)
    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self::from(f32x4::new(z, y, x, 1.0))
    }

    /// Fast load from a pointer to an array of four floats with layout
    /// `(w, x, y, z)` where `w` occupies the lowest address in memory.
    ///
    /// # tip
    ///
    /// This load operation is more efficient that modifying individual
    /// components back-to-back.
    ///
    /// # danger
    ///
    /// Unlike the component-wise constructor, the load here requires the
    /// homogeneous coordinate `w` to be supplied as well in the lowest
    /// address pointed to by `data`.
    pub fn load(&mut self, data: [f32; 4]) {
        self.p3 = f32x4::from_array(data);
    }

    /// Normalize this point (division is done via rcpps with an additional Newton-Raphson refinement).
    pub fn normalize(&mut self) {
        self.p3 = self.p3 * shuffle!(self.p3, [0, 0, 0, 0]).rcp_nr1();
    }

    /// Return a normalized copy of this point.
    pub fn normalized(mut self) -> Self {
        self.normalize();
        self
    }

    pub fn invert(&mut self) {
        let inv_norm = shuffle!(self.p3, [0, 0, 0, 0]).rcp_nr1();
        self.p3 = inv_norm * self.p3;
        self.p3 = inv_norm * self.p3;
    }

    pub fn inverse(mut self) -> Self {
        self.invert();
        self
    }

    pub fn reversed(&mut self) {
        self.p3 = self.p3 ^ f32x4::all(-0.0);
    }

    pub fn reverse(mut self) -> Self {
        self.reversed();
        self
    }
}
