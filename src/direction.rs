use core::arch::x86_64::*;
use crate::arch::f32x4;

/// Directions in **P**(&#x211d;&#x00B3;&#x2083;&#x2080;&#x2081;) are represented using
/// points at infinity (homogeneous coordinate 0). Having a homogeneous
/// coordinate of zero ensures that directions are translation-invariant.
#[derive(Clone, Copy)]
pub struct Direction {
    pub(crate) p3: f32x4,
}

impl Direction {
    /// Create a normalized direction
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self::from(f32x4::new(z, y, x, 0.0)).normalized()
    }

    /*
    /// Data should point to four floats with memory layout `(0.f, x, y, z)`
    /// where the zero occupies the lowest address in memory.
    pub fn load(&mut self, data: [f32; 4]) {
        debug_assert_eq!(data[0], 0.0, "Homogeneous coordinate of point data used to initialize a direction must be exactly zero");
        unsafe { self.p3 = _mm_loadu_ps(data.as_ptr()) }
    }
    */

    /// Normalize this direction by dividing all components by the magnitude
    /// (by default, `rsqrtps` is used with a single Newton-Raphson refinement iteration)
    pub fn normalize(&mut self) {
        unsafe {
            use crate::arch::{rsqrt_nr1, hi_dp_bc};
            let tmp = rsqrt_nr1(hi_dp_bc(self.p3.0, self.p3.0));
            self.p3 = _mm_mul_ps(self.p3.0, tmp).into();
        }
    }

    /// Return a normalized copy of this direction
    pub fn normalized(mut self) -> Self {
        self.normalize();
        self
    }
}
