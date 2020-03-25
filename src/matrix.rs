use crate::{arch::f32x4, Point};
use core::arch::x86_64::*;

/// 3x4 column-major matrix (used for converting rotors/motors to matrix form to
/// upload to shaders). Note that the storage requirement is identical to a
/// column major mat4x4 due to the SIMD representation.
#[doc(hidden)]
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct Mat3x4 {
    pub(crate) x: __m128,
    pub(crate) y: __m128,
    pub(crate) z: __m128,
    pub(crate) w: __m128,
}

/*
impl Mat3x4 {
    /// Apply the linear transformation represented by this matrix to a point
    /// packed with the layout (x, y, z, 1.f)
    pub unsafe fn apply(&self, xyzw: &__m128) -> __m128 {
        let out = _mm_mul_ps(self.x, swizzle!(*xyzw, 0, 0, 0, 0));
        let out = _mm_add_ps(out, _mm_mul_ps(self.y, swizzle!(*xyzw, 1, 1, 1, 1)));
        let out = _mm_add_ps(out, _mm_mul_ps(self.z, swizzle!(*xyzw, 2, 2, 2, 2)));
        let out = _mm_add_ps(out, _mm_mul_ps(self.w, swizzle!(*xyzw, 3, 3, 3, 3)));
        out
    }

    // TODO: provide a transpose function
}
*/

/// 4x4 column-major matrix (used for converting rotors/motors to matrix form to upload to shaders).
#[doc(hidden)]
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct Mat4x4 {
    pub(crate) x: f32x4,
    pub(crate) y: f32x4,
    pub(crate) z: f32x4,
    pub(crate) w: f32x4,
}

impl Mat4x4 {
    /// Apply the linear transformation represented by this matrix to a point
    /// packed with the layout (x, y, z, 1.f)
    pub fn apply(&self, xyzw: Point) -> Point {
        let x = self.x * shuffle!(xyzw.p3, [0, 0, 0, 0]);
        let y = self.y * shuffle!(xyzw.p3, [1, 1, 1, 1]);
        let z = self.z * shuffle!(xyzw.p3, [2, 2, 2, 2]);
        let w = self.w * shuffle!(xyzw.p3, [3, 3, 3, 3]);
        Point::from(x + y + z + w)
    }

    // TODO: provide a transpose function
}
