#![allow(dead_code)]

#[macro_use]
mod sse;

mod exp_log;
mod geometric_product;
mod inner_product;
mod matrix;
mod sandwitch;

pub use self::{
    exp_log::*, geometric_product::*, inner_product::*, matrix::*, sandwitch::*, sse::*,
};

use core::arch::x86_64::*;

#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct f32x4(pub(crate) __m128);

impl core::fmt::Debug for f32x4 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.into_array().iter()).finish()
    }
}

impl std::ops::Deref for f32x4 {
    type Target = __m128;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Into<[f32; 4]> for f32x4 {
    #[inline(always)]
    fn into(self) -> [f32; 4] {
        self.into_array()
    }
}

impl From<[f32; 4]> for f32x4 {
    #[inline(always)]
    fn from(array: [f32; 4]) -> Self {
        Self::from_array(array)
    }
}

impl From<__m128> for f32x4 {
    #[inline(always)]
    fn from(xmm: __m128) -> Self {
        Self(xmm)
    }
}

impl Into<__m128> for f32x4 {
    #[inline(always)]
    fn into(self) -> __m128 {
        self.0
    }
}

macro_rules! impl_bin_add {
    ($op:ident :: $fn:ident => $simd:ident) => {
        impl std::ops::$op for f32x4 {
            type Output = Self;
            #[inline(always)]
            fn $fn(self, other: Self) -> Self {
                Self(unsafe { $simd(self.0, other.0) })
            }
        }
    };
}

impl_bin_add!(Add::add => _mm_add_ps);
impl_bin_add!(Sub::sub => _mm_sub_ps);
impl_bin_add!(Mul::mul => _mm_mul_ps);

impl std::ops::Mul<f32> for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, s: f32) -> Self {
        self * Self::all(s)
    }
}

impl std::ops::Div<f32> for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, s: f32) -> Self {
        self * Self::all(s).rcp_nr1()
    }
}

impl std::ops::BitXor for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, other: Self) -> Self {
        Self(unsafe { _mm_xor_ps(self.0, other.0) })
    }
}

impl f32x4 {
    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self(unsafe { _mm_set_ps(x, y, z, w) })
    }

    #[inline(always)]
    pub fn flip_w() -> Self {
        Self(unsafe { _mm_set1_ps(-0.0) })
    }

    #[inline(always)]
    pub fn flip_xyz() -> Self {
        Self(unsafe { _mm_set_ps(-0.0, -0.0, -0.0, 0.0) })
    }

    #[inline(always)]
    pub fn all(s: f32) -> Self {
        Self(unsafe { _mm_set1_ps(s) })
    }

    #[inline(always)]
    pub fn first(self) -> f32 {
        unsafe {
            let mut out = 0.0;
            _mm_store_ss(&mut out, self.0);
            out
        }
    }

    #[inline(always)]
    pub fn from_array(data: [f32; 4]) -> Self {
        Self(unsafe { _mm_loadu_ps(data.as_ptr()) })
    }

    #[inline(always)]
    pub fn into_array(self) -> [f32; 4] {
        unsafe {
            let mut out = [0.0; 4];
            _mm_store_ps(out.as_mut_ptr(), self.0);
            out
        }
    }

    #[inline(always)]
    pub fn into_simd(self) -> __m128 {
        self.0
    }

    #[inline(always)]
    pub fn from_simd(simd: __m128) -> Self {
        Self(simd)
    }

    #[inline(always)]
    pub fn rcp_nr1(self) -> Self {
        rcp_nr1(self)
    }

    pub fn movehdup(self) -> Self {
        Self::from(unsafe { _mm_movehdup_ps(self.0) })
    }

    pub fn dp(a: Self, b: Self) -> Self {
        Self(unsafe { dp(a.0, b.0) })
    }

    pub fn hi_dp(a: Self, b: Self) -> Self {
        Self(unsafe { hi_dp(a.0, b.0) })
    }

    pub fn hi_dp_ss(a: Self, b: Self) -> Self {
        Self(unsafe { hi_dp_ss(a.0, b.0) })
    }

}
