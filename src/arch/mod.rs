#![allow(dead_code, unused_unsafe)]

#[macro_use]
pub mod sse;

mod exp_log;
mod matrix;
mod sandwitch;

pub use self::{exp_log::*, matrix::*, sandwitch::*, sse::*};

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

impl std::ops::BitAnd for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, other: Self) -> Self {
        Self(unsafe { _mm_and_ps(self.0, other.0) })
    }
}

impl std::ops::BitOr for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, other: Self) -> Self {
        Self(unsafe { _mm_or_ps(self.0, other.0) })
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
    pub fn all(s: f32) -> Self {
        Self(unsafe { _mm_set1_ps(s) })
    }

    #[inline(always)]
    pub fn zero() -> Self {
        Self(unsafe { _mm_setzero_ps() })
    }

    // 1/self (rcp)
    #[inline(always)]
    pub fn recip(self) -> Self {
        Self(unsafe { _mm_rcp_ps(self.0) })
    }

    #[inline(always)]
    pub fn flip_w() -> Self {
        Self::all(-0.0)
    }

    #[inline(always)]
    pub fn flip_xyz() -> Self {
        Self::new(-0.0, -0.0, -0.0, 0.0)
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
    pub fn set_scalar(s: f32) -> Self {
        Self(unsafe { _mm_set_ss(s) })
    }

    #[inline(always)]
    pub fn add_scalar(self, other: Self) -> Self {
        Self(unsafe { _mm_add_ss(self.0, other.0) })
    }

    #[inline(always)]
    pub fn mul_scalar(self, other: Self) -> Self {
        Self(unsafe { _mm_mul_ss(self.0, other.0) })
    }
}

impl f32x4 {
    #[inline(always)]
    pub fn rcp_nr1(self) -> Self {
        rcp_nr1(self)
    }

    #[inline(always)]
    pub fn sqrt_nr1(self) -> Self {
        Self(unsafe { sqrt_nr1(self.0) })
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

    pub fn blend1(self, b: Self) -> Self {
        if cfg!(target_feature = "sse4.1") {
            Self(unsafe { _mm_blend_ps(self.0, b.0, 1) })
        } else {
            self + b
        }
    }
}
