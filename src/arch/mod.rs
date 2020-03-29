#[cfg(target = "aarch64")]
pub mod neon;

#[macro_use]
pub mod sse;

mod sandwitch;

pub use self::{sandwitch::*, sse::*};

use core::arch::x86_64::*;

#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct f32x4(pub(crate) __m128);

impl core::fmt::Debug for f32x4 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.into_array().iter()).finish()
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
        impl core::ops::$op for f32x4 {
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
impl_bin_add!(BitAnd::bitand => _mm_and_ps);
impl_bin_add!(BitOr::bitor=> _mm_or_ps);
impl_bin_add!(BitXor::bitxor=> _mm_xor_ps);

impl core::ops::Mul<f32> for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, s: f32) -> Self {
        self * Self::all(s)
    }
}

impl core::ops::Div<f32> for f32x4 {
    type Output = Self;
    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, s: f32) -> Self {
        self * Self::all(s).rcp_nr1()
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
}

impl f32x4 {
    #[inline(always)]
    pub fn set0(s: f32) -> Self {
        Self(unsafe { _mm_set_ss(s) })
    }

    #[inline(always)]
    pub fn extract0(self) -> f32 {
        unsafe {
            let mut out = 0.0;
            _mm_store_ss(&mut out, self.0);
            out
        }
    }

    #[inline(always)]
    pub fn add0(self, other: Self) -> Self {
        Self(unsafe { _mm_add_ss(self.0, other.0) })
    }

    #[inline(always)]
    pub fn sub0(self, other: Self) -> Self {
        Self(unsafe { _mm_sub_ss(self.0, other.0) })
    }

    #[inline(always)]
    pub fn mul0(self, other: Self) -> Self {
        Self(unsafe { _mm_mul_ss(self.0, other.0) })
    }
}

impl f32x4 {
    fn cmpeq_ps(a: Self, b: Self) -> Self {
        Self(unsafe { _mm_cmpeq_ps(a.0, b.0) })
    }

    fn cmplt_ps(a: Self, b: Self) -> Self {
        Self(unsafe { _mm_cmplt_ps(a.0, b.0) })
    }

    fn andnot(self, other: Self) -> Self {
        Self(unsafe { _mm_andnot_ps(self.0, other.0) })
    }

    pub fn bit_eq_pair(a: (Self, Self), b: (Self, Self)) -> bool {
        unsafe {
            let eq0 = Self::cmpeq_ps(a.0, b.0);
            let eq1 = Self::cmpeq_ps(a.1, b.1);
            let eq = eq0 & eq1;
            _mm_movemask_ps(eq.0) == 0x0F
        }
    }

    pub fn bit_eq(self, other: Self) -> bool {
        unsafe { _mm_movemask_ps(_mm_cmpeq_ps(self.0, other.0)) == 0b1111 }
    }

    pub fn approx_eq(self, other: Self, epsilon: f32) -> bool {
        unsafe {
            let eps = _mm_set1_ps(epsilon);
            let cmp = _mm_cmplt_ps(
                _mm_andnot_ps(_mm_set1_ps(-0.0), _mm_sub_ps(self.0, other.0)),
                eps,
            );
            _mm_movemask_ps(cmp) != 0b1111
        }
    }

    pub fn approx_eq_pair(a: (Self, Self), b: (Self, Self), epsilon: f32) -> bool {
        unsafe {
            let eps = Self::all(epsilon);
            let neg = Self::all(-0.0);
            let cmp1 = Self::cmplt_ps(neg.andnot(a.0 - b.0), eps);
            let cmp2 = Self::cmplt_ps(neg.andnot(a.1 - b.1), eps);
            let cmp = cmp1 & cmp2;
            _mm_movemask_ps(cmp.0) == 0x0F
        }
    }
}

impl f32x4 {
    // Reciprocal with an additional single Newton-Raphson refinement
    #[inline(always)]
    pub fn rcp_nr1(self) -> Self {
        // f(x) = 1/x - a
        // f'(x) = -1/x^2
        // x_{n+1} = x_n - f(x)/f'(x)
        //         = 2x_n - a x_n^2 = x_n (2 - a x_n)

        // ~2.7x baseline with ~22 bits of accuracy
        let xn = self.recip();
        xn * (f32x4::all(2.0) - self * xn)
    }

    // Sqrt Newton-Raphson is evaluated in terms of rsqrt_nr1
    #[inline(always)]
    pub fn sqrt_nr1(self) -> Self {
        self * self.rsqrt_nr1()
    }

    // Reciprocal sqrt with an additional single Newton-Raphson refinement.
    #[inline(always)]
    pub fn rsqrt_nr1(self) -> Self {
        // f(x) = 1/x^2 - a
        // f'(x) = -1/(2x^(3/2))
        // Let x_n be the estimate, and x_{n+1} be the refinement
        // x_{n+1} = x_n - f(x)/f'(x)
        //         = 0.5 * x_n * (3 - a x_n^2)

        // From Intel optimization manual: expected performance is ~5.2x
        // baseline (sqrtps + divps) with ~22 bits of accuracy

        let xn = self.rsqrt();
        let xn3 = f32x4::all(3.0) - self * xn * xn;
        f32x4::all(0.5) * xn * xn3
    }

    #[inline(always)]
    pub fn rsqrt(self) -> Self {
        Self(unsafe { _mm_rsqrt_ps(self.0) })
    }

    pub fn movehdup(self) -> Self {
        Self::from(unsafe { _mm_movehdup_ps(self.0) })
    }

    pub fn moveldup(self) -> Self {
        Self::from(unsafe { _mm_moveldup_ps(self.0) })
    }

    pub fn movelh(self) -> Self {
        Self::from(unsafe { _mm_movelh_ps(self.0, self.0) })
    }

    pub fn movehl(self) -> Self {
        Self::from(unsafe { _mm_movehl_ps(self.0, self.0) })
    }

    pub fn movehl_ps(self, b: Self) -> Self {
        Self::from(unsafe { _mm_movehl_ps(self.0, b.0) })
    }

    pub fn dp(a: Self, b: Self) -> Self {
        dp(a, b)
    }

    pub fn dp_bc(a: Self, b: Self) -> Self {
        dp_bc(a, b)
    }

    pub fn hi_dp(a: Self, b: Self) -> Self {
        hi_dp(a, b)
    }

    pub fn hi_dp_ss(a: Self, b: Self) -> Self {
        hi_dp_ss(a, b)
    }

    pub fn hi_dp_bc(a: Self, b: Self) -> Self {
        hi_dp_bc(a, b)
    }

    pub fn cast_i32(a: i32, b: i32, c: i32, d: i32) -> Self {
        Self(unsafe { _mm_castsi128_ps(_mm_set_epi32(a, b, c, d)) })
    }

    pub fn unpack_high(self) -> Self {
        Self(unsafe { _mm_unpackhi_ps(self.0, self.0) })
    }

    pub fn unpack_low(self) -> Self {
        Self(unsafe { _mm_unpacklo_ps(self.0, self.0) })
    }

    pub fn blend1(self, b: Self) -> Self {
        if cfg!(target_feature = "sse4.1") {
            Self(unsafe { _mm_blend_ps(self.0, b.0, 1) })
        } else {
            //self + b
            self.add0(b)
        }
    }

    pub fn blend_and(self) -> Self {
        Self(unsafe {
            if cfg!(target_feature = "sse4.1") {
                _mm_blend_ps(self.0, _mm_setzero_ps(), 1)
            } else {
                _mm_and_ps(self.0, _mm_castsi128_ps(_mm_set_epi32(-1, -1, -1, 0)))
            }
        })
    }
}
