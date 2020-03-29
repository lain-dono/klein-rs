// Provide convenience macros and utilities for invoking x86 SSE

use super::f32x4;
use core::arch::x86_64::*;

// Little-endian XMM register swizzle
//
// swizzle(reg, 3, 2, 1, 0) is the identity.
//
// This is undef-ed at the bottom of klein.hpp so as not to
// pollute the macro namespace
macro_rules! swizzle {
    ($reg:expr, $x:expr, $y:expr, $z:expr, $w:expr) => {
        swizzle!($reg, $reg, $x, $y, $z, $w)
    };

    ($a:expr, $b:expr, $x:expr, $y:expr, $z:expr, $w:expr) => {{
        unsafe {
            core::arch::x86_64::_mm_shuffle_ps(
                $a,
                $b,
                core::arch::x86_64::_MM_SHUFFLE($x, $y, $z, $w),
            )
        }
    }};
}

macro_rules! shuffle {
    ($reg:expr, [$x:expr, $y:expr, $z:expr, $w:expr]) => {
        f32x4(swizzle!($reg.0, $reg.0, $x, $y, $z, $w));
    };
    ($a:expr, $b:expr, [$x:expr, $y:expr, $z:expr, $w:expr]) => {
        f32x4(swizzle!($a.0, $b.0, $x, $y, $z, $w))
    };
}

// DP high components and caller ignores returned high components
#[inline(always)]
pub fn hi_dp_ss(a: f32x4, b: f32x4) -> f32x4 {
    // 0 1 2 3 -> 1 + 2 + 3, 0, 0, 0
    let out = a * b;

    // 0 1 2 3 -> 1 1 3 3
    let hi = out.movehdup();

    // 0 1 2 3 + 1 1 3 3 -> (0 + 1, 1 + 1, 2 + 3, 3 + 3)
    let sum = hi + out;

    // unpacklo: 0 0 1 1
    let out = sum + out.unpack_low();

    // (1 + 2 + 3, _, _, _)
    out.movehl()
}

#[inline(always)]
pub fn hi_dp(a: f32x4, b: f32x4) -> f32x4 {
    if cfg!(target_feature = "sse4.1") {
        f32x4(unsafe { _mm_dp_ps(a.0, b.0, 0b1110_0001) })
    } else {
        // Equivalent to _mm_dp_ps(a, b, 0b11100001);

        // 0 1 2 3 -> 1 + 2 + 3, 0, 0, 0
        let out = a * b;

        // 0 1 2 3 -> 1 1 3 3
        let hi = out.movehdup();

        // 0 1 2 3 + 1 1 3 3 -> (0 + 1, 1 + 1, 2 + 3, 3 + 3)
        let sum = hi + out;

        // unpacklo: 0 0 1 1
        let out = sum + out.unpack_low();

        // (1 + 2 + 3, _, _, _)
        let out = out.movehl();

        out & f32x4::cast_i32(0, 0, 0, -1)
    }
}

#[inline(always)]
pub fn hi_dp_bc(a: f32x4, b: f32x4) -> f32x4 {
    if cfg!(target_feature = "sse4.1") {
        f32x4(unsafe { _mm_dp_ps(a.0, b.0, 0b1110_1111) })
    } else {
        // Multiply across and mask low component
        let out = a * b;

        // 0 1 2 3 -> 1 1 3 3
        let hi = out.movehdup();

        // 0 1 2 3 + 1 1 3 3 -> (0 + 1, 1 + 1, 2 + 3, 3 + 3)
        let sum = hi + out;

        // unpacklo: 0 0 1 1
        let out = sum + out.unpack_low();

        shuffle!(out, [2, 2, 2, 2])
    }
}

#[inline(always)]
pub fn dp(a: f32x4, b: f32x4) -> f32x4 {
    if cfg!(target_feature = "sse4.1") {
        f32x4(unsafe { _mm_dp_ps(a.0, b.0, 0b1111_0001) })
    } else {
        // Multiply across and shift right (shifting in zeros)
        let out = a * b;
        let hi = out.movehdup();

        // (a1 b1, a2 b2, a3 b3, 0) + (a2 b2, a2 b2, 0, 0)
        // = (a1 b1 + a2 b2, _, a3 b3, 0)
        let out = hi + out;
        let out = out.add0(hi.movehl_ps(out));

        out & f32x4::cast_i32(0, 0, 0, -1)
    }
}

#[inline(always)]
pub fn dp_bc(a: f32x4, b: f32x4) -> f32x4 {
    if cfg!(target_feature = "sse4.1") {
        f32x4(unsafe { _mm_dp_ps(a.0, b.0, 0xff) })
    } else {
        // Multiply across and shift right (shifting in zeros)
        let out = a * b;
        let hi = out.movehdup();

        // (a1 b1, a2 b2, a3 b3, 0) + (a2 b2, a2 b2, 0, 0)
        // = (a1 b1 + a2 b2, _, a3 b3, 0)
        let out = hi + out;
        let out = out.add0(hi.movehl_ps(out));

        shuffle!(out, [0, 0, 0, 0])
    }
}
