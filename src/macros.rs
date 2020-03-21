macro_rules! _expr_doc {
    ( $( #[doc = $doc:expr] $thing:item )* ) => (
        $(
            #[doc = $doc]
            $thing
        )*
    );
}

macro_rules! derive_attrs {
    (struct $ty:ty { $( $field:ident : { $($i:literal : $attr:ident),+ } ),+ } ) => {
        impl $ty {
            $(
                #[inline]
                pub fn $field (self) -> [f32; 4] {
                    unsafe {
                        let mut out = [0.0; 4];
                        _mm_store_ps(out.as_mut_ptr(), self.$field.into());
                        out
                    }
                }

                $( derive_attrs! { _attr $field $attr $i } )+
            )+
        }
    };

    (_attr $field:ident $attr:ident 0) => {
        #[inline]
        pub fn $attr (self) -> f32 {
            unsafe {
                let mut out = 0.0;
                _mm_store_ss(&mut out, self.$field);
                out
            }
        }
    };

    (_attr $field:ident $attr:ident $i:literal) => {
        #[inline]
        pub fn $attr (self) -> f32 { self.$field()[$i] }
    };
}

macro_rules! derive_debug {
    ($ty:ident { $( $field:ident: $simd:ty ),+ }) => {
        impl core::fmt::Debug for $ty {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_tuple(stringify!($ty))
                $(  .field(&self.$field()) )+
                    .finish()
            }
        }
    };
}

macro_rules! derive_conv {
    ($ty:ident { $( $field:ident: $simd:ty ),+ }) => {
        #[doc(hidden)]
        #[allow(unused_parens)]
        impl From<($($simd),+)> for $ty {
            #[inline(always)]
            fn from(($($field),+): ($($simd),+)) -> Self {
                Self { $($field),+ }
            }
        }

        #[doc(hidden)]
        #[allow(unused_parens)]
        impl Into<($($simd),+)> for $ty {
            #[inline(always)]
            fn into(self) -> ($($simd),+) {
                ($(self.$field),+)
            }
        }
    };
}

macro_rules! derive_ops {
    (PartialEq::eq for $ty:ty => $ty_fn:ident) => {
        impl std::cmp::PartialEq for $ty {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.$ty_fn(other)
            }
        }
    };

    (vector add/sub/scale/flip_w for $ty:ident { $( $field:ident: $simd:ty ),+ }) => {
        derive_ops!(vector add/sub/scale for $ty { $($field:$simd),+ });
        derive_ops!(vector flip_w for $ty { $($field),+ });
    };

    (vector add/sub/scale/flip_xyz for $ty:ident { $( $field:ident: $simd:ty ),+ }) => {
        derive_ops!(vector add/sub/scale for $ty { $($field:$simd),+ });
        derive_ops!(vector flip_xyz for $ty { $($field),+ });
    };

    (vector flip_w for $ty:ident { $($field:ident),+ }) => {
        /// Unary minus.
        impl std::ops::Neg for $ty {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                let flip = unsafe { _mm_set1_ps(-0.0) };
                Self {
                    $($field: unsafe { _mm_xor_ps(self.$field, flip) }),+
                }
            }
        }
    };

    (vector flip_xyz for $ty:ident { $($field:ident),+ }) => {
        /// Unary minus (leaves homogeneous coordinate untouched).
        impl std::ops::Neg for $ty {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                let flip = unsafe { _mm_set_ps(-0.0, -0.0, -0.0, 0.0) };
                Self {
                    $($field: unsafe { _mm_xor_ps(self.$field, flip) }),+
                }
            }
        }
    };

    (simd_binary $op:ident<$rhs:ident, $output:ident>::$op_fn:ident for $ty:ident { $( $field:ident ),+ } => $fn:ident) => {
        impl std::ops::$op<$rhs> for $ty {
            type Output = $output;
            #[inline(always)]
            fn $op_fn(self, rhs: $rhs) -> Self::Output {
                Self {
                    $($field: unsafe { $fn(self.$field, rhs.$field) }),+
                }
            }
        }
    };

    (vector add/sub/scale for $ty:ident { $( $field:ident:$simd:ty ),+ }) => {
        derive_debug!($ty { $($field: $simd),+ });
        derive_conv!($ty { $($field:$simd),+ });

        derive_ops!(simd_binary Add<Self, Self>::add for $ty { $( $field ),+ } => _mm_add_ps);
        derive_ops!(simd_binary Sub<Self, Self>::sub for $ty { $( $field ),+ } => _mm_sub_ps);

        impl std::ops::Mul<f32> for $ty {
            type Output = Self;
            #[inline(always)]
            fn mul(self, s: f32) -> Self::Output {
                let s = unsafe { _mm_set1_ps(s) };
                Self {
                    $($field: unsafe { _mm_mul_ps(self.$field, s) }),+
                }
            }
        }

        impl std::ops::Div<f32> for $ty {
            type Output = Self;
            #[inline(always)]
            fn div(self, s: f32) -> Self::Output {
                let s = unsafe { crate::arch::rcp_nr1(_mm_set1_ps(s)) };
                Self {
                    $($field: unsafe { _mm_mul_ps(self.$field, s) }),+
                }
            }
        }
    };
}

macro_rules! derive_f32x4 {
    (vector add/sub/scale/flip_w for $ty:ident { $( $field:ident: $simd:ty ),+ }) => {
        derive_f32x4!(vector add/sub/scale for $ty { $($field:$simd),+ });
        derive_f32x4!(vector flip_w for $ty { $($field),+ });
    };

    (vector add/sub/scale/flip_xyz for $ty:ident { $( $field:ident: $simd:ty ),+ }) => {
        derive_f32x4!(vector add/sub/scale for $ty { $($field:$simd),+ });
        derive_f32x4!(vector flip_xyz for $ty { $($field),+ });
    };

    (vector add/sub/scale for $ty:ident { $( $field:ident:$simd:ty ),+ }) => {
        derive_debug!($ty { $($field: $simd),+ });
        derive_conv!($ty { $($field:$simd),+ });

        impl std::ops::Add for $ty {
            type Output = Self;
            #[inline(always)] fn add(self, rhs: Self) -> Self::Output {
                Self {
                    $($field: self.$field + rhs.$field),+
                }
            }
        }

        impl std::ops::Sub for $ty {
            type Output = Self;
            #[inline(always)] fn sub(self, rhs: Self) -> Self::Output {
                Self {
                    $($field: self.$field - rhs.$field),+
                }
            }
        }

        impl std::ops::Mul<f32> for $ty {
            type Output = Self;
            #[inline(always)]
            fn mul(self, s: f32) -> Self::Output {
                Self { $($field: self.$field * s),+ }
            }
        }

        impl std::ops::Div<f32> for $ty {
            type Output = Self;
            #[inline(always)]
            fn div(self, s: f32) -> Self::Output {
                Self { $($field: self.$field * s),+ }
            }
        }
    };

    (vector flip_w for $ty:ident { $($field:ident),+ }) => {
        /// Unary minus.
        impl std::ops::Neg for $ty {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                let flip = f32x4::flip_w();
                Self { $($field: self.$field ^ flip),+ }
            }
        }
    };

    (vector flip_xyz for $ty:ident { $($field:ident),+ }) => {
        /// Unary minus (leaves homogeneous coordinate untouched).
        impl std::ops::Neg for $ty {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                let flip = f32x4::flip_xyz();
                Self { $($field: self.$field ^ flip),+ }
            }
        }
    };
}


use crate::arch::f32x4;
use crate::{Branch, Direction, IdealLine, Line, Motor, Plane, Point, Rotor, Translator};
use core::arch::x86_64::*;

derive_f32x4!(vector add/sub/scale/flip_w for Direction { p3: f32x4 });
derive_ops!(vector add/sub/scale/flip_xyz for Point { p3: __m128 });

derive_ops!(vector add/sub/scale/flip_w for IdealLine { p2: __m128 });
derive_ops!(vector add/sub/scale/flip_w for Branch { p1: __m128 });
derive_ops!(vector add/sub/scale/flip_w for Line { p1: __m128, p2: __m128 });
derive_ops!(vector add/sub/scale/flip_w for Motor { p1: __m128, p2: __m128 });
derive_ops!(vector add/sub/scale/flip_xyz for Plane { p0: __m128 });
derive_ops!(vector add/sub/scale/flip_w for Rotor { p1: __m128 });
derive_ops!(vector add/sub/scale for Translator { p2: __m128 });

derive_ops!(PartialEq::eq for Motor => motor_eq);
derive_ops!(PartialEq::eq for Rotor => rotor_eq);

derive_attrs!(struct Direction {
    p3: {1: x, 2: y, 3: z}
});
derive_attrs!(struct IdealLine {
    p2: {1: e01, 2: e02, 3: e03}
});
derive_attrs!(struct Branch {
    p1: {1: e23, 1: x, 2: e31, 2: y, 3: e12, 3: z}
});
derive_attrs!(struct Line {
    p1: {1: e23, 2: e31, 3: e12},
    p2: {1: e01, 2: e02, 3: e03}
});
derive_attrs!(struct Plane {
    p0: {0: e0, 0: d, 1: e1, 1: x, 2: e2, 2: y, 3: e3, 3: z}
});
derive_attrs!(struct Point {
    p3: {0: e123, 0: w, 1: e032, 1: x, 2: e013, 2: y, 3: e021, 3: z}
});
derive_attrs!(struct Translator {
    p2: {1: e01, 2: e02, 3: e03}
});
derive_attrs!(struct Rotor {
    p1: {0: scalar, 1: e23, 2: e13, 3: e12}
});
derive_attrs!(struct Motor {
    p1: {0: scalar, 1: e23, 2: e31, 3: e12},
    p2: {0: e0123, 1: e01, 2: e02, 3: e03}
});
