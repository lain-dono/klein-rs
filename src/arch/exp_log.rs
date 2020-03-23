// Provide routines for taking bivector/motor exponentials and logarithms.
use super::f32x4;

// Partition memory layouts
//     LSB --> MSB
// p0: (e0, e1, e2, e3)
// p1: (1, e23, e31, e12)
// p2: (e0123, e01, e02, e03)
// p3: (e123, e032, e013, e021)

// a := p1
// b := p2
// a + b is a general bivector but it is most likely *non-simple* meaning
// that it is neither purely real nor purely ideal.
// Exponentiates the bivector and returns the motor defined by partitions 1 and 2.
#[inline(always)]
pub fn exp(a: f32x4, b: f32x4) -> (f32x4, f32x4) {
    // The exponential map produces a continuous group of rotations about an
    // axis. We'd *like* to evaluate the exp(a + b) as exp(a)exp(b) but we
    // cannot do that in general because a and b do not commute (consider
    // the differences between the Taylor expansion of exp(ab) and exp(a)exp(b)).

    // First, we need to decompose the bivector into the sum of two
    // commutative bivectors (the product of these two parts will be a
    // scalar multiple of the pseudoscalar; see "Bivector times its ideal
    // axis and vice versa in demo.klein"). To do this, we compute the
    // squared norm of the bivector:
    //
    // NOTE: a sign flip is introduced since the square of a Euclidean
    // line is negative
    //
    // (a1^2 + a2^2 + a3^2) - 2(a1 b1 + a2 b2 + a3 b3) e0123

    // Broadcast dot(a, a) ignoring the scalar component to all components of a2

    let a2 = f32x4::hi_dp_bc(a, a);
    let ab = f32x4::hi_dp_bc(a, b);

    // Next, we need the sqrt of that quantity. Since e0123 squares to 0,
    // this has a closed form solution.
    //
    // sqrt(a1^2 + a2^2 + a3^2)
    //  - (a1 b1 + a2 b2 + a3 b3) / sqrt(a1^2 + a2^2 + a3^2) e0123
    //
    // (relabeling) = u + vI
    //
    // (square the above quantity yourself to quickly verify the claim)
    // Maximum relative error < 1.5*2e-12

    let a2_sqrt_rcp = a2.rsqrt_nr1();
    let u = a2 * a2_sqrt_rcp;
    // Don't forget the minus later!
    let minus_v = ab * a2_sqrt_rcp;

    // Last, we need the reciprocal of the norm to compute the normalized
    // bivector.
    //
    // 1 / sqrt(a1^2 + a2^2 + a3^2)
    //   + (a1 b1 + a2 b2 + a3 b3) / (a1^2 + a2^2 + a3^2)^(3/2) e0123
    //
    // The original bivector * the inverse norm gives us a normalized
    // bivector.
    let norm_real = a * a2_sqrt_rcp;
    let norm_ideal = b * a2_sqrt_rcp;
    // The real part of the bivector also interacts with the pseudoscalar to
    // produce a portion of the normalized ideal part
    // e12 e0123 = -e03, e31 e0123 = -e02, e23 e0123 = -e01
    // Notice how the products above actually commute
    let norm_ideal = norm_ideal - a * ab * a2_sqrt_rcp * a2.rcp_nr1();

    // The norm * our normalized bivector is the original bivector (a + b).
    // Thus, we have:
    //
    // (u + vI)n = u n + v n e0123
    //
    // Note that n and n e0123 are perpendicular (n e0123 lies on the ideal
    // plane, and all ideal components of n are extinguished after
    // polarization). As a result, we can now decompose the exponential.
    //
    // e^(u n + v n e0123) = e^(u n) e^(v n e0123) =
    // (cosu + sinu n) * (1 + v n e0123) =
    // cosu + sinu n + v n cosu e0123 + v sinu n^2 e0123 =
    // cosu + sinu n + v n cosu e0123 - v sinu e0123
    //
    // where we've used the fact that n is normalized and squares to -1.
    // Note the v here corresponds to minus_v
    let uv: [f32; 2] = [u.first(), minus_v.first()];

    let (sin, cos) = uv[0].sin_cos();

    let sinu = f32x4::all(sin);
    let p1 = f32x4::set_scalar(cos) + sinu * norm_real;

    // The second partition has contributions from both the real and ideal parts.
    let cosu = f32x4::new(cos, cos, cos, 0.0);
    let minus_vcosu = minus_v * cosu;
    let p2 = f32x4::set_scalar(uv[1] * sin) + sinu * norm_ideal + minus_vcosu * norm_real;

    (p1, p2)
}

#[inline(always)]
pub fn log(p1: f32x4, p2: f32x4) -> (f32x4, f32x4) {
    // The logarithm follows from the derivation of the exponential. Working
    // backwards, we ended up computing the exponential like so:
    //
    // cosu + sinu n + v n cosu e0123 - v sinu e0123 =
    // (cosu - v sinu e0123) + (sinu + v cosu e0123) n
    //
    // where n is the normalized bivector. If we compute the norm, that will
    // allow us to match it to sinu + vcosu e0123, which will then allow us
    // to deduce u and v.

    // The first thing we need to do is extract only the bivector components
    // from the motor.
    let bv_mask = f32x4::new(1.0, 1.0, 1.0, 0.0);
    let a = bv_mask * p1;
    let b = bv_mask * p2;

    // Next, we need to compute the norm as in the exponential.
    let a2 = f32x4::hi_dp_bc(a, a);
    // TODO: handle case when a2 is 0
    let ab = f32x4::hi_dp_bc(a, b);
    let a2_sqrt_rcp = a2.rsqrt_nr1();
    let s_scalar = (a2 * a2_sqrt_rcp).first();
    let t_scalar = (ab * a2_sqrt_rcp).first() * -1.0;
    // s + t e0123 is the norm of our bivector.

    // Store the scalar component
    let p_scalar = p1.first();

    // Store the pseudoscalar component
    let q_scalar = p2.first();

    // p = cosu
    // q = -v sinu
    // s_scalar = sinu
    // t_scalar = v cosu

    let p_zero = p_scalar.abs() < 1e-6;
    let (u, v) = if p_zero {
        (f32::atan2(-q_scalar, t_scalar), -q_scalar / s_scalar)
    } else {
        (f32::atan2(s_scalar, p_scalar), t_scalar / p_scalar)
    };

    // Now, (u + v e0123) * n when exponentiated will give us the motor, so
    // (u + v e0123) * n is the logarithm. To proceed, we need to compute
    // the normalized bivector.
    let norm_real = a * a2_sqrt_rcp;
    let norm_ideal = b * a2_sqrt_rcp;
    let norm_ideal = norm_ideal - a * ab * a2_sqrt_rcp * a2.rcp_nr1();

    let uvec = f32x4::all(u);
    let p1 = uvec * norm_real;
    let p2 = uvec * norm_ideal - f32x4::all(v) * norm_real;

    (p1, p2)
}
