use crate::arch::f32x4;

/// An ideal line represents a line at infinity and corresponds to the
/// multivector:
/// _a_**e**&#x2080;&#x2081; + _b_**e**&#x2080;&#x2082; + _c_**e**&#x2080;&#x2083;
#[derive(Clone, Copy)]
pub struct IdealLine {
    pub(crate) p2: f32x4,
}

impl IdealLine {
    pub fn new(a: f32, b: f32, c: f32) -> Self {
        Self::from(f32x4::new(c, b, a, 0.0))
    }

    pub fn ideal_norm(self) -> f32 {
        self.squared_ideal_norm().sqrt()
    }

    pub fn squared_ideal_norm(self) -> f32 {
        f32x4::hi_dp(self.p2, self.p2).extract0()
    }

    /// Reversion operator
    pub fn reverse(&mut self) {
        self.p2 = self.p2 ^ f32x4::new(-0.0, -0.0, -0.0, 0.0);
    }

    pub fn reversed(mut self) -> Self {
        self.reverse();
        self
    }
}

/// The `branch` both a line through the origin and also the principal branch of
/// the logarithm of a rotor.
///
/// The rotor branch will be most commonly constructed by taking the
/// logarithm of a normalized rotor. The branch may then be linearily scaled
/// to adjust the "strength" of the rotor, and subsequently re-exponentiated
/// to create the adjusted rotor.
///
/// # Example
///
///  Suppose we have a rotor $r$ and we wish to produce a rotor
///  `$\sqrt[4]{r}$` which performs a quarter of the rotation produced by
///  $r$. We can construct it like so:
///
///  ```c++
///      kln::branch b = r.log();
///      kln::rotor r_4 = (0.25f * b).exp();
///  ```
///
/// # Note
///
///  The branch of a rotor is technically a `line`, but because there are
///  no translational components, the branch is given its own type for
///  efficiency.
#[derive(Clone, Copy)]
pub struct Branch {
    pub(crate) p1: f32x4,
}

impl Branch {
    /// Construct the branch as the following multivector:
    ///
    /// $$a \mathbf{e}_{23} + b\mathbf{e}_{31} + c\mathbf{e}_{23}$$
    ///
    /// To convince yourself this is a line through the origin, remember that
    /// such a line can be generated using the geometric product of two planes
    /// through the origin.
    pub fn new(a: f32, b: f32, c: f32) -> Self {
        Self::from(f32x4::new(c, b, a, 0.0))
    }

    /// Returns the square root of the quantity produced by `squared_norm`.
    pub fn norm(self) -> f32 {
        self.squared_norm().sqrt()
    }

    /// If a line is constructed as the regressive product (join) of
    /// two points, the squared norm provided here is the squared
    /// distance between the two points (provided the points are
    /// normalized).
    ///
    /// Returns `d^2 + e^2 + f^2`.
    pub fn squared_norm(self) -> f32 {
        f32x4::hi_dp(self.p1, self.p1).extract0()
    }

    pub fn normalize(&mut self) {
        let inv_norm = f32x4::hi_dp_bc(self.p1, self.p1).rsqrt_nr1();
        self.p1 = self.p1 * inv_norm;
    }

    pub fn normalized(mut self) -> Self {
        self.normalize();
        self
    }

    pub fn invert(&mut self) {
        let inv_norm = f32x4::hi_dp_bc(self.p1, self.p1).rsqrt_nr1();
        self.p1 = self.p1 * inv_norm;
        self.p1 = self.p1 * inv_norm;
        self.p1 = f32x4::new(-0.0, -0.0, -0.0, 0.0) ^ self.p1;
    }

    pub fn inverse(mut self) -> Self {
        self.invert();
        self
    }

    pub fn reverse(&mut self) {
        self.p1 = self.p1 ^ f32x4::new(-0.0, -0.0, -0.0, 0.0);
    }

    pub fn reversed(mut self) -> Self {
        self.reverse();
        self
    }
}

/// A general line in `PGA` is given as a 6-coordinate bivector with a direct
/// correspondence to Plücker coordinates. All lines can be exponentiated using
/// the `exp` method to generate a motor.
#[derive(Clone, Copy)]
pub struct Line {
    // p1: (1, e12, e31, e23)
    pub(crate) p1: f32x4,
    // p2: (e0123, e01, e02, e03)
    pub(crate) p2: f32x4,
}

impl Line {
    /// A line is specifed by 6 coordinates which correspond to the line's
    /// [Plücker coordinates](https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates).
    /// The coordinates specified in this way correspond to the following
    /// multivector:
    ///
    /// $$a\mathbf{e}_{01} + b\mathbf{e}_{02} + c\mathbf{e}_{03} +\
    /// d\mathbf{e}_{23} + e\mathbf{e}_{31} + f\mathbf{e}_{12}$$
    #[allow(clippy::many_single_char_names)]
    pub fn new(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32) -> Self {
        Self {
            p1: f32x4::new(f, e, d, 0.0),
            p2: f32x4::new(c, b, a, 0.0),
        }
    }

    pub fn from_ideal_line(ideal_line: IdealLine) -> Self {
        Self {
            p1: f32x4::zero(),
            p2: ideal_line.p2,
        }
    }

    pub fn from_branch(branch: Branch) -> Self {
        Self {
            p1: branch.p1,
            p2: f32x4::zero(),
        }
    }

    /// Returns the square root of the quantity produced by
    /// `squared_norm`.
    pub fn norm(self) -> f32 {
        self.squared_norm().sqrt()
    }

    /// If a line is constructed as the regressive product (join) of
    /// two points, the squared norm provided here is the squared
    /// distance between the two points (provided the points are
    /// normalized). Returns $d^2 + e^2 + f^2$.
    pub fn squared_norm(self) -> f32 {
        f32x4::hi_dp(self.p1, self.p1).extract0()
    }

    /// Normalize a line such that $\ell^2 = -1$.
    pub fn normalize(&mut self) {
        // l = b + c where b is p1 and c is p2
        // l * ~l = |b|^2 - 2(b1 c1 + b2 c2 + b3 c3)e0123
        //
        // sqrt(l*~l) = |b| - (b1 c1 + b2 c2 + b3 c3)/|b| e0123
        //
        // 1/sqrt(l*~l) = 1/|b| + (b1 c1 + b2 c2 + b3 c3)/|b|^3 e0123
        //              = s + t e0123
        let b2 = f32x4::hi_dp_bc(self.p1, self.p1);
        let s = b2.rsqrt_nr1();
        let bc = f32x4::hi_dp_bc(self.p1, self.p2);
        let t = bc * b2.rcp_nr1() * s;

        // p1 * (s + t e0123) = s * p1 - t p1_perp
        self.p2 = self.p2 * s - self.p1 * t;
        self.p1 = self.p1 * s;
    }

    /// Return a normalized copy of this line
    pub fn normalized(mut self) -> Self {
        self.normalize();
        self
    }

    pub fn invert(&mut self) {
        // s, t computed as in the normalization
        let b2 = f32x4::hi_dp_bc(self.p1, self.p1);
        let s = b2.rsqrt_nr1();
        let bc = f32x4::hi_dp_bc(self.p1, self.p2);
        let b2_inv = b2.rcp_nr1();
        let t = bc * b2_inv * s;
        let neg = f32x4::new(-0.0, -0.0, -0.0, 0.0);

        // p1 * (s + t e0123)^2 = (s * p1 - t p1_perp) * (s + t e0123)
        // = s^2 p1 - s t p1_perp - s t p1_perp
        // = s^2 p1 - 2 s t p1_perp
        // p2 * (s + t e0123)^2 = s^2 p2
        // NOTE: s^2 = b2_inv
        let st = self.p1 * s * t;
        self.p2 = (self.p2 * b2_inv - (st + st)) ^ neg;
        self.p1 = (self.p1 * b2_inv) ^ neg;
    }

    pub fn inverse(mut self) -> Self {
        self.invert();
        self
    }

    pub fn reverse(&mut self) {
        let flip = f32x4::new(-0.0, -0.0, -0.0, 0.0);
        self.p1 = self.p1 ^ flip;
        self.p2 = self.p2 ^ flip;
    }

    pub fn reversed(mut self) -> Self {
        self.reverse();
        self
    }

    pub fn line_eq(self, other: Self) -> bool {
        f32x4::bit_eq_pair(self.into(), other.into())
    }

    pub fn approx_eq(self, other: Self, epsilon: f32) -> bool {
        f32x4::approx_eq_pair(self.into(), other.into(), epsilon)
    }
}
