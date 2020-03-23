use crate::{arch::f32x4, Branch, IdealLine, Line, Motor, Rotor, Translator};

impl Line {
    /// Exponentiate a line to produce a motor that posesses this line
    /// as its axis. This routine will be used most often when this line is
    /// produced as the logarithm of an existing rotor, then scaled to subdivide
    /// or accelerate the motor's action. The line need not be a _simple bivector_
    /// for the operation to be well-defined.
    pub fn exp(self) -> Motor {
        Motor::from(unsafe { crate::arch::exp(self.p1, self.p2) })
    }
}

impl Translator {
    /// Compute the logarithm of the translator, producing an ideal line axis.
    /// In practice, the logarithm of a translator is simply the ideal partition
    /// (without the scalar `1`).
    #[inline]
    pub fn log(self) -> IdealLine {
        IdealLine { p2: self.p2 }
    }

    /// Compute the square root of the provided translator.
    #[inline]
    pub fn sqrt(self) -> Self {
        self * 0.5
    }
}

impl IdealLine {
    /// Exponentiate an ideal line to produce a translation.
    ///
    /// The exponential of an ideal line
    /// $`a \mathbf{e}_{01} + b\mathbf{e}_{02} + c\mathbf{e}_{03}`$ is given as:
    ///
    /// $`\exp{\left[a\ee_{01} + b\ee_{02} + c\ee_{03}\right]} = 1 +\
    /// a\ee_{01} + b\ee_{02} + c\ee_{03}`$
    #[inline]
    pub fn exp(self) -> Translator {
        Translator { p2: self.p2 }
    }
}

impl Branch {
    /// Exponentiate a branch to produce a rotor.
    pub fn exp(self) -> Rotor {
        let p1 = f32x4::from(self.p1);

        // Compute the rotor angle
        let ang = f32x4::hi_dp(p1, p1).sqrt_nr1().first();
        let (sin, cos) = ang.sin_cos();

        let p1 = f32x4::all(sin / ang) * p1 + f32x4::set_scalar(cos);
        Rotor { p1: p1.into() }
    }

    #[inline]
    pub fn sqrt(self) -> Rotor {
        let p1 = f32x4::from(self.p1)
            .add_scalar(f32x4::set_scalar(1.0))
            .into();
        Rotor { p1 }.normalized()
    }
}

impl Rotor {
    /// Returns the principal branch of this rotor's logarithm. Invoking
    /// `exp` on the returned `kln::branch` maps back to this rotor.
    ///
    /// Given a rotor $\cos\alpha + \sin\alpha\left[a\ee_{23} + b\ee_{31} +\
    /// c\ee_{23}\right]$, the log is computed as simply
    /// $\alpha\left[a\ee_{23} + b\ee_{31} + c\ee_{23}\right]$.
    ///
    /// This map is only well-defined if the
    /// rotor is normalized such that $a^2 + b^2 + c^2 = 1$.
    #[inline]
    pub fn log(self) -> Branch {
        let p1 = f32x4::from(self.p1);
        let ang = p1.first().acos();
        let sin = f32x4::all(ang.sin());

        let p1 = p1 * sin.rcp_nr1() * f32x4::all(ang);

        let p1 = if cfg!(target_feature = "sse4.1") {
            p1.blend1(f32x4::zero())
        } else {
            p1 & f32x4::cast_i32(-1, -1, -1, 0)
        };
        Branch { p1: p1.into() }
    }

    /// Compute the square root of the provided rotor.
    #[inline]
    pub fn sqrt(self) -> Self {
        Self::from(f32x4::from(self.p1).add_scalar(f32x4::set_scalar(1.0))).normalized()
    }
}

impl Motor {
    #[inline]
    pub fn log(self) -> Line {
        Line::from(unsafe { crate::arch::log(self.p1, self.p2) })
    }

    /// Compute the square root of the provided motor.
    #[inline]
    pub fn sqrt(mut self) -> Self {
        self.p1 = f32x4::from(self.p1)
            .add_scalar(f32x4::set_scalar(1.0))
            .into();
        self.normalized()
    }
}
