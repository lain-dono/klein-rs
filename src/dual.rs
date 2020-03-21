/// A dual number is a multivector of the form _p_&#xFF0B;_q_**e**&#x2080;&#x2081;&#x2082;&#x2083;
#[derive(Clone, Copy)]
pub struct Dual {
    pub(crate) p: f32,
    pub(crate) q: f32,
}

impl Dual {
    pub fn new(p: f32, q: f32) -> Self {
        Self { p, q }
    }

    #[inline]
    pub fn scalar(self) -> f32 {
        self.p
    }

    #[inline]
    pub fn e0123(self) -> f32 {
        self.q
    }
}

impl std::ops::Add for Dual {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            p: self.p + rhs.p,
            q: self.q + rhs.q,
        }
    }
}

impl std::ops::Sub for Dual {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            p: self.p - rhs.p,
            q: self.q - rhs.q,
        }
    }
}

impl std::ops::Mul<f32> for Dual {
    type Output = Self;
    fn mul(self, s: f32) -> Self::Output {
        Self {
            p: self.p * s,
            q: self.q * s,
        }
    }
}

impl std::ops::Div<f32> for Dual {
    type Output = Self;
    fn div(self, s: f32) -> Self::Output {
        Self {
            p: self.p / s,
            q: self.q / s,
        }
    }
}
