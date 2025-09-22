use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub mod domain;
pub use domain::{Domain, DomainRef};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poly {
    coeffs: Vec<i64>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum PolyError {
    LengthMismatch { expected: usize, actual: usize },
}

impl std::fmt::Display for PolyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PolyError::LengthMismatch { expected, actual } => {
                write!(
                    f,
                    "coefficient length {actual} does not match expected = {expected}"
                )
            }
        }
    }
}

impl std::error::Error for PolyError {}

impl Poly {
    pub fn zero(n: usize) -> Self {
        Self { coeffs: vec![0; n] }
    }

    /// Build a polynomial from coefficients over integers (no reduction).
    pub fn from_coeffs(coeffs: Vec<i64>) -> Self {
        Self { coeffs }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.coeffs.len()
    }

    #[inline]
    pub fn coeffs(&self) -> &[i64] {
        &self.coeffs
    }

    /// Reduce coefficients modulo q and center into (-q/2, q/2].
    /// Returns a new Poly.
    pub fn reduce_mod(&self, q: u64) -> Poly {
        let q_i64 = q as i64;
        let half_q = q_i64 / 2;
        let coeffs = self
            .coeffs
            .iter()
            .map(|&c| {
                let mut r = c.rem_euclid(q_i64); // [0, q)
                if r > half_q {
                    r -= q_i64;
                }
                r
            })
            .collect();
        Poly { coeffs }
    }

    pub fn div_round(&self, q: u64) -> Poly {
        let coeffs = self
            .coeffs
            .iter()
            .map(|&v| Self::round_div(v as i128, q as i128) as i64)
            .collect();

        Poly { coeffs }
    }

    /// Multiply by a small scalar over integers.
    pub fn mul_scalar(&self, k: i64) -> Poly {
        let coeffs = self.coeffs.iter().map(|&c| c * k).collect();
        Poly { coeffs }
    }

    #[inline]
    fn floor_div(a: i128, b: i128) -> i128 {
        let d = a / b;
        let r = a % b;
        if r != 0 && ((a < 0) != (b < 0)) {
            d - 1
        } else {
            d
        }
    }

    #[inline]
    fn round_div(v: i128, d: i128) -> i128 {
        let quot = Self::floor_div(v, d);
        let rem = v - quot * d; // rem in [0, d)
        let half = (d + 1) / 2;
        if rem >= half { quot + 1 } else { quot }
    }
}

fn assert_same_len(a: &Poly, b: &Poly) {
    if a.len() != b.len() {
        panic!("length mismatch: {} vs {}", a.len(), b.len());
    }
}

// Integer Addition
impl<'a, 'b> Add<&'b Poly> for &'a Poly {
    type Output = Poly;
    fn add(self, rhs: &'b Poly) -> Poly {
        assert_same_len(self, rhs);
        let coeffs = self
            .coeffs
            .iter()
            .zip(rhs.coeffs.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        Poly { coeffs }
    }
}

impl Add<Poly> for Poly {
    type Output = Poly;
    fn add(mut self, rhs: Poly) -> Poly {
        self += &rhs;
        self
    }
}

impl Add<&Poly> for Poly {
    type Output = Poly;
    fn add(mut self, rhs: &Poly) -> Poly {
        self += rhs;
        self
    }
}

impl Add<Poly> for &Poly {
    type Output = Poly;
    fn add(self, rhs: Poly) -> Poly {
        self + &rhs
    }
}

impl AddAssign<&Poly> for Poly {
    fn add_assign(&mut self, rhs: &Poly) {
        assert_same_len(self, rhs);
        for (a, &b) in self.coeffs.iter_mut().zip(rhs.coeffs.iter()) {
            *a += b;
        }
    }
}

impl AddAssign<Poly> for Poly {
    fn add_assign(&mut self, rhs: Poly) {
        *self += &rhs;
    }
}

impl<'a, 'b> Sub<&'b Poly> for &'a Poly {
    type Output = Poly;
    fn sub(self, rhs: &'b Poly) -> Poly {
        assert_same_len(self, rhs);
        let coeffs = self
            .coeffs
            .iter()
            .zip(rhs.coeffs.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        Poly { coeffs }
    }
}

impl<'a> Neg for &'a Poly {
    type Output = Poly;
    fn neg(self) -> Poly {
        let coeffs = self.coeffs.iter().map(|&c| -c).collect();
        Poly { coeffs }
    }
}

impl Neg for Poly {
    type Output = Poly;
    fn neg(self) -> Poly {
        -&self
    }
}

impl<'a, 'b> Mul<&'b Poly> for &'a Poly {
    type Output = Poly;
    fn mul(self, rhs: &'b Poly) -> Poly {
        assert_same_len(self, rhs);
        let n = self.len();
        let mut acc = vec![0i128; n];
        for (i, &a) in self.coeffs.iter().enumerate() {
            for (j, &b) in rhs.coeffs.iter().enumerate() {
                let k = i + j;
                let term = (a as i128) * (b as i128);
                if k < n {
                    acc[k] += term;
                } else {
                    acc[k - n] -= term;
                }
            }
        }
        let coeffs = acc.into_iter().map(|v| v as i64).collect(); // Assume no overflow for small n; use checked if needed
        Poly { coeffs }
    }
}

// Scalar Mul
impl<'a> Mul<i64> for &'a Poly {
    type Output = Poly;
    fn mul(self, rhs: i64) -> Poly {
        self.mul_scalar(rhs)
    }
}

impl Mul<i64> for Poly {
    type Output = Poly;
    fn mul(mut self, rhs: i64) -> Poly {
        self *= rhs;
        self
    }
}

impl<'a> Mul<&'a Poly> for i64 {
    type Output = Poly;
    fn mul(self, rhs: &'a Poly) -> Poly {
        rhs.mul_scalar(self)
    }
}

impl Mul<Poly> for i64 {
    type Output = Poly;
    fn mul(self, rhs: Poly) -> Poly {
        self * &rhs
    }
}

impl MulAssign<&Poly> for Poly {
    fn mul_assign(&mut self, rhs: &Poly) {
        let r = (&*self) * rhs;
        *self = r;
    }
}

impl MulAssign<Poly> for Poly {
    fn mul_assign(&mut self, rhs: Poly) {
        *self *= &rhs;
    }
}

impl MulAssign<i64> for Poly {
    fn mul_assign(&mut self, rhs: i64) {
        for c in &mut self.coeffs {
            *c *= rhs;
        }
    }
}

impl SubAssign<&Poly> for Poly {
    fn sub_assign(&mut self, rhs: &Poly) {
        assert_same_len(self, rhs);
        for (a, &b) in self.coeffs.iter_mut().zip(rhs.coeffs.iter()) {
            *a -= b;
        }
    }
}

impl SubAssign<Poly> for Poly {
    fn sub_assign(&mut self, rhs: Poly) {
        *self -= &rhs;
    }
}
