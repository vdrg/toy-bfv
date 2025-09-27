use super::PolyLike;
use std::borrow::Cow;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub mod centered;
pub mod domain;
pub mod monomial;
pub mod rotation;

pub use monomial::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingPoly {
    q: u64,
    coeffs: Vec<u64>,
}

impl PolyLike for RingPoly {
    type CoeffType = u64;
    fn len(&self) -> usize {
        self.coeffs.len()
    }
    fn coeff(&self, i: usize) -> u64 {
        self.coeffs[i]
    }
    fn coeffs(&self) -> Cow<'_, [u64]> {
        Cow::Borrowed(&self.coeffs)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum RingPolyError {
    LengthMismatch { expected: usize, actual: usize },
}

impl std::fmt::Display for RingPolyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RingPolyError::LengthMismatch { expected, actual } => {
                write!(
                    f,
                    "coefficient length {actual} does not match expected = {expected}"
                )
            }
        }
    }
}

impl std::error::Error for RingPolyError {}

impl RingPoly {
    pub fn zero(q: u64, n: usize) -> Self {
        Self {
            q,
            coeffs: vec![0; n],
        }
    }

    /// Build a polynomial from coefficients over integers.
    pub fn from_coeffs(q: u64, coeffs: Vec<u64>) -> Self {
        Self { q, coeffs }
    }

    pub fn from_i128_coeffs(q: u64, coeffs: &[i128]) -> Self {
        let modulus = q as i128;
        let mapped = coeffs
            .iter()
            .map(|&v| ((v % modulus + modulus) % modulus) as u64)
            .collect();
        Self { q, coeffs: mapped }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.coeffs.len()
    }

    #[inline]
    pub fn coeffs(&self) -> &[u64] {
        &self.coeffs
    }

    #[inline]
    pub fn coeff(&self, i: usize) -> u64 {
        debug_assert!(i < self.len());
        self.coeffs[i]
    }

    // TODO: move these into a trait shared by poly and RotView
    #[inline]
    pub fn const_term(&self) -> u64 {
        self.coeff(0)
    }

    #[inline]
    pub fn q(&self) -> u64 {
        self.q
    }

    #[inline]
    pub fn constant(q: u64, n: usize, value: u64) -> RingPoly {
        let mut coeffs = vec![0u64; n];
        coeffs[0] = value % q;
        RingPoly::from_coeffs(q, coeffs)
    }

    pub fn mod_q(&self, new_q: u64) -> RingPoly {
        assert!(new_q != 0);
        let coeffs = self.coeffs.iter().map(|&c| c % new_q).collect();
        RingPoly { q: new_q, coeffs }
    }

    pub fn mod_q_centered(&self, new_q: u64) -> RingPoly {
        assert!(new_q != 0);
        let old_q = self.q as i128;
        let new_q_i = new_q as i128;
        let half = (old_q - 1) / 2;
        let coeffs = self
            .coeffs
            .iter()
            .map(|&c| {
                let mut x = c as i128;
                if x > half {
                    x -= old_q;
                }
                let mut r = x % new_q_i;
                if r < 0 {
                    r += new_q_i;
                }
                r as u64
            })
            .collect();
        RingPoly { q: new_q, coeffs }
    }

    /// Per-coeff: r = round(v * num / den), with v in [0, q).
    /// Uses the *unreduced* product v*num before dividing, then maps to [0, q).
    /// Ties are rounded up (half-up). Panics if den == 0.
    pub fn scale_round(&self, num: u64, den: u64) -> RingPoly {
        assert!(den != 0, "division by zero");

        let q = self.q;
        let half_q = q / 2;
        let num128 = num as u128;
        let den128 = den as u128;

        // round(|x| * num / den) with half-up rounding.
        #[inline]
        fn round_mul_div_u128(abs_x: u64, num128: u128, den128: u128) -> u64 {
            let prod = (abs_x as u128) * num128;
            let rounded = (prod + den128 / 2) / den128; // nearest, ties up
            rounded as u64
        }

        let coeffs = self
            .coeffs
            .iter()
            .copied()
            .map(|v| {
                if v <= half_q {
                    // x >= 0  (center-lift is x = v)
                    round_mul_div_u128(v, num128, den128) % q
                } else {
                    // x < 0   (center-lift is x = v - q, so |x| = q - v)
                    let t = round_mul_div_u128(q - v, num128, den128) % q;

                    // Return (-t) mod q
                    (q - t) % q
                }
            })
            .collect();

        RingPoly { q, coeffs }
    }

    pub fn div_round(&self, den: u64) -> RingPoly {
        assert!(den != 0, "division by zero");

        let den_i128 = den as i128;
        let q_i128 = self.q as i128;
        let half = (q_i128 - 1) / 2;

        let coeffs = self
            .coeffs
            .iter()
            .map(|&c| {
                let mut x = c as i128;
                if x > half {
                    x -= q_i128;
                }
                let rounded = if x >= 0 {
                    (x + den_i128 / 2) / den_i128
                } else {
                    (x - den_i128 / 2) / den_i128
                };
                (((rounded % q_i128) + q_i128) % q_i128) as u64
            })
            .collect();

        RingPoly { q: self.q, coeffs }
    }

    pub fn coeffs_centered_i128(&self) -> Vec<i128> {
        let q = self.q as i128;
        let half = q / 2;
        self.coeffs
            .iter()
            .map(|&c| {
                let v = c as i128;
                if v <= half { v } else { v - q }
            })
            .collect()
    }

    pub fn negacyclic_convolve_centered(&self, rhs: &RingPoly) -> Vec<i128> {
        assert_same_domain(self, rhs);
        let n = self.len();
        let a = self.coeffs_centered_i128();
        let b = rhs.coeffs_centered_i128();
        let mut acc = vec![0i128; n];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                let k = i + j;
                let term = ai * bj;
                if k < n {
                    acc[k] += term;
                } else {
                    acc[k - n] -= term;
                }
            }
        }
        acc
    }

    pub fn scale_round_raw(values: &[i128], num: u64, den: u64) -> Vec<i128> {
        assert!(den != 0, "division by zero");
        let num_i128 = num as i128;
        let den_i128 = den as i128;
        values
            .iter()
            .map(|&v| {
                let scaled = v * num_i128;
                if scaled >= 0 {
                    (scaled + den_i128 / 2) / den_i128
                } else {
                    (scaled - den_i128 / 2) / den_i128
                }
            })
            .collect()
    }

    /// Multiply by a scalar.
    pub fn mul_scalar(&self, k: u64) -> RingPoly {
        let q = self.q as u128;
        let k = k as u128;
        let coeffs = self
            .coeffs
            .iter()
            .map(|&c| ((c as u128 * k) % q) as u64)
            .collect();
        RingPoly { q: self.q, coeffs }
    }
}

pub fn assert_same_domain(a: &RingPoly, b: &RingPoly) {
    assert_same_len(a, b);
    if a.q != b.q {
        panic!("modulus mismatch: {} vs {}", a.q, b.q);
    }
}

fn assert_same_len(a: &RingPoly, b: &RingPoly) {
    if a.len() != b.len() {
        panic!("length mismatch: {} vs {}", a.len(), b.len());
    }
}

// Integer Addition (wrapping with mod q)
impl<'a, 'b> Add<&'b RingPoly> for &'a RingPoly {
    type Output = RingPoly;
    fn add(self, rhs: &'b RingPoly) -> RingPoly {
        assert_same_domain(self, rhs);
        let q = self.q;
        let coeffs = self
            .coeffs
            .iter()
            .zip(rhs.coeffs.iter())
            .map(|(&a, &b)| (a.wrapping_add(b)) % q)
            .collect();
        RingPoly { coeffs, q }
    }
}

impl Add<RingPoly> for RingPoly {
    type Output = RingPoly;
    fn add(mut self, rhs: RingPoly) -> RingPoly {
        self += &rhs;
        self
    }
}

impl Add<&RingPoly> for RingPoly {
    type Output = RingPoly;
    fn add(mut self, rhs: &RingPoly) -> RingPoly {
        self += rhs;
        self
    }
}

impl Add<RingPoly> for &RingPoly {
    type Output = RingPoly;
    fn add(self, rhs: RingPoly) -> RingPoly {
        self + &rhs
    }
}

impl AddAssign<&RingPoly> for RingPoly {
    fn add_assign(&mut self, rhs: &RingPoly) {
        assert_same_domain(self, rhs);
        let q = self.q;

        for (a, &b) in self.coeffs.iter_mut().zip(rhs.coeffs.iter()) {
            *a = (a.wrapping_add(b)) % q;
        }
    }
}

impl AddAssign<RingPoly> for RingPoly {
    fn add_assign(&mut self, rhs: RingPoly) {
        *self += &rhs;
    }
}

impl<'a, 'b> Sub<&'b RingPoly> for &'a RingPoly {
    type Output = RingPoly;
    fn sub(self, rhs: &'b RingPoly) -> RingPoly {
        assert_same_domain(self, rhs);
        let q = self.q;
        let coeffs = self
            .coeffs
            .iter()
            .zip(rhs.coeffs.iter())
            .map(|(&a, &b)| (a + q - (b % q)) % q)
            .collect();
        RingPoly { coeffs, q: self.q }
    }
}

// Negation (wrapping with mod q)
impl<'a> Neg for &'a RingPoly {
    type Output = RingPoly;
    fn neg(self) -> RingPoly {
        let coeffs = self
            .coeffs
            .iter()
            .map(|&c| (self.q - (c % self.q)) % self.q)
            .collect();
        RingPoly { coeffs, q: self.q }
    }
}

impl Neg for RingPoly {
    type Output = RingPoly;
    fn neg(self) -> RingPoly {
        -&self
    }
}

impl<'a, 'b> Mul<&'b RingPoly> for &'a RingPoly {
    type Output = RingPoly;
    fn mul(self, rhs: &'b RingPoly) -> RingPoly {
        assert_same_domain(self, rhs);
        let q = self.q;
        let n = self.len();
        let mut acc = vec![0u128; n];
        for (i, &a) in self.coeffs.iter().enumerate() {
            for (j, &b) in rhs.coeffs.iter().enumerate() {
                let k = i + j;
                let term = (a as u128) * (b as u128);
                if k < n {
                    acc[k] = (acc[k] + term) % q as u128;
                } else {
                    acc[k - n] = (acc[k - n] + q as u128 - (term % q as u128)) % q as u128;
                }
            }
        }
        let coeffs = acc.into_iter().map(|v| v as u64).collect();
        RingPoly { coeffs, q }
    }
}

impl<'a> Mul<RingPoly> for &'a RingPoly {
    type Output = RingPoly;
    fn mul(self, rhs: RingPoly) -> RingPoly {
        self * &rhs
    }
}

impl<'a> Mul<&'a RingPoly> for RingPoly {
    type Output = RingPoly;
    fn mul(self, rhs: &'a RingPoly) -> RingPoly {
        &self * rhs
    }
}

impl Mul<RingPoly> for RingPoly {
    type Output = RingPoly;
    fn mul(self, rhs: RingPoly) -> RingPoly {
        &self * &rhs
    }
}

// Scalar Mul
impl<'a> Mul<u64> for &'a RingPoly {
    type Output = RingPoly;
    fn mul(self, rhs: u64) -> RingPoly {
        self.mul_scalar(rhs)
    }
}

impl Mul<u64> for RingPoly {
    type Output = RingPoly;
    fn mul(mut self, rhs: u64) -> RingPoly {
        self *= rhs;
        self
    }
}

impl<'a> Mul<&'a RingPoly> for u64 {
    type Output = RingPoly;
    fn mul(self, rhs: &'a RingPoly) -> RingPoly {
        rhs.mul_scalar(self)
    }
}

impl Mul<RingPoly> for u64 {
    type Output = RingPoly;
    fn mul(self, rhs: RingPoly) -> RingPoly {
        self * &rhs
    }
}

impl MulAssign<&RingPoly> for RingPoly {
    fn mul_assign(&mut self, rhs: &RingPoly) {
        let r = (&*self) * rhs;
        *self = r;
    }
}

impl MulAssign<RingPoly> for RingPoly {
    fn mul_assign(&mut self, rhs: RingPoly) {
        *self *= &rhs;
    }
}

impl MulAssign<u64> for RingPoly {
    fn mul_assign(&mut self, rhs: u64) {
        for c in &mut self.coeffs {
            *c = (*c * rhs) % self.q;
        }
    }
}

impl SubAssign<&RingPoly> for RingPoly {
    fn sub_assign(&mut self, rhs: &RingPoly) {
        assert_same_domain(self, rhs);
        let q = self.q;
        for (a, &b) in self.coeffs.iter_mut().zip(rhs.coeffs.iter()) {
            *a = (*a + q - (b % q)) % q;
        }
    }
}

impl SubAssign<RingPoly> for RingPoly {
    fn sub_assign(&mut self, rhs: RingPoly) {
        *self -= &rhs;
    }
}
