use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
pub mod domain;
pub use domain::{Domain, DomainRef};

/// Reduce v into centered interval (-⌊q/2⌋, ⌈q/2⌉] for odd q.
#[inline]
pub fn redc_centered_i128(v: i128, q: i128) -> i64 {
    let mut r = v.rem_euclid(q); // [0,q)
    let half_up = q / 2; // floor(q/2)
    if r > half_up {
        r -= q;
    } // (-q/2, q/2]
    r as i64
}

#[inline]
pub fn redc_centered_i64(v: i64, q: i64) -> i64 {
    let mut r = v.rem_euclid(q); // [0,q)
    let half_up = q / 2;
    if r > half_up {
        r -= q;
    }
    r
}

#[inline]
fn assert_same_domain(a: &Poly, b: &Poly) {
    let same_ptr = a.domain.ptr_eq(&b.domain);
    if !(same_ptr || (a.n() == b.n() && a.q() == b.q())) {
        panic!(
            "domain mismatch: (n,q)=({}, {}) vs ({}, {})",
            a.n(),
            a.q(),
            b.n(),
            b.q()
        );
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poly {
    /// Invariant: coefficients are ALWAYS stored centered in (-⌊q/2⌋, ⌈q/2⌉].
    coeffs: Vec<i64>,
    domain: DomainRef,
}

#[derive(Debug, PartialEq, Eq)]
pub enum PolyError {
    LengthMismatch { expected: usize, actual: usize },
}
impl std::fmt::Display for PolyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PolyError::LengthMismatch { expected, actual } => write!(
                f,
                "coefficient length {actual} does not match n = {expected}"
            ),
        }
    }
}
impl std::error::Error for PolyError {}

impl Poly {
    pub fn zero(domain: &DomainRef) -> Self {
        Self {
            coeffs: vec![0; domain.n()],
            domain: domain.clone(),
        }
    }

    /// Build a polynomial, **centering** every coefficient for this domain's q.
    pub fn from_coeffs(domain: &DomainRef, mut coeffs: Vec<i64>) -> Result<Self, PolyError> {
        if coeffs.len() != domain.n() {
            return Err(PolyError::LengthMismatch {
                expected: domain.n(),
                actual: coeffs.len(),
            });
        }
        let q = domain.q() as i64;
        for c in &mut coeffs {
            *c = redc_centered_i64(*c, q);
        }
        Ok(Self {
            coeffs,
            domain: domain.clone(),
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.coeffs.len()
    }
    #[inline]
    pub fn n(&self) -> usize {
        self.domain.n()
    }
    #[inline]
    pub fn q(&self) -> u64 {
        self.domain.q()
    }
    #[inline]
    pub fn domain(&self) -> &DomainRef {
        &self.domain
    }
    /// Centered coefficients (internal storage).
    #[inline]
    pub fn coeffs(&self) -> &[i64] {
        &self.coeffs
    }
    /// Canonical [0,q) view (useful before NTT/serialization).
    pub fn coeffs_canonical_mod(&self) -> Vec<i64> {
        let q = self.q() as i64;
        self.coeffs
            .iter()
            .map(|&c| if c < 0 { c + q } else { c })
            .collect()
    }
    /// Trivial (already centered).
    #[inline]
    pub fn coeffs_centered(&self) -> Vec<i64> {
        self.coeffs.clone()
    }

    /// Convert to a different modulus, still centered.
    pub fn to_domain(&self, new: &DomainRef) -> Result<Poly, PolyError> {
        if self.len() != new.n() {
            return Err(PolyError::LengthMismatch {
                expected: new.n(),
                actual: self.len(),
            });
        }
        if self.domain.ptr_eq(new) {
            return Ok(self.clone());
        }
        let new_q = new.q() as i64;
        let v = self
            .coeffs
            .iter()
            .map(|&c| redc_centered_i64(c, new_q))
            .collect();
        Ok(Poly {
            coeffs: v,
            domain: new.clone(),
        })
    }

    /// Move into another domain (by value).
    pub fn into_domain(self, new: &DomainRef) -> Result<Poly, PolyError> {
        if self.len() != new.n() {
            return Err(PolyError::LengthMismatch {
                expected: new.n(),
                actual: self.len(),
            });
        }
        if self.domain.ptr_eq(new) {
            return Ok(self);
        }
        let new_q = new.q() as i64;
        let v = self
            .coeffs
            .into_iter()
            .map(|c| redc_centered_i64(c, new_q))
            .collect();
        Ok(Poly {
            coeffs: v,
            domain: new.clone(),
        })
    }

    /// Multiply by a small scalar (centered reduce).
    pub fn mul_scalar(&self, k: i64) -> Poly {
        let q = self.q() as i128;
        let coeffs = self
            .coeffs
            .iter()
            .map(|&c| redc_centered_i128((c as i128) * (k as i128), q))
            .collect();
        Poly {
            coeffs,
            domain: self.domain.clone(),
        }
    }
}

// ----- ring operations (centered) -----

impl<'a, 'b> Add<&'b Poly> for &'a Poly {
    type Output = Poly;
    fn add(self, rhs: &'b Poly) -> Poly {
        assert_same_domain(self, rhs);
        let q = self.q() as i128;
        let coeffs = self
            .coeffs
            .iter()
            .zip(&rhs.coeffs)
            .map(|(&a, &b)| redc_centered_i128(a as i128 + b as i128, q))
            .collect();
        Poly {
            coeffs,
            domain: self.domain.clone(),
        }
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

impl<'a, 'b> Sub<&'b Poly> for &'a Poly {
    type Output = Poly;
    fn sub(self, rhs: &'b Poly) -> Poly {
        assert_same_domain(self, rhs);
        let q = self.q() as i128;
        let coeffs = self
            .coeffs
            .iter()
            .zip(&rhs.coeffs)
            .map(|(&a, &b)| redc_centered_i128(a as i128 - b as i128, q))
            .collect();
        Poly {
            coeffs,
            domain: self.domain.clone(),
        }
    }
}

impl<'a> Neg for &'a Poly {
    type Output = Poly;
    fn neg(self) -> Poly {
        let q = self.q() as i64;
        let coeffs = self
            .coeffs
            .iter()
            .map(|&c| redc_centered_i64(-c, q))
            .collect();
        Poly {
            coeffs,
            domain: self.domain.clone(),
        }
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
        assert_same_domain(self, rhs);
        let n = self.n();
        let q = self.q() as i128;
        let mut acc = vec![0i128; n];
        for (i, &a) in self.coeffs.iter().enumerate() {
            for (j, &b) in rhs.coeffs.iter().enumerate() {
                let k = i + j;
                let term = (a as i128) * (b as i128);
                if k < n {
                    acc[k] += term;
                } else {
                    acc[k - n] -= term;
                } // x^n == -1
            }
        }
        let coeffs = acc.into_iter().map(|v| redc_centered_i128(v, q)).collect();
        Poly {
            coeffs,
            domain: self.domain.clone(),
        }
    }
}

// scalar ops
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

// assign variants
impl AddAssign<&Poly> for Poly {
    fn add_assign(&mut self, rhs: &Poly) {
        assert_same_domain(self, rhs);
        let q = self.q() as i128;
        for (a, &b) in self.coeffs.iter_mut().zip(&rhs.coeffs) {
            *a = redc_centered_i128((*a as i128) + (b as i128), q);
        }
    }
}
impl AddAssign<Poly> for Poly {
    fn add_assign(&mut self, rhs: Poly) {
        *self += &rhs;
    }
}

impl SubAssign<&Poly> for Poly {
    fn sub_assign(&mut self, rhs: &Poly) {
        assert_same_domain(self, rhs);
        let q = self.q() as i128;
        for (a, &b) in self.coeffs.iter_mut().zip(&rhs.coeffs) {
            *a = redc_centered_i128((*a as i128) - (b as i128), q);
        }
    }
}
impl SubAssign<Poly> for Poly {
    fn sub_assign(&mut self, rhs: Poly) {
        *self -= &rhs;
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
        let q = self.q() as i128;
        for c in &mut self.coeffs {
            *c = redc_centered_i128((*c as i128) * (rhs as i128), q);
        }
    }
}
