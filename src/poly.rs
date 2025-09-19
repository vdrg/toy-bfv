use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub mod domain;
pub use domain::{Domain, DomainRef};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poly {
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
            PolyError::LengthMismatch { expected, actual } => {
                write!(
                    f,
                    "coefficient length {actual} does not match n = {expected}"
                )
            }
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

    pub fn from_coeffs(domain: &DomainRef, mut coeffs: Vec<i64>) -> Result<Self, PolyError> {
        if coeffs.len() != domain.n() {
            return Err(PolyError::LengthMismatch {
                expected: domain.n(),
                actual: coeffs.len(),
            });
        }
        let q = domain.q() as i64;
        for c in &mut coeffs {
            *c = c.rem_euclid(q);
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

    #[inline]
    pub fn coeffs(&self) -> &[i64] {
        &self.coeffs
    }

    /// Multiply by a small scalar modulo q.
    pub fn mul_scalar(&self, k: i64) -> Poly {
        let q = self.q() as i128;
        let coeffs = self
            .coeffs
            .iter()
            .map(|&c| ((c as i128 * k as i128).rem_euclid(q) as i64))
            .collect();
        Poly {
            coeffs,
            domain: self.domain.clone(),
        }
    }

    /// Compute round((num/den) * self) coefficient-wise, then reduce into [0,q).
    pub fn scale_and_round(&self, num: i64, den: i64) -> Poly {
        let q = self.q() as i64;
        let num128 = num as i128;
        let den128 = den as i128;
        let coeffs = self
            .coeffs
            .iter()
            .map(|&c| {
                let v = (c as i128) * num128;
                let r = ((v + den128 / 2) / den128) as i64;
                r.rem_euclid(q)
            })
            .collect();
        Poly {
            coeffs,
            domain: self.domain.clone(),
        }
    }

    /// Embed residues mod `t` back into Z_q (requires t <= q).
    pub fn reduce_mod(&self, t: i64) -> Poly {
        let q = self.q() as i64;
        assert!(t > 0);
        assert!(t <= q, "reduce requires t <= q");
        let coeffs = self.coeffs.iter().map(|&c| c.rem_euclid(t)).collect();
        Poly {
            coeffs,
            domain: self.domain.clone(),
        }
    }

    /// Return coefficients in centered representation (-q/2, q/2].
    pub fn coeffs_centered(&self) -> Vec<i64> {
        let q = self.q() as i64;
        let half = (q + 1) / 2; // ceil(q/2)
        self.coeffs
            .iter()
            .map(|&v| if v >= half { v - q } else { v })
            .collect()
    }

    // Note: we keep the internal invariant that coefficients are stored in [0,q),
    // so we expose centered values via `coeffs_centered` without changing storage.
}

#[inline]
fn assert_same_domain(a: &Poly, b: &Poly) {
    // pointer-equality fast path; fallback to (n, q)
    let same_ptr = a.domain.ptr_eq(&b.domain);
    if !(same_ptr || (a.n() == b.n() && a.q() == b.q())) {
        panic!("domain mismatch");
    }
}

impl<'a, 'b> Add<&'b Poly> for &'a Poly {
    type Output = Poly;
    fn add(self, rhs: &'b Poly) -> Poly {
        assert_same_domain(self, rhs);
        let q = self.q() as i64;
        let coeffs = self
            .coeffs
            .iter()
            .zip(rhs.coeffs.iter())
            .map(|(&a, &b)| (a + b).rem_euclid(q))
            .collect();
        Poly {
            coeffs,
            domain: self.domain.clone(),
        }
    }
}

impl<'a, 'b> Sub<&'b Poly> for &'a Poly {
    type Output = Poly;
    fn sub(self, rhs: &'b Poly) -> Poly {
        assert_same_domain(self, rhs);
        let q = self.q() as i64;
        let coeffs = self
            .coeffs
            .iter()
            .zip(rhs.coeffs.iter())
            .map(|(&a, &b)| (a - b).rem_euclid(q))
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
        let coeffs = self.coeffs.iter().map(|&c| (-c).rem_euclid(q)).collect();
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
                    acc[k - n] -= term; // x^n == -1
                }
            }
        }
        let coeffs = acc.into_iter().map(|v| v.rem_euclid(q) as i64).collect();
        Poly {
            coeffs,
            domain: self.domain.clone(),
        }
    }
}

impl<'a> Mul<i64> for &'a Poly {
    type Output = Poly;
    fn mul(self, rhs: i64) -> Poly {
        self.mul_scalar(rhs)
    }
}

impl Mul<i64> for Poly {
    type Output = Poly;
    fn mul(mut self, rhs: i64) -> Poly {
        self *= rhs; // uses MulAssign<i64>
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

// Sub
impl Sub<Poly> for Poly {
    type Output = Poly;
    fn sub(mut self, rhs: Poly) -> Poly {
        self -= &rhs;
        self
    }
}
impl Sub<&Poly> for Poly {
    type Output = Poly;
    fn sub(mut self, rhs: &Poly) -> Poly {
        self -= rhs;
        self
    }
}
impl Sub<Poly> for &Poly {
    type Output = Poly;
    fn sub(self, rhs: Poly) -> Poly {
        self - &rhs
    }
}

// Mul
impl Mul<Poly> for Poly {
    type Output = Poly;
    fn mul(mut self, rhs: Poly) -> Poly {
        self *= &rhs;
        self
    }
}
impl Mul<&Poly> for Poly {
    type Output = Poly;
    fn mul(mut self, rhs: &Poly) -> Poly {
        self *= rhs;
        self
    }
}
impl Mul<Poly> for &Poly {
    type Output = Poly;
    fn mul(self, rhs: Poly) -> Poly {
        self * &rhs
    }
}

impl AddAssign<&Poly> for Poly {
    fn add_assign(&mut self, rhs: &Poly) {
        assert_same_domain(self, rhs);
        let q = self.q() as i64;
        for (a, &b) in self.coeffs.iter_mut().zip(rhs.coeffs.iter()) {
            *a = (*a + b).rem_euclid(q);
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
        let q = self.q() as i64;
        for (a, &b) in self.coeffs.iter_mut().zip(rhs.coeffs.iter()) {
            *a = (*a - b).rem_euclid(q);
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
        // reuse your &Poly * &Poly impl
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
            *c = (((*c as i128) * (rhs as i128)).rem_euclid(q)) as i64;
        }
    }
}
