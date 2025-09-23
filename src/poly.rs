use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub mod domain;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poly {
    q: u64,
    coeffs: Vec<u64>,
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

    #[inline]
    pub fn len(&self) -> usize {
        self.coeffs.len()
    }

    #[inline]
    pub fn coeffs(&self) -> &[u64] {
        &self.coeffs
    }

    #[inline]
    pub fn q(&self) -> u64 {
        self.q
    }

    /// Reduce coefficients modulo q.
    /// Returns the Poly mod q.
    // pub fn reduce_mod(&self, q: u64) -> Poly {
    //     if self.q <= q {
    //         return Poly {
    //             q,
    //             coeffs: self.coeffs.clone(),
    //         };
    //     }
    //
    //     println!("coeffs {:?}", &self.coeffs);
    //     let coeffs = self.coeffs.iter().map(|&c| c % q).collect();
    //     Poly { q, coeffs }
    // }

    pub fn reduce_mod(&self, q: u64) -> Poly {
        assert!(q != 0, "new_q cannot be zero");
        let m = q;
        let half = self.q / 2;
        let q_mod_m = self.q % m;
        let coeffs = self
            .coeffs
            .iter()
            .map(|&c| {
                let r = c % m;
                if c <= half { r } else { (r + m - q_mod_m) % m }
            })
            .collect();
        Poly { q: m, coeffs }
    }

    /// Per-coeff: r = round(v * num / den), with v in [0, q).
    /// Uses the *unreduced* product v*num before dividing, then maps to [0, q).
    /// Ties are rounded up (half-up). Panics if den == 0.
    pub fn scale_round(&self, num: u64, den: u64) -> Poly {
        assert!(den != 0, "division by zero");

        let q = self.q;
        let half_q = q / 2;
        let num128 = num as u128;
        let den128 = den as u128;

        // Helper: round(|x| * num / den) with half-up rounding.
        #[inline]
        fn round_mul_div_u128(abs_x: u64, num128: u128, den128: u128) -> u64 {
            let prod = (abs_x as u128) * num128;
            let rounded = (prod + den128 / 2) / den128; // nearest, ties up
            // println!(
            //     "abs_x {:?} prod {:?} rounded {:?} den {:?}",
            //     abs_x, prod, rounded, den128
            // );
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

        Poly { q, coeffs }
    }

    pub fn div_round(&self, den: u64) -> Poly {
        self.scale_round(1, den)
    }

    /// Multiply by a scalar.
    pub fn mul_scalar(&self, k: u64) -> Poly {
        let q = self.q as u128;
        let k = k as u128;
        let coeffs = self
            .coeffs
            .iter()
            .map(|&c| ((c as u128 * k) % q) as u64)
            .collect();
        Poly { q: self.q, coeffs }
    }
}

fn assert_same_domain(a: &Poly, b: &Poly) {
    assert_same_len(a, b);
    if a.q != b.q {
        panic!("modulus mismatch: {} vs {}", a.q, b.q);
    }
}

fn assert_same_len(a: &Poly, b: &Poly) {
    if a.len() != b.len() {
        panic!("length mismatch: {} vs {}", a.len(), b.len());
    }
}

// Integer Addition (wrapping with mod q)
impl<'a, 'b> Add<&'b Poly> for &'a Poly {
    type Output = Poly;
    fn add(self, rhs: &'b Poly) -> Poly {
        assert_same_domain(self, rhs);
        let q = self.q;
        let coeffs = self
            .coeffs
            .iter()
            .zip(rhs.coeffs.iter())
            .map(|(&a, &b)| (a.wrapping_add(b)) % q)
            .collect();
        Poly { coeffs, q }
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
        assert_same_domain(self, rhs);
        let q = self.q;

        for (a, &b) in self.coeffs.iter_mut().zip(rhs.coeffs.iter()) {
            *a = (a.wrapping_add(b)) % q;
        }
    }
}

impl AddAssign<Poly> for Poly {
    fn add_assign(&mut self, rhs: Poly) {
        *self += &rhs;
    }
}

// Integer Subtraction (wrapping with mod q, positive reps)
impl<'a, 'b> Sub<&'b Poly> for &'a Poly {
    type Output = Poly;
    fn sub(self, rhs: &'b Poly) -> Poly {
        assert_same_domain(self, rhs);
        let q = self.q;
        let coeffs = self
            .coeffs
            .iter()
            .zip(rhs.coeffs.iter())
            .map(|(&a, &b)| (a + q - (b % q)) % q)
            .collect();
        Poly { coeffs, q: self.q }
    }
}

// Negation (wrapping with mod q)
impl<'a> Neg for &'a Poly {
    type Output = Poly;
    fn neg(self) -> Poly {
        let coeffs = self
            .coeffs
            .iter()
            .map(|&c| (self.q - (c % self.q)) % self.q)
            .collect();
        Poly { coeffs, q: self.q }
    }
}

impl Neg for Poly {
    type Output = Poly;
    fn neg(self) -> Poly {
        -&self
    }
}

// Integer Multiplication (over ℤ[x]/(x^n +1), using u128 for acc to avoid overflow)
impl<'a, 'b> Mul<&'b Poly> for &'a Poly {
    type Output = Poly;
    fn mul(self, rhs: &'b Poly) -> Poly {
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
        Poly { coeffs, q }
    }
}

// Scalar Mul
impl<'a> Mul<u64> for &'a Poly {
    type Output = Poly;
    fn mul(self, rhs: u64) -> Poly {
        self.mul_scalar(rhs)
    }
}

impl Mul<u64> for Poly {
    type Output = Poly;
    fn mul(mut self, rhs: u64) -> Poly {
        self *= rhs;
        self
    }
}

impl<'a> Mul<&'a Poly> for u64 {
    type Output = Poly;
    fn mul(self, rhs: &'a Poly) -> Poly {
        rhs.mul_scalar(self)
    }
}

impl Mul<Poly> for u64 {
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

impl MulAssign<u64> for Poly {
    fn mul_assign(&mut self, rhs: u64) {
        for c in &mut self.coeffs {
            *c = (*c * rhs) % self.q;
        }
    }
}

impl SubAssign<&Poly> for Poly {
    fn sub_assign(&mut self, rhs: &Poly) {
        assert_same_domain(self, rhs);
        let q = self.q;
        for (a, &b) in self.coeffs.iter_mut().zip(rhs.coeffs.iter()) {
            *a = (*a + q - (b % q)) % q;
        }
    }
}

impl SubAssign<Poly> for Poly {
    fn sub_assign(&mut self, rhs: Poly) {
        *self -= &rhs;
    }
}
