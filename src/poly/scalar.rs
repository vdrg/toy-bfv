use crate::poly::{EvalAt, RingPoly};

use super::PolyLike;
use std::borrow::Cow;

/// Univariate polynomial P(y) with possibly signed coefficients.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalarPoly<T = u64> {
    pub coeffs: Vec<T>,
}

impl<T: Copy + PartialEq + From<u8>> PolyLike for ScalarPoly<T> {
    type CoeffType = T;
    fn len(&self) -> usize {
        self.coeffs.len()
    }
    fn coeff(&self, i: usize) -> T {
        self.coeffs[i]
    }
    fn coeffs(&self) -> Cow<'_, [T]> {
        Cow::Borrowed(&self.coeffs)
    }
}

impl ScalarPoly {
    pub fn new(coeffs: Vec<u64>) -> Self {
        // reduce once for hygiene
        Self { coeffs }
    }
}

// Evaluate P(y) for y in R_q (Horner)
impl EvalAt<RingPoly> for ScalarPoly<u64> {
    type Output = RingPoly;
    fn eval_at(&self, x: &RingPoly) -> RingPoly {
        let q = x.q();
        let n = x.len();
        if self.coeffs.is_empty() {
            return RingPoly::zero(q, n);
        }
        // start with top coefficient as constant polynomial
        let mut acc = {
            let a = (self.coeffs[self.len() - 1] % q + q) % q;
            let mut v = vec![0u64; n];
            v[0] = a;
            RingPoly::from_coeffs(q, v)
        };
        for idx in (0..self.len() - 1).rev() {
            acc = &acc * x;
            let a = (self.coeffs[idx] % q + q) % q;
            if a != 0 {
                let mut v = vec![0u64; n];
                v[0] = a;
                acc += &RingPoly::from_coeffs(q, v);
            }
        }
        acc
    }
}
