use super::RingPoly;
use crate::poly::Poly;
use std::borrow::Cow;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Monomial {
    pub n: usize,
    pub k: usize, // represents x^k
}

#[inline]
pub fn xk(n: usize, k: usize) -> Monomial {
    Monomial { n, k }
}

impl Poly for Monomial {
    type CoeffType = u64;
    fn len(&self) -> usize {
        self.n
    }
    fn coeff(&self, i: usize) -> u64 {
        if (i % self.n) == self.k { 1 } else { 0 }
    }
    fn coeffs(&self) -> Cow<'_, [u64]> {
        let mut v = vec![0u64; self.n];
        v[self.k] = 1;
        Cow::Owned(v)
    }
}

// Multiply RingPoly by x^k  (O(n), no extra allocs besides the output RingPoly)
impl<'a> std::ops::Mul<Monomial> for &'a RingPoly {
    type Output = RingPoly;
    fn mul(self, m: Monomial) -> RingPoly {
        let n = self.len();
        let q = self.q();
        let k = m.k % n;
        if k == 0 {
            return self.clone();
        }
        let mut out = vec![0u64; n];
        // Negacyclic rotation: wrap with a minus
        for i in 0..n {
            let a = self.coeffs[i];
            if a == 0 {
                continue;
            }
            let j = i + k;
            if j < n {
                // +a at j
                out[j] = (out[j] + a) % q;
            } else {
                // -(a) at j-n
                let idx = j - n;
                let neg = if a == 0 { 0 } else { (q - a) % q };
                out[idx] = (out[idx] + neg) % q;
            }
        }
        RingPoly { q, coeffs: out }
    }
}

impl std::ops::Mul<Monomial> for RingPoly {
    type Output = RingPoly;
    fn mul(self, m: Monomial) -> RingPoly {
        (&self) * m
    }
}
impl<'a> std::ops::Mul<&'a RingPoly> for Monomial {
    type Output = RingPoly;
    fn mul(self, rhs: &'a RingPoly) -> RingPoly {
        rhs * self
    }
}
impl std::ops::Mul<RingPoly> for Monomial {
    type Output = RingPoly;
    fn mul(self, rhs: RingPoly) -> RingPoly {
        (&rhs) * self
    }
}
