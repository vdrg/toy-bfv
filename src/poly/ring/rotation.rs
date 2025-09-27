use super::{RingPoly, xk};

pub struct RotView<'a> {
    p: &'a RingPoly,
    k: usize, // rotation by x^k
}

impl<'a> RotView<'a> {
    #[inline]
    pub fn new(p: &'a RingPoly, k: usize) -> Self {
        Self { p, k: k % p.len() }
    }

    /// Coefficient at index i of x^k * p (mod x^n+1), computed on the fly.
    #[inline]
    pub fn coeff(&self, i: usize) -> u64 {
        let n = self.p.len();
        let q = self.p.q();
        let k = self.k;
        debug_assert!(i < n);

        // We need p_{i-k} for i>=k, else -(p_{i-k+n}) if i<k
        if i >= k {
            self.p.coeff(i - k) % q
        } else {
            let a = self.p.coeff(i + n - k) % q;
            if a == 0 { 0 } else { (q - a) % q } // negacyclic wrap
        }
    }

    #[inline]
    pub fn const_term(&self) -> u64 {
        self.coeff(0)
    }

    /// Materialize the rotated polynomial by multiplying by monomial
    pub fn to_poly(&self) -> RingPoly {
        self.p * xk(self.p.len(), self.k)
    }
}

impl RingPoly {
    #[inline]
    pub fn rotate(&self, k: usize) -> RotView<'_> {
        RotView::new(self, k)
    }
}
