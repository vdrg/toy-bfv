use std::ops::{Add, Mul, Neg, Sub};
use std::sync::Arc;

use rand::Rng;
use rand::seq::SliceRandom;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Domain {
    n: usize,
    q: u64,
}

impl Domain {
    pub fn new(n: usize, q: u64) -> Self {
        assert!(n.is_power_of_two(), "n must be a power of two");
        assert!(q % 2 == 1, "use odd q");
        Self { n, q }
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn q(&self) -> u64 {
        self.q
    }

    #[inline]
    fn arc(&self) -> Arc<Domain> {
        Arc::new(self.clone())
    }

    /// Ternary in {-1, 0, 1} per coefficient.
    pub fn sample_ternary<R: Rng>(&self, rng: &mut R) -> Poly {
        let mut v = vec![0i64; self.n];
        for c in &mut v {
            *c = (rng.random_range(0..3) as i64) - 1;
        }
        Poly::from_coeffs(self.arc(), v)
    }

    /// Uniform in [0, q) per coefficient.
    pub fn sample_uniform_mod_q<R: Rng>(&self, rng: &mut R) -> Poly {
        let mut v = vec![0i64; self.n];
        for c in &mut v {
            *c = rng.random_range(0..self.q) as i64;
        }
        Poly::from_coeffs(self.arc(), v)
    }

    /// Binary in {0,1} per coefficient.
    pub fn sample_binary01<R: Rng>(&self, rng: &mut R) -> Poly {
        let mut v = vec![0i64; self.n];
        for c in &mut v {
            *c = rng.random_range(0..2) as i64;
        }
        Poly::from_coeffs(self.arc(), v)
    }

    /// Rademacher in {-1,1} per coefficient.
    pub fn sample_rademacher<R: Rng>(&self, rng: &mut R) -> Poly {
        let mut v = vec![0i64; self.n];
        for c in &mut v {
            let bit: i64 = rng.random_range(0..2) as i64;
            *c = 2 * bit - 1; // 0->-1, 1->+1
        }
        Poly::from_coeffs(self.arc(), v)
    }

    /// Sparse ternary with exactly h non-zeros set to +/-1.
    pub fn sample_sparse_ternary<R: Rng>(&self, h: usize, rng: &mut R) -> Poly {
        assert!(h <= self.n, "h must be <= n");
        let mut v = vec![0i64; self.n];
        // choose h distinct indices uniformly
        let mut idx: Vec<usize> = (0..self.n).collect();
        idx.shuffle(rng);
        for &i in idx.iter().take(h) {
            let sign: i64 = if rng.random_range(0..2) == 0 { -1 } else { 1 };
            v[i] = sign;
        }
        Poly::from_coeffs(self.arc(), v)
    }

    /// Centered binomial distribution CBD(k): sum_k Ber(1/2) - sum_k Ber(1/2).
    pub fn sample_cbd<R: Rng>(&self, k: usize, rng: &mut R) -> Poly {
        let mut v = vec![0i64; self.n];
        for c in &mut v {
            let mut left = k;
            let mut a_sum: u32 = 0;
            let mut b_sum: u32 = 0;
            while left > 0 {
                let take = left.min(64);
                let mask: u64 = if take == 64 { !0 } else { (1u64 << take) - 1 };
                let a = rng.random::<u64>() & mask;
                let b = rng.random::<u64>() & mask;
                a_sum += a.count_ones();
                b_sum += b.count_ones();
                left -= take;
            }
            *c = (a_sum as i64) - (b_sum as i64);
        }
        Poly::from_coeffs(self.arc(), v)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poly {
    pub coeffs: Vec<i64>,
    ctx: Arc<Domain>,
}

impl Poly {
    pub fn zero(ctx: &Arc<Domain>) -> Self {
        Self {
            coeffs: vec![0; ctx.n()],
            ctx: Arc::clone(ctx),
        }
    }

    pub fn from_coeffs(ctx: Arc<Domain>, coeffs: Vec<i64>) -> Self {
        assert_eq!(coeffs.len(), ctx.n());
        let mut p = Self { coeffs, ctx };
        p.reduce_mod_q_inplace();
        p
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.coeffs.len()
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.ctx.n()
    }

    #[inline]
    pub fn q(&self) -> u64 {
        self.ctx.q()
    }

    #[inline]
    pub fn domain(&self) -> &Arc<Domain> {
        &self.ctx
    }

    fn reduce_mod_q_inplace(&mut self) {
        let q = self.q() as i64;
        for c in &mut self.coeffs {
            let mut v = *c % q;
            if v < 0 {
                v += q;
            }
            *c = v;
        }
    }

    /// Multiply by a small scalar modulo q.
    pub fn mul_scalar(&self, k: i64) -> Poly {
        let q = self.q() as i64;
        let mut out = Poly::zero(self.domain());
        for i in 0..self.len() {
            let prod = (self.coeffs[i] as i128) * (k as i128);
            let mut v = (prod % (q as i128)) as i64;
            if v < 0 {
                v += q;
            }
            out.coeffs[i] = v;
        }
        out
    }

    /// Compute round((num/den) * self) coefficient-wise, then reduce into [0,q).
    pub fn scale_and_round(&self, num: i64, den: i64) -> Poly {
        let mut out = Poly::zero(self.domain());
        let q = self.q() as i64;
        let num128 = num as i128;
        let den128 = den as i128;
        for i in 0..self.len() {
            let v = self.coeffs[i] as i128 * num128;
            let r = ((v + den128 / 2) / den128) as i64;
            let mut w = r % q;
            if w < 0 {
                w += q;
            }
            out.coeffs[i] = w;
        }
        out
    }

    /// Reduce each coefficient modulo a small positive integer m.
    pub fn mod_small(&self, m: i64) -> Poly {
        debug_assert!(m > 0);
        let mut out = Poly::zero(self.domain());
        for i in 0..self.len() {
            let mut w = self.coeffs[i] % m;
            if w < 0 { w += m; }
            out.coeffs[i] = w;
        }
        out
    }
}

impl<'a, 'b> Add<&'b Poly> for &'a Poly {
    type Output = Poly;
    fn add(self, rhs: &'b Poly) -> Poly {
        assert!(
            self.n() == rhs.n() && self.q() == rhs.q(),
            "domain mismatch"
        );
        let q = self.q() as i64;
        let mut out = Poly::zero(self.domain());
        for i in 0..self.len() {
            let mut s = self.coeffs[i] + rhs.coeffs[i];
            s %= q;
            if s < 0 {
                s += q;
            }
            out.coeffs[i] = s;
        }
        out
    }
}

impl<'a, 'b> Sub<&'b Poly> for &'a Poly {
    type Output = Poly;
    fn sub(self, rhs: &'b Poly) -> Poly {
        assert!(
            self.n() == rhs.n() && self.q() == rhs.q(),
            "domain mismatch"
        );
        let q = self.q() as i64;
        let mut out = Poly::zero(self.domain());
        for i in 0..self.len() {
            let mut d = self.coeffs[i] - rhs.coeffs[i];
            d %= q;
            if d < 0 {
                d += q;
            }
            out.coeffs[i] = d;
        }
        out
    }
}

impl<'a> Neg for &'a Poly {
    type Output = Poly;
    fn neg(self) -> Poly {
        let q = self.q() as i64;
        let mut out = Poly::zero(self.domain());
        for i in 0..self.len() {
            let v = self.coeffs[i];
            let mut n = (-v) % q;
            if n < 0 {
                n += q;
            }
            out.coeffs[i] = n;
        }
        out
    }
}

impl<'a, 'b> Mul<&'b Poly> for &'a Poly {
    type Output = Poly;
    fn mul(self, rhs: &'b Poly) -> Poly {
        assert!(
            self.n() == rhs.n() && self.q() == rhs.q(),
            "domain mismatch"
        );
        let n = self.n();
        let q = self.q() as i128;
        let mut acc = vec![0i128; n];
        for i in 0..n {
            for j in 0..n {
                let k = i + j;
                let term = (self.coeffs[i] as i128) * (rhs.coeffs[j] as i128);
                if k < n {
                    acc[k] += term;
                } else {
                    acc[k - n] -= term; // x^n == -1
                }
            }
        }
        let mut out = Poly::zero(self.domain());
        for i in 0..n {
            let mut v = acc[i] % q;
            if v < 0 {
                v += q;
            }
            out.coeffs[i] = v as i64;
        }
        out
    }
}

// Scalar multiplication: &Poly * i64 and i64 * &Poly
impl<'a> Mul<i64> for &'a Poly {
    type Output = Poly;
    fn mul(self, rhs: i64) -> Poly {
        self.mul_scalar(rhs)
    }
}

impl<'a> Mul<&'a Poly> for i64 {
    type Output = Poly;
    fn mul(self, rhs: &'a Poly) -> Poly {
        rhs.mul_scalar(self)
    }
}

impl Poly {
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
