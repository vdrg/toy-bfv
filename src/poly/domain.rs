use super::Poly;
use rand::Rng;
use std::ops::Deref;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Domain {
    n: usize,
    q: u64,
}

#[derive(Eq, PartialEq, Clone, Debug)]
pub struct DomainRef(Arc<Domain>);

impl Deref for DomainRef {
    type Target = Domain;
    fn deref(&self) -> &Domain {
        &self.0
    }
}

impl DomainRef {
    pub fn new(n: usize, q: u64) -> Self {
        assert!(n.is_power_of_two(), "n must be a power of two");
        assert!(q % 2 == 1, "use odd q");
        Self(Arc::new(Domain { n, q }))
    }

    #[inline]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn q(&self) -> u64 {
        self.q
    }

    pub fn sample_ternary<R: Rng + ?Sized>(&self, rng: &mut R) -> Poly {
        let mut v = vec![0i64; self.n];
        for c in &mut v {
            *c = rng.random_range(0..3) as i64 - 1;
        }
        Poly::from_coeffs(self, v).unwrap()
    }

    /// Uniform in [0, q) per coefficient.
    pub fn sample_uniform_mod_q<R: Rng + ?Sized>(&self, rng: &mut R) -> Poly {
        let mut v = vec![0i64; self.n];
        for c in &mut v {
            *c = rng.random_range(0..self.q) as i64;
        }
        Poly::from_coeffs(self, v).unwrap()
    }

    /// Centered binomial distribution CBD(k): sum_k Ber(1/2) - sum_k Ber(1/2).
    pub fn sample_cbd<R: Rng + ?Sized>(&self, k: usize, rng: &mut R) -> Poly {
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
        Poly::from_coeffs(self, v).expect("length matches domain")
    }
}
