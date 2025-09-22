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
        assert!(q % 2 == 1, "q must be odd");
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

    /// Uniform sample in centered Z_q: (-⌊q/2⌋, ⌈q/2⌉]
    pub fn sample_uniform_centered<R: Rng + ?Sized>(&self, rng: &mut R) -> crate::poly::Poly {
        let q = self.q as i64;
        let half = q / 2;
        let mut v = vec![0i64; self.n];
        for c in &mut v {
            *c = rng.random_range(-half..half) as i64; // [0, q)
        }
        crate::poly::Poly::from_coeffs(v)
    }

    /// Ternary in {-1,0,1} (already centered).
    pub fn sample_ternary<R: Rng + ?Sized>(&self, rng: &mut R) -> crate::poly::Poly {
        let mut v = vec![0i64; self.n];
        for c in &mut v {
            *c = rng.random_range(0..3) - 1;
        }
        crate::poly::Poly::from_coeffs(v)
    }

    /// CBD noise; returns centered coefficients.
    pub fn sample_cbd<R: Rng + ?Sized>(&self, std_dev: f64, rng: &mut R) -> crate::poly::Poly {
        // CBD(k) has Var=k/2 ⇒ k = 2*std_dev^2 (rounded).
        let k = (2.0 * std_dev * std_dev).round().max(0.0) as usize;
        let mut v = vec![0i64; self.n];
        for c in &mut v {
            let mut a_sum: u32 = 0;
            let mut b_sum: u32 = 0;
            let mut rem = k;
            while rem > 0 {
                let chunk = rem.min(64);
                let mask = if chunk == 64 { !0 } else { (1u64 << chunk) - 1 };
                let a = rng.random::<u64>() & mask;
                let b = rng.random::<u64>() & mask;
                a_sum += a.count_ones();
                b_sum += b.count_ones();
                rem -= chunk;
            }
            *c = (a_sum as i64) - (b_sum as i64); // already centered
        }
        crate::poly::Poly::from_coeffs(v)
    }
}
