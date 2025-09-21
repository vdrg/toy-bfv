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

    /// Samples a polynomial with coefficients in {-1, 0, 1}.
    pub fn sample_ternary<R: Rng + ?Sized>(&self, rng: &mut R) -> Poly {
        let mut v = vec![0i64; self.n];
        for c in &mut v {
            *c = rng.random_range(0..3) - 1;
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

    /// Samples from the Centered Binomial Distribution (CBD), which is an
    /// efficient approximation of a discrete Gaussian.
    /// The standard deviation determines the variance of the distribution.
    // Q: is this enough for cryptographic applications?
    pub fn sample_cbd<R: Rng + Sized>(&self, std_dev: f64, rng: &mut R) -> Poly {
        // The parameter k for CBD is derived from the variance (std_dev^2).
        // For CBD(k), the variance is k/2. So, k = 2 * variance.
        let k = (2.0 * std_dev * std_dev).round() as usize;
        let mut v = vec![0i64; self.n];

        for c in &mut v {
            let mut a_sum: u32 = 0;
            let mut b_sum: u32 = 0;
            let mut remaining = k;

            // Process in chunks of 64 bits for performance.
            while remaining > 0 {
                let chunk_size = remaining.min(64);
                let mask = if chunk_size == 64 {
                    !0
                } else {
                    (1u64 << chunk_size) - 1
                };

                let a_chunk = rng.random::<u64>() & mask;
                let b_chunk = rng.random::<u64>() & mask;

                a_sum += a_chunk.count_ones();
                b_sum += b_chunk.count_ones();

                remaining -= chunk_size;
            }
            *c = (a_sum as i64) - (b_sum as i64);
        }

        Poly::from_coeffs(self, v).expect("length matches domain")
    }
}
