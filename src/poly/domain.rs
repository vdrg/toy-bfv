use super::Poly;
use rand::Rng;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Domain {
    n: usize,
    q: u64,
}

impl Domain {
    pub fn new(n: usize, q: u64) -> Self {
        assert!(n.is_power_of_two(), "n must be a power of two");
        assert!(q % 2 == 1, "q must be odd");
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

    /// Uniform sample in Z_q
    pub fn sample_uniform<R: Rng + ?Sized>(&self, rng: &mut R) -> Poly {
        let q = self.q;
        let mut coeffs = vec![0u64; self.n];
        for c in &mut coeffs {
            *c = rng.random_range(0..q); // [0, q)
        }
        Poly::from_coeffs(q, coeffs)
    }

    /// Ternary in {-1,0,1} mod q
    pub fn sample_ternary<R: Rng + ?Sized>(&self, rng: &mut R) -> Poly {
        let q = self.q;
        let mut coeffs = vec![0u64; self.n];
        for c in &mut coeffs {
            let r = rng.random_range(0..3);
            *c = if r == 2 { q - 1 } else { r }
        }
        Poly::from_coeffs(q, coeffs)
    }

    /// CBD noise (approximates a discrete gaussian)
    pub fn sample_cbd<R: Rng + ?Sized>(&self, std_dev: f64, rng: &mut R) -> Poly {
        let q = self.q;
        // CBD(k) has Var=k/2 => k = 2*std_dev^2 (rounded).
        let k = (2.0 * std_dev * std_dev).round().max(0.0) as usize;
        let mut v = vec![0u64; self.n];
        for c in &mut v {
            let mut a_sum: u64 = 0;
            let mut b_sum: u64 = 0;
            let mut rem = k;
            while rem > 0 {
                let chunk = rem.min(64);
                let mask = if chunk == 64 { !0 } else { (1u64 << chunk) - 1 };
                let a = rng.random::<u64>() & mask;
                let b = rng.random::<u64>() & mask;
                a_sum += a.count_ones() as u64;
                b_sum += b.count_ones() as u64;
                rem -= chunk;
            }
            *c = a_sum + (q - b_sum % q); // already centered
        }
        Poly::from_coeffs(q, v)
    }
}
