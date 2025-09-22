use crate::poly::{DomainRef, Poly};
use rand::Rng;

/// Holds the cryptographic parameters for a BFV scheme instance.
/// This struct separates the configuration of the scheme from its implementation.
pub struct BfvParameters {
    /// R_q, the ciphertext polynomial ring.
    pub rq: DomainRef,
    /// R_t, the plaintext polynomial ring.
    pub rt: DomainRef,
    /// The scaling factor `p` for Version 2 relinearization.
    pub relin_p: u64,
    pub relin_error_std_dev: f64,
    /// The standard deviation for the error distribution.
    pub error_std_dev: f64,
}

impl BfvParameters {
    /// Creates a new set of BFV parameters.
    pub fn new(
        n: usize,
        q: u64,
        t: u64,
        relin_p: u64,
        relin_error_std_dev: f64,
        error_std_dev: f64,
    ) -> Self {
        // TODO: q | p?
        assert!(
            relin_p > 0,
            "Relinearization scaling factor p must be positive."
        );
        assert!(
            t > 0 && t < q,
            "Plaintext modulus t must satisfy 0 < t < q."
        );
        Self {
            rq: DomainRef::new(n, q),
            rt: DomainRef::new(n, t),
            relin_p,
            relin_error_std_dev,
            error_std_dev,
        }
    }
}

/// Provides a default, 128-bit secure parameter set for the BFV scheme.
impl Default for BfvParameters {
    /// Returns a parameter set providing approximately 128 bits of security.
    ///
    /// These parameters are based on the recommendations from the Homomorphic Encryption
    /// Standard. They are suitable for evaluating circuits with a small multiplicative depth.
    /// - n = 4096: The polynomial degree.
    /// - q = 1099511922689: A 40-bit prime ciphertext modulus (2^40 + 2^28 + 1).
    /// - t = 65537: A 17-bit prime plaintext modulus, allowing for a large message space.
    /// - relin_p = q^3
    /// - error_std_dev = 3.2: A standard value for the error distribution.
    fn default() -> Self {
        let q: u64 = 1099511922689;
        // TODO: is this reasonable?
        let p = q.pow(3);
        Self::new(4096, q, 65537, p, 7.0e10, 3.2)
    }
}

pub struct BFV {
    /// The parameters for this BFV instance. Made public to allow access
    /// to the underlying domains for message creation.
    pub params: BfvParameters,
    delta: u64,
}

type SecretKey = Poly;
type PublicKey = (Poly, Poly); // (b, a)

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelinearizationKey(Poly, Poly);

pub struct Keys {
    pub secret: SecretKey,
    pub public: PublicKey,
    pub relin: RelinearizationKey,
}

pub struct Ciphertext(pub(super) Poly, pub(super) Poly);

impl BFV {
    pub fn new(params: BfvParameters) -> Self {
        let q = params.rq.q();
        Self {
            delta: q / params.rt.q(),
            params,
        }
    }

    #[inline]
    pub fn q(&self) -> u64 {
        self.params.rq.q()
    }

    #[inline]
    pub fn t(&self) -> u64 {
        self.params.rt.q()
    }

    #[inline]
    pub fn n(&self) -> usize {
        // Both domains have the same n
        self.params.rt.n()
    }

    pub fn keygen<R: Rng>(&self, rng: &mut R) -> Keys {
        let s = self.params.rq.sample_ternary(rng);
        let e = self.params.rq.sample_cbd(self.params.error_std_dev, rng);
        let a = self.params.rq.sample_uniform_centered(rng);
        // RLWE
        let b = -(&a * &s + &e);
        // Relinearization key (V2)
        let s2 = &s * &s;
        let p = self.params.relin_p;
        let rlk_domain = DomainRef::new(self.n(), p * self.q());
        // Q: what should be the distribution for the error?
        let rlk_e = rlk_domain.sample_cbd(self.params.relin_error_std_dev, rng);
        let rlk_a = rlk_domain.sample_uniform_centered(rng);
        // b = -(a*s + e) + p*s^2 (mod p*q)
        let p_s2 = &s2.to_domain(&rlk_domain).unwrap() * (p as i64);
        let rlk_b = -(&rlk_a * &s.to_domain(&rlk_domain).unwrap() + &rlk_e) + p_s2;
        Keys {
            secret: s,
            public: (b, a),
            relin: RelinearizationKey(rlk_b, rlk_a),
        }
    }

    pub fn encrypt<R: Rng>(&self, m: &Poly, pk: &PublicKey, rng: &mut R) -> Ciphertext {
        let delta = self.delta as i64;
        let u = self.params.rq.sample_ternary(rng);
        let e1 = self.params.rq.sample_cbd(self.params.error_std_dev, rng);
        let e2 = self.params.rq.sample_cbd(self.params.error_std_dev, rng);
        // Change m's domain to Rq
        let m = m.to_domain(&self.params.rq).expect("n matches");
        let c0 = &pk.0 * &u + e1 + m * delta;
        let c1 = &pk.1 * &u + e2;
        Ciphertext(c0, c1)
    }

    pub fn decrypt(&self, ct: &Ciphertext, sk: &SecretKey) -> Poly {
        // v = c0 + c1*s = m*Δ + noise (mod q)
        let v = &ct.0 + &ct.1 * sk;
        // round(v * t / q)
        let scaled_coeffs = self.scale_and_round_coeffs(&v);
        // Embed into mod t
        Poly::from_coeffs(&self.params.rt, scaled_coeffs).unwrap()
    }

    pub fn add(&self, ct1: &Ciphertext, ct2: &Ciphertext) -> Ciphertext {
        Ciphertext(&ct1.0 + &ct2.0, &ct1.1 + &ct2.1)
    }

    fn integer_ring_mul(a: &Poly, b: &Poly) -> Vec<i128> {
        let n = a.n();
        let mut acc = vec![0i128; n];
        for (i, &a_c) in a.coeffs().iter().enumerate() {
            for (j, &b_c) in b.coeffs().iter().enumerate() {
                let k = i + j;
                let term = (a_c as i128) * (b_c as i128);
                if k < n {
                    acc[k] += term;
                } else {
                    acc[k - n] -= term;
                }
            }
        }
        acc
    }

    fn integer_ring_add(a: &Vec<i128>, b: &Vec<i128>) -> Vec<i128> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
    }

    /// Ciphertext multiplication with component-wise scale-and-round by t/q
    /// followed by relinearization using the provided relin key.
    pub fn mul(&self, ct1: &Ciphertext, ct2: &Ciphertext, rlk: &RelinearizationKey) -> Ciphertext {
        let c0_raw = Self::integer_ring_mul(&ct1.0, &ct2.0);
        let c1_raw1 = Self::integer_ring_mul(&ct1.0, &ct2.1);
        let c1_raw2 = Self::integer_ring_mul(&ct1.1, &ct2.0);
        let c1_raw = Self::integer_ring_add(&c1_raw1, &c1_raw2);
        let c2_raw = Self::integer_ring_mul(&ct1.1, &ct2.1);
        let mut c0_coeffs = self.scale_and_round_coeffs_raw(&c0_raw);
        let mut c1_coeffs = self.scale_and_round_coeffs_raw(&c1_raw);
        let c2_coeffs = self.scale_and_round_coeffs_raw(&c2_raw);
        let p = self.params.relin_p as i128;
        let r0c = rlk.0.coeffs();
        let r1c = rlk.1.coeffs();
        let term0_coeffs = self.poly_mul_and_scale_coeffs(&c2_coeffs, &r0c, p);
        let term1_coeffs = self.poly_mul_and_scale_coeffs(&c2_coeffs, &r1c, p);
        for i in 0..self.n() {
            c0_coeffs[i] += term0_coeffs[i];
            c1_coeffs[i] += term1_coeffs[i];
        }
        let c0 = Poly::from_coeffs(&self.params.rq, c0_coeffs).unwrap();
        let c1 = Poly::from_coeffs(&self.params.rq, c1_coeffs).unwrap();
        Ciphertext(c0, c1)
    }

    fn poly_mul_and_scale_coeffs(
        &self,
        poly1_coeffs: &[i64],
        poly2_coeffs: &[i64],
        p: i128,
    ) -> Vec<i64> {
        let n = poly1_coeffs.len();
        let mut acc = vec![0i128; n];
        for (i, &a) in poly1_coeffs.iter().enumerate() {
            for (j, &b) in poly2_coeffs.iter().enumerate() {
                let k = i + j;
                let term = (a as i128) * (b as i128);
                if k < n {
                    acc[k] += term;
                } else {
                    acc[k - n] -= term;
                }
            }
        }
        acc.into_iter()
            .map(|v| Self::round_div(v, p) as i64)
            .collect()
    }

    fn scale_and_round_coeffs_raw(&self, raw: &Vec<i128>) -> Vec<i64> {
        let t = self.t() as i128;
        let q = self.q() as i128;
        raw.iter()
            .map(|&v| Self::round_div(v * t, q) as i64)
            .collect()
    }

    /// This computes round(p * t/q) coefficient-wise. Since coefficients are already centered,
    /// we skip the manual centering step.
    fn scale_and_round_coeffs(&self, poly: &Poly) -> Vec<i64> {
        let t = self.t() as i128;
        let q = self.q() as i128;
        poly.coeffs()
            .iter()
            .map(|&c| {
                let centered = c as i128;
                Self::round_div(centered * t, q) as i64
            })
            .collect()
    }

    #[inline]
    fn floor_div(a: i128, b: i128) -> i128 {
        let d = a / b;
        let r = a % b;
        if r != 0 && ((a < 0) != (b < 0)) {
            d - 1
        } else {
            d
        }
    }

    #[inline]
    fn round_div(v: i128, d: i128) -> i128 {
        let quot = Self::floor_div(v, d);
        let rem = v - quot * d; // rem in [0, d)
        let half = (d + 1) / 2;
        if rem >= half { quot + 1 } else { quot }
    }
}

#[cfg(test)]
mod tests {
    use super::{BFV, BfvParameters, Keys};
    use crate::poly::Poly;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn testing_params() -> BfvParameters {
        // q must be < p. Let's use a large p for testing.
        let q: u64 = 12289;
        // TODO: is this reasonable?
        let p: u64 = (1u64 << 30) + 1;
        BfvParameters::new(8, q, 17, p, 2.0, 2.0)
        // BfvParameters::default()
    }

    fn setup_bfv_and_keys() -> (BFV, Keys, StdRng) {
        let params = testing_params();
        let bfv = BFV::new(params);
        let mut rng = StdRng::seed_from_u64(42);
        let keys = bfv.keygen(&mut rng);
        (bfv, keys, rng)
    }

    #[test]
    fn test_bfv_zero_polynomial() {
        let (bfv, keys, mut rng) = setup_bfv_and_keys();
        let m_zero = Poly::zero(&bfv.params.rt);
        let ct = bfv.encrypt(&m_zero, &keys.public, &mut rng);
        let decrypted = bfv.decrypt(&ct, &keys.secret);
        assert_eq!(decrypted.coeffs(), m_zero.coeffs());
    }

    #[test]
    fn test_bfv_constant_polynomial() {
        let (bfv, keys, mut rng) = setup_bfv_and_keys();
        let n = bfv.n();
        let m_const = Poly::from_coeffs(&bfv.params.rt, vec![5i64; n]).expect("len matches");
        let ct = bfv.encrypt(&m_const, &keys.public, &mut rng);
        let decrypted = bfv.decrypt(&ct, &keys.secret);
        assert_eq!(decrypted.coeffs(), m_const.coeffs());
    }

    #[test]
    fn test_bfv_random_polynomial() {
        let (bfv, keys, mut rng) = setup_bfv_and_keys();
        let m = bfv.params.rt.sample_uniform_centered(&mut rng);
        let ct = bfv.encrypt(&m, &keys.public, &mut rng);
        let decrypted = bfv.decrypt(&ct, &keys.secret);
        assert_eq!(decrypted.coeffs(), m.coeffs());
    }

    #[test]
    fn test_bfv_single_coefficient() {
        let (bfv, keys, mut rng) = setup_bfv_and_keys();
        let n = bfv.n();
        let mut coeffs = vec![0i64; n];
        coeffs[0] = 10;
        let m_single = Poly::from_coeffs(&bfv.params.rt, coeffs).expect("len matches");
        let ct = bfv.encrypt(&m_single, &keys.public, &mut rng);
        let decrypted = bfv.decrypt(&ct, &keys.secret);
        assert_eq!(decrypted.coeffs(), m_single.coeffs());
    }

    #[test]
    fn test_bfv_add() {
        let (bfv, keys, mut rng) = setup_bfv_and_keys();
        let m1 = bfv.params.rt.sample_uniform_centered(&mut rng);
        let m2 = bfv.params.rt.sample_uniform_centered(&mut rng);
        let ct1 = bfv.encrypt(&m1, &keys.public, &mut rng);
        let ct2 = bfv.encrypt(&m2, &keys.public, &mut rng);
        let added = bfv.add(&ct1, &ct2);
        let decrypted = bfv.decrypt(&added, &keys.secret);
        assert_eq!(decrypted.coeffs(), (&m1 + &m2).coeffs());
    }

    #[test]
    fn test_bfv_mul_basic() {
        let (bfv, keys, mut rng) = setup_bfv_and_keys();
        let m1 = bfv.params.rt.sample_uniform_centered(&mut rng);
        let m2 = bfv.params.rt.sample_uniform_centered(&mut rng);
        let ct1 = bfv.encrypt(&m1, &keys.public, &mut rng);
        let ct2 = bfv.encrypt(&m2, &keys.public, &mut rng);
        let c_prod = bfv.mul(&ct1, &ct2, &keys.relin);
        let dec = bfv.decrypt(&c_prod, &keys.secret);
        assert_eq!(dec.coeffs(), (&m1 * &m2).coeffs());
    }
}
