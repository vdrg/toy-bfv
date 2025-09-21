use rand::Rng;

use crate::poly::{DomainRef, Poly};

/// Holds the cryptographic parameters for a BFV scheme instance.
/// This struct separates the configuration of the scheme from its implementation.
pub struct BfvParameters {
    /// R_q, the ciphertext polynomial ring.
    pub rq: DomainRef,
    /// R_t, the plaintext polynomial ring.
    pub rt: DomainRef,
    /// The gadget base `w` used for relinearization.
    pub relin_base: u64,
    /// The standard deviation for the error distribution.
    pub error_std_dev: f64,
    /// Flag indicating if parameters are set for the optimized bootstrapping case.
    /// This is true if q is a power of two and t divides q.
    pub is_bootstrapping_optimized: bool, // TODO: not used for now
}

impl BfvParameters {
    /// Creates a new set of BFV parameters.
    pub fn new(n: usize, q: u64, t: u64, relin_base: u64, error_std_dev: f64) -> Self {
        assert!(
            relin_base >= 2,
            "Relinearization base must be >= 2 for decomposition to work."
        );
        assert!(
            t > 0 && t < q,
            "Plaintext modulus t must satisfy 0 < t < q."
        );

        let is_bootstrapping_optimized = q.is_power_of_two() && (q % t == 0);

        Self {
            rq: DomainRef::new(n, q),
            rt: DomainRef::new(n, t),
            relin_base,
            error_std_dev,
            is_bootstrapping_optimized,
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
    /// - relin_base = 2^16: A reasonable gadget base for the relinearization key.
    /// - error_std_dev = 3.2: A standard value for the error distribution.
    fn default() -> Self {
        Self::new(4096, 1099511922689, 65537, 1 << 16, 3.2)
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

// Encryptions of w^j * s^2 for j=0..L-1
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelinearizationKey(Vec<(Poly, Poly)>); // (a_j, b_j)

impl RelinearizationKey {
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &(Poly, Poly)> {
        self.0.iter()
    }
}

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
        let a = self.params.rq.sample_uniform_mod_q(rng);

        // RLWE
        let b = -(&a * &s + &e);

        // Relinearization key (V1)
        let s2 = &s * &s;

        let base = self.params.relin_base;
        let q = self.q();

        let l = q.ilog(base) as usize;
        let mut pairs = Vec::with_capacity(l + 1);

        let mut w_i: i64 = 1;

        for _ in 0..=l {
            let a_j = self.params.rq.sample_uniform_mod_q(rng);
            let e_j = self.params.rq.sample_cbd(self.params.error_std_dev, rng);

            // r0 = - (a_i * s + e_i) + w^i * s^2
            let r0 = -(&a_j * &s + &e_j) + &s2 * w_i;

            pairs.push((r0, a_j));

            w_i = (w_i as i128 * base as i128).rem_euclid(q as i128) as i64;
        }

        Keys {
            secret: s,
            public: (b, a),
            relin: RelinearizationKey(pairs),
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
        // v = c0 + c1*s = m*Δ + noise  (mod q)
        let v = &ct.0 + &ct.1 * sk;

        // round(v * t / q)
        let scaled_coeffs = self.scale_and_round_coeffs(&v);

        // Embed into mod t
        Poly::from_coeffs(&self.params.rt, scaled_coeffs).unwrap()
    }

    pub fn add(&self, ct1: &Ciphertext, ct2: &Ciphertext) -> Ciphertext {
        Ciphertext(&ct1.0 + &ct2.0, &ct1.1 + &ct2.1)
    }

    /// Ciphertext multiplication with component-wise scale-and-round by t/q
    /// followed by relinearization using the provided relin key.
    pub fn mul(&self, ct1: &Ciphertext, ct2: &Ciphertext, rlk: &RelinearizationKey) -> Ciphertext {
        let c0 = &ct1.0 * &ct2.0;
        let c1 = &ct1.0 * &ct2.1 + &ct1.1 * &ct2.0;
        let c2 = &ct1.1 * &ct2.1;

        // Scaling during multiplication results in polynomials in the ciphertext domain R_q.
        let c0_scaled = self.scale_and_round_coeffs(&c0);
        let c1_scaled = self.scale_and_round_coeffs(&c1);
        let c2_scaled = self.scale_and_round_coeffs(&c2);

        let mut c0 = Poly::from_coeffs(&self.params.rq, c0_scaled).unwrap();
        let mut c1 = Poly::from_coeffs(&self.params.rq, c1_scaled).unwrap();
        let c2 = Poly::from_coeffs(&self.params.rq, c2_scaled).unwrap();

        // Decompose c2 in base w (coeff-wise)
        let digits = self.decompose_base_w(&c2, rlk.len());

        // Relinearize
        for (c2_i, (b_i, a_i)) in digits.iter().zip(rlk.iter()) {
            c0 += c2_i * b_i;
            c1 += c2_i * a_i;
        }

        Ciphertext(c0, c1)
    }

    /// This computes round(p * t/q) coefficient-wise, interpreting the input
    /// polynomial's coefficients in a centered interval.
    fn scale_and_round_coeffs(&self, poly: &Poly) -> Vec<i64> {
        let t = self.t() as i128;
        let q = self.q() as i128;
        let half_q = (q + 1) / 2; // ceil(q/2)

        poly.coeffs()
            .iter()
            .map(|&c| {
                // Center the coefficient from [0, q) to (-q/2, q/2]
                let centered = if (c as i128) >= half_q {
                    c as i128 - q
                } else {
                    c as i128
                };

                // Scale by t/q and round
                let v = centered * t;
                let rounded = (v + v.signum() * q / 2) / q;

                // The result remains in the ciphertext domain R_q
                // NOTE: we don't take results mod q
                rounded as i64
            })
            .collect()
    }

    /// Centered base-w decomposition with L digits in R_q.
    /// Conventions:
    /// - Centering matches decrypt: coefficient c is treated as negative iff c > ceil(q/2).
    /// - Digits are balanced: d_j ∈ [ -floor(w/2),  ceil(w/2) - 1 ].
    /// - For each coefficient x̃ (centered), we produce digits d_0..d_{L-1} such that
    ///     x̃ = Σ_j d_j * w^j   (over ℤ),
    ///   and return digit polynomials D_j ∈ R_q with coefficients reduced into [0, q).
    fn decompose_base_w(&self, poly: &Poly, l: usize) -> Vec<Poly> {
        debug_assert_eq!(poly.q(), self.q(), "decomposition must be in the same R_q");

        let w = self.params.relin_base as i64;

        let digit_rows: Vec<Vec<i64>> = poly
            .coeffs_centered()
            .into_iter()
            // [j][i] = j-th digit at coefficient i
            .map(|c| Self::decompose_coeff(c, w, l))
            .collect();

        // Transpose to get the digit polynomials
        (0..l)
            .map(|j| {
                let coeffs = (0..poly.n()).map(|i| digit_rows[i][j]).collect();
                Poly::from_coeffs(&self.params.rq, coeffs).expect("len = n")
            })
            .collect()
    }

    fn decompose_coeff(x: i64, w: i64, l: usize) -> Vec<i64> {
        let thr_w = (w + 1) / 2;
        let mut current = x;
        let mut digits = Vec::with_capacity(l);

        for _ in 0..l {
            let mut r = current.rem_euclid(w);
            if r >= thr_w {
                r -= w;
            }
            digits.push(r as i64);
            current = (current - r) / w;
        }

        debug_assert!(current == 0, "Decomposition failed, L might be too small");
        digits
    }
}

#[cfg(test)]
mod tests {
    use super::{BFV, BfvParameters, Ciphertext, Keys};
    use crate::poly::Poly;
    use rand::rng;

    fn testing_params() -> BfvParameters {
        BfvParameters::new(8, 65537, 17, 1 << 10, 3.2)
        // BfvParameters::default()
    }

    fn setup_bfv_and_keys() -> (BFV, Keys, rand::rngs::ThreadRng) {
        let params = testing_params();
        let bfv = BFV::new(params);
        let mut rng = rng();
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
        let m = bfv.params.rt.sample_uniform_mod_q(&mut rng);
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

        let m1 = bfv.params.rt.sample_uniform_mod_q(&mut rng);
        let m2 = bfv.params.rt.sample_uniform_mod_q(&mut rng);

        let ct1 = bfv.encrypt(&m1, &keys.public, &mut rng);
        let ct2 = bfv.encrypt(&m2, &keys.public, &mut rng);
        let added = bfv.add(&ct1, &ct2);

        let decrypted = bfv.decrypt(&added, &keys.secret);

        assert_eq!(decrypted.coeffs(), (&m1 + &m2).coeffs());
    }

    #[test]
    fn test_bfv_mul_basic() {
        let (bfv, keys, mut rng) = setup_bfv_and_keys();

        let m1 = bfv.params.rt.sample_uniform_mod_q(&mut rng);
        let m2 = bfv.params.rt.sample_uniform_mod_q(&mut rng);

        let ct1 = bfv.encrypt(&m1, &keys.public, &mut rng);
        let ct2 = bfv.encrypt(&m2, &keys.public, &mut rng);

        let Ciphertext(ref c1_0, ref c1_1) = ct1;
        let Ciphertext(ref c2_0, ref c2_1) = ct2;

        let c0 = &*c1_0 * &*c2_0;
        let c1 = &*c1_0 * &*c2_1 + &*c1_1 * &*c2_0;
        let c2 = &*c1_1 * &*c2_1;

        println!("c0 before={:?}", c0.coeffs());
        println!("c1 before={:?}", c1.coeffs());
        println!("c2 before={:?}", c2.coeffs());
        println!("relin.len={:?}", keys.relin.len());

        let digits = bfv.decompose_base_w(&c2, keys.relin.len());
        println!("digits={:?}", digits);

        let c_prod = bfv.mul(&ct1, &ct2, &keys.relin);
        let dec = bfv.decrypt(&c_prod, &keys.secret);

        assert_eq!(dec.coeffs(), (&m1 * &m2).coeffs());
    }
}
