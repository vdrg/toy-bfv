use crate::poly::{Poly, domain::Domain};
use rand::Rng;

/// Holds the cryptographic parameters for a BFV scheme instance.
/// This struct separates the configuration of the scheme from its implementation.
pub struct BfvParameters {
    /// R_q, the ciphertext polynomial ring.
    pub rq: Domain,
    /// R_t, the plaintext polynomial ring.
    pub rt: Domain,
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
            rq: Domain::new(n, q),
            rt: Domain::new(n, t),
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
        let a = self.params.rq.sample_uniform(rng);

        // RLWE
        let b = -(&a * &s + &e);

        println!("after b");
        // Relinearization key (V2)
        let p = self.params.relin_p;
        let pq = p * self.q();
        // let s = s.reduce_mod(pq);
        // let s2 = &s * &s;
        println!("after s");

        let rlk_domain = Domain::new(self.n(), pq);
        let rlk_e = rlk_domain.sample_cbd(self.params.relin_error_std_dev, rng);
        println!("rlk_e coeffs: {:?}", rlk_e.coeffs());
        let rlk_a = rlk_domain.sample_uniform(rng);
        println!("rlk_a coeffs: {:?}", rlk_a.coeffs());

        let s2 = (&s * &s).reduce_mod(pq);
        println!("s2 coeffs: {:?}", s2.coeffs());
        let spq = s.reduce_mod(pq);
        println!("spq coeffs: {:?}", spq.coeffs());
        // b = -(a*s + e) + p*s^2 (mod p*q)
        let rlk_b = (-(&rlk_a * &spq + &rlk_e) + p * &s2).reduce_mod(pq);
        println!("rlk_b coeffs: {:?}", rlk_b.coeffs());

        Keys {
            secret: s,
            public: (b, a),
            relin: RelinearizationKey(rlk_b, rlk_a),
        }
    }

    pub fn encrypt<R: Rng>(&self, m: &Poly, pk: &PublicKey, rng: &mut R) -> Ciphertext {
        let q = self.q();
        let delta = self.delta;
        let u = self.params.rq.sample_ternary(rng);
        println!("u coeffs: {:?}", u.coeffs());
        let e1 = self.params.rq.sample_cbd(self.params.error_std_dev, rng);
        println!("e1 coeffs: {:?}", e1.coeffs());
        let e2 = self.params.rq.sample_cbd(self.params.error_std_dev, rng);
        println!("e2 coeffs: {:?}", e2.coeffs());

        // Change m's domain to Rq
        let m = m.reduce_mod(q);
        println!("m lifted coeffs: {:?}", m.coeffs());
        let c0 = &pk.0 * &u + e1 + &m * delta;
        println!("m lifted coeffs: {:?}", m.coeffs());
        let c1 = &pk.1 * &u + e2;
        println!("ct1 coeffs: {:?}", c1.coeffs());
        Ciphertext(c0, c1)
    }

    pub fn decrypt(&self, ct: &Ciphertext, sk: &SecretKey) -> Poly {
        let t = self.t();
        let q = self.q();

        println!(
            "before mult in decrypt {:?} {:?} {:?}",
            &ct.0.q(),
            &ct.1.q(),
            &sk.q()
        );
        // v = c0 + c1*s = m*Δ + noise (mod q)
        let v = &ct.0 + &ct.1 * sk;

        println!("v {:?}", &v);

        println!(
            "scale(v, t, q).mod(t) {:?}",
            &v.scale_round(t, q).reduce_mod(t)
        );

        // round(v * t / q) mod q
        v.scale_round(t, q).reduce_mod(t)
    }

    pub fn add(&self, ct1: &Ciphertext, ct2: &Ciphertext) -> Ciphertext {
        Ciphertext(&ct1.0 + &ct2.0, &ct1.1 + &ct2.1)
    }

    /// Ciphertext multiplication with component-wise scale-and-round by t/q
    /// followed by relinearization using the provided relin key.
    pub fn mul(&self, ct1: &Ciphertext, ct2: &Ciphertext, rlk: &RelinearizationKey) -> Ciphertext {
        let t = self.t();
        let q = self.q();
        let p = self.params.relin_p;

        let mut c0 = (&ct1.0 * &ct2.0).scale_round(t, q).reduce_mod(q);
        println!("initial c0 coeffs: {:?}", c0.coeffs());
        let mut c1 = (&ct1.0 * &ct2.1 + &ct1.1 * &ct2.0)
            .scale_round(t, q)
            .reduce_mod(q);
        println!("initial c1 coeffs: {:?}", c1.coeffs());

        let c2 = (&ct1.1 * &ct2.1).scale_round(t, q).reduce_mod(p * q);

        println!(
            "c2 {:?} rlk.0 {:?} prod {:?}",
            c2.coeffs(),
            rlk.0.coeffs(),
            (&c2 * &rlk.0).coeffs()
        );
        let relin_term0 = (&c2 * &rlk.0).div_round(p);
        println!(
            "relin_term0 (after div_round) coeffs: {:?}",
            relin_term0.coeffs()
        );
        let reduced_term0 = relin_term0.reduce_mod(q);
        println!("reduced_term0 coeffs: {:?}", reduced_term0.coeffs());
        c0 += (&c2 * &rlk.0).div_round(p).reduce_mod(q);
        let relin_term1 = (&c2 * &rlk.1).div_round(p);
        println!(
            "relin_term1 (after div_round) coeffs: {:?}",
            relin_term1.coeffs()
        );
        let reduced_term1 = relin_term1.reduce_mod(q);
        println!("reduced_term1 coeffs: {:?}", reduced_term1.coeffs());
        c1 += (&c2 * &rlk.1).div_round(p).reduce_mod(q);

        println!("c0 {:?} c1 {:?}", &c0.coeffs(), &c1.coeffs());

        Ciphertext(c0, c1)
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
        BfvParameters::new(8, q, 17, p, 0.0, 0.0)
        // BfvParameters::default()
    }

    fn setup_bfv_and_keys() -> (BFV, Keys, StdRng) {
        let params = testing_params();
        let bfv = BFV::new(params);
        let mut rng = StdRng::seed_from_u64(0);
        let keys = bfv.keygen(&mut rng);
        println!("after keygen");
        (bfv, keys, rng)
    }

    #[test]
    fn test_bfv_zero_polynomial() {
        let (bfv, keys, mut rng) = setup_bfv_and_keys();
        let m_zero = Poly::zero(bfv.q(), bfv.n());
        let ct = bfv.encrypt(&m_zero, &keys.public, &mut rng);
        let decrypted = bfv.decrypt(&ct, &keys.secret);
        assert_eq!(decrypted.coeffs(), m_zero.coeffs());
    }

    #[test]
    fn test_bfv_constant_polynomial() {
        let (bfv, keys, mut rng) = setup_bfv_and_keys();
        let n = bfv.n();
        let m_const = Poly::from_coeffs(bfv.q(), vec![5u64; n]);
        let ct = bfv.encrypt(&m_const, &keys.public, &mut rng);
        let decrypted = bfv.decrypt(&ct, &keys.secret);
        assert_eq!(decrypted.coeffs(), m_const.coeffs());
    }

    #[test]
    fn test_bfv_random_polynomial() {
        let (bfv, keys, mut rng) = setup_bfv_and_keys();
        let m = bfv.params.rt.sample_uniform(&mut rng);
        let ct = bfv.encrypt(&m, &keys.public, &mut rng);
        let decrypted = bfv.decrypt(&ct, &keys.secret);
        assert_eq!(decrypted.coeffs(), m.coeffs());
    }

    #[test]
    fn test_bfv_single_coefficient() {
        let (bfv, keys, mut rng) = setup_bfv_and_keys();
        let t = bfv.t();
        let n = bfv.n();
        let mut coeffs = vec![0u64; n];
        coeffs[0] = 10;
        let m_single = Poly::from_coeffs(t, coeffs);
        let ct = bfv.encrypt(&m_single, &keys.public, &mut rng);
        let decrypted = bfv.decrypt(&ct, &keys.secret);
        assert_eq!(decrypted.coeffs(), m_single.coeffs());
    }

    #[test]
    fn test_bfv_add() {
        let (bfv, keys, mut rng) = setup_bfv_and_keys();
        let m1 = bfv.params.rt.sample_uniform(&mut rng);
        let m2 = bfv.params.rt.sample_uniform(&mut rng);
        let ct1 = bfv.encrypt(&m1, &keys.public, &mut rng);
        let ct2 = bfv.encrypt(&m2, &keys.public, &mut rng);
        let added = bfv.add(&ct1, &ct2);
        let decrypted = bfv.decrypt(&added, &keys.secret);
        assert_eq!(decrypted.coeffs(), (&m1 + &m2).coeffs());
    }

    #[test]
    fn test_bfv_add_multiple() {
        let (bfv, keys, mut rng) = setup_bfv_and_keys();
        let m1 = bfv.params.rt.sample_uniform(&mut rng);
        let m2 = bfv.params.rt.sample_uniform(&mut rng);
        let m3 = bfv.params.rt.sample_uniform(&mut rng);
        let ct1 = bfv.encrypt(&m1, &keys.public, &mut rng);
        let ct2 = bfv.encrypt(&m2, &keys.public, &mut rng);
        let ct3 = bfv.encrypt(&m3, &keys.public, &mut rng);
        let added = bfv.add(&ct1, &ct2);
        let added = bfv.add(&added, &ct3);
        let decrypted = bfv.decrypt(&added, &keys.secret);
        assert_eq!(decrypted.coeffs(), (&m1 + &m2 + &m3).coeffs());
    }

    #[test]
    fn test_bfv_mul_basic() {
        let (bfv, keys, mut rng) = setup_bfv_and_keys();
        let m1 = bfv.params.rt.sample_uniform(&mut rng);
        println!("m1 coeffs: {:?}", m1.coeffs());
        let m2 = bfv.params.rt.sample_uniform(&mut rng);
        println!("m2 coeffs: {:?}", m2.coeffs());
        let ct1 = bfv.encrypt(&m1, &keys.public, &mut rng);
        let ct2 = bfv.encrypt(&m2, &keys.public, &mut rng);
        let c_prod = bfv.mul(&ct1, &ct2, &keys.relin);
        println!("cprod[0] {:?}", &c_prod.0.coeffs());
        let dec = bfv.decrypt(&c_prod, &keys.secret);
        println!("after decrypt");
        assert_eq!(dec.coeffs(), (&m1 * &m2).coeffs());
    }
}
