use crate::{
    ciphertext::{Noise, RLWECiphertext},
    poly::{RingPoly, domain::Domain},
};
use rand::Rng;

// pub mod bootstrap;
pub mod params;

pub use params::*;

#[derive(Clone, Debug)]
pub struct BFV {
    /// The parameters for this BFV instance. Made public to allow access
    /// to the underlying domains for message creation.
    pub params: BfvParameters,
    delta: u64,
}

type SecretKey = RingPoly;
type PublicKey = (RingPoly, RingPoly); // (b, a)

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelinearizationKey(RingPoly, RingPoly);

pub struct Keys {
    pub secret: SecretKey,
    pub public: PublicKey,
    pub relin: RelinearizationKey,
}

impl BFV {
    pub fn new(params: BfvParameters) -> Self {
        let q = params.rq.q();
        Self {
            delta: q / params.rt.q(),
            params,
        }
    }

    pub fn with_modulus(&self, q: u64, t: u64) -> Self {
        let mut p = self.params.clone();
        p.rq = Domain::new(self.n(), q);
        p.rt = Domain::new(self.n(), t);
        BFV::new(p)
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
        let p = self.params.relin_p;
        let pq = p * self.q();

        let s = self.params.rq.sample_binary(rng);
        let e = self.params.rq.sample_cbd(self.params.error_std_dev, rng);
        let a = self.params.rq.sample_uniform(rng);

        // RLWE
        let b = -(&a * &s + &e);

        let pk = (b, a);

        // Relinearization key (V2)
        let rlk_domain = Domain::new(self.n(), pq);
        let rlk_e = rlk_domain.sample_cbd(self.params.relin_error_std_dev, rng);
        let rlk_a = rlk_domain.sample_uniform(rng);

        let s2 = (&s * &s).mod_q_centered(pq);
        let spq = s.mod_q_centered(pq);

        // b = -(a*s + e) + p*s^2 (mod p*q)
        let rlk_b = -(&rlk_a * &spq + &rlk_e) + p * &s2;

        Keys {
            secret: s,
            public: pk,
            relin: RelinearizationKey(rlk_b, rlk_a),
        }
    }

    pub fn encrypt<R: Rng>(&self, m: &RingPoly, pk: &PublicKey, rng: &mut R) -> RLWECiphertext {
        let q = self.q();
        let delta = self.delta;
        let u = self.params.rq.sample_binary(rng);
        let e1 = self.params.rq.sample_cbd(self.params.error_std_dev, rng);
        let e2 = self.params.rq.sample_cbd(self.params.error_std_dev, rng);

        // Change m's domain to Rq
        let m = m.mod_q_centered(q);

        let c0 = &pk.0 * &u + e1 + &m * delta;
        let c1 = &pk.1 * &u + e2;

        // TODO: correctly track noise
        RLWECiphertext::new(c0, c1)
    }

    pub fn decrypt(&self, ct: &RLWECiphertext, sk: &SecretKey) -> RingPoly {
        let (t, q) = (self.t(), self.q());

        let v = &ct[0] + &ct[1] * sk;

        v.scale_round(t, q).mod_q_centered(self.t())
    }

    pub fn add(&self, ct1: &RLWECiphertext, ct2: &RLWECiphertext) -> RLWECiphertext {
        let c0 = &ct1[0] + &ct2[0];
        let c1 = &ct1[1] + &ct2[1];

        RLWECiphertext::new(c0, c1).with_noise(self.noise_after_add(&ct1.noise(), &ct2.noise()))
    }

    /// RLWECiphertext multiplication with component-wise scale-and-round by t/q
    /// followed by relinearization using the provided relin key.
    pub fn mul(
        &self,
        ct1: &RLWECiphertext,
        ct2: &RLWECiphertext,
        rlk: &RelinearizationKey,
    ) -> RLWECiphertext {
        let (t, q, p) = (self.t(), self.q(), self.params.relin_p);
        let pq = p.checked_mul(q).expect("p*q overflow");

        // Eq. (3): component-wise round(t/q) on ℤ-convolutions, then mod q
        let c0 = (ct1[0].z() * ct2[0].z()).round_scale(t, q).mod_q(q);

        let c1 = (ct1[0].z() * ct2[1].z() + ct1[1].z() * ct2[0].z())
            .round_scale(t, q)
            .mod_q(q);

        // FV.SH.Relin (Version 2) on the scaled c2
        let c2 = (ct1[1].z() * ct2[1].z())
            .round_scale(t, q)
            .mod_q(q)
            .mod_q_centered(pq); // scaled c2 in R_q

        let r0 = (&c2 * &rlk.0).div_round(p).mod_q(q); // round((c2*rlk[0])/p) mod q
        let r1 = (&c2 * &rlk.1).div_round(p).mod_q(q); // idem for rlk[1]

        RLWECiphertext::new(c0 + r0, c1 + r1)
            .with_noise(self.noise_after_mul(&ct1.noise(), &ct2.noise()))
    }

    #[inline]
    fn noise_after_add(&self, a: &Noise, b: &Noise) -> Noise {
        Noise::new(a.bound() + b.bound())
    }

    /// Mul (BFV basic mul + component t/q scaling):
    /// (t/q)·(‖e1‖ + ‖e2‖) + relinearization error
    #[inline]
    fn noise_after_mul(&self, a: &Noise, b: &Noise) -> Noise {
        let scale = self.t() as f64 / self.q() as f64;
        let relin = self.params.relin_error_std_dev; // very crude
        Noise::new(scale * (a.bound() + b.bound()) + relin)
    }
}

#[cfg(test)]
mod tests {
    use super::{BFV, BfvParameters, Keys};
    use crate::poly::{Poly, RingPoly};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn testing_params() -> BfvParameters {
        // TODO: use parameters that make sense
        let q: u64 = 1_073_741_825;
        let p: u64 = 2_147_483_647;

        BfvParameters::new(16, q, 17, p, 3.2, 100.0)
        // BfvParameters::default()
    }

    fn setup_bfv_and_keys() -> (BFV, Keys, StdRng) {
        let params = testing_params();
        let bfv = BFV::new(params);
        let mut rng = StdRng::seed_from_u64(0);
        let keys = bfv.keygen(&mut rng);
        (bfv, keys, rng)
    }

    #[test]
    fn test_bfv_zero_polynomial() {
        let (bfv, keys, mut rng) = setup_bfv_and_keys();
        let m_zero = RingPoly::zero(bfv.q(), bfv.n());
        let ct = bfv.encrypt(&m_zero, &keys.public, &mut rng);
        let decrypted = bfv.decrypt(&ct, &keys.secret);
        assert_eq!(decrypted.coeffs(), m_zero.coeffs());
    }

    #[test]
    fn test_bfv_constant_polynomial() {
        let (bfv, keys, mut rng) = setup_bfv_and_keys();
        let n = bfv.n();
        let m_const = RingPoly::from_coeffs(bfv.q(), vec![5u64; n]);
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
        let m_single = RingPoly::from_coeffs(t, coeffs);
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
        let m2 = bfv.params.rt.sample_uniform(&mut rng);
        let ct1 = bfv.encrypt(&m1, &keys.public, &mut rng);
        let ct2 = bfv.encrypt(&m2, &keys.public, &mut rng);
        let c_prod = bfv.mul(&ct1, &ct2, &keys.relin);
        let dec = bfv.decrypt(&c_prod, &keys.secret);
        assert_eq!(dec.coeffs(), (&m1 * &m2).coeffs());
    }

    #[test]
    fn test_bfv_mul_multiple() {
        let (bfv, keys, mut rng) = setup_bfv_and_keys();
        let m1 = bfv.params.rt.sample_uniform(&mut rng);
        let m2 = bfv.params.rt.sample_uniform(&mut rng);
        let m3 = bfv.params.rt.sample_uniform(&mut rng);
        let ct1 = bfv.encrypt(&m1, &keys.public, &mut rng);
        let ct2 = bfv.encrypt(&m2, &keys.public, &mut rng);
        let ct3 = bfv.encrypt(&m3, &keys.public, &mut rng);
        let c_prod = bfv.mul(&ct1, &ct2, &keys.relin);
        let c_prod = bfv.mul(&c_prod, &ct3, &keys.relin);
        let dec = bfv.decrypt(&c_prod, &keys.secret);
        assert_eq!(dec.coeffs(), (&(&m1 * &m2) * &m3).coeffs());
    }
}
