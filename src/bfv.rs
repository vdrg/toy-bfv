use rand::Rng;

use crate::poly::{DomainRef, Poly};

pub struct BFV {
    domain: DomainRef,
    t: u64,
    delta: u64,
}

type SecretKey = Poly;
type PublicKey = (Poly, Poly); // (b, a)

pub struct Keys {
    pub secret: SecretKey,
    pub public: PublicKey,
}

type Ciphertext = (Poly, Poly);

impl BFV {
    pub fn new(domain: &DomainRef, t: u64) -> Self {
        let q = domain.q();
        assert!(
            t > 0 && t <= q,
            "plaintext modulus t must satisfy 0 < t <= q"
        );
        Self {
            domain: domain.clone(),
            t,
            delta: q / t,
        }
    }

    pub fn keygen<R: Rng>(&self, rng: &mut R) -> Keys {
        let s = self.domain.sample_ternary(rng);
        let e = self.domain.sample_ternary(rng);
        let a = self.domain.sample_uniform_mod_q(rng);

        // RLWE: b = e - a*s  (mod q)
        let b = -(&a * &s) + &e;

        Keys {
            secret: s,
            public: (b, a),
        }
    }

    pub fn encrypt<R: Rng>(&self, m: &Poly, pk: &PublicKey, rng: &mut R) -> Ciphertext {
        let delta = self.delta as i64;

        let u = self.domain.sample_ternary(rng);
        let e1 = self.domain.sample_ternary(rng);
        let e2 = self.domain.sample_ternary(rng);

        let c0 = &pk.0 * &u + e1 + m * delta;

        let c1 = &pk.1 * &u + e2;

        (c0, c1)
    }

    pub fn decrypt(&self, ct: &Ciphertext, sk: &SecretKey) -> Poly {
        let t = self.t as i64;
        let q = self.domain.q() as i64;

        // v = c0 + c1*s = m*Δ + noise  (mod q)
        let v = &ct.0 + &ct.1 * sk;

        // round(v * t / q) mod t  (returned embedded in Z_q with residues < t)
        v.scale_and_round(t, q).reduce_mod(t)
    }

    pub fn add(&self, ct1: &Ciphertext, ct2: &Ciphertext) -> Ciphertext {
        return (&ct1.0 + &ct2.0, &ct1.1 + &ct2.1);
    }
}

#[cfg(test)]
mod tests {
    use super::{BFV, Keys};
    use crate::poly::{DomainRef, Poly};
    use rand::Rng;
    use rand::rng;

    fn setup_bfv_and_keys() -> (DomainRef, BFV, Keys, rand::rngs::ThreadRng) {
        let n = 8; // tiny for tests
        let q = 12289; // toy prime
        let t = 17; // plaintext modulus
        let domain = DomainRef::new(n, q);
        let bfv = BFV::new(&domain, t);
        let mut rng = rng();
        let keys = bfv.keygen(&mut rng);
        (domain, bfv, keys, rng)
    }

    #[test]
    fn test_bfv_zero_polynomial() {
        let (domain, bfv, keys, mut rng) = setup_bfv_and_keys();
        let m_zero = Poly::zero(&domain);
        let ct = bfv.encrypt(&m_zero, &keys.public, &mut rng);
        let decrypted = bfv.decrypt(&ct, &keys.secret);
        assert_eq!(decrypted.coeffs(), m_zero.coeffs());
    }

    #[test]
    fn test_bfv_constant_polynomial() {
        let (domain, bfv, keys, mut rng) = setup_bfv_and_keys();
        let n = domain.n();
        let m_const = Poly::from_coeffs(&domain, vec![5i64; n]).expect("len matches");
        let ct = bfv.encrypt(&m_const, &keys.public, &mut rng);
        let decrypted = bfv.decrypt(&ct, &keys.secret);
        assert_eq!(decrypted.coeffs(), m_const.coeffs());
    }

    #[test]
    fn test_bfv_random_polynomial() {
        let (domain, bfv, keys, mut rng) = setup_bfv_and_keys();
        let n = domain.n();
        let t = bfv.t;
        let coeffs: Vec<i64> = (0..n).map(|_| rng.random_range(0..t) as i64).collect();
        let m_random = Poly::from_coeffs(&domain, coeffs).expect("len matches");
        let ct = bfv.encrypt(&m_random, &keys.public, &mut rng);
        let decrypted = bfv.decrypt(&ct, &keys.secret);
        assert_eq!(decrypted.coeffs(), m_random.coeffs());
    }

    #[test]
    fn test_bfv_single_coefficient() {
        let (domain, bfv, keys, mut rng) = setup_bfv_and_keys();
        let n = domain.n();
        let mut coeffs = vec![0i64; n];
        coeffs[0] = 10;
        let m_single = Poly::from_coeffs(&domain, coeffs).expect("len matches");
        let ct = bfv.encrypt(&m_single, &keys.public, &mut rng);
        let decrypted = bfv.decrypt(&ct, &keys.secret);
        assert_eq!(decrypted.coeffs(), m_single.coeffs());
    }

    #[test]
    fn test_bfv_add() {
        let (domain, bfv, keys, mut rng) = setup_bfv_and_keys();
        let n = domain.n();
        let t = bfv.t;
        let coeffs1: Vec<i64> = (0..n).map(|_| rng.random_range(0..t) as i64).collect();
        let m1 = Poly::from_coeffs(&domain, coeffs1).expect("len matches");

        let coeffs2: Vec<i64> = (0..n).map(|_| rng.random_range(0..t) as i64).collect();
        let m2 = Poly::from_coeffs(&domain, coeffs2).expect("len matches");

        let ct1 = bfv.encrypt(&m1, &keys.public, &mut rng);
        let ct2 = bfv.encrypt(&m2, &keys.public, &mut rng);
        let added = bfv.add(&ct1, &ct2);

        let decrypted = bfv.decrypt(&added, &keys.secret);

        assert_eq!(
            decrypted.coeffs(),
            (&m1 + &m2).reduce_mod(t as i64).coeffs()
        );
    }
}
