use std::sync::Arc;

use rand::Rng;

use crate::poly::{Domain, Poly};

pub struct BFV {
    ctx: Arc<Domain>,
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
    pub fn new(n: usize, q: u64, t: u64) -> Self {
        let ctx = Arc::new(Domain::new(n, q));
        Self {
            ctx,
            t,
            delta: q / t,
        }
    }

    pub fn keygen<R: Rng>(&self, rng: &mut R) -> Keys {
        let s = self.ctx.sample_ternary(rng);
        let e = self.ctx.sample_ternary(rng);
        let a = self.ctx.sample_uniform_mod_q(rng);

        // Standard RLWE: b = -a*s + e (mod q) so decryption cancels a*s*u.
        let as_prod = &a * &s;
        let neg_as = -&as_prod;
        let b = &neg_as + &e;

        Keys {
            secret: s,
            public: (b, a),
        }
    }

    pub fn encrypt<R: Rng>(&self, m: &Poly, pk: &PublicKey, rng: &mut R) -> Ciphertext {
        let u = self.ctx.sample_ternary(rng);
        let e1 = self.ctx.sample_ternary(rng);
        let e2 = self.ctx.sample_ternary(rng);

        let bu = &pk.0 * &u;
        let tmp = &bu + &e1;
        let m_scaled = m * (self.delta as i64);
        let c0 = &tmp + &m_scaled;

        let au = &pk.1 * &u;
        let c1 = &au + &e2;

        (c0, c1)
    }

    pub fn decrypt(&self, ct: &Ciphertext, sk: &SecretKey) -> Poly {
        let c1s = &ct.1 * sk;
        let v = &ct.0 + &c1s;
        let w = v.scale_and_round(self.t as i64, self.ctx.q() as i64);
        w.mod_small(self.t as i64)
    }
}

#[cfg(test)]
mod tests {
    use super::{BFV, Keys};
    use crate::poly::Poly;
    use rand::Rng;
    use rand::rng;
    use std::sync::Arc;

    fn setup_bfv_and_keys() -> (BFV, Keys, rand::rngs::ThreadRng) {
        let n = 8; // small power-of-two for tests
        let q = 12289; // typical toy prime
        let t = 17; // plaintext modulus
        let bfv = BFV::new(n, q, t);
        let mut rng = rng();
        let keys = bfv.keygen(&mut rng);
        (bfv, keys, rng)
    }

    #[test]
    fn test_bfv_zero_polynomial() {
        let (bfv, keys, mut rng) = setup_bfv_and_keys();
        let ctx: Arc<_> = Arc::clone(keys.public.0.domain());
        let m_zero = Poly::zero(&ctx);
        let ct = bfv.encrypt(&m_zero, &keys.public, &mut rng);
        let decrypted = bfv.decrypt(&ct, &keys.secret);
        assert_eq!(decrypted.coeffs, m_zero.coeffs);
    }

    #[test]
    fn test_bfv_constant_polynomial() {
        let (bfv, keys, mut rng) = setup_bfv_and_keys();
        let ctx: Arc<_> = Arc::clone(keys.public.0.domain());
        let n = ctx.n();
        let m_const = Poly::from_coeffs(Arc::clone(&ctx), vec![5i64; n]);
        let ct = bfv.encrypt(&m_const, &keys.public, &mut rng);
        let decrypted = bfv.decrypt(&ct, &keys.secret);
        assert_eq!(decrypted.coeffs, m_const.coeffs);
    }

    #[test]
    fn test_bfv_random_polynomial() {
        let (bfv, keys, mut rng) = setup_bfv_and_keys();
        let ctx: Arc<_> = Arc::clone(keys.public.0.domain());
        let n = ctx.n();
        let t = bfv.t;
        let coeffs = (0..n).map(|_| rng.random_range(0..t) as i64).collect();
        let m_random = Poly::from_coeffs(Arc::clone(&ctx), coeffs);
        let ct = bfv.encrypt(&m_random, &keys.public, &mut rng);
        let decrypted = bfv.decrypt(&ct, &keys.secret);
        assert_eq!(decrypted.coeffs, m_random.coeffs);
    }

    #[test]
    fn test_bfv_single_coefficient() {
        let (bfv, keys, mut rng) = setup_bfv_and_keys();
        let ctx: Arc<_> = Arc::clone(keys.public.0.domain());
        let n = ctx.n();
        let mut coeffs = vec![0i64; n];
        coeffs[0] = 10;
        let m_single = Poly::from_coeffs(Arc::clone(&ctx), coeffs);
        let ct = bfv.encrypt(&m_single, &keys.public, &mut rng);
        let decrypted = bfv.decrypt(&ct, &keys.secret);
        assert_eq!(decrypted.coeffs, m_single.coeffs);
    }
}
