use crate::poly::RingPoly;

#[derive(Clone, Debug)]
pub struct Noise {
    bound: f64,
}

impl Noise {
    pub fn zero() -> Self {
        Self { bound: 0.0 }
    }
    pub fn new(bound: f64) -> Self {
        Self { bound }
    }
    #[inline]
    pub fn bound(&self) -> f64 {
        self.bound
    }
}

#[derive(Clone, Debug)]
pub struct RLWECiphertext {
    c0: RingPoly,
    c1: RingPoly,
    noise: Noise,
}

impl RLWECiphertext {
    pub fn new(c0: RingPoly, c1: RingPoly) -> Self {
        Self {
            c0,
            c1,
            noise: Noise::zero(),
        }
    }

    pub fn zero(q: u64, n: usize) -> Self {
        RLWECiphertext::new(RingPoly::zero(q, n), RingPoly::zero(q, n))
    }

    pub fn with_noise(mut self, noise: Noise) -> Self {
        self.noise = noise;
        self
    }
    #[inline]
    pub fn parts(&self) -> (&RingPoly, &RingPoly) {
        (&self.c0, &self.c1)
    }
    #[inline]
    pub fn noise(&self) -> &Noise {
        &self.noise
    }

    /// Change ciphertext modulus by centered reduction (both components).
    pub fn mod_q_centered(&self, new_q: u64) -> RLWECiphertext {
        RLWECiphertext::new(
            self.parts().0.mod_q_centered(new_q),
            self.parts().1.mod_q_centered(new_q),
        )
    }

    /// Multiply a ciphertext by a public scalar k (mod q).
    #[inline]
    pub fn mul_scalar(&self, k: u64) -> RLWECiphertext {
        RLWECiphertext::new(&self[0] * k, &self[1] * k)
    }

    #[inline]
    pub fn neg(&self) -> RLWECiphertext {
        RLWECiphertext::new(-&self[0], -&self[1])
    }
}

impl std::ops::Index<usize> for RLWECiphertext {
    type Output = RingPoly;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.c0,
            1 => &self.c1,
            _ => panic!(
                "RLWECiphertext index out of range: {} (expected 0 or 1)",
                index
            ),
        }
    }
}

impl std::ops::IndexMut<usize> for RLWECiphertext {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.c0,
            1 => &mut self.c1,
            _ => panic!(
                "RLWECiphertext index out of range: {} (expected 0 or 1)",
                index
            ),
        }
    }
}

// TODO: implement std::ops ?
