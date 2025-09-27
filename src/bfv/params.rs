use crate::poly::domain::Domain;

/// Holds the cryptographic parameters for a BFV scheme instance.
/// This struct separates the configuration of the scheme from its implementation.
#[derive(Clone, Debug)]
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
    /// These parameters are based on the recommendations from the Homomorphic Encryption Sandard
    /// - n = 4096: The polynomial degree.
    /// - q = 1099511922689: A 40-bit prime ciphertext modulus (2^40 + 2^28 + 1).
    /// - t = 65537: A 17-bit prime plaintext modulus, allowing for a large message space.
    /// - relin_p = q^3
    /// - error_std_dev = 3.2: standard deviation for the error distribution.
    fn default() -> Self {
        let q: u64 = 1099511922689;
        // TODO: this will overflow!
        let p = q.pow(3);
        Self::new(4096, q, 65537, p, 7.0e10, 3.2)
    }
}
