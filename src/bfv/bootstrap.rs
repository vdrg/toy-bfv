use rand::Rng;

use super::{BFV, Ciphertext, RelinearizationKey};
use crate::{
    bfv::PublicKey,
    poly::{Poly, xk},
};

// SR ciphertext bits (each bit is a ciphertext under t=2)
type BitRow = Vec<Ciphertext>;

/// Bootstrapping key: encryptions of secret coefficients s_j ∈ {0,1}
pub type BootKey = Vec<Ciphertext>; // bsk[j] = Enc( s_j · x^0 )

#[inline]
fn bitlen_nonzero(x: u64) -> usize {
    (64 - x.leading_zeros()) as usize
}

impl BFV {
    pub fn bootstrap<R: Rng>(
        &self,
        ct: &Ciphertext,
        bk: &BootKey,
        rlk: &RelinearizationKey,
        pk: &PublicKey,
        rng: &mut R,
        sr_bits: Option<usize>,
    ) -> Ciphertext {
        let n = self.n();

        // accumulator
        let mut acc = Ciphertext::zero(self.q(), n);

        for k in 0..n {
            // Rotate input by multiplying by monomial,
            // so original coeff k becomes constant term
            let ct_rot = Ciphertext::new(&ct[0] * xk(n - k), &ct[1] * xk(n - k));
            // Run constant-slot bootstrap
            let bit0 = self.bootstrap_constant_coeff(&ct_rot, bk, rlk, pk, rng, sr_bits);
            // Rotate result back into position k
            let bit_at_k = Ciphertext::new(&bit0[0] * xk(k), &bit0[1] * xk(k));
            // Accumulate
            acc = self.add(&acc, &bit_at_k);
        }
        acc
    }

    // Constant-slot bootstrap: returns Enc(bit at coefficient 0).
    pub fn bootstrap_constant_coeff<R: Rng>(
        &self,
        ct: &Ciphertext,
        bk: &BootKey,
        rlk: &RelinearizationKey,
        pk: &PublicKey,
        rng: &mut R,
        sr_bits: Option<usize>,
    ) -> Ciphertext {
        // SR: for t=2, SR≥2 (rounding bit and one below).
        let sr = sr_bits.unwrap_or(2);
        assert!(sr >= 2, "t=2 requires SR ≥ 2");

        let n_bits_q = bitlen_nonzero(self.q() - 1);

        // 1) rows
        let rows = build_rows_for_constant(self, ct, bk, sr, n_bits_q, pk, rng);
        // 2) reduce with CSA
        let (a, b) = csa_tree(self, rows, rlk);
        // 3) ripple add
        let (w_bits, _cout) = ripple_add(self, &a, &b, rlk);
        // 4) rounding rule for t=2: result = bit(rb) XOR bit(rb-1)
        let (rb, below) = rounding_indices_t2(self.q(), sr);
        self.add(&w_bits[rb], &w_bits[below])
    }

    /// Enc(0) using pk (fresh randomness).
    fn enc_zero<R: Rng>(&self, pk: &PublicKey, rng: &mut R) -> Ciphertext {
        let pt0 = Poly::zero(self.t(), self.n());
        self.encrypt(&pt0, pk, rng)
    }

    /// Add a **plaintext bit at coefficient 0** (t=2) into a ciphertext: XOR on the constant slot.
    fn add_plain_bit_at_const(&self, ct: &Ciphertext, bit: u64) -> Ciphertext {
        if (bit & 1) == 0 {
            return ct.clone();
        }
        // e0 = [1,0,0,...]; lift to R_q centered, then scale by Δ and add to c0.
        let mut e0 = vec![0u64; self.n()];
        e0[0] = 1;
        let one_t = Poly::from_coeffs(self.t(), e0);
        let one_q = one_t.mod_q_centered(self.q()) * self.delta;
        let c0 = &ct[0] + &one_q;
        Ciphertext::new(c0, ct[1].clone())
    }

    /// bit · CT (t=2): AND gate with a *plaintext* bit (0 or 1). If bit==0 return Enc(0), else ct.
    fn gate_plain_bit<R: Rng>(
        &self,
        ct: &Ciphertext,
        bit: u64,
        pk: &PublicKey,
        rng: &mut R,
    ) -> Ciphertext {
        if (bit & 1) == 1 {
            ct.clone()
        } else {
            self.enc_zero(pk, rng)
        }
    }

    /// Generate bootstrapping key bsk[j] = Enc(s_j at coefficient 0).
    pub fn boot_keygen<R: Rng>(&self, sk: &Poly, pk: &PublicKey, rng: &mut R) -> BootKey {
        let n = self.n();
        let mut out = Vec::with_capacity(n);
        for &sj in sk.coeffs() {
            let bit = sj & 1;
            // Plaintext with bit only at index 0
            let mut ptv = vec![0u64; n];
            ptv[0] = bit;
            let pt = Poly::from_coeffs(self.t(), ptv);
            out.push(self.encrypt(&pt, pk, rng));
        }

        out
    }
}

/// Extract top SR bits of value v, where n_bits = bitlen(q-1). Bit 0 is LSB within the top window.
#[inline]
fn top_sr_bits_u64(v: u64, n_bits: usize, sr: usize) -> u64 {
    let shift = n_bits.saturating_sub(sr);
    (v >> shift) & ((1u64 << sr) - 1)
}

/// Rows for constant-slot sum:
///   Row 0: top SR bits of const(c0) as Enc(bits).
///   Rows j: Enc(s_j) AND top SR bits of const(x^j * c1).
fn build_rows_for_constant<R: Rng>(
    bfv: &BFV,
    ct: &Ciphertext,
    bk: &BootKey,
    sr: usize,
    n_bits_q: usize,
    pk: &PublicKey,
    rng: &mut R,
) -> Vec<BitRow> {
    let n = bfv.n();
    let mut rows = Vec::with_capacity(1 + n);

    // Row 0 from c0's constant term
    let d0_const = ct[0].rotate(0).const_term();
    let d0_bits = top_sr_bits_u64(d0_const, n_bits_q, sr);

    let mut r0 = Vec::with_capacity(sr);
    for b in 0..sr {
        let ct0 = bfv.enc_zero(pk, rng);
        r0.push(bfv.add_plain_bit_at_const(&ct0, (d0_bits >> b) & 1));
    }
    rows.push(r0);

    // Rows j from c1 and Enc(s_j)
    for j in 0..n {
        let wj_const = ct[1].rotate(j).const_term();
        let wj_bits = top_sr_bits_u64(wj_const, n_bits_q, sr);
        let s_j = &bk[j];

        let mut rj = Vec::with_capacity(sr);
        for b in 0..sr {
            let bit = (wj_bits >> b) & 1;
            // (s_j AND bit) as ciphertext: bit ? Enc(s_j) : Enc(0)
            rj.push(bfv.gate_plain_bit(s_j, bit, pk, rng));
        }
        rows.push(rj);
    }
    rows
}

fn csa(
    bfv: &BFV,
    a: &BitRow,
    b: &BitRow,
    c: &BitRow,
    rlk: &RelinearizationKey,
) -> (BitRow, BitRow) {
    let sr = a.len();
    assert_eq!(sr, b.len());
    assert_eq!(sr, c.len());
    let mut sum = Vec::with_capacity(sr);
    let mut carry = Vec::with_capacity(sr);

    for i in 0..sr {
        let s_ab = bfv.add(&a[i], &b[i]); // a ^ b
        let s_abc = bfv.add(&s_ab, &c[i]); // (a ^ b) ^ c
        let ab = bfv.mul(&a[i], &b[i], rlk); // a & b
        let bc = bfv.mul(&b[i], &c[i], rlk); // b & c
        let ac = bfv.mul(&a[i], &c[i], rlk); // a & c
        let ab_xor_bc = bfv.add(&ab, &bc);
        let maj = bfv.add(&ab_xor_bc, &ac); // majority(a,b,c)
        sum.push(s_abc);
        carry.push(maj);
    }
    // shift carry left by 1 (insert Enc(0) as LSB)
    let enc0 = Ciphertext::zero(bfv.q(), bfv.n());
    carry.insert(0, enc0);
    carry.pop();
    (sum, carry)
}

fn csa_tree(bfv: &BFV, mut rows: Vec<BitRow>, rlk: &RelinearizationKey) -> (BitRow, BitRow) {
    if rows.len() == 1 {
        let z = Ciphertext::new(Poly::zero(bfv.q(), bfv.n()), Poly::zero(bfv.q(), bfv.n()));
        let zero = vec![z; rows[0].len()];
        rows.push(zero);
    }
    while rows.len() > 2 {
        let a = rows.remove(0);
        let b = rows.remove(0);
        let c = rows.remove(0);
        let (s, carry) = csa(bfv, &a, &b, &c, rlk);
        rows.push(s);
        rows.push(carry);
    }
    (rows.remove(0), rows.remove(0))
}

fn ripple_add(
    bfv: &BFV,
    a: &BitRow,
    b: &BitRow,
    rlk: &RelinearizationKey,
) -> (Vec<Ciphertext>, Ciphertext) {
    let sr = a.len();
    assert_eq!(sr, b.len());
    let mut out = Vec::with_capacity(sr);
    let mut carry = Ciphertext::new(Poly::zero(bfv.q(), bfv.n()), Poly::zero(bfv.q(), bfv.n()));
    for i in 0..sr {
        let ab = bfv.add(&a[i], &b[i]);
        let sum = bfv.add(&ab, &carry);
        let ab_and = bfv.mul(&a[i], &b[i], rlk);
        let bc_and = bfv.mul(&b[i], &carry, rlk);
        let ac_and = bfv.mul(&a[i], &carry, rlk);
        let tmp = bfv.add(&ab_and, &bc_and);
        carry = bfv.add(&tmp, &ac_and);
        out.push(sum);
    }
    (out, carry)
}

fn rounding_indices_t2(q: u64, sr: usize) -> (usize /*rb*/, usize /*below*/) {
    // q = 2^n => bitlen(q-1) = n. For t=2, Δ = 2^(n-1), rounding bit index = k - n + SR - 1 with k=n-1 => rb = SR-2.
    let n = bitlen_nonzero(q - 1);
    let k = n - 1;
    let rb = (k as isize - n as isize + sr as isize - 1) as usize; // = sr - 2
    println!("n {:?}", n);
    println!("k {:?}", k);
    println!("rb {:?}", rb);
    let below = rb.checked_sub(1).expect("SR must be ≥ 2");
    (rb, below)
}

#[cfg(test)]
mod tests {
    use crate::bfv::{BFV, BfvParameters, Keys};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn testing_params() -> BfvParameters {
        let n: usize = 16;
        let t: u64 = 2;
        let q: u64 = 2 << 9;
        let p: u64 = 2_147_483_647;

        BfvParameters::new(n, q, t, p, 3.2, 100.0)
    }

    fn setup_bfv_and_keys() -> (BFV, Keys, StdRng) {
        let params = testing_params();
        let bfv = BFV::new(params);
        let mut rng = StdRng::seed_from_u64(0);
        let keys = bfv.keygen(&mut rng);
        (bfv, keys, rng)
    }

    #[test]
    fn test_simple_bootstrapping() {
        let (bfv, keys, mut rng) = setup_bfv_and_keys();
        let m = bfv.params.rt.sample_uniform(&mut rng);
        let ct = bfv.encrypt(&m, &keys.public, &mut rng);
        let bct = bfv.bootstrap(&ct, &keys.boot, &keys.relin, &keys.public, &mut rng, None);
        let decrypted = bfv.decrypt(&bct, &keys.secret);
        assert_eq!(decrypted.coeffs(), m.coeffs());
    }
}
