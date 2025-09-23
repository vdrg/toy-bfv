use super::Poly;

/// By-value wrapper around `&Poly` to drive integer/centered arithmetic.
#[derive(Copy, Clone)]
pub struct ZRef<'a> {
    pub(crate) p: &'a Poly,
}
impl<'a> ZRef<'a> {
    #[inline]
    pub fn new(p: &'a Poly) -> Self {
        ZRef { p }
    }
}

/// Integer (centered) negacyclic convolution result.
#[derive(Clone, Debug)]
pub struct ZExpr {
    pub(crate) coeffs: Vec<i128>,
}

// ZRef * ZRef -> ZExpr  (integer conv on centered reps)
impl<'a, 'b> std::ops::Mul<ZRef<'b>> for ZRef<'a> {
    type Output = ZExpr;
    fn mul(self, rhs: ZRef<'b>) -> ZExpr {
        ZExpr {
            coeffs: self.p.negacyclic_convolve_centered(rhs.p),
        }
    }
}

// ZExpr + ZExpr, unary -
impl std::ops::Add for ZExpr {
    type Output = ZExpr;
    fn add(mut self, rhs: ZExpr) -> ZExpr {
        assert_eq!(self.coeffs.len(), rhs.coeffs.len(), "ZExpr len mismatch");
        for (a, b) in self.coeffs.iter_mut().zip(rhs.coeffs.into_iter()) {
            *a += b;
        }
        self
    }
}
impl std::ops::Neg for ZExpr {
    type Output = ZExpr;
    fn neg(mut self) -> ZExpr {
        for v in &mut self.coeffs {
            *v = -*v;
        }
        self
    }
}

// Finishers
impl ZExpr {
    #[inline]
    pub fn from_poly(p: &Poly) -> Self {
        Self {
            coeffs: p.coeffs_centered_i128(),
        }
    }
    #[inline]
    pub fn round_scale(mut self, num: u64, den: u64) -> ZExpr {
        self.coeffs = Poly::scale_round_raw(&self.coeffs, num, den);
        self
    }
    #[inline]
    pub fn mod_q(self, q: u64) -> Poly {
        Poly::from_i128_coeffs(q, &self.coeffs)
    }
    #[inline]
    pub fn round_scale_mod_q(self, num: u64, den: u64, q: u64) -> Poly {
        let s = Poly::scale_round_raw(&self.coeffs, num, den);
        Poly::from_i128_coeffs(q, &s)
    }
}

// ----- In-module, path-importable macro -----

// Simple, robust form: wrap a single Poly expression and return a ZRef.
// Use as: z!(&a) * z!(&b) + z!(&c) * z!(&d)
#[macro_export]
macro_rules! z {
    ( $e:expr ) => {
        $crate::poly::dsl::ZRef::new(&$e)
    };
}

// Normalize an operand to `&Poly`: strip leading `&` and parentheses
// (Internal helpers removed; not needed with the simple wrapper form.)

// Re-export macros so callers can `use crate::poly::dsl::*;` and call `z!(...)` directly.
pub use crate::z;
