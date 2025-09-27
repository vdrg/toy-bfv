use super::RingPoly;

#[derive(Copy, Clone)]
pub struct ZView<'a>(pub &'a RingPoly);

#[derive(Clone, Debug)]
pub struct ZExpr {
    pub(crate) coeffs: Vec<i128>,
}

impl RingPoly {
    #[inline]
    pub fn z(&self) -> ZView<'_> {
        ZView(self)
    }
}

impl<'a, 'b> std::ops::Mul<ZView<'b>> for ZView<'a> {
    type Output = ZExpr;
    fn mul(self, rhs: ZView<'b>) -> ZExpr {
        assert_eq!(self.0.q(), rhs.0.q(), "mod mismatch");
        assert_eq!(self.0.len(), rhs.0.len(), "len mismatch");
        ZExpr {
            coeffs: self.0.negacyclic_convolve_centered(rhs.0),
        }
    }
}

impl std::ops::Add for ZExpr {
    type Output = ZExpr;
    fn add(mut self, rhs: ZExpr) -> ZExpr {
        assert_eq!(self.coeffs.len(), rhs.coeffs.len(), "len mismatch");
        for (a, b) in self.coeffs.iter_mut().zip(rhs.coeffs.into_iter()) {
            *a += b;
        }
        self
    }
}

impl std::ops::Neg for ZExpr {
    type Output = ZExpr;
    fn neg(mut self) -> ZExpr {
        for a in &mut self.coeffs {
            *a = -*a;
        }
        self
    }
}

impl ZExpr {
    #[inline]
    pub fn round_scale(mut self, num: u64, den: u64) -> ZExpr {
        self.coeffs = RingPoly::scale_round_raw(&self.coeffs, num, den);
        self
    }
    #[inline]
    pub fn mod_q(self, q: u64) -> RingPoly {
        RingPoly::from_i128_coeffs(q, &self.coeffs)
    }
}
