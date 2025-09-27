use std::borrow::Cow;

pub mod ring;
pub mod scalar;

pub use self::ring::*;
pub use self::scalar::*;

pub trait PolyLike {
    type CoeffType: Copy + PartialEq + From<u8>;

    fn len(&self) -> usize;
    fn coeff(&self, i: usize) -> Self::CoeffType;
    fn coeffs(&self) -> Cow<'_, [Self::CoeffType]>;

    #[inline]
    fn const_term(&self) -> Self::CoeffType {
        self.coeff(0)
    }
    #[inline]
    fn degree(&self) -> Option<usize> {
        let v = self.coeffs();
        v.iter().rposition(|&c| c != Self::CoeffType::from(0))
    }
}

pub trait EvalAt<Arg> {
    type Output;
    fn eval_at(&self, x: &Arg) -> Self::Output;
}
