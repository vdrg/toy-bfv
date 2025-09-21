use rand::rng;
use toy_bfv::bfv::{BFV, BfvParameters};

fn main() {
    let params = BfvParameters::default();
    let bfv = BFV::new(params);
    let mut rng = rng();
    let keys = bfv.keygen(&mut rng);
}
