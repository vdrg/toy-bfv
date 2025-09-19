use rand::rng;
use toy_bfv::bfv::BFV;

fn main() {
    let n = 1024;
    let q = 32767; // q must be odd
    let t = 256;
    let bfv = BFV::new(n, q, t);
    let mut rng = rng();
    let keys = bfv.keygen(&mut rng);
}
