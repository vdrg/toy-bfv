use rand::rng;
use toy_bfv::bfv::BFV;
use toy_bfv::poly::DomainRef;

fn main() {
    let n = 1024;
    let q = 32767;
    let t = 256;
    let domain = DomainRef::new(n, q);
    let bfv = BFV::new(&domain, t);
    let mut rng = rng();
    let keys = bfv.keygen(&mut rng);
}
