use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::{SeedableRng, rngs::StdRng};
use toy_bfv::bfv::*; // adjust path

fn bench_bfv(c: &mut Criterion) {
    // Tiny toy params; for deeper multiplications increase q (see discussion)
    let q: u64 = 12289;
    let t: u64 = 17;
    let p: u64 = (1u64 << 30) + 1;
    let params = BfvParameters::new(1 << 8, q, t, p, 0.0, 0.0); // n=256 example
    let bfv = BFV::new(params);

    let mut rng = StdRng::seed_from_u64(0);
    let keys = bfv.keygen(&mut rng);

    let m1 = bfv.params.rt.sample_uniform(&mut rng);
    let m2 = bfv.params.rt.sample_uniform(&mut rng);

    c.bench_function("encrypt", |b| {
        b.iter(|| {
            let ct = bfv.encrypt(black_box(&m1), &keys.public, &mut rng);
            black_box(ct);
        })
    });

    let ct1 = bfv.encrypt(&m1, &keys.public, &mut rng);
    let ct2 = bfv.encrypt(&m2, &keys.public, &mut rng);

    c.bench_function("add", |b| b.iter(|| black_box(bfv.add(&ct1, &ct2))));
    c.bench_function("mul", |b| {
        b.iter(|| black_box(bfv.mul(&ct1, &ct2, &keys.relin)))
    });
}

criterion_group!(benches, bench_bfv);
criterion_main!(benches);
